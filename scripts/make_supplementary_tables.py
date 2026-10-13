"""Supplementary Tables S2-S6 for the cheese manuscript, as formatted CSV files.

Reads ``reports/experiments/<split>/`` (after ``run_inference.py`` and
``analyze_experiments.py``) and writes ``tableS2.csv`` ... ``tableS6.csv`` to
``reports/supplementary_tables/``. The cells are already formatted for the
supplementary Word file:

- S2: full validation, source split (all models, input modes and origins);
- S3: the same on held-out cheese types;
- S4: all paired comparisons with Holm-adjusted p-values (both splits);
- S5: RMSE by process stage and forecast horizon, per origin and input mode (source split);
- S6: split robustness at origin 0 (5-fold seeds 0-2 and leave-one-source-out);
- S7, S8: LAB-mapping sensitivity (verified links, Ec/Sa source excluded): RQ2/RQ3
  comparisons and skill at origin 0.

Example (from the repository root):
    python scripts/make_supplementary_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cheese_fate.analysis import HORIZON_LABELS, horizon_bin
from cheese_fate.evaluation import summarize_censored

ROOT = Path(__file__).resolve().parents[1]
ORGANISMS = {"escherichia_coli": "E. coli", "staphylococcus_aureus": "S. aureus"}
NAMES = {
    "persistence": "Persistence", "mechanistic": "Mechanistic", "gbt": "GBT",
    "gbt_nolab": "GBT, no LAB", "gru": "GRU", "lstm": "LSTM",
    "residual_gbt": "Residual hybrid", "residual_gbt_nolab": "Residual hybrid, no LAB",
    "node_hybrid": "Neural-ODE hybrid",
}
ORDER = list(NAMES)
MODES = {"measured": "Measured", "naive": "Naive", "coupled": "Coupled"}
SPLITS = {"source_id": "Sources", "food": "Cheese types"}
FAMILIES = {"RQ1 model vs model": "RQ1 models", "RQ2 forecast vs measured": "RQ2 input modes",
            "RQ3 LAB vs no LAB": "RQ3 LAB features"}
S5_MODELS = ["persistence", "mechanistic", "gbt", "gru", "lstm", "residual_gbt", "node_hybrid"]


def ci(value, lo, hi, digits=2) -> str:
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def pval(p) -> str:
    if pd.isna(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def num(value, digits=2) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def days(origin_h) -> str:
    return f"{origin_h / 24:g}"


def sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(_m=frame["model"].map(ORDER.index),
                         _o=frame["mode"].map(list(MODES).index))
    return frame.sort_values(["origin_h", "_o", "_m"]).drop(columns=["_m", "_o"])


def validation(folder: Path) -> pd.DataFrame:
    rows = []
    for slug, organism in ORGANISMS.items():
        table = pd.read_csv(folder / f"{slug}_validation.csv")
        censored = pd.read_csv(folder / f"{slug}_summary.csv").set_index(
            ["model", "mode", "origin_h"])["censored_agreement"]
        for r in sort_rows(table).itertuples(index=False):
            reference = r.model == "persistence"
            rows.append({
                "Organism": organism, "Model": NAMES[r.model], "Input mode": MODES[r.mode],
                "Origin (d)": days(r.origin_h), "n obs": int(r.n_detected),
                "RMSE [95% CI]": ci(r.rmse, r.rmse_lo, r.rmse_hi),
                "Skill [95% CI]": "0 (reference)" if reference else ci(r.skill, r.skill_lo, r.skill_hi),
                "p (skill > 0)": "–" if reference else pval(r.p_skill),
                "Bf": num(r.bf), "Af": num(r.af), "ASZ": num(r.asz), "APZ": num(r.apz),
                "Censored agreement": num(censored.get((r.model, r.mode, r.origin_h))),
            })
    return pd.DataFrame(rows)


def label(pair: str) -> tuple[str, str]:
    model, mode = pair.split("/")
    return NAMES[model], MODES[mode]


def comparisons(root: Path) -> pd.DataFrame:
    rows = []
    for slug, organism in ORGANISMS.items():
        for split, split_name in SPLITS.items():
            table = pd.read_csv(root / split / f"{slug}_comparisons.csv")
            table = table.assign(_f=table["family"].map(list(FAMILIES).index))
            for r in table.sort_values(["_f", "origin_h"], kind="stable").itertuples(index=False):
                (a, mode_a), (b, mode_b) = label(r.a), label(r.b)
                if r.family.startswith("RQ2"):
                    comparison, mode = a, f"{mode_a} − {mode_b}"
                else:
                    comparison, mode = f"{a} − {b}", mode_a
                rows.append({
                    "Organism": organism, "Split": split_name, "Family": FAMILIES[r.family],
                    "Origin (d)": days(r.origin_h), "Comparison": comparison, "Input mode": mode,
                    "ΔRMSE [95% CI]": ci(r.difference, r.difference_lo, r.difference_hi),
                    "p": pval(r.p_two_sided), "Holm-adjusted p": pval(r.p_holm),
                })
    return pd.DataFrame(rows)


def strata(predictions: pd.DataFrame) -> pd.DataFrame:
    """RMSE per model in every (mode, origin, stratum), the stratum being all, a stage or a horizon."""
    frame = predictions[predictions["model"].isin(S5_MODELS)].assign(
        horizon=lambda f: horizon_bin(f["horizon_h"]).astype(str))
    parts = [("All", frame.assign(stratum="All")),
             ("Stage", frame.assign(stratum=frame["stage"].str.capitalize())),
             ("Horizon", frame.assign(stratum=frame["horizon"] + ""))]
    rows = []
    for kind, part in parts:
        for (mode, origin, stratum, model), group in part.groupby(
                ["mode", "origin_h", "stratum", "model"], observed=True):
            metrics = summarize_censored(group)
            rows.append({"mode": mode, "origin_h": origin, "kind": kind, "stratum": stratum,
                         "model": model, "rmse": metrics["rmse"], "n": int(metrics["n_detected"])})
    return pd.DataFrame(rows)


def by_stage_and_horizon(root: Path) -> pd.DataFrame:
    rank = {"All": 0, "Manufacture": 1, "Ripening": 2,
            **{h: 3 + i for i, h in enumerate(HORIZON_LABELS)}}
    rows = []
    for slug, organism in ORGANISMS.items():
        predictions = pd.read_csv(root / "source_id" / f"{slug}_predictions.csv.gz")
        table = strata(predictions)
        wide = table.pivot_table(index=["mode", "origin_h", "kind", "stratum"], columns="model",
                                 values="rmse").reset_index()
        counts = table[table["model"] == "persistence"].set_index(
            ["mode", "origin_h", "kind", "stratum"])["n"]
        wide["n"] = counts.reindex(pd.MultiIndex.from_frame(
            wide[["mode", "origin_h", "kind", "stratum"]])).to_numpy()
        wide = wide[wide["n"] > 0]
        # A single stage (later origins are all ripening) would only repeat the "All" row.
        stages = wide[wide["kind"] == "Stage"].groupby(["mode", "origin_h"])["stratum"].transform("size")
        wide = wide.drop(stages[stages == 1].index)
        wide = wide.assign(_o=wide["mode"].map(list(MODES).index),
                           _s=wide["stratum"].map(rank))
        for r in wide.sort_values(["_o", "origin_h", "_s"]).to_dict("records"):
            rows.append({
                "Organism": organism, "Input mode": MODES[r["mode"]],
                "Origin (d)": days(r["origin_h"]),
                "Stratum": r["stratum"] if r["kind"] != "Horizon" else f"Horizon {r['stratum']}",
                "n obs": int(r["n"]),
                **{NAMES[m]: num(r.get(m)) for m in S5_MODELS},
            })
    return pd.DataFrame(rows)


def robustness(root: Path) -> pd.DataFrame:
    splits = {"source_id": "5-fold, seed 0 (primary)", "source_id_seed1": "5-fold, seed 1",
              "source_id_seed2": "5-fold, seed 2", "source_id_loso": "Leave-one-source-out"}
    rows = []
    for slug, organism in ORGANISMS.items():
        for folder, split_name in splits.items():
            run = json.loads((root / folder / f"{slug}_run.json").read_text())
            table = pd.read_csv(root / folder / f"{slug}_validation.csv")
            table = sort_rows(table[table["origin_h"] == 0])
            for r in table.itertuples(index=False):
                reference = r.model == "persistence"
                rows.append({
                    "Organism": organism, "Validation": split_name, "Model": NAMES[r.model],
                    "Input mode": MODES[r.mode], "Folds": f"{run['folds']} ({r.n_sources})",
                    "RMSE [95% CI]": ci(r.rmse, r.rmse_lo, r.rmse_hi),
                    "Skill [95% CI]": "0 (reference)" if reference else ci(r.skill, r.skill_lo, r.skill_hi),
                    "p (skill > 0)": "–" if reference else pval(r.p_skill),
                })
    return pd.DataFrame(rows)


VARIANTS = {"": "Primary", "_lablinks": "Verified LAB links", "_noEcSa": "Ec/Sa source excluded"}


def lab_sensitivity_comparisons(root: Path) -> pd.DataFrame:
    """RQ2 and RQ3 comparisons at origin 0, one column per LAB-mapping variant."""
    rows = []
    for slug, organism in ORGANISMS.items():
        for split, split_name in SPLITS.items():
            cells = {}
            for suffix, variant in VARIANTS.items():
                table = pd.read_csv(root / f"{split}{suffix}" / f"{slug}_comparisons.csv")
                table = table[(table["origin_h"] == 0) & ~table["family"].str.startswith("RQ1")]
                for r in table.itertuples(index=False):
                    cells.setdefault((r.family, r.a, r.b), {})[variant] = (
                        f"{ci(r.difference, r.difference_lo, r.difference_hi)}; {pval(r.p_holm)}")
            for (family, a, b), values in cells.items():
                (name_a, mode_a), (name_b, mode_b) = label(a), label(b)
                comparison = name_a if family.startswith("RQ2") else f"{name_a} − {name_b}"
                mode = f"{mode_a} − {mode_b}" if family.startswith("RQ2") else mode_a
                rows.append({"Organism": organism, "Split": split_name, "Family": FAMILIES[family],
                             "Comparison": comparison, "Input mode": mode,
                             **{v: values.get(v, "–") for v in VARIANTS.values()}})
    return pd.DataFrame(rows)


def lab_sensitivity_skill(root: Path) -> pd.DataFrame:
    """Skill at origin 0 per model and input mode, one column per LAB-mapping variant."""
    rows = []
    for slug, organism in ORGANISMS.items():
        for split, split_name in SPLITS.items():
            tables = {variant: sort_rows(pd.read_csv(root / f"{split}{suffix}" / f"{slug}_validation.csv"))
                      for suffix, variant in VARIANTS.items()}
            primary = tables["Primary"]
            primary = primary[(primary["origin_h"] == 0) & (primary["model"] != "persistence")]
            for r in primary.itertuples(index=False):
                cells = {}
                for variant, table in tables.items():
                    hit = table[(table["origin_h"] == 0) & (table["model"] == r.model)
                                & (table["mode"] == r.mode)].iloc[0]
                    cells[variant] = ci(hit.skill, hit.skill_lo, hit.skill_hi)
                rows.append({"Organism": organism, "Split": split_name, "Model": NAMES[r.model],
                             "Input mode": MODES[r.mode], **cells})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", type=Path, default=ROOT / "reports/experiments")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports/supplementary_tables")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = {
        "tableS2": validation(args.experiments / "source_id"),
        "tableS3": validation(args.experiments / "food"),
        "tableS4": comparisons(args.experiments),
        "tableS5": by_stage_and_horizon(args.experiments),
        "tableS6": robustness(args.experiments),
        "tableS7": lab_sensitivity_comparisons(args.experiments),
        "tableS8": lab_sensitivity_skill(args.experiments),
    }
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows", flush=True)


if __name__ == "__main__":
    main()
