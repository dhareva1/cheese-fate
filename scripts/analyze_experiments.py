"""Analyse the cross-validated experiment outputs: metrics, RQ1-RQ6 tables, figures.

Reads ``reports/experiments/<group>/`` (written by ``run_experiments.py``) and writes CSV
tables, figures and ``summary.md`` next to them.

Example (from the repository root):
    python scripts/analyze_experiments.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cheese_fate import analysis, figures
from cheese_fate.combase import load_exports
from cheese_fate.experiment import MODELS
from cheese_fate.tasks import MANUFACTURE_END_H, build_records

ROOT = Path(__file__).resolve().parents[1]
ORGANISMS = ("Escherichia coli", "Staphylococcus aureus")
ALPHA = 0.1


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def markdown(frame: pd.DataFrame, digits: int = 2) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if math.isnan(value) else ("∞" if math.isinf(value) else f"{value:.{digits}f}")
        return str(value)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def analyse(organism: str, folder: Path, export) -> dict:
    name = slug(organism)
    predictions = pd.read_csv(folder / f"{name}_predictions.csv.gz")
    dense = pd.read_csv(folder / f"{name}_dense.csv.gz")
    shap = pd.read_csv(folder / f"{name}_shap.csv.gz")
    forecasts = pd.read_csv(folder / f"{name}_forecaster.csv.gz")
    out = {"organism": organism}

    summary = analysis.summary_table(predictions)
    summary.to_csv(folder / f"{name}_summary.csv", index=False)
    analysis.stage_table(predictions).to_csv(folder / f"{name}_stages.csv", index=False)
    gap = analysis.forecast_gap(predictions)
    gap.to_csv(folder / f"{name}_forecast_gap.csv", index=False)
    lab = analysis.lab_effect(predictions)
    lab.to_csv(folder / f"{name}_lab_effect.csv", index=False)
    accuracy = analysis.forecaster_accuracy(forecasts)
    accuracy.to_csv(folder / f"{name}_forecaster_accuracy.csv", index=False)
    out.update(summary=summary, gap=gap, lab=lab, forecaster=accuracy)

    measured = summary[(summary["mode"] == "measured") & (summary["origin_h"] == 0)]
    candidates = measured[measured["model"] != "persistence"]
    best = candidates.loc[candidates["rmse"].idxmin(), "model"]
    out["best"] = best

    # RQ4: deacidification test on out-of-fold SHAP values.
    observations = pd.read_csv(ROOT / f"data/processed/cheese/observations_{name}.csv")
    records, _ = build_records(observations, export)
    rebounds = analysis.rebound_table(records)
    tests, per_record = {}, {}
    for model in sorted(shap["model"].unique()):
        attribution = analysis.ph_attribution(shap, rebounds, model)
        attribution.to_csv(folder / f"{name}_ph_attribution_{model}.csv", index=False)
        tests[model] = analysis.rebound_test(attribution)
        per_record[model] = attribution
    out.update(rebound_tests=tests, attribution=per_record)

    # RQ5: cross-conformal bands, coverage per model, time-to-safe for the best model.
    origin0 = predictions[predictions["origin_h"] == 0]
    coverage, bands_of = [], {}
    for (model, mode), rows in origin0.groupby(["model", "mode"]):
        bands = analysis.cross_conformal(rows, ALPHA)
        bands_of[model, mode] = bands
        coverage.append({"model": model, "mode": mode, "records": len(bands),
                         "coverage": bands["covered"].mean(), "q_mean": bands["q"].mean()})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(folder / f"{name}_conformal.csv", index=False)
    # Time-to-safe for the best model with two bands: the pre-registered two-sided
    # whole-curve band, and a one-sided upper band over ripening (from MANUFACTURE_END_H),
    # the guarantee a safety certificate actually needs.
    safe, ripening = [], []
    for mode in ("measured", "coupled"):
        key = (best, mode)
        if key not in bands_of:
            continue
        rows = origin0[(origin0["model"] == best) & (origin0["mode"] == mode)]
        curve = dense[(dense["model"] == best) & (dense["mode"] == mode)]
        upper = analysis.cross_conformal(rows, ALPHA, one_sided=True, after_h=MANUFACTURE_END_H)
        ripening.append({"model": best, "mode": mode, "records": len(upper),
                         "coverage": upper["covered"].mean(), "q_mean": upper["q"].mean()})
        for band, bands in (("two-sided", bands_of[key]), ("upper, ripening", upper)):
            after = MANUFACTURE_END_H if band.startswith("upper") else None
            table = analysis.time_to_safe(curve, rows, bands, after_h=after)
            safe.append(table.assign(model=best, mode=mode, band=band))
    ripening = pd.DataFrame(ripening)
    ripening.to_csv(folder / f"{name}_conformal_upper_ripening.csv", index=False)
    out["ripening"] = ripening
    safe = pd.concat(safe, ignore_index=True)
    safe.to_csv(folder / f"{name}_time_to_safe.csv", index=False)
    out.update(coverage=coverage, safe=safe, bands=bands_of, dense=dense, origin0=origin0)
    return out


def safe_summary(safe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (band, mode, target), g in safe.groupby(["band", "mode", "target"]):
        reached = np.isfinite(g["predicted_safe_h"])
        observed = np.isfinite(g["observed_safe_h"])
        past = g[g["observed_past_60d"] & g["observed_safe_from_60d"].notna()]
        certified = past[past["model_safe_at_60d"] == True]  # noqa: E712
        rows.append({
            "band": band, "mode": mode, "target": target, "curves": len(g),
            "observed_reached": observed.mean(),
            "certified": reached.mean(),
            "median_certified_days": float(np.median(g.loc[reached, "predicted_safe_h"])) / 24
            if reached.any() else np.nan,
            "certified_confirmed": g.loc[reached, "confirmed"].astype(float).mean()
            if reached.any() else np.nan,
            "past_60d": len(past),
            "safe_from_60d_observed": past["observed_safe_from_60d"].astype(float).mean()
            if len(past) else np.nan,
            "certified_at_60d": (past["model_safe_at_60d"] == True).mean()  # noqa: E712
            if len(past) else np.nan,
            "certified_at_60d_but_unsafe": int((certified["observed_safe_from_60d"] == False).sum()),  # noqa: E712
        })
    return pd.DataFrame(rows)


def write_summary(results: list[dict], folder: Path, group: str) -> None:
    lines = [f"# Experiment results (folds grouped by `{group}`)", "",
             "Generated by `scripts/analyze_experiments.py`. RMSE etc. in log10 CFU/g on "
             "detected held-out counts; `skill` = 1 − RMSE/RMSE(persistence) on the same "
             "targets (positive beats persistence). Origin 0 is the curve start.", ""]
    for r in results:
        organism, summary = r["organism"], r["summary"]
        lines += [f"## *{organism}*", ""]
        table = summary[summary["origin_h"] == 0].pivot_table(
            index="model", columns="mode", values="rmse").reindex(
            [m for m in MODELS if m in summary["model"].unique()])
        skill = summary[(summary["origin_h"] == 0) & (summary["mode"] == "measured")].set_index("model")
        table = table.assign(
            sd_measured=skill["rmse_fold_sd"], skill_measured=skill["skill"],
            acceptable_zone=skill["acceptable_zone"], censored_agreement=skill["censored_agreement"],
        ).reset_index()
        lines += ["### RQ1 – accuracy by model and input mode (origin 0, pooled RMSE)", "",
                  markdown(table), "", f"Best non-reference model (measured inputs): **{r['best']}**.", ""]

        later = summary[(summary["origin_h"] > 0) & summary["model"].isin(
            ["persistence", "mechanistic", r["best"]])]
        later = later.pivot_table(index=["origin_h", "model"], columns="mode", values="rmse").reset_index()
        lines += ["### RQ2 – forecast origins day 7 and day 30 (pooled RMSE)", "", markdown(later), ""]
        gap = r["gap"]
        gap = gap[(gap["model"] == r["best"]) & (gap["origin_h"] == 0)].drop(columns=["model"])
        lines += [f"Gap by horizon, {r['best']}, origin 0:", "", markdown(gap), ""]
        fc = r["forecaster"]
        fc = fc[fc["horizon"] == "all"].drop(columns=["horizon"])
        lines += ["Coupled forecaster vs hold-last-value (RMSE in pH units, aw units, log10 LAB):",
                  "", markdown(fc, 3), ""]

        lines += ["### RQ3 – effect of same-batch LAB (RMSE with LAB − without; negative = LAB helps)",
                  "", markdown(r["lab"][r["lab"]["origin_h"] == 0], 3), ""]

        tests = pd.DataFrame([{"model": m, **t} for m, t in r["rebound_tests"].items()])
        lines += ["### RQ4 – deacidification (pre-registered SHAP test)", "",
                  "Curves with a measured pH rise > 0.3 after the pH minimum; attribution = summed "
                  "out-of-fold SHAP of all pH features after the minimum, mean per curve.", "",
                  markdown(tests, 3), ""]

        lines += ["### RQ5 – conformal coverage (90% target, whole-curve bands, origin 0)", "",
                  markdown(r["coverage"].sort_values(["mode", "coverage"]), 3), "",
                  f"One-sided upper band over ripening (≥ 48 h), {r['best']}:", "",
                  markdown(r["ripening"], 3), "",
                  f"Time-to-safe ({r['best']}); `certified` = share of curves whose upper band "
                  "reaches the target within the observed period:", "",
                  markdown(safe_summary(r["safe"]), 2), ""]
    (folder / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, default=ROOT / "data/raw/combase_cheese/exports")
    parser.add_argument("--experiments", type=Path, default=ROOT / "reports/experiments")
    parser.add_argument("--groups", nargs="+", default=["source_id", "food"])
    args = parser.parse_args()
    export = load_exports(args.exports)
    for group in args.groups:
        folder = args.experiments / group
        organisms = [o for o in ORGANISMS if (folder / f"{slug(o)}_predictions.csv.gz").exists()]
        if not organisms:
            continue
        results = [analyse(o, folder, export) for o in organisms]
        write_summary(results, folder, group)

        out = folder / "figures"
        order = [m for m in MODELS]
        figures.model_accuracy({r["organism"]: r["summary"] for r in results},
                               out / "model_accuracy.png", order)
        for r in results:
            figures.forecast_gap({r["organism"]: r["gap"]}, r["best"],
                                 out / f"forecast_gap_{slug(r['organism'])}.png")
        figures.rebound_scatter({r["organism"]: r["attribution"]["gbt"] for r in results
                                 if "gbt" in r["attribution"]}, "gbt",
                                analysis.REBOUND_THRESHOLD, out / "ph_rebound_gbt.png")
        figures.time_to_safe({r["organism"]: r["safe"][(r["safe"]["mode"] == "measured")
                                                      & (r["safe"]["band"] == "upper, ripening")]
                              for r in results}, out / "time_to_safe.png")
        for r in results:
            origin0 = r["origin0"]
            rows = origin0[(origin0["model"] == r["best"]) & (origin0["mode"] == "measured")]
            counts = rows.groupby("record_id").agg(n=("time_h", "size"), end=("time_h", "max"))
            chosen = counts[(counts["n"] >= 6) & (counts["end"] > 720)].sort_values("n").index
            chosen = list(chosen[np.linspace(0, len(chosen) - 1, min(6, len(chosen))).astype(int)])
            if chosen:
                figures.example_curves(r["dense"][r["dense"]["mode"] == "measured"], rows,
                                       r["bands"][r["best"], "measured"], chosen, r["best"],
                                       out / f"example_curves_{slug(r['organism'])}.png")
        (folder / "analysis.json").write_text(json.dumps(
            {r["organism"]: {"best": r["best"], "rebound_tests": r["rebound_tests"]}
             for r in results}, indent=2))
        print(f"{group}: wrote {folder / 'summary.md'}")


if __name__ == "__main__":
    main()
