"""Skill across split seeds and leave-one-source-out, from ``*_validation.csv`` tables.

Run ``run_inference.py --names source_id source_id_seed1 source_id_seed2 source_id_loso``
first. Writes ``reports/experiments/robustness.csv`` and ``robustness.md``.

Example (from the repository root):
    python scripts/summarize_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORGANISMS = {"escherichia_coli": "Escherichia coli",
             "staphylococcus_aureus": "Staphylococcus aureus"}
RUNS = {"source_id": "5-fold, seed 0", "source_id_seed1": "5-fold, seed 1",
        "source_id_seed2": "5-fold, seed 2", "source_id_loso": "leave-one-source-out"}
MODELS = ["mechanistic", "gbt", "gru", "lstm", "residual_gbt", "node_hybrid"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / "reports/experiments")
    args = parser.parse_args()

    rows = []
    for run, label in RUNS.items():
        for slug, organism in ORGANISMS.items():
            path = args.root / run / f"{slug}_validation.csv"
            if not path.exists():
                continue
            table = pd.read_csv(path).query("origin_h == 0 and mode in ('measured', 'naive')")
            for _, r in table[table["model"].isin(MODELS)].iterrows():
                rows.append({"organism": organism, "run": label, "model": r["model"],
                             "mode": r["mode"], "rmse": r["rmse"], "skill": r["skill"],
                             "skill_lo": r["skill_lo"], "skill_hi": r["skill_hi"],
                             "p_skill": r["p_skill"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.root / "robustness.csv", index=False)

    lines = ["# Robustness of skill across splits", "",
             "Skill at origin 0 with 95% source-bootstrap interval, from `run_inference.py`.", ""]
    for organism in ORGANISMS.values():
        for mode in ("measured", "naive"):
            part = frame[(frame["organism"] == organism) & (frame["mode"] == mode)]
            if part.empty:
                continue
            runs = [label for label in RUNS.values() if label in set(part["run"])]
            lines += [f"## *{organism}*, {mode} inputs", "",
                      "| model | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
            for model in MODELS:
                cells = []
                for label in runs:
                    r = part[(part["model"] == model) & (part["run"] == label)]
                    cells.append("" if r.empty else
                                 f"{r.skill.iloc[0]:.2f} [{r.skill_lo.iloc[0]:.2f}, {r.skill_hi.iloc[0]:.2f}]")
                lines.append(f"| {model} | " + " | ".join(cells) + " |")
            lines.append("")
    (args.root / "robustness.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
