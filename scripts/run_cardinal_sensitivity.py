"""Sensitivity of the mechanistic baseline to its fixed cardinal values.

The *E. coli* aw_min and pH_max and the *S. aureus* pH limits are provisional
(``organisms.py``). Each variant shifts one fixed value, refits the free parameters by the
same grouped cross-validation as ``run_baseline.py`` (folds by source, seed 0) and records the
pooled out-of-fold RMSE on detected counts against persistence.

Writes ``reports/sensitivity/cardinal.csv``.

Example (from the repository root):
    python scripts/run_cardinal_sensitivity.py
"""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cheese_fate.baseline import _persistence, cross_validate
from cheese_fate.combase import load_exports
from cheese_fate.mechanistic import curves_from_observations
from cheese_fate.organisms import BOUNDS, FREE, PRESETS

ROOT = Path(__file__).resolve().parents[1]

# One fixed value changed per variant; ranges cover values reported for each organism.
VARIANTS = {
    "Escherichia coli": [
        {}, {"ph_min": 4.0}, {"ph_min": 4.6}, {"ph_max": 9.0}, {"ph_max": 10.0},
        {"aw_min": 0.935}, {"aw_min": 0.97}, {"t_min": 4.0},
    ],
    "Staphylococcus aureus": [
        {}, {"ph_min": 4.0}, {"ph_min": 4.5}, {"ph_max": 9.8}, {"ph_opt": 6.5},
        {"aw_min": 0.859}, {"t_min": 6.06, "t_max": 47.9},
    ],
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pooled_rmse(frame: pd.DataFrame) -> float:
    detected = frame[~frame["is_initial"] & ~frame["censored"].astype(bool)]
    return float(np.sqrt(np.mean((detected["predicted"] - detected["log_count"]) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, default=ROOT / "data/raw/combase_cheese/exports")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/cheese")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/sensitivity")
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    conditions = load_exports(args.exports).conditions
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for organism, variants in VARIANTS.items():
        observations = pd.read_csv(args.processed / f"observations_{slug(organism)}.csv")
        curves, _ = curves_from_observations(observations, conditions)
        groups = observations.drop_duplicates("record_id").set_index("record_id")["source_id"]
        for change in variants:
            params = replace(PRESETS[organism].params, **change)
            predictions, _ = cross_validate(curves, groups.to_dict(), params, FREE, BOUNDS,
                                            n_splits=5, seed=0, n_jobs=args.jobs)
            row = {"organism": organism,
                   "variant": ", ".join(f"{k}={v}" for k, v in change.items()) or "preset",
                   "rmse": pooled_rmse(predictions),
                   "rmse_persistence": pooled_rmse(_persistence(predictions))}
            row["skill"] = 1.0 - row["rmse"] / row["rmse_persistence"]
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(args.out / "cardinal.csv", index=False)


if __name__ == "__main__":
    main()
