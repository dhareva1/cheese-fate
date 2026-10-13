"""Cross-validate the mechanistic baseline per organism with folds grouped by source.

Example (from the repository root):
    python scripts/run_baseline.py --organisms "Escherichia coli"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from cheese_fate.baseline import cross_validate, fit_all
from cheese_fate.combase import load_exports
from cheese_fate.mechanistic import curves_from_observations
from cheese_fate.organisms import BOUNDS, FREE, PRESETS

ROOT = Path(__file__).resolve().parents[1]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, default=ROOT / "data/raw/combase_cheese/exports")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/cheese")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/baseline")
    parser.add_argument("--organisms", nargs="+", default=list(PRESETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    conditions = load_exports(args.exports).conditions
    args.out.mkdir(parents=True, exist_ok=True)
    for organism in args.organisms:
        name = slug(organism)
        observations = pd.read_csv(args.processed / f"observations_{name}.csv")
        curves, excluded = curves_from_observations(observations, conditions)
        excluded.to_csv(args.out / f"{name}_excluded.csv", index=False)
        groups = observations.drop_duplicates("record_id").set_index("record_id")["source_id"]
        preset = PRESETS[organism]
        print(f"{organism}: {len(curves)} curves, {len(excluded)} excluded", flush=True)

        predictions, folds = cross_validate(
            curves, groups.to_dict(), preset.params, FREE, BOUNDS,
            n_splits=args.folds, seed=args.seed, n_jobs=args.jobs,
        )
        predictions.to_csv(args.out / f"{name}_cv_predictions.csv", index=False)
        folds.to_csv(args.out / f"{name}_cv_folds.csv", index=False)
        metrics = ["rmse", "mae", "r2", "acceptable_zone", "censored_agreement"]
        metrics += [f"naive_{m}" for m in metrics]
        summary = folds[metrics].agg(["mean", "std"]).round(3)
        print(summary.to_string(), flush=True)

        full = fit_all(curves, preset.params, FREE, BOUNDS)
        (args.out / f"{name}_full_fit.json").write_text(
            json.dumps(
                {
                    "organism": organism,
                    "free": FREE,
                    "bounds": BOUNDS,
                    "sources": preset.sources,
                    "n_curves": len(curves),
                    "folds": args.folds,
                    "seed": args.seed,
                    "parameters": full,
                },
                indent=2,
            )
        )
        print(f"full-data fit: { {k: round(full[k], 3) for k in FREE} }", flush=True)


if __name__ == "__main__":
    main()
