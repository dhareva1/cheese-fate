"""Cross-validate all models in every input mode and origin, per organism.

Writes, per organism, to ``reports/experiments/<group>/``: target predictions, dense
trajectories (for time-to-safe), out-of-fold SHAP values, forecaster accuracy and a run
record (split, seeds, configuration, per-fold fit details). Analyse with
``analyze_experiments.py``.

Example (from the repository root):
    python scripts/run_experiments.py --organisms "Escherichia coli"
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cheese_fate.combase import load_exports
from cheese_fate.experiment import (
    MODELS, Config, cross_validate, fold_assignment, fold_parameters,
)
from cheese_fate.mechanistic import curves_from_observations
from cheese_fate.organisms import BOUNDS, FREE, PRESETS
from cheese_fate.tasks import build_records

ROOT = Path(__file__).resolve().parents[1]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, default=ROOT / "data/raw/combase_cheese/exports")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/cheese")
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/baseline")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/experiments")
    parser.add_argument("--organisms", nargs="+", default=list(PRESETS))
    parser.add_argument("--group", choices=("source_id", "food"), default="source_id",
                        help="held-out unit: source (primary) or cheese type (generalisation)")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--train-mode", choices=("measured", "naive"), default="measured",
                        help="input mode of the training tasks")
    parser.add_argument("--name", help="output folder name (default: the group); use it for "
                        "variants such as leave-one-source-out or naive training")
    parser.add_argument("--folds", type=int, default=5,
                        help="number of grouped folds; 0 = one fold per group (leave-one-out)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--lab-links", type=Path,
                        help="CSV (record_id, lab_batch_id) of verified pathogen-to-LAB pairs "
                        "that the batch rule cannot link (sensitivity analysis)")
    parser.add_argument("--exclude-sources", nargs="+", default=[],
                        help="source IDs to leave out entirely (sensitivity analysis)")
    args = parser.parse_args()

    export = load_exports(args.exports)
    out = args.out / (args.name or args.group)
    out.mkdir(parents=True, exist_ok=True)
    config = Config(models=tuple(args.models), seed=args.seed, epochs=args.epochs,
                    patience=args.patience, train_mode=args.train_mode)
    lab_links = {}
    if args.lab_links:
        links = pd.read_csv(args.lab_links)
        lab_links = dict(zip(links["record_id"], links["lab_batch_id"]))
    for organism in args.organisms:
        name = slug(organism)
        started = time.time()
        observations = pd.read_csv(args.processed / f"observations_{name}.csv")
        observations = observations[~observations["source_id"].isin(args.exclude_sources)]
        records, excluded = build_records(observations, export, lab_links)
        n_folds = args.folds or len({getattr(r, args.group) for r in records})
        assignment = fold_assignment(records, args.group, n_folds, args.seed)

        # Reuse the baseline's fold parameters only when its split is the same.
        reuse = None
        baseline = args.baseline / f"{name}_cv_folds.csv"
        if args.group == "source_id" and baseline.exists():
            reuse = pd.read_csv(baseline)
            expected = [";".join(sorted({r.source_id for r, f in zip(records, assignment) if f == k}))
                        for k in range(n_folds)]
            if len(reuse) != n_folds or list(reuse.sort_values("fold")["test_groups"]) != expected:
                reuse = None
        curves = None
        if reuse is None:
            curves, _ = curves_from_observations(observations, export.conditions)
        preset = PRESETS[organism]
        params = fold_parameters(curves, assignment, preset.params, FREE, BOUNDS,
                                 reuse=reuse, n_jobs=args.jobs)
        print(f"{organism}: {len(records)} records, mechanistic parameters "
              f"{'reused from baseline' if reuse is not None else 'refitted'}", flush=True)

        result = cross_validate(records, assignment, params, config, n_jobs=args.jobs)
        result.predictions.to_csv(out / f"{name}_predictions.csv.gz", index=False)
        result.dense.to_csv(out / f"{name}_dense.csv.gz", index=False)
        result.shap.to_csv(out / f"{name}_shap.csv.gz", index=False)
        result.forecaster.to_csv(out / f"{name}_forecaster.csv.gz", index=False)
        (out / f"{name}_run.json").write_text(json.dumps({
            "organism": organism,
            "group": args.group,
            "folds": n_folds,
            "train_mode": args.train_mode,
            "seed": args.seed,
            "n_records": len(records),
            "n_excluded": len(excluded),
            "lab_links": str(args.lab_links) if args.lab_links else None,
            "n_lab_linked": sum(r.record_id in lab_links and "lab" in r.series for r in records),
            "excluded_sources": args.exclude_sources,
            "assignment": {r.record_id: int(f) for r, f in zip(records, assignment)},
            "mechanistic": [{k: getattr(p, k) for k in FREE} for p in params],
            "mechanistic_reused": reuse is not None,
            "config": asdict(config),
            "fold_info": result.info["folds"],
            "minutes": round((time.time() - started) / 60, 1),
        }, indent=2, default=str))
        print(f"{organism}: done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
