"""Build model-ready observation tables and a data audit from the ComBase cheese exports.

Example (from the repository root):
    python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from cheese_fate.combase import load_exports
from cheese_fate.curation import audit, build_observations

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Escherichia coli", "Staphylococcus aureus")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, default=ROOT / "data/raw/combase_cheese/exports")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/cheese")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--organisms", nargs="+", default=list(TARGETS))
    parser.add_argument("--min-points", type=int, default=5)
    args = parser.parse_args()

    export = load_exports(args.exports)
    print(
        f"loaded {len(export.records)} records, {len(export.counts)} counts, "
        f"{len(export.conditions)} condition values"
    )

    args.reports.mkdir(parents=True, exist_ok=True)
    summary = audit(export, min_points=args.min_points)
    summary.to_csv(args.reports / "data_audit.csv")
    print(summary.to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    for organism in args.organisms:
        observations = build_observations(export, [organism], min_points=args.min_points)
        path = args.out / f"observations_{slug(organism)}.csv"
        observations.to_csv(path, index=False)
        print(
            f"{organism}: {observations['record_id'].nunique()} curves, "
            f"{len(observations)} observations -> {path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
