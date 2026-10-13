"""Grouped cross-validation of the mechanistic baseline.

Parameters are fitted on the training sources only and used to predict the held-out
sources. Initial points of each curve are given, not predicted, so they are excluded from
the metrics. Every fold also scores a persistence reference (each later count equals the
initial count, columns ``naive_*``); a useful model must beat it.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd

from cheese_fate.evaluation import grouped_folds, summarize_censored
from cheese_fate.mechanistic import Curve, Parameters, fit, predict


def cross_validate(
    curves: list[Curve],
    groups: dict[str, str],
    initial: Parameters,
    free: tuple[str, ...],
    bounds: dict[str, tuple[float, float]],
    n_splits: int = 5,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return out-of-fold predictions and one row per fold (fitted parameters and metrics).

    ``groups`` maps each record ID to its group (e.g. ``source_id``); whole groups are held
    out together. ``n_jobs > 1`` fits folds in parallel processes.
    """
    record_groups = np.array([groups[curve.record_id] for curve in curves])
    splits = list(grouped_folds(record_groups, n_splits, seed))
    jobs = [
        (fold, [curves[i] for i in train], [curves[i] for i in test], initial, free, bounds)
        for fold, (train, test) in enumerate(splits)
    ]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            outputs = list(pool.map(_run_fold, jobs))
    else:
        outputs = [_run_fold(job) for job in jobs]

    predictions, folds = [], []
    for (fold, (train, test)), (held_out, fitted, converged) in zip(enumerate(splits), outputs):
        predictions.append(held_out)
        folds.append(
            {
                "fold": fold,
                "train_curves": len(train),
                "test_curves": len(test),
                "test_groups": ";".join(sorted(set(record_groups[test]))),
                "converged": converged,
                **{name: getattr(fitted, name) for name in free},
                **summarize_censored(held_out[~held_out["is_initial"]]),
                **{
                    f"naive_{key}": value
                    for key, value in summarize_censored(
                        _persistence(held_out)[~held_out["is_initial"]]
                    ).items()
                    if not key.startswith("n_")
                },
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def _persistence(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reference model: every later count equals the curve's initial count."""
    initial = predictions[predictions["is_initial"]].set_index("record_id")["log_count"]
    return predictions.assign(predicted=predictions["record_id"].map(initial))


def _run_fold(job) -> tuple[pd.DataFrame, Parameters, bool]:
    fold, train, test, initial, free, bounds = job
    fitted, result = fit(train, initial, free=free, bounds=bounds)
    return predict(test, fitted).assign(fold=fold), fitted, bool(result.success)


def fit_all(
    curves: list[Curve],
    initial: Parameters,
    free: tuple[str, ...],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Fit on every curve; returns all parameter values for reporting."""
    fitted, result = fit(curves, initial, free=free, bounds=bounds)
    return {**asdict(fitted), "converged": bool(result.success), "cost": float(result.cost)}
