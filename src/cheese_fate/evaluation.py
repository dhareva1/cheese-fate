"""Metrics and grouped splits for generalisation across sources and cheese types."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def _paired(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = ~(np.isnan(observed) | np.isnan(predicted))
    return observed[keep], predicted[keep]


def rmse(observed, predicted) -> float:
    o, p = _paired(observed, predicted)
    return float(np.sqrt(np.mean((o - p) ** 2)))


def mae(observed, predicted) -> float:
    o, p = _paired(observed, predicted)
    return float(np.mean(np.abs(o - p)))


def r2(observed, predicted) -> float:
    o, p = _paired(observed, predicted)
    return float(1.0 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))


def acceptable_zone(observed, predicted, lower: float = -0.5, upper: float = 0.5) -> float:
    """Fraction of log-count residuals (observed - predicted) within ``[lower, upper]``."""
    o, p = _paired(observed, predicted)
    residual = o - p
    return float(np.mean((residual >= lower) & (residual <= upper)))


def summarize(observed, predicted) -> dict[str, float]:
    return {
        "rmse": rmse(observed, predicted),
        "mae": mae(observed, predicted),
        "r2": r2(observed, predicted),
        "acceptable_zone": acceptable_zone(observed, predicted),
    }


def summarize_censored(frame: pd.DataFrame, tolerance: float = 0.5) -> dict[str, float]:
    """Metrics for a prediction table with ``log_count``, ``predicted`` and ``censored``.

    Regression metrics use detected observations only. For "not detected" observations,
    ``censored_agreement`` is the fraction predicted at or below the detection limit plus
    ``tolerance``.
    """
    detected = frame[~frame["censored"]]
    censored = frame[frame["censored"]]
    metrics = summarize(detected["log_count"], detected["predicted"])
    metrics["n_detected"] = float(len(detected))
    metrics["n_censored"] = float(len(censored))
    metrics["censored_agreement"] = (
        float(np.mean(censored["predicted"] <= censored["log_count"] + tolerance))
        if len(censored)
        else float("nan")
    )
    return metrics


def leave_one_group_out(groups: pd.Series | np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(group, train_index, test_index)`` holding out one group at a time.

    Use ``source_id`` for leave-one-source-out and ``food`` for leave-one-cheese-type-out;
    all rows of a record share both, so records are never split.
    """
    groups = np.asarray(groups)
    for group in pd.unique(groups):
        test = groups == group
        yield group, np.flatnonzero(~test), np.flatnonzero(test)


def grouped_folds(
    groups: pd.Series | np.ndarray, n_splits: int = 5, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_index, test_index)`` for ``n_splits`` folds of whole groups.

    Groups are shuffled with ``seed`` and assigned greedily to the fold with the fewest rows,
    so folds have similar sizes even when groups are imbalanced.
    """
    groups = np.asarray(groups)
    names, sizes = np.unique(groups, return_counts=True)
    if n_splits > names.size:
        raise ValueError(f"n_splits={n_splits} exceeds the number of groups ({names.size})")
    order = np.random.default_rng(seed).permutation(names.size)
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of, load = {}, np.zeros(n_splits)
    for index in order:
        fold = int(np.argmin(load))
        fold_of[names[index]] = fold
        load[fold] += sizes[index]
    assignment = np.array([fold_of[g] for g in groups])
    for fold in range(n_splits):
        yield np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
