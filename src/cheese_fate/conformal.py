"""Record-level split conformal bands and time-to-safe estimates (research direction C).

Observations within one curve are correlated, so the nonconformity score is computed per
record: the largest absolute log-count error over that curve. A band of half-width ``q``
around a new curve's prediction then covers the *whole* curve with probability at least
``1 - alpha`` under exchangeability of records (finite-sample split conformal guarantee).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def record_scores(
    frame: pd.DataFrame,
    observed: str = "log_count",
    predicted: str = "predicted",
    record: str = "record_id",
) -> pd.Series:
    """Return the maximum absolute error per record."""
    errors = (frame[observed] - frame[predicted]).abs()
    return errors.groupby(frame[record]).max()


def conformal_halfwidth(scores: np.ndarray | pd.Series, alpha: float = 0.1) -> float:
    """Return the split-conformal quantile of calibration scores at level ``1 - alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(np.asarray(scores, dtype=float))
    n = scores.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    return float(scores[rank - 1])


def time_to_target(
    times: np.ndarray,
    upper_bound: np.ndarray,
    target_log: float,
) -> float:
    """First time from which the upper bound stays at or below ``target_log``; inf if never."""
    times = np.asarray(times, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=float)
    order = np.argsort(times, kind="stable")
    times, upper_bound = times[order], upper_bound[order]
    above = np.flatnonzero(upper_bound > target_log)
    if above.size == 0:
        return float(times[0])
    last_above = above[-1]
    return float(times[last_above + 1]) if last_above + 1 < times.size else math.inf
