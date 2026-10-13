"""Point-prediction models on :class:`cheese_fate.tasks.Task` objects.

Every model has ``fit(tasks)`` and ``predict(tasks, times=None)``; ``predict`` returns one
array of log10 counts per task, at the task's target times or at ``times[i]`` for task ``i``.
Predictions are clipped to a physically meaningful range of log counts.

- :class:`Persistence`: every later count equals the anchor count (the reference to beat).
- :class:`Mechanistic`: the dynamic mechanistic model with fold-fitted parameters.
- :class:`BoostedTrees`: ML-only gradient-boosted trees on :data:`tasks.FEATURES`.
- :class:`ResidualBoostedTrees`: hybrid; trees correct the mechanistic prediction.

Trees are fitted to left-censored targets ("not detected" means ``count <= limit``) by
Tobit-style EM imputation (:class:`CensoredBoosting`), so a censored point pulls predictions
below its limit rather than onto it. Hyperparameters are tuned by grouped cross-validation inside the
training data only (nested validation).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from cheese_fate.evaluation import grouped_folds
from cheese_fate.mechanistic import Parameters, simulate
from cheese_fate.tasks import FEATURES, LAB_FEATURES, Task, features

LOG_RANGE = (-2.0, 10.0)  # log10 CFU/g; the mechanistic floor up to dense pathogen growth


def target_times(tasks: list[Task], times) -> list[np.ndarray]:
    if times is None:
        return [task.times for task in tasks]
    return [np.asarray(t, dtype=float) for t in times]


def censored_residuals(observed, predicted, censored) -> np.ndarray:
    """Prediction minus observation; censored points count only when over-predicted."""
    residual = np.asarray(predicted, float) - np.asarray(observed, float)
    return np.where(censored, np.maximum(residual, 0.0), residual)


class Persistence:
    name = "persistence"

    def fit(self, tasks):
        return self

    def predict(self, tasks, times=None):
        return [np.full(t.shape, task.anchor_log) for task, t in zip(tasks, target_times(tasks, times))]


class Mechanistic:
    """Dynamic Baranyi model with inactivation, started from each task's anchor count."""

    name = "mechanistic"

    def __init__(self, params: Parameters):
        self.params = params

    def fit(self, tasks):
        return self

    def predict(self, tasks, times=None):
        out = []
        for task, t in zip(tasks, target_times(tasks, times)):
            full = np.concatenate([[task.anchor_time], t])
            out.append(simulate(full, task.anchor_log, task.profile, self.params)[1:])
        return out


def fit_trees(params: dict, X: pd.DataFrame, y) -> HistGradientBoostingRegressor:
    """Fit histogram boosting; columns missing in every training row are filled with 0.

    scikit-learn cannot bin an all-missing column. A constant column never gets a split, so
    predictions on later rows that do have values are unaffected.
    """
    empty = X.columns[X.isna().all()]
    if len(empty):
        X = X.assign(**{column: 0.0 for column in empty})
    return HistGradientBoostingRegressor(**params).fit(X, y)


class CensoredBoosting:
    """Histogram gradient boosting fitted to left-censored targets (Tobit-style EM).

    Each round fits the trees, estimates the residual SD on detected points, and replaces
    every censored target by its expected value below the limit under a normal residual,
    ``E[y | y < L] = mu - sigma * phi(z) / Phi(z)`` with ``z = (L - mu) / sigma``. The imputed
    value is always below the limit, so predictions can fall below the detection limit
    where the detected data imply die-off.
    """

    def __init__(self, rounds: int = 4, **params):
        self.rounds = rounds
        self.params = params

    def fit(self, X, y, censored):
        y = np.asarray(y, float)
        censored = np.asarray(censored, bool)
        target = y.copy()
        for round_ in range(self.rounds):
            self.model_ = fit_trees(self.params, X, target)
            if not censored.any() or round_ == self.rounds - 1:
                break
            mu = self.model_.predict(X)
            detected = ~censored
            sigma = max(float(np.sqrt(np.mean((y[detected] - mu[detected]) ** 2))), 0.1) \
                if detected.any() else 1.0
            z = (y[censored] - mu[censored]) / sigma
            target = y.copy()
            target[censored] = mu[censored] - sigma * np.exp(
                stats.norm.logpdf(z) - stats.norm.logcdf(z))
        return self

    def predict(self, X):
        return self.model_.predict(X)


GRID = tuple(
    {"max_leaf_nodes": leaves, "min_samples_leaf": leaf}
    for leaves in (7, 15, 31)
    for leaf in (10, 40)
)
BASE_PARAMS = {"learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1.0}


def tune(X, y, censored, groups, seed: int = 0, inner_folds: int = 3) -> tuple[dict, pd.DataFrame]:
    """Pick tree hyperparameters by grouped inner cross-validation (censored RMSE)."""
    n_groups = np.unique(groups).size
    folds = list(grouped_folds(groups, min(inner_folds, n_groups), seed))
    rows = []
    for candidate in GRID:
        params = {**BASE_PARAMS, **candidate, "random_state": seed}
        errors = []
        for train, test in folds:
            model = CensoredBoosting(**params).fit(X.iloc[train], y[train], censored[train])
            errors.append(censored_residuals(y[test], model.predict(X.iloc[test]), censored[test]))
        rows.append({**candidate, "rmse": float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))})
    scores = pd.DataFrame(rows)
    best = scores.loc[scores["rmse"].idxmin(), list(GRID[0])].to_dict()
    return {**BASE_PARAMS, **{k: int(v) for k, v in best.items()}, "random_state": seed}, scores


class BoostedTrees:
    """ML-only model: trees predict the change from the anchor count."""

    name = "gbt"

    def __init__(self, feature_names=FEATURES, seed: int = 0, name: str | None = None):
        self.feature_names = list(feature_names)
        self.seed = seed
        if name:
            self.name = name

    def design(self, tasks, times=None) -> pd.DataFrame:
        frames = [features(task, t) for task, t in zip(tasks, target_times(tasks, times))]
        return pd.concat(frames, ignore_index=True)[self.feature_names]

    def _offset(self, tasks, times=None) -> np.ndarray:
        return np.concatenate([np.full(t.shape, task.anchor_log)
                               for task, t in zip(tasks, target_times(tasks, times))])

    def fit(self, tasks):
        X = self.design(tasks)
        offset = self._offset(tasks)
        y = np.concatenate([task.log_counts for task in tasks]) - offset
        censored = np.concatenate([task.censored for task in tasks])
        groups = np.concatenate([[task.record.source_id] * task.times.size for task in tasks])
        self.params_, self.tuning_ = tune(X, y, censored, groups, seed=self.seed)
        self.model_ = CensoredBoosting(**self.params_).fit(X, y, censored)
        return self

    def predict(self, tasks, times=None):
        X = self.design(tasks, times)
        flat = np.clip(self._offset(tasks, times) + self.model_.predict(X), *LOG_RANGE)
        sizes = [t.size for t in target_times(tasks, times)]
        return np.split(flat, np.cumsum(sizes)[:-1])


class ResidualBoostedTrees(BoostedTrees):
    """Hybrid: trees learn the mechanistic model's residual from conditions, LAB and the
    mechanistic prediction itself."""

    name = "residual_gbt"

    def __init__(self, mechanistic: Mechanistic, feature_names=FEATURES, seed: int = 0,
                 name: str | None = None):
        super().__init__(feature_names, seed, name)
        self.mechanistic = mechanistic

    def design(self, tasks, times=None) -> pd.DataFrame:
        X = super().design(tasks, times)
        mech = np.concatenate(self.mechanistic.predict(tasks, times))
        anchor = self._anchor(tasks, times)
        return X.assign(mech_log=mech, mech_change=mech - anchor)

    def _anchor(self, tasks, times=None):
        return BoostedTrees._offset(self, tasks, times)

    def _offset(self, tasks, times=None) -> np.ndarray:
        return np.concatenate(self.mechanistic.predict(tasks, times))


def without_lab(feature_names=FEATURES) -> list[str]:
    return [name for name in feature_names if name not in LAB_FEATURES]


def with_params(params: Parameters, values: dict) -> Parameters:
    return replace(params, **{k: float(v) for k, v in values.items()})
