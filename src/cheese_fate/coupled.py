"""Coupled forecaster of LAB, pH and aw trajectories (research direction A).

In forecast mode, pH, aw and LAB after the origin are predicted states rather than inputs.
They are forecast in a chain, LAB -> pH -> aw: acidification follows LAB growth, and the pH
forecast feeds the aw forecast. Each link is a gradient-boosted model driven by:

- the temperature plan from the origin to the forecast time (known in advance);
- the last values of LAB, pH and aw known at the origin and how old they are;
- the forecasts of the earlier links at the same time;
- the forecast horizon, process time and raw-milk flag.

A link predicts the change from its own last known value; a variable never measured before
the origin is predicted as an absolute value. Training pairs ``(origin, later measurement)``
come from the measured trajectories of the training records (one record per batch, since a
batch shares its trajectories). The earlier-link forecasts used as training features are
cross-fitted over grouped inner folds, so they carry realistic forecast error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from cheese_fate.evaluation import grouped_folds
from cheese_fate.models import fit_trees
from cheese_fate.tasks import OBSERVED, Record, Task, features, known_series, make_task, time_grid

CHAIN = ("lab", "ph", "aw")
RANGES = {"lab": (-2.0, 11.0), "ph": (3.5, 8.0), "aw": (0.6, 1.0)}
PARAMS = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15,
          "min_samples_leaf": 20, "l2_regularization": 1.0}


def last_known(known: dict, variable: str, cutoff: float) -> tuple[float, float]:
    """Last value of ``variable`` at or before ``cutoff`` and its age in hours."""
    if variable not in known:
        return np.nan, np.nan
    x, y = known[variable]
    keep = x <= cutoff
    if not keep.any():
        return np.nan, np.nan
    return float(y[keep][-1]), float(cutoff - x[keep][-1])


def design(record: Record, known: dict, cutoff: float, times, hats: dict) -> pd.DataFrame:
    """Forecast features for future ``times`` given what is known at ``cutoff``."""
    times = np.asarray(times, dtype=float)
    plan = {"temperature_c": known["temperature_c"]} if "temperature_c" in known else {}
    probe = Task(record, cutoff, cutoff, cutoff, np.nan, times, times, times, plan, "plan")
    temperature = features(probe, times)
    frame = pd.DataFrame({
        "horizon_h": times - cutoff,
        "log_horizon": np.log1p(times - cutoff),
        "time_h": times,
        "origin_h": np.full(times.shape, cutoff),
        "temp_now": temperature["temp_now"].to_numpy(),
        "temp_mean": temperature["temp_mean"].to_numpy(),
        "hours_warm": temperature["hours_warm"].to_numpy(),
        "hours_hot": temperature["hours_hot"].to_numpy(),
        "raw_milk": np.full(times.shape, record.raw_milk),
    })
    for variable in OBSERVED:
        value, age = last_known(known, variable, cutoff)
        frame[f"{variable}_last"] = value
        frame[f"{variable}_age"] = age
    for variable, values in hats.items():
        frame[f"{variable}_hat"] = values
    return frame


class CoupledForecaster:
    """Chained gradient-boosted forecaster of LAB, pH and aw after a forecast origin."""

    def __init__(self, chain=CHAIN, seed: int = 0, inner_folds: int = 3, max_origins: int = 8):
        self.chain = tuple(chain)
        self.seed, self.inner_folds, self.max_origins = seed, inner_folds, max_origins

    def _pairs(self, records: list[Record], variable: str):
        """Yield ``(record, cutoff, known, target_times, target_values)`` training pairs."""
        for record in records:
            if variable not in record.series or variable in record.static:
                continue
            x, y = record.series[variable]
            if x.size < 2:
                continue
            origins = np.unique(x[:-1])
            if origins.size > self.max_origins:
                origins = origins[np.linspace(0, origins.size - 1, self.max_origins).astype(int)]
            if record.times[0] < x[0]:
                origins = np.append(origins, record.times[0])
            for cutoff in origins:
                later = x > cutoff
                if later.any():
                    yield record, float(cutoff), known_series(record, cutoff), x[later], y[later]

    def fit(self, records: list[Record]):
        records = list({record.batch_id: record for record in records}.values())
        self.models_ = {}
        for i, variable in enumerate(self.chain):
            pairs = list(self._pairs(records, variable))
            if not pairs:
                continue
            hats = self._cross_fitted(records, pairs, self.chain[:i])
            X = pd.concat([design(r, known, cutoff, t, hat)
                           for (r, cutoff, known, t, _), hat in zip(pairs, hats)],
                          ignore_index=True)
            y = np.concatenate([values for *_, values in pairs])
            last = X[f"{variable}_last"].to_numpy()
            has_last = ~np.isnan(last)
            absolute_columns = [c for c in X.columns if c not in (f"{variable}_last", f"{variable}_age")]
            entry = {"columns": list(X.columns), "absolute_columns": absolute_columns,
                     "n_pairs": int(y.size)}
            if has_last.sum() >= 20:
                entry["change"] = fit_trees({**PARAMS, "random_state": self.seed},
                                            X[has_last], y[has_last] - last[has_last])
            entry["absolute"] = fit_trees({**PARAMS, "random_state": self.seed},
                                          X[absolute_columns], y)
            self.models_[variable] = entry
        return self

    def _cross_fitted(self, records, pairs, earlier) -> list[dict]:
        if not earlier:
            return [{} for _ in pairs]
        groups = np.array([record.source_id for record in records])
        n_folds = min(self.inner_folds, np.unique(groups).size)
        fold_of = {}
        submodels = []
        for fold, (train, test) in enumerate(grouped_folds(groups, n_folds, self.seed)):
            submodels.append(CoupledForecaster(earlier, self.seed, self.inner_folds,
                                               self.max_origins).fit([records[j] for j in train]))
            fold_of.update({records[j].batch_id: fold for j in test})
        return [submodels[fold_of[r.batch_id]].chain_values(r, known, cutoff, t)
                for r, cutoff, known, t, _ in pairs]

    def _predict(self, variable: str, X: pd.DataFrame) -> np.ndarray:
        entry = self.models_.get(variable)
        if entry is None:
            return np.full(len(X), np.nan)
        X = X[entry["columns"]]
        last = X[f"{variable}_last"].to_numpy()
        values = entry["absolute"].predict(X[entry["absolute_columns"]])
        if "change" in entry and not np.isnan(last).all():
            values = np.where(np.isnan(last), values, last + entry["change"].predict(X))
        return np.clip(values, *RANGES[variable])

    def chain_values(self, record: Record, known: dict, cutoff: float, times) -> dict:
        """Forecast every chain variable at ``times`` (all after ``cutoff``)."""
        hats = {}
        for variable in self.chain:
            hats[variable] = self._predict(variable, design(record, known, cutoff, times, hats))
        return hats

    def forecast(self, record: Record, cutoff: float, end: float) -> dict:
        """Trajectories known at ``cutoff`` extended with forecasts up to ``end``.

        Only variables the record actually measures are forecast; the others stay missing,
        as in measured-input mode.
        """
        known = known_series(record, cutoff)
        plan = {"temperature_c": known["temperature_c"]} if "temperature_c" in known else {}
        grid = time_grid(cutoff, [max(end, cutoff)], plan)
        grid = grid[grid > cutoff]
        if grid.size == 0:
            return known
        hats = self.chain_values(record, known, cutoff, grid)
        series = dict(known)
        for variable in self.chain:
            if variable not in record.series or variable in record.static:
                continue
            x, y = known.get(variable, (np.array([]), np.array([])))
            series[variable] = (np.concatenate([x, grid]), np.concatenate([y, hats[variable]]))
        return series


def coupled_task(task: Task, forecaster: CoupledForecaster, end: float | None = None) -> Task:
    """Return ``task`` with pH, aw and LAB after its cutoff replaced by forecasts."""
    end = float(task.times.max()) if end is None else end
    return task.with_series(forecaster.forecast(task.record, task.cutoff, end), "coupled")


def evaluate(forecaster: CoupledForecaster, records: list[Record], origins) -> pd.DataFrame:
    """Score forecasts and naive hold-last forecasts against later measurements."""
    rows = []
    for record in records:
        for origin in origins:
            task = make_task(record, origin)
            if task is None:
                continue
            known = known_series(record, task.cutoff)
            for variable in forecaster.chain:
                if variable not in record.series or variable in record.static:
                    continue
                x, y = record.series[variable]
                later = x > task.cutoff
                if not later.any():
                    continue
                hats = forecaster.chain_values(record, known, task.cutoff, x[later])
                naive, _ = last_known(known, variable, task.cutoff)
                rows.append(pd.DataFrame({
                    "record_id": record.record_id, "variable": variable, "origin_h": origin,
                    "time_h": x[later], "horizon_h": x[later] - task.cutoff,
                    "measured": y[later], "forecast": hats[variable], "naive": naive,
                }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
