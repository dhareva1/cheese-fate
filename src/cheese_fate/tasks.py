"""Prediction tasks: what a model is allowed to know when it predicts one curve.

A task anchors a curve at a known (detected) count and asks for the counts at later target
times, given condition trajectories for temperature, pH, aw and same-batch LAB. Every model
(persistence, mechanistic, ML and hybrid) consumes the same tasks, so input modes are compared
on identical targets:

- ``measured``: the full measured trajectories (the conventional upper bound);
- ``naive``: pH, aw and LAB measured up to the forecast origin only, then held constant;
- ``coupled``: pH, aw and LAB after the origin come from a forecaster
  (:mod:`cheese_fate.coupled`).

The temperature plan is known in every mode. Constant values from the ComBase record sheet
(``origin == "static"``) describe the product and are treated as known in every mode.

Origins follow the study protocol (section 5.1): origin 0 is the curve start (its
first detected count, as in the mechanistic baseline); origins at day 7 and day 30 exist only
for curves that start before them and are observed after them. A task at a later origin is
anchored at the last detected count at or before the origin and scored on counts after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from cheese_fate.combase import ComBaseExport
from cheese_fate.curation import lab_by_batch
from cheese_fate.mechanistic import ConditionProfile, curves_from_observations

CONDITIONS = ("temperature_c", "ph", "aw")
OBSERVED = ("ph", "aw", "lab")  # known only up to the forecast origin
ORIGINS_H = (0.0, 168.0, 720.0)
MODES = ("measured", "naive", "coupled")
MANUFACTURE_END_H = 48.0  # process stage boundary for reporting
HOT_C = 45.0  # above the growth range of both pathogens (thermal exposure)
WARM_C = 20.0  # lower bound of the warm, growth-favourable band

@dataclass(frozen=True, eq=False)
class Record:
    """One pathogen curve with its metadata and measured trajectories.

    ``times``, ``log_counts`` and ``censored`` start at the first detected count, as in
    :func:`cheese_fate.mechanistic.curves_from_observations`. ``series`` maps ``temperature_c``,
    ``ph``, ``aw`` and ``lab`` (same-batch LAB, maximum over LAB records) to sorted
    ``(times, values)``; ``static`` names the variables that hold one record-sheet value.
    """

    record_id: str
    source_id: str
    food: str
    batch_id: str
    raw_milk: float
    detection_limit: float
    times: np.ndarray
    log_counts: np.ndarray
    censored: np.ndarray
    series: dict
    static: frozenset


@dataclass(frozen=True, eq=False)
class Task:
    """Predict ``log_counts`` at ``times`` from ``anchor_log`` at ``anchor_time``.

    ``cutoff`` is the knowledge cutoff (the origin, or the curve start for origin 0);
    ``series`` holds the trajectories the model may use, as they are in ``mode``.
    """

    record: Record
    origin_h: float
    cutoff: float
    anchor_time: float
    anchor_log: float
    times: np.ndarray
    log_counts: np.ndarray
    censored: np.ndarray
    series: dict
    mode: str

    @cached_property
    def profile(self) -> ConditionProfile:
        return ConditionProfile({v: s for v, s in self.series.items() if v in CONDITIONS})

    def with_series(self, series: dict, mode: str) -> "Task":
        return Task(
            self.record, self.origin_h, self.cutoff, self.anchor_time, self.anchor_log,
            self.times, self.log_counts, self.censored, series, mode,
        )


def build_records(
    observations: pd.DataFrame, export: ComBaseExport, lab_links: dict[str, str] | None = None
) -> tuple[list[Record], pd.DataFrame]:
    """Return model-ready records and the table of excluded records with reasons.

    ``lab_links`` maps a pathogen record ID to the LAB batch to use instead of its own batch,
    for verified pairs that the record-name rule cannot link (sensitivity analysis only).
    """
    lab_links = lab_links or {}
    curves, excluded = curves_from_observations(observations, export.conditions)
    meta = observations.drop_duplicates("record_id").set_index("record_id")
    conditions = {key: rows for key, rows in export.conditions.groupby("record_id")}
    lab = {key: rows for key, rows in lab_by_batch(export).groupby("batch_id")}
    records = []
    for curve in curves:
        info = meta.loc[curve.record_id]
        series, static = {}, set()
        rows = conditions.get(curve.record_id)
        if rows is not None:
            for variable, group in rows.groupby("variable"):
                group = group.sort_values("time_h", kind="stable")
                series[variable] = (
                    group["time_h"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float)
                )
                if (group["origin"] == "static").all():
                    static.add(variable)
        lab_rows = lab.get(lab_links.get(curve.record_id, info["batch_id"]))
        if lab_rows is not None:
            lab_rows = lab_rows.sort_values("time_h", kind="stable")
            series["lab"] = (
                lab_rows["time_h"].to_numpy(dtype=float),
                lab_rows["lab_log_count"].to_numpy(dtype=float),
            )
        raw_milk = info["raw_milk"]
        records.append(
            Record(
                record_id=curve.record_id,
                source_id=str(info["source_id"]),
                food=str(info["food"]),
                batch_id=str(info["batch_id"]),
                raw_milk=float(raw_milk) if pd.notna(raw_milk) else np.nan,
                detection_limit=float(info["detection_limit"]),
                times=curve.times,
                log_counts=curve.log_counts,
                censored=curve.censored,
                series=series,
                static=frozenset(static),
            )
        )
    return records, excluded


def value_at(variable: str, series: dict, times) -> np.ndarray:
    """Evaluate a trajectory: temperature as steps, others linearly; held beyond the range."""
    times = np.asarray(times, dtype=float)
    if variable not in series:
        return np.full(times.shape, np.nan)
    x, y = series[variable]
    if variable == "temperature_c":
        index = np.clip(np.searchsorted(x, times, side="right") - 1, 0, None)
        return y[index]
    return np.interp(times, x, y)


def known_series(record: Record, cutoff: float) -> dict:
    """Trajectories a forecaster at ``cutoff`` knows: future pH, aw and LAB are dropped."""
    known = {}
    for variable, (x, y) in record.series.items():
        if variable not in OBSERVED or variable in record.static:
            known[variable] = (x, y)
            continue
        keep = x <= cutoff
        if keep.any():
            known[variable] = (x[keep], y[keep])
    return known


def make_task(record: Record, origin_h: float, series: dict | None = None, mode: str = "measured"):
    """Return the task for ``record`` at ``origin_h``, or None if the origin does not apply."""
    times, index = record.times, np.arange(record.times.size)
    if origin_h <= 0.0:
        anchor, cutoff = 0, float(times[0])
        targets = index > 0
    else:
        if times[0] >= origin_h:
            return None
        known = np.flatnonzero((times <= origin_h) & ~record.censored)
        anchor, cutoff = int(known[-1]), float(origin_h)
        targets = (index > anchor) & (times > origin_h)
    if not targets.any():
        return None
    if series is None:
        series = record.series if mode == "measured" else known_series(record, cutoff)
    return Task(
        record=record,
        origin_h=float(origin_h),
        cutoff=cutoff,
        anchor_time=float(times[anchor]),
        anchor_log=float(record.log_counts[anchor]),
        times=times[targets],
        log_counts=record.log_counts[targets],
        censored=record.censored[targets],
        series=series,
        mode=mode,
    )


def time_grid(
    start: float, times, series: dict, ratio: float = 1.15, first: float = 0.25
) -> np.ndarray:
    """Time points from ``start`` to ``max(times)`` for integrating along a curve.

    Geometric spacing (``first`` hours, growing by ``ratio``) resolves manufacture finely and
    ripening coarsely; every trajectory breakpoint and every query time is included, so step
    temperatures and linear trajectories are integrated exactly.
    """
    times = np.asarray(times, dtype=float)
    end = float(times.max()) if times.size else start
    span = end - start
    steps = int(np.ceil(np.log(max(span / first, 1.0)) / np.log(ratio))) + 1
    geometric = start + first * ratio ** np.arange(steps)
    breakpoints = [x for x, _ in series.values()]
    breakpoints = np.concatenate(breakpoints) if breakpoints else np.array([])
    breakpoints = breakpoints[(breakpoints > start) & (breakpoints < end)]
    return np.unique(np.concatenate([[start], geometric[geometric < end], breakpoints, times]))


def geometric_grid(start: float, end: float, ratio: float = 1.15, first: float = 0.25) -> np.ndarray:
    """``start`` plus geometric offsets (``first`` hours, growing by ``ratio``) up to ``end``.

    The grid depends only on the start and is extended just past ``end``, so a longer horizon
    only appends points: the grid never depends on how often conditions were measured or on
    which times are queried.
    """
    span = max(end - start, 0.0)
    steps = int(np.ceil(np.log(max(span / first, 1.0)) / np.log(ratio))) + 1
    return np.concatenate([[start], start + first * ratio ** np.arange(steps)])


FEATURES = (
    "elapsed_h", "log_elapsed", "time_h", "anchor_log",
    "temp_now", "temp_mean", "temp_max", "hours_hot", "hours_warm",
    "ph_now", "ph_anchor", "ph_mean", "ph_min_hist", "ph_rebound",
    "aw_now", "aw_mean",
    "lab_now", "lab_anchor", "lab_mean", "lab_max_hist",
    "raw_milk",
)
LAB_FEATURES = ("lab_now", "lab_anchor", "lab_mean", "lab_max_hist")


def features(task: Task, query_times=None) -> pd.DataFrame:
    """Return one row of :data:`FEATURES` per query time (default: the task's target times).

    Means and hours are integrated from the anchor to the query time; ``*_hist`` extremes and
    ``ph_rebound`` (current pH minus the lowest pH so far) cover the whole history up to the
    query time. Only ``task.series`` is used, so forecast-mode tasks cannot leak future
    measurements. Missing trajectories give NaN.
    """
    query = task.times if query_times is None else np.asarray(query_times, dtype=float)
    grid = time_grid(task.anchor_time, query, task.series)
    position = np.searchsorted(grid, query)
    elapsed = grid - task.anchor_time
    width = np.diff(grid)
    out = {
        "elapsed_h": elapsed,
        "log_elapsed": np.log1p(elapsed),
        "time_h": grid,
        "anchor_log": np.full(grid.shape, task.anchor_log),
        "raw_milk": np.full(grid.shape, task.record.raw_milk),
    }

    def running_mean(values, step):
        segment = values[:-1] if step else (values[:-1] + values[1:]) / 2.0
        integral = np.concatenate([[0.0], np.cumsum(segment * width)])
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(elapsed > 0, integral / np.where(elapsed > 0, elapsed, 1.0), values)

    temperature = value_at("temperature_c", task.series, grid)
    out["temp_now"] = temperature
    out["temp_mean"] = running_mean(temperature, step=True)
    out["temp_max"] = np.maximum.accumulate(temperature) if temperature.size else temperature
    out["hours_hot"] = np.concatenate([[0.0], np.cumsum((temperature[:-1] > HOT_C) * width)])
    warm = (temperature[:-1] > WARM_C) & (temperature[:-1] <= HOT_C)
    out["hours_warm"] = np.concatenate([[0.0], np.cumsum(warm * width)])
    if "temperature_c" not in task.series:
        for name in ("hours_hot", "hours_warm"):
            out[name] = np.full(grid.shape, np.nan)

    for variable, prefix in (("ph", "ph"), ("aw", "aw"), ("lab", "lab")):
        values = value_at(variable, task.series, grid)
        out[f"{prefix}_now"] = values
        out[f"{prefix}_mean"] = running_mean(values, step=False)
        out[f"{prefix}_anchor"] = np.full(
            grid.shape, value_at(variable, task.series, [task.anchor_time])[0]
        )
    out["ph_min_hist"] = _history_extreme("ph", task.series, grid, np.minimum)
    out["ph_rebound"] = out["ph_now"] - out["ph_min_hist"]
    out["lab_max_hist"] = _history_extreme("lab", task.series, grid, np.maximum)
    return pd.DataFrame({name: np.asarray(out[name])[position] for name in FEATURES})


def _history_extreme(variable: str, series: dict, grid: np.ndarray, ufunc) -> np.ndarray:
    """Running minimum or maximum of a linear trajectory from its first point to each time."""
    if variable not in series:
        return np.full(grid.shape, np.nan)
    x, _ = series[variable]
    union = np.unique(np.concatenate([x, grid]))
    running = ufunc.accumulate(value_at(variable, series, union))
    return running[np.searchsorted(union, grid)]


def stage(times) -> np.ndarray:
    return np.where(np.asarray(times, dtype=float) < MANUFACTURE_END_H, "manufacture", "ripening")
