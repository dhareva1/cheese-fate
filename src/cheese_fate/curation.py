"""Build model-ready observation tables from tidy ComBase data.

Log counts are never interpolated. Condition values (temperature, pH, aw) and same-batch LAB
counts are aligned onto each count time within one record or batch only, and every aligned
value that was not measured at exactly that time is flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cheese_fate.combase import ComBaseExport

LAB = "Lactic acid bacteria"
# ComBase stores "not detected" (below the detection limit) as the symbol -0.01 in log-count
# cells (Baranyi & Tamplin, 2004). A few sources use +0.01 the same way; both are symbols,
# not counts. The detection limit itself is not recorded.
ND_SYMBOLS = (-0.01, 0.01)
DEFAULT_DETECTION_LIMIT = 1.0  # log10 CFU/g, used only when a source has no detected count
CONDITION_VARIABLES = ("temperature_c", "ph", "aw")
# Temperature follows process set-points (steps); pH and aw drift between measurements.
DEFAULT_METHODS = {"temperature_c": "step", "ph": "linear", "aw": "linear"}


def align(
    times: np.ndarray,
    source_times: np.ndarray,
    source_values: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align a measured series onto ``times``; return values and an interpolated flag.

    ``step`` carries the last measured value forward (also after the final measurement) and
    is undefined before the first one. ``linear`` interpolates between measurements and is
    undefined outside the measured range. Undefined values are NaN.
    """
    times = np.asarray(times, dtype=float)
    order = np.argsort(source_times, kind="stable")
    x = np.asarray(source_times, dtype=float)[order]
    y = np.asarray(source_values, dtype=float)[order]
    values = np.full(times.shape, np.nan)
    if x.size == 0:
        return values, np.zeros(times.shape, dtype=bool)

    if method == "step":
        index = np.searchsorted(x, times, side="right") - 1
        valid = index >= 0
        values[valid] = y[index[valid]]
    elif method == "linear":
        valid = (times >= x[0]) & (times <= x[-1])
        values[valid] = np.interp(times[valid], x, y)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    measured = np.isin(times, x)
    interpolated = ~np.isnan(values) & ~measured
    return values, interpolated


def lab_by_batch(export: ComBaseExport) -> pd.DataFrame:
    """Return one LAB series per batch: the maximum LAB log count at each sampled time.

    Batches often have several LAB records (different media or LAB groups); the maximum is
    used as the dominant LAB population at that time.
    """
    lab_records = export.records.loc[export.records["organism"] == LAB, ["record_id", "batch_id"]]
    data = export.counts.merge(lab_records, on="record_id")
    return (
        data.groupby(["batch_id", "time_h"], as_index=False)["log_count"]
        .max()
        .rename(columns={"log_count": "lab_log_count"})
    )


def build_observations(
    export: ComBaseExport,
    organisms: list[str] | tuple[str, ...],
    min_points: int = 5,
    methods: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return one row per log-count observation for the selected organisms.

    Columns: record and grouping IDs, ``time_h``, ``log_count``, aligned
    ``temperature_c``, ``ph``, ``aw`` and same-batch ``lab_log_count`` (linear within the LAB
    sampling range), each with an ``<column>_interpolated`` flag. Records with fewer than
    ``min_points`` distinct count times are excluded.
    """
    methods = {**DEFAULT_METHODS, **(methods or {})}
    records = export.records[export.records["organism"].isin(organisms)]
    counts = export.counts.merge(records, on="record_id")
    n_times = counts.groupby("record_id")["time_h"].transform("nunique")
    counts = counts[n_times >= min_points]

    conditions = {key: group for key, group in export.conditions.groupby(["record_id", "variable"])}
    lab = {key: group for key, group in lab_by_batch(export).groupby("batch_id")}

    frames = []
    for record_id, group in counts.groupby("record_id", sort=True):
        group = group.sort_values("time_h", kind="stable").copy()
        times = group["time_h"].to_numpy(dtype=float)
        for variable in CONDITION_VARIABLES:
            series = conditions.get((record_id, variable))
            if series is None:
                group[variable] = np.nan
                group[f"{variable}_interpolated"] = False
                if variable == "temperature_c":
                    group["temperature_origin"] = "missing"
                continue
            # A constant value from the record sheet holds for the whole record.
            method = "step" if (series["origin"] == "static").all() else methods[variable]
            values, flags = align(times, series["time_h"], series["value"], method)
            group[variable] = values
            group[f"{variable}_interpolated"] = flags
            if variable == "temperature_c":
                group["temperature_origin"] = series["origin"].iloc[0]
        lab_series = lab.get(group["batch_id"].iloc[0])
        if lab_series is None:
            group["lab_log_count"] = np.nan
            group["lab_log_count_interpolated"] = False
        else:
            values, flags = align(
                times, lab_series["time_h"], lab_series["lab_log_count"], "linear"
            )
            group["lab_log_count"] = values
            group["lab_log_count_interpolated"] = flags
        frames.append(group)

    if not frames:
        raise ValueError(f"No records with at least {min_points} count times for {organisms}")
    columns = [
        "record_id", "batch_id", "source_id", "organism", "food", "raw_milk", "time_h",
        "log_count", "temperature_origin",
        *(c for v in (*CONDITION_VARIABLES, "lab_log_count") for c in (v, f"{v}_interpolated")),
    ]
    return flag_censoring(pd.concat(frames, ignore_index=True)[columns])


def is_nd_symbol(log_counts: pd.Series) -> np.ndarray:
    return np.isclose(log_counts.to_numpy(dtype=float)[:, None], ND_SYMBOLS).any(axis=1)


def flag_censoring(
    observations: pd.DataFrame, default_limit: float = DEFAULT_DETECTION_LIMIT
) -> pd.DataFrame:
    """Mark ComBase "not detected" symbols as left-censored observations.

    Adds ``log_count_raw`` (the value as exported), ``censored``, ``detection_limit`` and
    ``detection_limit_imputed``. The detection limit of a (source, organism) pair is estimated
    as its smallest detected count, an upper bound of the true limit; pairs without any
    detected count get ``default_limit``. For censored rows ``log_count`` is set to the
    detection limit, i.e. the observation means "log count <= log_count".
    """
    data = observations.copy()
    data["log_count_raw"] = data["log_count"]
    data["censored"] = is_nd_symbol(data["log_count"])
    detected = data.loc[~data["censored"]]
    limits = detected.groupby(["source_id", "organism"])["log_count"].min()
    keys = pd.MultiIndex.from_frame(data[["source_id", "organism"]])
    limit = limits.reindex(keys).to_numpy()
    data["detection_limit_imputed"] = np.isnan(limit)
    data["detection_limit"] = np.where(np.isnan(limit), default_limit, limit)
    data.loc[data["censored"], "log_count"] = data.loc[data["censored"], "detection_limit"]
    return data


def ph_rebound(times: np.ndarray, ph: np.ndarray) -> tuple[float, float]:
    """Return the time of the pH minimum and the largest pH rise after it."""
    times = np.asarray(times, dtype=float)
    ph = np.asarray(ph, dtype=float)
    keep = ~np.isnan(ph)
    if keep.sum() < 2:
        return np.nan, np.nan
    times, ph = times[keep], ph[keep]
    lowest = int(np.argmin(ph))
    return float(times[lowest]), float(ph[lowest:].max() - ph[lowest])


def audit(export: ComBaseExport, min_points: int = 5) -> pd.DataFrame:
    """Summarise data availability per organism for records with ``min_points`` counts."""
    n_counts = export.counts.groupby("record_id")["time_h"].nunique()
    duration = export.counts.groupby("record_id")["time_h"].max()
    by_var = export.conditions.groupby(["record_id", "variable"])["value"]
    distinct = by_var.nunique().unstack().reindex(columns=list(CONDITION_VARIABLES))
    value_range = (by_var.max() - by_var.min()).unstack().reindex(
        columns=list(CONDITION_VARIABLES)
    )
    lab_batches = set(export.records.loc[export.records["organism"] == LAB, "batch_id"])

    data = export.records.set_index("record_id")
    data["n_counts"] = n_counts.reindex(data.index).fillna(0)
    data = data[data["n_counts"] >= min_points]
    data["duration_h"] = duration.reindex(data.index)
    data["temperature_varies"] = distinct["temperature_c"].reindex(data.index).fillna(0) > 1
    data["ph_varies"] = value_range["ph"].reindex(data.index).fillna(0) > 0.3
    data["has_aw"] = distinct["aw"].reindex(data.index).notna()
    data["lab_same_batch"] = data["batch_id"].isin(lab_batches)

    summary = data.groupby("organism").agg(
        curves=("n_counts", "size"),
        temperature_varies=("temperature_varies", "sum"),
        ph_varies=("ph_varies", "sum"),
        has_aw=("has_aw", "sum"),
        lab_same_batch=("lab_same_batch", "sum"),
        sources=("source_id", "nunique"),
        foods=("food", "nunique"),
        median_counts=("n_counts", "median"),
        median_duration_h=("duration_h", "median"),
    )
    return summary.sort_values("curves", ascending=False)
