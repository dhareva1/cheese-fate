"""Load ComBase Browser Excel exports into tidy tables.

Each export workbook has a ``Data Records`` sheet and, when any record in the file has them,
long-format time-series sheets (``Logcs``, ``Temperatures``, ``pHs``, ``Aws``). Times are in
hours as exported by ComBase. Sheets are concatenated across workbooks and exact duplicate
rows are dropped; no values are modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RECORD_SHEET = "Data Records"
COUNT_SHEET = "Logcs"
CONDITION_SHEETS = {
    "Temperatures": ("Temperature", "temperature_c"),
    "pHs": ("pH", "ph"),
    "Aws": ("Aw", "aw"),
}
# Constant conditions are stored as a single number in the record sheet instead of a series.
STATIC_COLUMNS = {"Temperature (C)": "temperature_c", "pH": "ph", "Aw": "aw"}
# Values outside these ranges are physically implausible for cheese and are discarded
# (e.g. some records list "Aw" values of 0.24-0.33).
PLAUSIBLE = {"temperature_c": (-5.0, 100.0), "ph": (2.0, 10.0), "aw": (0.6, 1.0)}


@dataclass(frozen=True)
class ComBaseExport:
    """Tidy ComBase data.

    ``records``: one row per record with ``record_id, batch_id, source_id, organism, food,
    conditions, raw_milk``.
    ``counts``: ``record_id, time_h, log_count``.
    ``conditions``: ``record_id, variable, time_h, value, origin`` with variable in
    ``temperature_c``, ``ph`` or ``aw``. ``origin`` is ``series`` for time series and
    ``static`` for a constant value from the record sheet (stored at time 0), used only when
    the record has no series for that variable. Implausible values are dropped.
    """

    records: pd.DataFrame
    counts: pd.DataFrame
    conditions: pd.DataFrame


def batch_id(record_id: str) -> str:
    """Return the batch part of a ComBase record ID by removing a trailing ``_<number>``.

    Records from one cheese batch share this stem and differ in the organism suffix,
    e.g. ``BitBo1_2`` (E. coli) and ``BitBo1_10`` (LAB).
    """
    head, sep, tail = record_id.rpartition("_")
    return head if sep and tail.isdigit() else record_id


def source_id(record_id: str) -> str:
    """Return the leading alphabetic prefix of a record ID, a proxy for the source study."""
    prefix = ""
    for char in record_id:
        if not char.isalpha():
            break
        prefix += char
    return prefix or record_id


def load_exports(paths: str | Path | list[Path]) -> ComBaseExport:
    """Load one export directory (all ``*.xlsx``) or an explicit list of workbooks."""
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        files = sorted(root.glob("*.xlsx")) if root.is_dir() else [root]
    else:
        files = [Path(path) for path in paths]
    if not files:
        raise FileNotFoundError(f"No .xlsx exports found in {paths}")

    records, counts, conditions = [], [], []
    for file in files:
        workbook = pd.ExcelFile(file)
        records.append(pd.read_excel(workbook, RECORD_SHEET))
        if COUNT_SHEET in workbook.sheet_names:
            counts.append(pd.read_excel(workbook, COUNT_SHEET))
        for sheet, (column, variable) in CONDITION_SHEETS.items():
            if sheet in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet).rename(columns={column: "value"})
                conditions.append(frame.assign(variable=variable))

    raw_records = pd.concat(records, ignore_index=True).drop_duplicates(subset="Record ID")
    series = _tidy_conditions(conditions).assign(origin="series")
    static = _static_conditions(raw_records)
    has_series = pd.MultiIndex.from_frame(series[["record_id", "variable"]].drop_duplicates())
    static = static[~pd.MultiIndex.from_frame(static[["record_id", "variable"]]).isin(has_series)]
    return ComBaseExport(
        records=_tidy_records(raw_records),
        counts=_tidy_counts(counts),
        conditions=_plausible(
            pd.concat([f for f in (series, static) if not f.empty] or [static], ignore_index=True)
        ),
    )


def _static_conditions(raw_records: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column, variable in STATIC_COLUMNS.items():
        if column not in raw_records.columns:
            continue
        values = pd.to_numeric(raw_records[column], errors="coerce")
        keep = values.notna()
        frames.append(
            pd.DataFrame(
                {
                    "record_id": raw_records.loc[keep, "Record ID"].astype(str).str.strip(),
                    "variable": variable,
                    "time_h": 0.0,
                    "value": values[keep],
                    "origin": "static",
                }
            )
        )
    columns = ["record_id", "variable", "time_h", "value", "origin"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _plausible(conditions: pd.DataFrame) -> pd.DataFrame:
    low = conditions["variable"].map(lambda v: PLAUSIBLE[v][0])
    high = conditions["variable"].map(lambda v: PLAUSIBLE[v][1])
    keep = conditions["value"].between(low, high)
    data = conditions[keep]
    return data.sort_values(["record_id", "variable", "time_h"], kind="stable").reset_index(
        drop=True
    )


def _tidy_records(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.rename(
        columns={
            "Record ID": "record_id",
            "Organism": "organism",
            "Food Name": "food",
            "Conditions": "conditions",
        }
    )
    data["record_id"] = data["record_id"].astype(str).str.strip()
    data["batch_id"] = data["record_id"].map(batch_id)
    data["source_id"] = data["record_id"].map(source_id)
    data["conditions"] = data["conditions"].fillna("")
    data["raw_milk"] = data["conditions"].str.contains(r"\bRaw\b", regex=True)
    columns = ["record_id", "batch_id", "source_id", "organism", "food", "conditions", "raw_milk"]
    return data[columns].reset_index(drop=True)


def _tidy_counts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["record_id", "time_h", "log_count"])
    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    data = data.rename(columns={"Record ID": "record_id", "Time": "time_h", "Logc": "log_count"})
    data["record_id"] = data["record_id"].astype(str).str.strip()
    return data.sort_values(["record_id", "time_h"], kind="stable").reset_index(drop=True)


def _tidy_conditions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["record_id", "variable", "time_h", "value"])
    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    data = data.rename(columns={"Record ID": "record_id", "Time": "time_h"})
    data["record_id"] = data["record_id"].astype(str).str.strip()
    return data[["record_id", "variable", "time_h", "value"]]
