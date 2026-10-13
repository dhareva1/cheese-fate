"""Forecast-mode evaluation protocol (research direction A).

Conventional dynamic models receive the measured pH, aw and LAB trajectories for the whole
period they predict. In forecast mode, a model standing at ``origin_h`` may only use:

- the process temperature plan for the whole period (it is set by the cheesemaker);
- pH, aw, LAB and pathogen counts observed up to and including ``origin_h``;
- static record metadata (cheese type, raw milk, source).

The gap between measured-input and forecast-mode accuracy quantifies how much safety
prediction depends on future pH measurements.
"""

from __future__ import annotations

import pandas as pd

KNOWN_IN_ADVANCE = ("temperature_c",)
OBSERVED_ONLY = ("ph", "aw", "lab_log_count", "log_count")


def forecast_inputs(observations: pd.DataFrame, origin_h: float) -> pd.DataFrame:
    """Return a copy in which values a forecaster could not know at ``origin_h`` are masked.

    Masked values are set to NaN; rows are kept so predictions can be aligned with targets.
    A ``horizon_h`` column gives the time ahead of the origin (negative for history).
    """
    data = observations.copy()
    future = data["time_h"] > origin_h
    for column in OBSERVED_ONLY:
        if column in data.columns:
            data.loc[future, column] = float("nan")
    data["horizon_h"] = data["time_h"] - origin_h
    return data


def forecast_targets(observations: pd.DataFrame, origin_h: float) -> pd.DataFrame:
    """Return the future pathogen observations a forecast made at ``origin_h`` is scored on."""
    future = observations[observations["time_h"] > origin_h]
    return future[["record_id", "time_h", "log_count"]].assign(
        horizon_h=future["time_h"] - origin_h
    )
