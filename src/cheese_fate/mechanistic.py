"""Dynamic mechanistic baseline: Baranyi growth with cardinal secondary models plus inactivation.

State is ``(log10 N, ln q)``. Inside the growth region

    d log10N/dt = mu(T, pH, aw) * alpha(q) * (1 - 10**(log10N - log10Nmax)) / ln(10)
    d ln q/dt   = mu(T, pH, aw),   alpha = q / (1 + q)

with ``mu = mu_opt * gamma_T * gamma_pH * gamma_aw`` (CTMI for temperature, CPM for pH, a
linear term for aw). Outside the growth region (``mu == 0``) the population declines
log-linearly at rate 1/D (log10 per hour) in one of two regimes:

- non-thermal (T <= t_max), e.g. acid stress during ripening:
  ``log10 D = log_d_ref - (min(T, t_max) - t_ref) / z_t + (pH - ph_ref) / z_ph``
- thermal (T > t_max), e.g. milk heating and curd cooking:
  ``log10 D = log_d_thermal - (T - t_thermal_ref) / z_thermal``

Keeping the regimes separate stops the sub-lethal temperature term from being extrapolated
to cooking temperatures. The growth ceiling of a curve is ``min(log_n_max, log_n0 +
headroom)``: in cheese, pathogens stop well below broth maxima at a level that depends on the
batch (e.g. suppression by LAB), so a finite ``headroom`` bounds growth relative to the
starting count.

Curd formation concentrates cells retained in the curd while whey drains (about tenfold for a
10% cheese yield). Curves that start in milk (before ``concentration_time_h``) therefore get
an instantaneous jump of ``log_concentration`` at that time. The population never falls below ``log_n_floor`` (survivors below
one cell per 100 g are not resolved; the decline fades out smoothly near the floor), and
the inactivation rate is capped at ``max_inactivation_rate`` log10/h: faster kill cannot be
distinguished once the floor is reached, while uncapped rates make the ODE so stiff that the
solver stalls. Default values are starting points for fitting; see
:mod:`cheese_fate.organisms` for organism-specific presets and their sources.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult, least_squares

LN10 = np.log(10.0)


@dataclass(frozen=True)
class Parameters:
    mu_opt: float = 1.0
    t_min: float = 6.0
    t_opt: float = 37.0
    t_max: float = 46.0
    ph_min: float = 4.4
    ph_opt: float = 7.0
    ph_max: float = 9.5
    aw_min: float = 0.90
    log_n_max: float = 9.0
    headroom: float = float("inf")
    log_concentration: float = 0.0
    concentration_time_h: float = 1.0
    log_n_floor: float = -2.0
    h0: float = 1.0
    log_d_ref: float = 2.5
    t_ref: float = 10.0
    z_t: float = 10.0
    ph_ref: float = 5.0
    z_ph: float = 1.0
    log_d_thermal: float = 0.0
    t_thermal_ref: float = 60.0
    z_thermal: float = 6.0
    max_inactivation_rate: float = 50.0


@dataclass(frozen=True)
class Curve:
    """One observed curve for fitting or prediction.

    ``censored`` marks observations that only bound the count from above
    (``log_count <= value``), e.g. ComBase "not detected" results.
    """

    record_id: str
    times: np.ndarray
    log_counts: np.ndarray
    censored: np.ndarray
    profile: "ConditionProfile"


def gamma_temperature(t: float, p: Parameters) -> float:
    """Cardinal temperature model with inflection (CTMI)."""
    if t <= p.t_min or t >= p.t_max:
        return 0.0
    num = (t - p.t_max) * (t - p.t_min) ** 2
    den = (p.t_opt - p.t_min) * (
        (p.t_opt - p.t_min) * (t - p.t_opt) - (p.t_opt - p.t_max) * (p.t_opt + p.t_min - 2 * t)
    )
    return float(np.clip(num / den, 0.0, 1.0))


def gamma_ph(ph: float, p: Parameters) -> float:
    """Cardinal pH model (CPM)."""
    if np.isnan(ph):
        return 1.0
    if ph <= p.ph_min or ph >= p.ph_max:
        return 0.0
    num = (ph - p.ph_min) * (ph - p.ph_max)
    return float(np.clip(num / (num - (ph - p.ph_opt) ** 2), 0.0, 1.0))


def gamma_aw(aw: float, p: Parameters) -> float:
    """Linear water-activity term; missing aw is treated as non-limiting."""
    if np.isnan(aw):
        return 1.0
    return float(np.clip((aw - p.aw_min) / (1.0 - p.aw_min), 0.0, 1.0))


class ConditionProfile:
    """Conditions as functions of time, held constant beyond the measured range.

    Temperature is a step series (process set-points); pH and aw are linear between
    measurements. A variable without measurements is NaN; missing pH and aw are treated as
    non-limiting, while temperature is required for simulation.
    """

    def __init__(self, series: dict[str, tuple[np.ndarray, np.ndarray]]):
        self._series = {}
        for variable, (times, values) in series.items():
            order = np.argsort(times, kind="stable")
            self._series[variable] = (
                np.asarray(times, dtype=float)[order],
                np.asarray(values, dtype=float)[order],
            )

    @property
    def has_temperature(self) -> bool:
        return "temperature_c" in self._series

    def breakpoints(self) -> np.ndarray:
        times = [t for t, _ in self._series.values()]
        return np.unique(np.concatenate(times)) if times else np.array([])

    def __call__(self, t: float) -> tuple[float, float, float]:
        return tuple(self._value(variable, t) for variable in ("temperature_c", "ph", "aw"))

    def _value(self, variable: str, t: float) -> float:
        if variable not in self._series:
            return np.nan
        times, values = self._series[variable]
        if variable == "temperature_c":
            index = max(int(np.searchsorted(times, t, side="right")) - 1, 0)
            return float(values[index])
        return float(np.interp(t, times, values))

    @classmethod
    def from_conditions(cls, conditions: pd.DataFrame, record_id: str) -> "ConditionProfile":
        """Build from the tidy ``conditions`` table of a :class:`ComBaseExport`."""
        rows = conditions[conditions["record_id"] == record_id]
        return cls(
            {
                variable: (group["time_h"].to_numpy(), group["value"].to_numpy())
                for variable, group in rows.groupby("variable")
            }
        )


def growth_rate(t: float, profile: ConditionProfile, p: Parameters) -> float:
    temperature, ph, aw = profile(t)
    return p.mu_opt * gamma_temperature(temperature, p) * gamma_ph(ph, p) * gamma_aw(aw, p)


def inactivation_rate(t: float, profile: ConditionProfile, p: Parameters) -> float:
    """Log10 reductions per hour (1 / D)."""
    temperature, ph, _ = profile(t)
    if temperature > p.t_max:
        log_d = p.log_d_thermal - (temperature - p.t_thermal_ref) / p.z_thermal
    else:
        ph = p.ph_ref if np.isnan(ph) else ph
        log_d = p.log_d_ref - (temperature - p.t_ref) / p.z_t + (ph - p.ph_ref) / p.z_ph
    return float(min(10.0 ** -log_d, p.max_inactivation_rate))


def simulate(
    times: np.ndarray,
    log_n0: float,
    profile: ConditionProfile,
    params: Parameters,
) -> np.ndarray:
    """Predict log10 counts at ``times`` (hours) from ``log_n0`` at the earliest time."""
    if not profile.has_temperature:
        raise ValueError("Simulation requires a temperature profile")
    times = np.asarray(times, dtype=float)
    q0 = 1.0 / np.expm1(params.h0)
    ceiling = min(params.log_n_max, log_n0 + params.headroom)

    def rhs(t, state):
        log_n, ln_q = state
        mu = growth_rate(t, profile, params)
        if mu > 0.0:
            alpha = 1.0 / (1.0 + np.exp(-ln_q))
            limit = 1.0 - 10.0 ** (log_n - ceiling)
            return [mu * alpha * limit / LN10, mu]
        # Smooth floor: the decline fades out approaching log_n_floor. A hard switch makes the
        # right-hand side discontinuous and stalls the solver.
        fade = max(1.0 - 10.0 ** (params.log_n_floor - log_n), 0.0)
        return [-inactivation_rate(t, profile, params) * fade, 0.0]

    # Integrate segment by segment so solver steps never straddle a temperature step change.
    end = float(times.max())
    concentrate = params.log_concentration != 0.0 and times.min() < params.concentration_time_h
    extra = [params.concentration_time_h] if concentrate else []
    knots = np.unique(np.concatenate([[times.min(), end], profile.breakpoints(), extra]))
    knots = knots[(knots >= times.min()) & (knots <= end)]
    state = np.array([log_n0, np.log(q0)])
    predictions = np.full(times.shape, np.nan)
    predictions[times == knots[0]] = log_n0
    for start, stop in zip(knots[:-1], knots[1:]):
        inside = (times > start) & (times <= stop)
        t_eval = np.unique(np.append(times[inside], stop))
        solution = solve_ivp(rhs, (start, stop), state, t_eval=t_eval, rtol=1e-6, atol=1e-8)
        if not solution.success:
            # Stiff segments (very fast kill near the floor) can defeat the explicit solver.
            solution = solve_ivp(
                rhs, (start, stop), state, t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-8
            )
        if not solution.success:
            raise RuntimeError(f"Integration failed on [{start}, {stop}]: {solution.message}")
        lookup = dict(zip(solution.t, solution.y[0]))
        predictions[inside] = [lookup[t] for t in times[inside]]
        state = solution.y[:, -1]
        if concentrate and stop == params.concentration_time_h:
            state[0] += params.log_concentration
        state[0] = max(state[0], params.log_n_floor)
    return np.maximum(predictions, params.log_n_floor)


def curves_from_observations(
    observations: pd.DataFrame, conditions: pd.DataFrame
) -> tuple[list[Curve], pd.DataFrame]:
    """Build fit-ready curves and a table explaining every excluded record.

    A curve starts at its first detected count, which serves as the initial value; earlier
    "not detected" points are dropped because the starting level below the detection limit
    is unknown. Records without a temperature profile, without any detected count, or with
    no observation after the first detected one are excluded.
    """
    curves, excluded = [], []
    for record_id, group in observations.groupby("record_id", sort=True):
        group = group.sort_values("time_h", kind="stable")
        profile = ConditionProfile.from_conditions(conditions, record_id)
        detected = np.flatnonzero(~group["censored"].to_numpy())
        if not profile.has_temperature:
            reason = "no temperature profile"
        elif detected.size == 0:
            reason = "no detected count"
        elif detected[0] == len(group) - 1:
            reason = "no observation after first detected count"
        else:
            reason = None
        if reason:
            excluded.append({"record_id": record_id, "reason": reason})
            continue
        start = detected[0]
        curves.append(
            Curve(
                record_id=record_id,
                times=group["time_h"].to_numpy(dtype=float)[start:],
                log_counts=group["log_count"].to_numpy(dtype=float)[start:],
                censored=group["censored"].to_numpy(dtype=bool)[start:],
                profile=profile,
            )
        )
    return curves, pd.DataFrame(excluded, columns=["record_id", "reason"])


def curve_residuals(curve: Curve, params: Parameters) -> np.ndarray:
    """Residuals after the initial point; censored points only penalise over-prediction."""
    predicted = simulate(curve.times, curve.log_counts[0], curve.profile, params)[1:]
    observed = curve.log_counts[1:]
    residual = predicted - observed
    return np.where(curve.censored[1:], np.maximum(residual, 0.0), residual)


def fit(
    curves: list[Curve],
    initial: Parameters,
    free: tuple[str, ...] = ("mu_opt", "h0", "log_d_ref", "z_ph", "log_d_thermal"),
    bounds: dict[str, tuple[float, float]] | None = None,
    max_nfev: int = 400,
) -> tuple[Parameters, OptimizeResult]:
    """Fit ``free`` parameters to several curves jointly by (censored) least squares.

    ``max_nfev`` bounds the number of residual evaluations so a fit always terminates; check
    ``result.success`` (``status == 0`` means the budget ran out).
    """
    names = {f.name for f in fields(Parameters)}
    unknown = set(free) - names
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    bounds = bounds or {}
    lower = [bounds.get(name, (-np.inf, np.inf))[0] for name in free]
    upper = [bounds.get(name, (-np.inf, np.inf))[1] for name in free]

    def residuals(vector):
        params = replace(initial, **dict(zip(free, vector)))
        return np.concatenate([curve_residuals(curve, params) for curve in curves])

    x0 = [getattr(initial, name) for name in free]
    result = least_squares(
        residuals, x0, bounds=(lower, upper), x_scale="jac", max_nfev=max_nfev
    )
    return replace(initial, **dict(zip(free, result.x))), result


def predict(curves: list[Curve], params: Parameters) -> pd.DataFrame:
    """Return one row per curve observation with the mechanistic prediction."""
    frames = [
        pd.DataFrame(
            {
                "record_id": curve.record_id,
                "time_h": curve.times,
                "log_count": curve.log_counts,
                "censored": curve.censored,
                "predicted": simulate(curve.times, curve.log_counts[0], curve.profile, params),
                "is_initial": np.arange(curve.times.size) == 0,
            }
        )
        for curve in curves
    ]
    return pd.concat(frames, ignore_index=True)
