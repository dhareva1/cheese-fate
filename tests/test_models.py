import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cheese_fate.conformal import conformal_halfwidth, record_scores, time_to_target
from cheese_fate.evaluation import (
    acceptable_zone,
    grouped_folds,
    leave_one_group_out,
    rmse,
    summarize,
)
from cheese_fate.forecast import forecast_inputs, forecast_targets
from cheese_fate.evaluation import summarize_censored
from cheese_fate.mechanistic import (
    ConditionProfile,
    Curve,
    Parameters,
    curve_residuals,
    curves_from_observations,
    fit,
    gamma_ph,
    gamma_temperature,
    simulate,
)

PARAMS = Parameters()


def constant(temperature, ph=6.5):
    return ConditionProfile({"temperature_c": ([0.0], [temperature]), "ph": ([0.0], [ph])})


def test_cardinal_terms_are_bounded_and_peak_at_optimum():
    assert gamma_temperature(PARAMS.t_opt, PARAMS) == pytest.approx(1.0)
    assert gamma_temperature(PARAMS.t_min, PARAMS) == 0.0
    assert gamma_temperature(60.0, PARAMS) == 0.0
    assert gamma_ph(PARAMS.ph_opt, PARAMS) == pytest.approx(1.0)
    assert gamma_ph(4.0, PARAMS) == 0.0
    assert gamma_ph(float("nan"), PARAMS) == 1.0


def test_growth_is_bounded_by_maximum_population():
    times = np.linspace(0, 200, 21)
    log_n = simulate(times, 3.0, constant(30.0), PARAMS)
    assert log_n[0] == 3.0
    assert np.all(np.diff(log_n) >= -1e-4)  # solver tolerance near the asymptote
    assert log_n[-1] == pytest.approx(PARAMS.log_n_max, abs=0.05)


def test_population_declines_outside_growth_region():
    times = np.array([0.0, 100.0, 200.0])
    log_n = simulate(times, 5.0, constant(10.0, ph=4.2), PARAMS)
    assert log_n[1] < 5.0 and log_n[2] < log_n[1]
    # Lower pH shortens D, so decline is faster.
    faster = simulate(times, 5.0, constant(10.0, ph=3.8), PARAMS)
    assert faster[2] < log_n[2]


def test_thermal_regime_is_separate_from_ripening_inactivation():
    times = np.array([0.0, 0.5])
    hot = ConditionProfile({"temperature_c": ([0.0], [72.0])})
    # Thermal D at 72 C: 10**(0 - 12/6) = 0.01 h -> 0.5 h removes 50 log, floored.
    assert simulate(times, 6.0, hot, PARAMS)[1] == PARAMS.log_n_floor
    # log_d_thermal alone sets the kill above t_max: D = 10**(3 - 12/6) = 10 h -> 0.05 log,
    # independent of the non-thermal parameters.
    weak = replace(PARAMS, log_d_thermal=3.0, log_d_ref=0.0, z_t=1.0)
    assert simulate(times, 6.0, hot, weak)[1] == pytest.approx(5.95, abs=1e-4)


def test_population_never_drops_below_floor():
    times = np.array([0.0, 1000.0])
    log_n = simulate(times, 1.0, constant(10.0, ph=3.5), replace(PARAMS, log_d_ref=0.0))
    assert log_n[1] == PARAMS.log_n_floor


def test_censored_residual_only_penalises_overprediction():
    profile = constant(10.0, ph=4.0)  # decline
    times = np.array([0.0, 100.0, 200.0])
    predicted = simulate(times, 5.0, profile, PARAMS)
    below = Curve("r", times, np.array([5.0, 9.0, 9.0]), np.array([False, True, True]), profile)
    above = Curve("r", times, np.array([5.0, 1.0, 1.0]), np.array([False, True, True]), profile)
    assert np.all(curve_residuals(below, PARAMS) == 0.0)
    assert curve_residuals(above, PARAMS) == pytest.approx(np.maximum(predicted[1:] - 1.0, 0))
    assert curve_residuals(above, PARAMS)[0] > 0


def test_curves_start_at_first_detected_count_and_report_exclusions():
    obs = pd.DataFrame(
        {
            "record_id": ["a"] * 4 + ["b"] * 2 + ["c"] * 2,
            "time_h": [0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 0.0, 10.0],
            "log_count": [1.5, 3.0, 4.0, 1.5, 1.5, 1.5, 3.0, 2.0],
            "censored": [True, False, False, True, True, True, False, False],
        }
    )
    conditions = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "variable": ["temperature_c", "temperature_c"],
            "time_h": [0.0, 0.0],
            "value": [20.0, 20.0],
        }
    )
    curves, excluded = curves_from_observations(obs, conditions)
    assert [c.record_id for c in curves] == ["a"]
    assert curves[0].times.tolist() == [10.0, 20.0, 30.0]
    assert curves[0].censored.tolist() == [False, False, True]
    assert excluded.set_index("record_id")["reason"].to_dict() == {
        "b": "no detected count",
        "c": "no temperature profile",
    }


def test_summarize_censored_scores_detected_and_censored_separately():
    frame = pd.DataFrame(
        {
            "log_count": [3.0, 4.0, 1.5, 1.5],
            "predicted": [3.0, 4.5, 1.8, 3.0],
            "censored": [False, False, True, True],
        }
    )
    metrics = summarize_censored(frame)
    assert metrics["n_detected"] == 2 and metrics["n_censored"] == 2
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["censored_agreement"] == 0.5


def test_simulate_respects_temperature_steps():
    times = np.array([0.0, 10.0, 20.0, 30.0])
    profile = ConditionProfile({"temperature_c": ([0.0, 10.0], [30.0, 2.0])})
    log_n = simulate(times, 3.0, profile, replace(PARAMS, h0=0.0001))
    assert log_n[1] > 3.5          # growth during the warm step
    assert log_n[3] < log_n[1]     # decline after cooling below t_min


def test_fit_recovers_growth_rate_from_synthetic_curve():
    times = np.linspace(0, 60, 13)
    profile = constant(30.0)
    truth = replace(PARAMS, mu_opt=0.8)
    counts = simulate(times, 2.0, profile, truth)
    curve = Curve("r", times, counts, np.zeros(times.size, dtype=bool), profile)
    fitted, result = fit([curve], PARAMS, free=("mu_opt",), bounds={"mu_opt": (0.01, 5)})
    assert result.success
    assert fitted.mu_opt == pytest.approx(0.8, rel=1e-3)


def test_forecast_masks_only_unknowable_future_values():
    obs = pd.DataFrame(
        {
            "record_id": ["r"] * 3,
            "time_h": [0.0, 24.0, 48.0],
            "log_count": [3.0, 4.0, 2.0],
            "temperature_c": [37.0, 12.0, 12.0],
            "ph": [6.6, 5.2, 5.5],
            "lab_log_count": [6.0, 8.0, 8.5],
        }
    )
    masked = forecast_inputs(obs, origin_h=24.0)
    assert masked["temperature_c"].tolist() == [37.0, 12.0, 12.0]
    assert masked["ph"].isna().tolist() == [False, False, True]
    assert masked["log_count"].isna().tolist() == [False, False, True]
    assert masked["horizon_h"].tolist() == [-24.0, 0.0, 24.0]
    assert obs["ph"].notna().all()
    assert forecast_targets(obs, 24.0)["log_count"].tolist() == [2.0]


def test_conformal_halfwidth_and_record_scores():
    frame = pd.DataFrame(
        {
            "record_id": ["a", "a", "b"],
            "log_count": [1.0, 2.0, 3.0],
            "predicted": [1.5, 1.0, 3.2],
        }
    )
    assert record_scores(frame).to_dict() == pytest.approx({"a": 1.0, "b": 0.2})
    scores = np.arange(1, 20)  # n = 19 -> rank ceil(20 * 0.9) = 18
    assert conformal_halfwidth(scores, alpha=0.1) == 18
    assert conformal_halfwidth([1.0, 2.0], alpha=0.1) == math.inf


def test_time_to_target_requires_staying_below():
    times = [0, 10, 20, 30, 40]
    assert time_to_target(times, [5, 2, 4, 1, 0.5], target_log=3) == 30
    assert time_to_target(times, [5, 4, 4, 4, 4], target_log=3) == math.inf
    assert time_to_target(times, [1, 1, 1, 1, 1], target_log=3) == 0


def test_metrics_ignore_missing_pairs():
    observed = [1.0, 2.0, np.nan, 4.0]
    predicted = [1.0, 2.5, 3.0, 4.0]
    assert rmse(observed, predicted) == pytest.approx(np.sqrt(0.25 / 3))
    assert acceptable_zone(observed, predicted) == 1.0
    assert set(summarize(observed, predicted)) == {"rmse", "mae", "r2", "acceptable_zone"}


def test_group_splits_never_split_a_group():
    groups = np.array(["s1", "s1", "s2", "s3", "s3", "s3", "s4"])
    held_out = [g for g, _, _ in leave_one_group_out(groups)]
    assert held_out == ["s1", "s2", "s3", "s4"]
    seen = []
    for train, test in grouped_folds(groups, n_splits=2, seed=1):
        assert not set(groups[train]) & set(groups[test])
        seen.extend(test)
    assert sorted(seen) == list(range(len(groups)))
    with pytest.raises(ValueError):
        next(grouped_folds(groups, n_splits=5))


def test_cross_validate_holds_out_whole_sources():
    from cheese_fate.baseline import cross_validate

    times = np.linspace(0, 48, 7)
    truth = replace(PARAMS, mu_opt=0.5)
    curves, groups = [], {}
    for i, temperature in enumerate([20.0, 25.0, 30.0, 35.0]):
        profile = constant(temperature)
        counts = simulate(times, 2.0, profile, truth)
        curves.append(Curve(f"r{i}", times, counts, np.zeros(times.size, dtype=bool), profile))
        groups[f"r{i}"] = f"s{i // 2}"
    predictions, folds = cross_validate(
        curves, groups, PARAMS, free=("mu_opt",), bounds={"mu_opt": (0.01, 5)}, n_splits=2
    )
    assert len(folds) == 2
    assert sorted(folds["test_groups"]) == ["s0", "s1"]
    assert folds["mu_opt"].to_numpy() == pytest.approx([0.5, 0.5], rel=1e-3)
    assert set(predictions["record_id"]) == {"r0", "r1", "r2", "r3"}
    assert folds["rmse"].max() < 1e-3
    assert (folds["naive_rmse"] > folds["rmse"]).all()


def test_headroom_caps_growth_relative_to_start():
    times = np.linspace(0, 200, 11)
    capped = simulate(times, 3.0, constant(30.0), replace(PARAMS, headroom=1.5))
    assert capped[-1] == pytest.approx(4.5, abs=0.05)
    high_start = simulate(times, 8.5, constant(30.0), replace(PARAMS, headroom=1.5))
    assert high_start[-1] == pytest.approx(PARAMS.log_n_max, abs=0.05)


def test_concentration_jump_applies_only_to_curves_starting_in_milk():
    params = replace(PARAMS, log_concentration=1.0, concentration_time_h=1.0)
    still = ConditionProfile({"temperature_c": ([0.0], [4.0])})  # no growth, slow decline
    from_milk = simulate(np.array([0.0, 0.5, 2.0]), 3.0, still, params)
    assert from_milk[1] == pytest.approx(3.0, abs=0.01)
    assert from_milk[2] == pytest.approx(4.0, abs=0.01)
    from_curd = simulate(np.array([2.0, 5.0]), 3.0, still, params)
    assert from_curd[1] == pytest.approx(3.0, abs=0.01)


def test_extreme_kill_parameters_finish_quickly():
    import time

    profile = ConditionProfile({"temperature_c": ([0.0, 0.2, 0.5], [37.0, 75.0, 12.0])})
    harsh = replace(PARAMS, log_d_thermal=-4.0, z_thermal=3.0, log_d_ref=0.0)
    start = time.perf_counter()
    log_n = simulate(np.array([0.0, 0.5, 500.0]), 5.0, profile, harsh)
    assert time.perf_counter() - start < 2.0
    # 0.3 h at the capped 50 log/h reaches the floor; at 12 C (> t_min) survivors regrow.
    assert log_n[1] == pytest.approx(harsh.log_n_floor, abs=0.05)
    assert log_n[2] > log_n[1]
