import math

import numpy as np
import pandas as pd
import pytest

from cheese_fate import analysis
from cheese_fate.coupled import CoupledForecaster, coupled_task
from cheese_fate.mechanistic import Parameters
from cheese_fate.models import (
    BoostedTrees, CensoredBoosting, Mechanistic, Persistence, ResidualBoostedTrees, without_lab,
)
from cheese_fate.sequence import NeuralODEHybrid, RecurrentModel
from cheese_fate.tasks import Record, features, known_series, make_task, time_grid


def record(record_id="A1_1", source="A", ph_drop=1.5, start=6.0, seed=0, n=8):
    """Synthetic curve: pH falls then rebounds, LAB grows, counts decline with pH."""
    rng = np.random.default_rng(seed)
    times = np.array([0.0, 6, 24, 72, 168, 400, 720, 1440])[:n]
    ph_t = np.array([0.0, 24, 168, 720, 1440])
    ph = np.array([6.6, 6.6 - ph_drop, 6.6 - ph_drop - 0.1, 6.6 - ph_drop + 0.4, 6.6 - ph_drop + 0.6])
    lab_t = np.array([0.0, 24, 168, 1440])
    lab = np.array([5.0, 8.5, 8.8, 7.5])
    counts = start - 0.002 * times + rng.normal(0, 0.1, times.size)
    censored = np.zeros(times.size, bool)
    censored[-1] = True
    counts[-1] = 1.0
    return Record(
        record_id=record_id, source_id=source, food="Test cheese", batch_id=record_id.split("_")[0],
        raw_milk=1.0, detection_limit=1.0, times=times, log_counts=counts, censored=censored,
        series={"temperature_c": (np.array([0.0, 2.0, 48.0]), np.array([32.0, 50.0, 12.0])),
                "ph": (ph_t, ph), "lab": (lab_t, lab)},
        static=frozenset(),
    )


def dataset(n=12):
    return [record(f"S{i}_1", f"S{i % 6}", ph_drop=1.0 + 0.1 * i, start=4 + i % 4, seed=i)
            for i in range(n)]


def test_tasks_anchor_at_origin_and_mask_the_future():
    r = record()
    start = make_task(r, 0.0)
    assert start.anchor_time == 0.0 and start.times[0] == 6.0
    later = make_task(r, 168.0, mode="naive")
    assert later.anchor_time == 168.0 and later.cutoff == 168.0
    assert np.all(later.times > 168.0)
    assert later.series["ph"][0].max() <= 168.0 and later.series["lab"][0].max() <= 168.0
    assert np.array_equal(later.series["temperature_c"][0], r.series["temperature_c"][0])
    assert make_task(record(n=4), 720.0) is None


def test_naive_features_do_not_depend_on_future_measurements():
    r = record()
    changed = record()
    changed.series["ph"][1][-1] = 3.9  # future pH after the day-7 origin
    a = features(make_task(r, 168.0, mode="naive"))
    b = features(make_task(changed, 168.0, mode="naive"))
    pd.testing.assert_frame_equal(a, b)
    measured_a = features(make_task(r, 168.0))
    measured_b = features(make_task(changed, 168.0))
    assert not measured_a["ph_now"].equals(measured_b["ph_now"])


def test_features_integrate_step_temperature_exactly():
    task = make_task(record(), 0.0)
    row = features(task, [48.0]).iloc[0]
    assert row["hours_hot"] == pytest.approx(46.0)  # 50 C from 2 h to 48 h
    assert row["temp_mean"] == pytest.approx((2 * 32 + 46 * 50) / 48)
    late = features(task, [1440.0]).iloc[0]
    assert late["ph_rebound"] == pytest.approx(0.7)
    assert late["lab_max_hist"] == pytest.approx(8.8)
    grid = time_grid(0.0, [100.0], task.series)
    assert {2.0, 48.0, 100.0} <= set(grid)


def test_censored_boosting_only_penalises_overprediction():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 1, 400)})
    truth = 3.0 - 4.0 * X["x"].to_numpy()
    censored = truth < 1.0
    observed = np.where(censored, 1.0, truth)
    model = CensoredBoosting(rounds=4, max_iter=100, random_state=0).fit(X, observed, censored)
    low = X["x"].to_numpy() > 0.8
    assert model.predict(X[low]).mean() < 0.9  # free to go below the limit
    naive = CensoredBoosting(rounds=1, max_iter=100, random_state=0).fit(X, observed, censored)
    assert naive.predict(X[low]).mean() == pytest.approx(1.0, abs=0.1)


def test_reference_and_tree_models_predict_every_target():
    records = dataset()
    tasks = [make_task(r, 0.0) for r in records]
    mech = Mechanistic(Parameters(t_min=6.0, t_max=46.0))
    for model in (Persistence(), mech, BoostedTrees(), ResidualBoostedTrees(mech, without_lab())):
        predicted = model.fit(tasks).predict(tasks[:2])
        assert [p.shape for p in predicted] == [t.times.shape for t in tasks[:2]]
        assert all(np.isfinite(p).all() for p in predicted)
    assert np.allclose(Persistence().predict(tasks[:1])[0], tasks[0].anchor_log)


def test_neural_ode_starts_from_the_mechanistic_model_and_trains():
    records = dataset()
    tasks = [make_task(r, 0.0) for r in records]
    mech = Mechanistic(Parameters(t_min=6.0, t_max=46.0))
    hybrid = NeuralODEHybrid(mech, epochs=1, patience=1, hidden=8)
    hybrid.fit(tasks)  # one step from a zero output layer barely moves it
    untrained = NeuralODEHybrid(mech, epochs=0, hidden=8)
    untrained.scaler_ = hybrid.scaler_
    untrained.module_ = hybrid._module(hybrid._batch(tasks[:1]).inputs.shape[-1])
    assert np.allclose(untrained.predict(tasks[:2])[0],
                       np.maximum(mech.predict(tasks[:2])[0], -2.0), atol=1e-4)
    gru = RecurrentModel("gru", epochs=5, patience=5, hidden=8).fit(tasks)
    dense = gru.predict(tasks[:1], [np.array([10.0, 500.0])])
    assert dense[0].shape == (2,)
    alone = gru.predict(tasks[:1], [np.array([500.0])])
    assert alone[0][0] == pytest.approx(dense[0][1], abs=1e-5)  # independent of other queries


def test_coupled_forecaster_keeps_the_past_and_fills_measured_variables():
    records = dataset()
    forecaster = CoupledForecaster(seed=0).fit(records)
    task = make_task(records[0], 168.0, mode="naive")
    coupled = coupled_task(task, forecaster)
    x, y = coupled.series["ph"]
    known_x, known_y = known_series(records[0], 168.0)["ph"]
    assert np.allclose(x[: known_x.size], known_x) and np.allclose(y[: known_y.size], known_y)
    assert x.max() >= task.times.max()
    assert "aw" not in coupled.series  # never measured for this record
    later = coupled.series["ph"][1][x > 168.0]
    assert np.all((later >= 3.5) & (later <= 8.0))


def test_cross_conformal_and_time_to_safe():
    rows = []
    for i in range(30):
        t = np.array([24.0, 400.0, 1500.0])
        error = 0.1 * (i % 10)
        rows.append(pd.DataFrame({
            "record_id": f"r{i}", "fold": i % 3, "food": "c", "time_h": t,
            "anchor_time": 0.0, "anchor_log": 5.0, "detection_limit": 1.0,
            "log_count": [5.0, 3.0, 1.0], "censored": [False, False, True],
            "predicted": np.array([5.0, 3.0, 0.5]) + error,
        }))
    targets = pd.concat(rows, ignore_index=True)
    bands = analysis.cross_conformal(targets, alpha=0.2)
    assert bands["covered"].mean() >= 0.7
    assert analysis.observed_safe_time(targets[targets["record_id"] == "r0"], 3.0) == 400.0
    dense = targets[["record_id", "time_h", "predicted"]]
    safe = analysis.time_to_safe(dense, targets, bands)
    first = safe[(safe["record_id"] == "r0") & (safe["target"] == "reduction")].iloc[0]
    assert first["predicted_safe_h"] == 1500.0 and first["confirmed"] == True  # noqa: E712
    assert first["model_safe_at_60d"] == False  # noqa: E712


def test_rebound_test_reports_direction():
    per_record = pd.DataFrame({"rebound": np.linspace(0.4, 1.4, 12),
                               "ph_attribution": np.linspace(0.05, 0.6, 12)})
    result = analysis.rebound_test(per_record)
    assert result["n_rebound_curves"] == 12
    assert result["wilcoxon_p_greater"] < 0.01 and result["spearman_rho"] == pytest.approx(1.0)
    assert math.isfinite(result["spearman_rho_all"])
