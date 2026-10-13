import numpy as np
import pandas as pd
import pytest

from cheese_fate import inference


def predictions(n_sources=12, per_source=20, seed=0, noise_model=0.5, noise_ref=1.0):
    """Two models on the same targets: 'good' has smaller errors than 'persistence'."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sources):
        observed = rng.normal(4.0, 1.0, per_source)
        for model, noise in (("good", noise_model), ("persistence", noise_ref)):
            rows.append(pd.DataFrame({
                "model": model, "mode": "measured", "origin_h": 0.0, "source_id": f"S{s}",
                "log_count": observed, "censored": False,
                "predicted": observed + rng.normal(0.0, noise, per_source),
            }))
    frame = pd.concat(rows, ignore_index=True)
    # A censored row must be ignored by every metric.
    extra = frame.iloc[:2].assign(censored=True, predicted=100.0)
    return pd.concat([frame, extra], ignore_index=True)


def test_point_metrics_match_direct_computation():
    frame = predictions()
    table = inference.validation_table(frame, n_boot=200).set_index("model")
    detected = frame[(frame["model"] == "good") & ~frame["censored"]]
    error = detected["predicted"] - detected["log_count"]
    assert table.loc["good", "rmse"] == pytest.approx(np.sqrt(np.mean(error ** 2)))
    assert table.loc["good", "bf"] == pytest.approx(10 ** error.mean())
    assert table.loc["good", "af"] == pytest.approx(10 ** error.abs().mean())
    residual = -error
    assert table.loc["good", "asz"] == pytest.approx(np.mean(residual.abs() <= 0.5))
    assert table.loc["good", "apz"] == pytest.approx(np.mean((residual >= -1) & (residual <= 0.5)))
    assert table.loc["good", "rmse_lo"] <= table.loc["good", "rmse"] <= table.loc["good", "rmse_hi"]


def test_skill_is_significant_for_a_clearly_better_model():
    table = inference.validation_table(predictions(), n_boot=500).set_index("model")
    assert table.loc["good", "skill"] > 0.3
    assert table.loc["good", "skill_lo"] > 0
    assert table.loc["good", "p_skill"] < 0.01
    assert table.loc["good", "sources_better"] == 12


def test_skill_test_is_calibrated_for_an_equal_model():
    """Under no true difference, p < 0.05 should occur in about 5% of datasets."""
    p = [inference.validation_table(predictions(noise_model=1.0, seed=s), n_boot=300, seed=s)
         .set_index("model").loc["good", "p_skill"] for s in range(60)]
    assert np.mean(np.array(p) < 0.05) <= 0.15


def test_paired_difference_sign_and_target_check():
    frame = predictions()
    result = inference.paired_difference(
        frame, [(("good", "measured"), ("persistence", "measured"), 0.0)], n_boot=500)
    assert result.loc[0, "difference"] < 0
    assert result.loc[0, "difference_hi"] < 0
    assert result.loc[0, "p_two_sided"] < 0.01
    # Different target sets must be rejected.
    shorter = frame.drop(frame[(frame["model"] == "good")].index[:3])
    with pytest.raises(ValueError):
        inference.paired_difference(
            shorter, [(("good", "measured"), ("persistence", "measured"), 0.0)], n_boot=10)


def test_bootstrap_resamples_whole_sources():
    weights = inference.bootstrap_weights(7, 50, seed=1)
    assert weights.shape == (50, 7)
    assert (weights.sum(axis=1) == 7).all()


def test_holm():
    adjusted = inference.holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
