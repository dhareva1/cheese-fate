"""Uncertainty and paired comparisons for cross-validated predictions.

Observations within a source study are correlated, so every interval and test here treats the
**source** as the unit: sources are resampled with replacement (cluster bootstrap) and pooled
metrics are recomputed from per-source sums. Comparisons are paired: both models are scored
on the same held-out targets in each resample.

- :func:`source_sums`: per-source sufficient statistics on detected targets.
- :func:`validation_table`: pooled RMSE, skill, bias and accuracy factors and the acceptable
  simulation zone, each with a source-bootstrap 95% interval.
- :func:`paired_difference`: RMSE(a) − RMSE(b) with interval and bootstrap p-value.
- :func:`holm`: Holm–Bonferroni adjustment.

Validation criteria follow predictive microbiology practice:

- bias factor ``Bf = 10^mean(predicted − observed)`` and accuracy factor
  ``Af = 10^mean|predicted − observed|`` on log10 counts (Ross, 1996; Baranyi et al., 1999);
- acceptable simulation zone (ASZ; Møller et al., 2013): the share of observed log counts
  within ±0.5 log of the prediction; a model is acceptable when this share is at least 0.70;
- acceptable prediction zone (APZ; Oscar, 2005), reported alongside: the share of residuals
  ``observed − predicted`` within −1 log (fail-safe) and +0.5 log (fail-dangerous).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ASZ_HALF_WIDTH = 0.5
APZ_LOWER, APZ_UPPER = -1.0, 0.5
ASZ_PASS = 0.70
KEYS = ["model", "mode", "origin_h"]
SUMS = ["n", "sse", "bias", "abs", "asz", "apz"]


def source_sums(predictions: pd.DataFrame) -> pd.DataFrame:
    """Sums per model, mode, origin and source over detected held-out targets."""
    detected = predictions[~predictions["censored"].astype(bool)]
    error = detected["predicted"] - detected["log_count"]
    residual = -error
    frame = detected[KEYS + ["source_id"]].assign(
        n=1.0, sse=error ** 2, bias=error, abs=error.abs(),
        asz=(residual.abs() <= ASZ_HALF_WIDTH).astype(float),
        apz=((residual >= APZ_LOWER) & (residual <= APZ_UPPER)).astype(float),
    )
    return frame.groupby(KEYS + ["source_id"], sort=True)[SUMS].sum().reset_index()


def bootstrap_weights(n_sources: int, n_boot: int, seed: int) -> np.ndarray:
    """``n_boot × n_sources`` counts of how often each source is drawn in each resample."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_sources, np.full(n_sources, 1.0 / n_sources), size=n_boot)


def _matrix(sums: pd.DataFrame, sources: np.ndarray) -> dict[tuple, np.ndarray]:
    """Per (model, mode, origin): a ``n_sources × 5`` array of sums, zero for absent sources."""
    out = {}
    for key, group in sums.groupby(KEYS, sort=True):
        table = group.set_index("source_id")[SUMS].reindex(sources).fillna(0.0)
        out[key] = table.to_numpy()
    return out


def _metrics(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Pooled metrics from summed statistics (last axis = SUMS)."""
    n = totals[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "rmse": np.sqrt(totals[..., 1] / n),
            "bias": totals[..., 2] / n,
            "bf": 10.0 ** (totals[..., 2] / n),
            "af": 10.0 ** (totals[..., 3] / n),
            "asz": totals[..., 4] / n,
            "apz": totals[..., 5] / n,
        }


def _interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def _p_greater(values: np.ndarray) -> float:
    """One-sided bootstrap p-value that the statistic is not above zero."""
    values = values[np.isfinite(values)]
    return float((np.sum(values <= 0.0) + 1) / (values.size + 1))


def validation_table(predictions: pd.DataFrame, reference: str = "persistence",
                     n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Pooled validation metrics with source-bootstrap 95% intervals.

    ``skill = 1 − RMSE/RMSE(reference)`` is paired: both RMSEs come from the same resample.
    ``p_skill`` is the one-sided bootstrap p-value for skill > 0. ``sources_better`` counts
    held-out sources on which the model's RMSE is below the reference's.
    """
    sums = source_sums(predictions)
    sources = np.sort(sums["source_id"].unique())
    arrays = _matrix(sums, sources)
    weights = bootstrap_weights(sources.size, n_boot, seed)
    rows = []
    for (model, mode, origin), array in arrays.items():
        ref = arrays.get((reference, mode, origin))
        point = _metrics(array.sum(axis=0))
        boot = _metrics(weights @ array)
        row = {"model": model, "mode": mode, "origin_h": origin, "n_detected": int(array[:, 0].sum()),
               "n_sources": int((array[:, 0] > 0).sum())}
        for name in ("rmse", "bias", "bf", "af", "asz", "apz"):
            low, high = _interval(boot[name])
            row.update({name: float(point[name]), f"{name}_lo": low, f"{name}_hi": high})
        row["asz_pass"] = bool(point["asz"] >= ASZ_PASS)
        if ref is not None:
            ref_point = _metrics(ref.sum(axis=0))["rmse"]
            ref_boot = _metrics(weights @ ref)["rmse"]
            skill_boot = 1.0 - boot["rmse"] / ref_boot
            row["skill"] = float(1.0 - point["rmse"] / ref_point)
            row["skill_lo"], row["skill_hi"] = _interval(skill_boot)
            row["p_skill"] = _p_greater(skill_boot) if model != reference else float("nan")
            present = (array[:, 0] > 0) & (ref[:, 0] > 0)
            per_source = np.sqrt(array[present, 1] / array[present, 0])
            per_source_ref = np.sqrt(ref[present, 1] / ref[present, 0])
            row["sources_better"] = int(np.sum(per_source < per_source_ref))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(predictions: pd.DataFrame, comparisons: list[tuple],
                      n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """RMSE(a) − RMSE(b) for each ``((model_a, mode_a), (model_b, mode_b), origin_h)``.

    Negative means *a* is more accurate. ``p_two_sided`` is the bootstrap two-sided p-value
    for a difference of zero. Both arms must be scored on the same targets, which holds for
    every model within a mode and origin, and across modes because the targets are the same
    held-out counts.
    """
    sums = source_sums(predictions)
    sources = np.sort(sums["source_id"].unique())
    arrays = _matrix(sums, sources)
    weights = bootstrap_weights(sources.size, n_boot, seed)
    rows = []
    for (model_a, mode_a), (model_b, mode_b), origin in comparisons:
        a, b = arrays.get((model_a, mode_a, origin)), arrays.get((model_b, mode_b, origin))
        if a is None or b is None:
            continue
        if not np.array_equal(a[:, 0], b[:, 0]):
            raise ValueError(f"{model_a}/{mode_a} and {model_b}/{mode_b} are scored on different targets")
        point = _metrics(a.sum(axis=0))["rmse"] - _metrics(b.sum(axis=0))["rmse"]
        boot = _metrics(weights @ a)["rmse"] - _metrics(weights @ b)["rmse"]
        low, high = _interval(boot)
        finite = boot[np.isfinite(boot)]
        p = 2.0 * min(np.sum(finite >= 0.0) + 1, np.sum(finite <= 0.0) + 1) / (finite.size + 1)
        rows.append({"a": f"{model_a}/{mode_a}", "b": f"{model_b}/{mode_b}", "origin_h": origin,
                     "rmse_a": float(_metrics(a.sum(axis=0))["rmse"]),
                     "rmse_b": float(_metrics(b.sum(axis=0))["rmse"]),
                     "difference": float(point), "difference_lo": low, "difference_hi": high,
                     "p_two_sided": float(min(p, 1.0))})
    return pd.DataFrame(rows)


def holm(p_values) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (p.size - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
