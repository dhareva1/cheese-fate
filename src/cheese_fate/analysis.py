"""Analyses of cross-validated predictions (methodology sections 5-7).

- :func:`fold_metrics`, :func:`summary_table`: accuracy per model, mode and origin, with the
  persistence reference (RQ1, RQ6).
- :func:`forecast_gap`: accuracy lost in forecast mode, per origin and horizon (RQ2).
- :func:`lab_effect`: paired per-fold change from adding same-batch LAB (RQ3).
- :func:`ph_attribution`, :func:`rebound_test`: the pre-registered deacidification test on
  out-of-fold SHAP values (RQ4).
- :func:`cross_conformal`, :func:`time_to_safe`: record-level conformal bands and the ripening
  time to a target level (RQ5).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from cheese_fate.conformal import conformal_halfwidth, time_to_target
from cheese_fate.curation import ph_rebound
from cheese_fate.evaluation import summarize_censored
from cheese_fate.tasks import Record

HORIZON_EDGES = (0.0, 48.0, 168.0, 720.0, 2160.0, math.inf)
HORIZON_LABELS = ("0-2 d", "2-7 d", "7-30 d", "30-90 d", ">90 d")
SIXTY_DAYS_H = 1440.0
REBOUND_THRESHOLD = 0.3


def horizon_bin(horizon_h) -> pd.Categorical:
    return pd.cut(horizon_h, HORIZON_EDGES, labels=HORIZON_LABELS, right=True)


def _metrics(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(by, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **summarize_censored(group)})
    return pd.DataFrame(rows)


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics per model, mode, origin and fold."""
    return _metrics(predictions, ["model", "mode", "origin_h", "fold"])


def summary_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics plus fold mean and SD of RMSE, and skill against persistence.

    ``skill`` is ``1 - RMSE / RMSE(persistence)`` on the same targets: positive means the
    model beats persistence.
    """
    pooled = _metrics(predictions, ["model", "mode", "origin_h"])
    folds = fold_metrics(predictions)
    spread = folds.groupby(["model", "mode", "origin_h"])["rmse"].agg(
        rmse_fold_mean="mean", rmse_fold_sd="std").reset_index()
    table = pooled.merge(spread, on=["model", "mode", "origin_h"])
    reference = table[table["model"] == "persistence"].set_index(["mode", "origin_h"])["rmse"]
    keys = pd.MultiIndex.from_frame(table[["mode", "origin_h"]])
    table["skill"] = 1.0 - table["rmse"] / reference.reindex(keys).to_numpy()
    return table


def stage_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per process stage (manufacture / ripening)."""
    return _metrics(predictions, ["model", "mode", "origin_h", "stage"])


def forecast_gap(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pooled RMSE per model, origin and horizon in each mode, and the forecast-mode gaps."""
    frame = predictions.assign(horizon=horizon_bin(predictions["horizon_h"]))
    table = _metrics(frame, ["model", "origin_h", "horizon", "mode"])
    table["horizon"] = pd.Categorical(table["horizon"], HORIZON_LABELS, ordered=True)
    wide = table.pivot_table(index=["model", "origin_h", "horizon"], columns="mode",
                             values="rmse", observed=True).reset_index()
    counts = table[table["mode"] == "measured"].set_index(["model", "origin_h", "horizon"])
    wide["n_detected"] = counts["n_detected"].reindex(
        pd.MultiIndex.from_frame(wide[["model", "origin_h", "horizon"]])).to_numpy()
    for mode in ("naive", "coupled"):
        if mode in wide:
            wide[f"gap_{mode}"] = wide[mode] - wide["measured"]
    wide.columns.name = None
    return wide.sort_values(["model", "origin_h", "horizon"]).reset_index(drop=True)


def lab_effect(predictions: pd.DataFrame, pairs=(("gbt", "gbt_nolab"),
                                                 ("residual_gbt", "residual_gbt_nolab"))) -> pd.DataFrame:
    """Per-fold RMSE change from adding LAB features (negative = LAB helps)."""
    folds = fold_metrics(predictions).set_index(["model", "mode", "origin_h", "fold"])["rmse"]
    rows = []
    for with_lab, without in pairs:
        if with_lab not in folds.index.get_level_values(0) or without not in folds.index.get_level_values(0):
            continue
        difference = (folds.loc[with_lab] - folds.loc[without]).dropna()
        for (mode, origin), values in difference.groupby(level=[0, 1]):
            rows.append({
                "model": with_lab, "mode": mode, "origin_h": origin,
                "rmse_change_mean": values.mean(), "rmse_change_sd": values.std(),
                "folds_improved": int((values < 0).sum()), "folds": int(values.size),
            })
    return pd.DataFrame(rows)


def forecaster_accuracy(forecasts: pd.DataFrame) -> pd.DataFrame:
    """RMSE of the coupled forecaster and of hold-last-value, per variable/origin/horizon."""
    frame = forecasts.dropna(subset=["measured"]).assign(
        horizon=horizon_bin(forecasts["horizon_h"]),
        error_forecast=lambda d: (d["forecast"] - d["measured"]) ** 2,
        error_naive=lambda d: (d["naive"] - d["measured"]) ** 2,
    )
    table = frame.groupby(["variable", "origin_h", "horizon"], observed=True).agg(
        n=("measured", "size"), rmse_forecast=("error_forecast", "mean"),
        rmse_naive=("error_naive", "mean"))
    table[["rmse_forecast", "rmse_naive"]] **= 0.5
    overall = frame.groupby(["variable", "origin_h"]).agg(
        n=("measured", "size"), rmse_forecast=("error_forecast", "mean"),
        rmse_naive=("error_naive", "mean"))
    overall[["rmse_forecast", "rmse_naive"]] **= 0.5
    overall = overall.assign(horizon="all").set_index("horizon", append=True)
    return pd.concat([table, overall]).reset_index()


# -- RQ4: deacidification -------------------------------------------------------------------

def rebound_table(records: list[Record]) -> pd.DataFrame:
    """Measured pH minimum time and largest rise after it, per record."""
    rows = []
    for record in records:
        if "ph" in record.series and "ph" not in record.static:
            t_min, rise = ph_rebound(*record.series["ph"])
        else:
            t_min, rise = np.nan, np.nan
        rows.append({"record_id": record.record_id, "ph_min_time": t_min, "rebound": rise})
    return pd.DataFrame(rows)


def ph_attribution(shap: pd.DataFrame, rebounds: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per record: mean SHAP attribution of all pH features after the pH minimum.

    The attribution is in log10 counts (the model's predicted change, or its correction for
    the residual hybrid); positive means the pH inputs raise the prediction, i.e. slower
    die-off.
    """
    frame = shap[shap["model"] == model].merge(rebounds, on="record_id")
    columns = [c for c in frame.columns if c.startswith("shap_ph_")]
    frame = frame.assign(ph_attribution=frame[columns].sum(axis=1))
    after = frame[frame["time_h"] > frame["ph_min_time"]]
    return after.groupby("record_id").agg(
        rebound=("rebound", "first"), ph_attribution=("ph_attribution", "mean"),
        n=("ph_attribution", "size")).reset_index()


def rebound_test(per_record: pd.DataFrame, threshold: float = REBOUND_THRESHOLD) -> dict:
    """Pre-registered RQ4 test on curves with a pH rebound above ``threshold``.

    (a) the post-minimum pH attribution is positive (one-sided Wilcoxon signed-rank);
    (b) it increases with the rebound size (one-sided Spearman). The Spearman correlation
    over all curves with a pH series is reported as context.
    """
    rebound = per_record[per_record["rebound"] > threshold]
    result = {"n_rebound_curves": int(len(rebound)), "n_all_curves": int(len(per_record))}
    if len(rebound) >= 5:
        result["median_attribution"] = float(rebound["ph_attribution"].median())
        result["wilcoxon_p_greater"] = float(
            stats.wilcoxon(rebound["ph_attribution"], alternative="greater").pvalue)
        rho = stats.spearmanr(rebound["rebound"], rebound["ph_attribution"],
                              alternative="greater")
        result["spearman_rho"], result["spearman_p_greater"] = float(rho.statistic), float(rho.pvalue)
    if len(per_record) >= 5:
        rho = stats.spearmanr(per_record["rebound"], per_record["ph_attribution"])
        result["spearman_rho_all"], result["spearman_p_all"] = float(rho.statistic), float(rho.pvalue)
    return result


# -- RQ5: conformal bands and time-to-safe ----------------------------------------------------

def record_scores(predictions: pd.DataFrame, one_sided: bool = False,
                  after_h: float | None = None) -> pd.Series:
    """Largest error per record; a censored point counts only when over-predicted.

    ``one_sided`` scores only under-prediction (observed above predicted), the error that
    matters for an upper safety bound; censored points then never count. ``after_h`` restricts
    the score to observations from that process time on (records without any score 0).
    """
    error = predictions["log_count"] - predictions["predicted"]  # positive = under-predicted
    if one_sided:
        error = np.where(predictions["censored"], 0.0, error)
    else:
        error = np.where(predictions["censored"], np.maximum(-error, 0.0), np.abs(error))
    error = pd.Series(error, index=predictions.index)
    if after_h is not None:
        error = error.where(predictions["time_h"] >= after_h)
    return error.groupby(predictions["record_id"]).max().fillna(0.0)


def cross_conformal(predictions: pd.DataFrame, alpha: float = 0.1, one_sided: bool = False,
                    after_h: float | None = None) -> pd.DataFrame:
    """Band half-width per held-out fold, calibrated on the other folds' out-of-fold errors.

    Returns one row per record with its fold, score, half-width ``q`` and whether the band
    covers the curve. Two-sided (default, pre-registered): ``prediction +/- q`` covers the
    whole curve; censored points need only the lower band below the limit. One-sided with
    ``after_h``: every detected count from ``after_h`` on lies below ``prediction + q``.
    """
    scores = record_scores(predictions, one_sided, after_h)
    fold_of = predictions.groupby("record_id")["fold"].first()
    rows = []
    for fold in sorted(fold_of.unique()):
        calibration = scores[fold_of != fold]
        q = conformal_halfwidth(calibration.to_numpy(), alpha)
        held_out = scores[fold_of == fold]
        rows.append(pd.DataFrame({"record_id": held_out.index, "fold": fold,
                                  "score": held_out.to_numpy(), "q": q,
                                  "covered": held_out.to_numpy() <= q}))
    return pd.concat(rows, ignore_index=True)


def observed_safe_time(targets: pd.DataFrame, target_log: float) -> float:
    """Earliest time from which every observation is at or below ``target_log``.

    A "not detected" point counts as at or below the target only when its detection limit
    is; inf if the last observation is above the target or unresolved.
    """
    targets = targets.sort_values("time_h")
    below = np.where(targets["censored"], targets["detection_limit"] <= target_log + 1e-9,
                     targets["log_count"] <= target_log + 1e-9)
    if below.all():
        return float(targets["time_h"].iloc[0])
    last_bad = np.flatnonzero(~below)[-1]
    return float(targets["time_h"].iloc[last_bad + 1]) if last_bad + 1 < len(targets) else math.inf


def time_to_safe(
    dense: pd.DataFrame, targets: pd.DataFrame, bands: pd.DataFrame, reduction: float = 2.0,
    after_h: float | None = None,
) -> pd.DataFrame:
    """Predicted and observed time to safety per record, against two targets.

    Targets: ``reduction`` log below the anchor count, and the detection limit. The
    predicted time is the earliest time from which the upper band (``prediction + q``)
    stays at or below the target within the observed period (inf if never). It is
    *confirmed* when every later observation is also at or below the target. The 60-day
    columns compare the conformal verdict at day 60 with what was observed from day 60 on.
    ``after_h`` limits the search to times from which the band is valid (a ripening band).
    """
    if after_h is not None:
        dense = dense[dense["time_h"] >= after_h]
    rows = []
    q_of = bands.set_index("record_id")["q"]
    for record_id, curve in dense.groupby("record_id"):
        observed = targets[targets["record_id"] == record_id]
        if observed.empty or record_id not in q_of.index:
            continue
        anchor = float(observed["anchor_log"].iloc[0])
        limit = float(observed["detection_limit"].iloc[0])
        end = float(observed["time_h"].max())
        upper = curve["predicted"].to_numpy() + q_of[record_id]
        times = curve["time_h"].to_numpy()
        for name, target_log in (("reduction", anchor - reduction), ("detection_limit", limit)):
            predicted = time_to_target(times, upper, target_log)
            observed_time = observed_safe_time(observed, target_log)
            later = observed[observed["time_h"] >= predicted]
            confirmed = (observed_safe_time(later, target_log) <= predicted + 1e-9
                         if math.isfinite(predicted) and not later.empty else np.nan)
            at_60 = observed[observed["time_h"] >= SIXTY_DAYS_H]
            rows.append({
                "record_id": record_id, "food": observed["food"].iloc[0], "target": name,
                "target_log": target_log, "anchor_log": anchor, "end_h": end,
                "q": q_of[record_id], "predicted_safe_h": predicted,
                "observed_safe_h": observed_time, "confirmed": confirmed,
                "observed_past_60d": not at_60.empty,
                "observed_safe_from_60d": (observed_safe_time(at_60, target_log)
                                           <= at_60["time_h"].min() + 1e-9) if not at_60.empty else np.nan,
                "model_safe_at_60d": predicted <= SIXTY_DAYS_H if end >= SIXTY_DAYS_H else np.nan,
            })
    return pd.DataFrame(rows)


# -- RQ2 probe: why measured future trajectories do not help across sources -------------------

def between_group_share(frame: pd.DataFrame, columns, group: str = "source_id") -> pd.DataFrame:
    """Share of each column's variance that lies between groups (one-way eta squared).

    A high share means the column mostly identifies the group (source); a model can then use
    it as a proxy for source-specific behaviour that does not transfer to unseen sources.
    """
    rows = []
    for column in columns:
        data = frame[[group, column]].dropna()
        if data.empty:
            continue
        total = ((data[column] - data[column].mean()) ** 2).sum()
        means = data.groupby(group)[column].transform("mean")
        between = ((means - data[column].mean()) ** 2).sum()
        rows.append({"variable": column, "n": len(data), "groups": data[group].nunique(),
                     "between_share": between / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def gap_by_group(predictions: pd.DataFrame, model: str, group: str = "source_id",
                 modes=("measured", "naive"), origin_h: float = 0.0) -> pd.DataFrame:
    """Squared-error sum per group in two modes and their difference (detected points)."""
    rows = predictions[(predictions["model"] == model) & (predictions["origin_h"] == origin_h)
                       & ~predictions["censored"] & predictions["mode"].isin(modes)]
    rows = rows.assign(squared_error=(rows["predicted"] - rows["log_count"]) ** 2)
    table = rows.pivot_table(index=group, columns="mode", values="squared_error", aggfunc="sum")
    table["n"] = rows[rows["mode"] == modes[0]].groupby(group).size()
    table["difference"] = table[modes[0]] - table[modes[1]]
    table = table.sort_values("difference", ascending=False)
    table["cumulative_share"] = table["difference"].cumsum() / table["difference"].sum()
    return table.reset_index()


def outside_training_range(train: pd.DataFrame, test: pd.DataFrame, columns,
                           quantiles=(0.05, 0.95)) -> pd.Series:
    """Fraction of test rows with each column outside the training rows' quantile range."""
    out = {}
    for column in columns:
        low, high = train[column].quantile(list(quantiles))
        values = test[column].dropna()
        out[column] = float(((values < low) | (values > high)).mean()) if len(values) else np.nan
    return pd.Series(out)
