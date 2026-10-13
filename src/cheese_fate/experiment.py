"""Grouped cross-validation of all models in every input mode (methodology sections 4-7).

One fold:

1. fits the coupled LAB/pH/aw forecaster on the training records;
2. fits every model on training tasks anchored at each origin, in ``Config.train_mode``
   (measured inputs by default; ``naive`` trains on what a forecaster knows at the origin);
3. predicts the held-out records' tasks in every mode (measured, naive, coupled) and origin;
4. predicts dense trajectories (origin 0, measured and coupled modes) for time-to-safe;
5. computes out-of-fold SHAP values for the tree models (measured mode, origin 0).

Mechanistic parameters are fitted per fold beforehand (:func:`fold_parameters`), so the
mechanistic model and the hybrids use parameters that never saw the held-out sources.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cheese_fate.coupled import CoupledForecaster, coupled_task, evaluate
from cheese_fate.evaluation import grouped_folds
from cheese_fate.mechanistic import Curve, Parameters, fit
from cheese_fate.models import (
    BoostedTrees, Mechanistic, Persistence, ResidualBoostedTrees, with_params, without_lab,
)
from cheese_fate.sequence import NeuralODEHybrid, RecurrentModel
from cheese_fate.tasks import MODES, ORIGINS_H, Record, make_task, stage, time_grid

MODELS = (
    "persistence", "mechanistic", "gbt", "gbt_nolab", "gru", "lstm",
    "residual_gbt", "residual_gbt_nolab", "node_hybrid",
)
TREE_MODELS = ("gbt", "residual_gbt")


@dataclass
class Config:
    models: tuple = MODELS
    origins: tuple = ORIGINS_H
    modes: tuple = MODES
    seed: int = 0
    epochs: int = 400
    patience: int = 40
    hidden: int = 32
    threads: int = 2
    shap: bool = True
    train_mode: str = "measured"  # input mode of the training tasks: measured or naive


@dataclass
class FoldResult:
    predictions: pd.DataFrame
    dense: pd.DataFrame
    shap: pd.DataFrame
    forecaster: pd.DataFrame
    info: dict = field(default_factory=dict)


def fold_assignment(records: list[Record], group: str, n_splits: int, seed: int) -> np.ndarray:
    """Fold index per record; identical to the baseline split for the same group and seed."""
    groups = np.array([getattr(record, group) for record in records])
    assignment = np.empty(len(records), dtype=int)
    for fold, (_, test) in enumerate(grouped_folds(groups, n_splits, seed)):
        assignment[test] = fold
    return assignment


def fold_parameters(
    curves: list[Curve], assignment: np.ndarray, initial: Parameters, free, bounds,
    reuse: pd.DataFrame | None = None, n_jobs: int = 1,
) -> list[Parameters]:
    """Mechanistic parameters per fold, fitted on that fold's training curves.

    ``reuse`` is a baseline ``*_cv_folds.csv`` table for the same split; its fitted values are
    used instead of refitting when it has one row per fold.
    """
    n_folds = int(assignment.max()) + 1
    if reuse is not None and len(reuse) == n_folds:
        return [with_params(initial, row[list(free)].to_dict())
                for _, row in reuse.sort_values("fold").iterrows()]
    jobs = [([c for c, f in zip(curves, assignment) if f != fold], initial, free, bounds)
            for fold in range(n_folds)]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return list(pool.map(_fit_mechanistic, jobs))
    return [_fit_mechanistic(job) for job in jobs]


def _fit_mechanistic(job) -> Parameters:
    train, initial, free, bounds = job
    return fit(train, initial, free=free, bounds=bounds)[0]


def build_models(params: Parameters, config: Config) -> dict:
    mech = Mechanistic(params)
    neural = {"epochs": config.epochs, "patience": config.patience, "hidden": config.hidden,
              "seed": config.seed}
    factories = {
        "persistence": lambda: Persistence(),
        "mechanistic": lambda: mech,
        "gbt": lambda: BoostedTrees(seed=config.seed),
        "gbt_nolab": lambda: BoostedTrees(without_lab(), config.seed, name="gbt_nolab"),
        "gru": lambda: RecurrentModel("gru", **neural),
        "lstm": lambda: RecurrentModel("lstm", **neural),
        "residual_gbt": lambda: ResidualBoostedTrees(mech, seed=config.seed),
        "residual_gbt_nolab": lambda: ResidualBoostedTrees(
            mech, without_lab(), config.seed, name="residual_gbt_nolab"),
        "node_hybrid": lambda: NeuralODEHybrid(mech, **neural),
    }
    return {name: factories[name]() for name in config.models}


def run_fold(job) -> FoldResult:
    # Histogram boosting on a few thousand rows is far slower with many OpenMP threads
    # (oversubscription), so each fold uses one; folds run in parallel processes instead.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1), warnings.catch_warnings():
        # Rejected trial steps of the ODE solver overflow harmlessly before being retried.
        warnings.filterwarnings("ignore", "overflow encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        return _run_fold(job)


def _run_fold(job) -> FoldResult:
    fold, train, test, params, config = job
    import torch

    torch.set_num_threads(config.threads)
    info = {"fold": fold, "train_records": len(train), "test_records": len(test),
            "test_groups": ";".join(sorted({r.source_id for r in test}))}
    started = time.time()
    forecaster = CoupledForecaster(seed=config.seed).fit(train)
    info["forecaster_s"] = round(time.time() - started, 1)

    train_tasks = [t for r in train for o in config.origins
                   if (t := make_task(r, o, mode=config.train_mode)) is not None]
    models = build_models(params, config)
    for name, model in models.items():
        started = time.time()
        model.fit(train_tasks)
        info[f"{name}_fit_s"] = round(time.time() - started, 1)
        if hasattr(model, "params_"):
            info[f"{name}_params"] = {k: model.params_[k] for k in ("max_leaf_nodes", "min_samples_leaf")}
        if hasattr(model, "best_epoch_"):
            info[f"{name}_best_epoch"] = model.best_epoch_
            info[f"{name}_valid_rmse"] = round(model.valid_rmse_, 3)

    tasks = {}
    for origin in config.origins:
        measured = [t for r in test if (t := make_task(r, origin)) is not None]
        for mode in config.modes:
            if mode == "measured":
                tasks[mode, origin] = measured
            elif mode == "naive":
                tasks[mode, origin] = [make_task(t.record, origin, mode="naive") for t in measured]
            else:
                tasks[mode, origin] = [coupled_task(t, forecaster) for t in measured]

    frames = []
    for (mode, origin), mode_tasks in tasks.items():
        if not mode_tasks:
            continue
        for name, model in models.items():
            predicted = model.predict(mode_tasks)
            frames.append(_rows(mode_tasks, predicted, name, mode, origin, fold))
    predictions = pd.concat(frames, ignore_index=True)

    dense_frames = []
    for mode in ("measured", "coupled"):
        mode_tasks = tasks.get((mode, config.origins[0]), [])
        grids = [time_grid(t.anchor_time, t.times, {}) for t in mode_tasks]
        for name, model in models.items():
            predicted = model.predict(mode_tasks, grids)
            for task, grid, values in zip(mode_tasks, grids, predicted):
                dense_frames.append(pd.DataFrame({
                    "record_id": task.record.record_id, "model": name, "mode": mode,
                    "fold": fold, "time_h": grid, "predicted": values,
                }))
    dense = pd.concat(dense_frames, ignore_index=True) if dense_frames else pd.DataFrame()

    shap_frame = pd.DataFrame()
    if config.shap:
        shap_frame = _shap(models, tasks.get(("measured", config.origins[0]), []), fold)
    forecaster_eval = evaluate(forecaster, test, config.origins).assign(fold=fold)
    return FoldResult(predictions, dense, shap_frame, forecaster_eval, info)


def _rows(tasks, predicted, name, mode, origin, fold) -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": np.concatenate([[t.record.record_id] * t.times.size for t in tasks]),
        "source_id": np.concatenate([[t.record.source_id] * t.times.size for t in tasks]),
        "food": np.concatenate([[t.record.food] * t.times.size for t in tasks]),
        "model": name, "mode": mode, "origin_h": origin, "fold": fold,
        "anchor_time": np.concatenate([np.full(t.times.size, t.anchor_time) for t in tasks]),
        "anchor_log": np.concatenate([np.full(t.times.size, t.anchor_log) for t in tasks]),
        "time_h": np.concatenate([t.times for t in tasks]),
        "horizon_h": np.concatenate([t.times - t.cutoff for t in tasks]),
        "stage": np.concatenate([stage(t.times) for t in tasks]),
        "log_count": np.concatenate([t.log_counts for t in tasks]),
        "censored": np.concatenate([t.censored for t in tasks]),
        "detection_limit": np.concatenate([np.full(t.times.size, t.record.detection_limit)
                                           for t in tasks]),
        "predicted": np.concatenate(predicted),
    })


def _shap(models: dict, tasks: list, fold: int) -> pd.DataFrame:
    import shap

    frames = []
    for name in TREE_MODELS:
        model = models.get(name)
        if model is None or not tasks:
            continue
        X = model.design(tasks)
        values = shap.TreeExplainer(model.model_.model_).shap_values(X)
        frame = pd.DataFrame(values, columns=[f"shap_{c}" for c in X.columns])
        frame = pd.concat([X.add_prefix("x_").reset_index(drop=True), frame], axis=1)
        frames.append(frame.assign(
            model=name, fold=fold,
            record_id=np.concatenate([[t.record.record_id] * t.times.size for t in tasks]),
            time_h=np.concatenate([t.times for t in tasks]),
        ))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cross_validate(
    records: list[Record], assignment: np.ndarray, params: list[Parameters], config: Config,
    n_jobs: int = 1,
) -> FoldResult:
    """Run every fold and concatenate the results."""
    jobs = [
        (fold, [r for r, f in zip(records, assignment) if f != fold],
         [r for r, f in zip(records, assignment) if f == fold], params[fold], config)
        for fold in range(len(params))
    ]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(run_fold, jobs))
    else:
        results = [run_fold(job) for job in jobs]
    return FoldResult(
        predictions=pd.concat([r.predictions for r in results], ignore_index=True),
        dense=pd.concat([r.dense for r in results], ignore_index=True),
        shap=pd.concat([r.shap for r in results], ignore_index=True),
        forecaster=pd.concat([r.forecaster for r in results], ignore_index=True),
        info={"folds": [r.info for r in results]},
    )
