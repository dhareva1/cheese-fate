"""Neural models integrated along each curve's time grid (PyTorch).

Both models run over :func:`cheese_fate.tasks.geometric_grid` from the anchor (fine during
manufacture, coarse during ripening) and are read at any query time by linear interpolation
between grid points, so irregular and sparse sampling needs no fixed time step. The grid
depends only on the anchor, never on how often conditions were measured or which times are
queried, so the number of recurrent or Euler steps carries no source-specific artifact and a
curve's prediction at a time does not depend on the other query times. Features on the grid
are still integrated exactly (:func:`cheese_fate.tasks.features`).

- :class:`RecurrentModel` (ML-only, ``gru`` or ``lstm``): a recurrent network reads the
  conditions at each grid point and the step length, and outputs the change from the anchor.
- :class:`NeuralODEHybrid` (universal differential equation): the population is the
  mechanistic trajectory ``m(t)`` plus a correction state ``d(t)`` with

      dd/dt = r_max * tanh(MLP(conditions(t), LAB(t), m(t), dm/dt, d(t)))

  integrated by explicit Euler on the grid. The output layer starts at zero, so training
  starts from the mechanistic model, and the bounded rate keeps the integration stable.

Training is full-batch Adam on the censored squared error (censored points only penalise
over-prediction) with early stopping on held-out training sources.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import torch
from torch import nn

from cheese_fate.evaluation import grouped_folds
from cheese_fate.models import LOG_RANGE, Mechanistic, target_times
from cheese_fate.tasks import FEATURES, Task, features, geometric_grid

MASKED = ("ph_now", "aw_now", "lab_now")


class _Scaler:
    def fit(self, X: np.ndarray):
        with warnings.catch_warnings():  # all-missing columns get mean 0 and SD 1 below
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        self.std_ = np.where(~np.isfinite(self.std_) | (self.std_ < 1e-8), 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num((X - self.mean_) / self.std_, nan=0.0)


class _Batch:
    """Padded tensors for a list of tasks: inputs on the grid and gathered targets."""

    def __init__(self, inputs, dt, mech, anchor, positions, weights, y, censored, target_mask,
                 sizes):
        # ``mech`` holds the mechanistic prediction at the query times (exact, not interpolated)
        self.inputs, self.dt, self.mech, self.anchor = inputs, dt, mech, anchor
        self.positions, self.weights = positions, weights
        self.y, self.censored, self.target_mask = y, censored, target_mask
        self.sizes = sizes

    def subset(self, index):
        index = torch.as_tensor(index)
        return _Batch(self.inputs[index], self.dt[index], self.mech[index], self.anchor[index],
                      self.positions[index], self.weights[index], self.y[index],
                      self.censored[index], self.target_mask[index],
                      [self.sizes[i] for i in index.tolist()])

    def read(self, on_grid: torch.Tensor) -> torch.Tensor:
        """Values at the query times, interpolated linearly between grid points."""
        left = torch.gather(on_grid, 1, self.positions)
        right = torch.gather(on_grid, 1, self.positions + 1)
        return left + self.weights * (right - left)


class _GridModel:
    """Shared batching, scaling and training for the grid models."""

    name = "grid"
    uses_mechanistic = False

    def __init__(self, feature_names=FEATURES, hidden: int = 32, epochs: int = 400,
                 patience: int = 40, lr: float = 3e-3, weight_decay: float = 1e-4,
                 seed: int = 0, mechanistic: Mechanistic | None = None, name: str | None = None):
        self.feature_names = list(feature_names)
        self.hidden, self.epochs, self.patience = hidden, epochs, patience
        self.lr, self.weight_decay, self.seed = lr, weight_decay, seed
        self.mechanistic = mechanistic
        if name:
            self.name = name

    # -- data -------------------------------------------------------------------------
    def _raw(self, task: Task, times: np.ndarray):
        grid = geometric_grid(task.anchor_time, float(np.max(times)))
        frame = features(task, grid)
        X = frame[self.feature_names].to_numpy(dtype=float)
        dt = np.append(np.diff(grid), 0.0)
        masks = np.column_stack([~np.isnan(frame[name].to_numpy()) for name in MASKED])
        if self.uses_mechanistic:
            mech = self.mechanistic.predict([task], [grid])[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(dt > 0, np.diff(mech, append=mech[-1]) / np.where(dt > 0, dt, 1), 0)
            X = np.column_stack([X, mech, np.clip(rate, -5.0, 5.0)])
        else:
            mech = np.full(grid.shape, task.anchor_log)
        X = np.column_stack([X, np.log1p(dt)])
        return grid, X, masks, dt, mech

    def _batch(self, tasks, times=None, fit_scaler: bool = False) -> _Batch:
        targets = target_times(tasks, times)
        raw = [self._raw(task, t) for task, t in zip(tasks, targets)]
        if fit_scaler:
            self.scaler_ = _Scaler().fit(np.vstack([X for _, X, *_ in raw]))
        n, length = len(tasks), max(g.size for g, *_ in raw)
        width = raw[0][1].shape[1] + len(MASKED)
        m = max(t.size for t in targets)
        inputs = np.zeros((n, length, width), np.float32)
        dt = np.zeros((n, length), np.float32)
        mech = np.zeros((n, m), np.float32)
        positions = np.zeros((n, m), np.int64)
        weights = np.zeros((n, m), np.float32)
        y = np.zeros((n, m), np.float32)
        censored = np.zeros((n, m), bool)
        target_mask = np.zeros((n, m), bool)
        for i, ((grid, X, masks, step, trajectory), task, t) in enumerate(zip(raw, tasks, targets)):
            inputs[i, : grid.size] = np.column_stack([self.scaler_.transform(X), masks])
            dt[i, : grid.size] = step
            if self.uses_mechanistic:
                mech[i, : t.size] = self.mechanistic.predict([task], [t])[0]
            left = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, grid.size - 2)
            positions[i, : t.size] = left
            weights[i, : t.size] = np.clip((t - grid[left]) / (grid[left + 1] - grid[left]), 0, 1)
            target_mask[i, : t.size] = True
            if times is None:
                y[i, : t.size] = task.log_counts
                censored[i, : t.size] = task.censored
        anchor = np.array([task.anchor_log for task in tasks], np.float32)
        return _Batch(*(torch.as_tensor(a) for a in (inputs, dt, mech, anchor, positions, weights,
                                                     y, censored, target_mask)),
                      [t.size for t in targets])

    # -- training ---------------------------------------------------------------------
    def _loss(self, batch: _Batch, module) -> torch.Tensor:
        predicted = module(batch)
        residual = predicted - batch.y
        residual = torch.where(batch.censored, torch.relu(residual), residual)
        return (residual[batch.target_mask] ** 2).mean()

    def fit(self, tasks: list[Task]):
        torch.manual_seed(self.seed)
        batch = self._batch(tasks, fit_scaler=True)
        groups = np.array([task.record.source_id for task in tasks])
        n_folds = min(5, np.unique(groups).size)
        train, valid = next(grouped_folds(groups, n_folds, self.seed))
        self.module_ = self._module(batch.inputs.shape[-1])
        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        train_batch, valid_batch = batch.subset(train), batch.subset(valid)
        best, best_state, best_epoch, history = np.inf, None, 0, []
        for epoch in range(self.epochs):
            self.module_.train()
            optimizer.zero_grad()
            loss = self._loss(train_batch, self.module_)
            loss.backward()
            nn.utils.clip_grad_norm_(self.module_.parameters(), 1.0)
            optimizer.step()
            self.module_.eval()
            with torch.no_grad():
                valid_loss = float(self._loss(valid_batch, self.module_))
            history.append((epoch, loss.item(), valid_loss))
            if valid_loss < best - 1e-5:
                best, best_state, best_epoch = valid_loss, copy.deepcopy(self.module_.state_dict()), epoch
            elif epoch - best_epoch >= self.patience:
                break
        self.module_.load_state_dict(best_state)
        self.best_epoch_, self.valid_rmse_, self.history_ = best_epoch, float(np.sqrt(best)), history
        return self

    def predict(self, tasks, times=None):
        batch = self._batch(tasks, times)
        self.module_.eval()
        with torch.no_grad():
            predicted = self.module_(batch).numpy()
        return [np.clip(predicted[i, :size], *LOG_RANGE) for i, size in enumerate(batch.sizes)]


class _Recurrent(nn.Module):
    def __init__(self, width: int, hidden: int, cell: str):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(width, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: _Batch) -> torch.Tensor:
        states, _ = self.rnn(batch.inputs)
        change = self.head(states).squeeze(-1)
        on_grid = batch.anchor[:, None] + change
        return batch.read(on_grid)


class RecurrentModel(_GridModel):
    """ML-only GRU or LSTM over the condition sequence; predicts the change from the anchor."""

    def __init__(self, cell: str = "gru", **kwargs):
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        kwargs.setdefault("name", cell)
        super().__init__(**kwargs)
        self.cell = cell

    def _module(self, width: int) -> nn.Module:
        return _Recurrent(width, self.hidden, self.cell)


class _Correction(nn.Module):
    def __init__(self, width: int, hidden: int, max_rate: float, floor: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width + 1, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_rate, self.floor = max_rate, floor

    def forward(self, batch: _Batch) -> torch.Tensor:
        n, length, _ = batch.inputs.shape
        correction = torch.zeros(n)
        states = [correction]
        for k in range(length - 1):
            x = torch.cat([batch.inputs[:, k], correction[:, None]], dim=1)
            rate = self.max_rate * torch.tanh(self.net(x).squeeze(-1))
            correction = correction + batch.dt[:, k] * rate
            states.append(correction)
        correction = batch.read(torch.stack(states, dim=1))
        return torch.clamp(batch.mech + correction, min=self.floor)


class NeuralODEHybrid(_GridModel):
    """Mechanistic trajectory plus a learned correction ODE (universal differential equation).

    ``max_rate`` bounds the correction in log10 per hour.
    """

    name = "node_hybrid"
    uses_mechanistic = True

    def __init__(self, mechanistic: Mechanistic, max_rate: float = 0.5, **kwargs):
        super().__init__(mechanistic=mechanistic, **kwargs)
        self.max_rate = max_rate

    def _module(self, width: int) -> nn.Module:
        return _Correction(width, self.hidden, self.max_rate, LOG_RANGE[0])
