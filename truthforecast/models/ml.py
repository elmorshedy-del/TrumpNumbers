"""Gradient boosting on quantiles, and ensembles.

The GBM optimizes pinball loss directly, so it is scored on exactly the thing it
was trained on. That is not a small detail: train a model against an improper
loss and it learns to distort, quietly and effectively.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastTask, Model

warnings.filterwarnings("ignore")

GBM_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _features(index: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
    """Calendar plus lag features for a daily count series."""
    n = len(values)
    dow = index.dayofweek.to_numpy()
    feats = [
        dow,
        index.month.to_numpy(),
        (dow >= 5).astype(float),
    ]
    for lag in (1, 2, 3, 7, 14):
        lagged = np.full(n, np.nan)
        if n > lag:
            lagged[lag:] = values[:-lag]
        feats.append(lagged)
    for win in (7, 28):
        roll = pd.Series(values).shift(1).rolling(win, min_periods=2).mean().to_numpy()
        feats.append(roll)
    roll_sd = pd.Series(values).shift(1).rolling(28, min_periods=5).std().to_numpy()
    feats.append(roll_sd)
    return np.column_stack(feats)


class QuantileGBM(Model):
    """LightGBM quantile regression on the remaining-week total.

    Trained directly on the target the leaderboard scores: for each historical
    week and each weekday cut, predict the sum of that week's remaining days.
    One model per quantile, assembled into a predictive distribution.
    """

    name = "lgbm-quantile"
    family = "ml"
    description = "LightGBM quantile regression over calendar and lag features."
    assumptions = "Recent history and the calendar carry the signal; enough rows to learn from."
    fails_when = "Data is thin, or the pattern it learned has since changed."
    refit_every_weeks = 4

    def __init__(self):
        self.models: dict[float, object] = {}
        self.fallback: np.ndarray | None = None
        self.trained = False

    def _build_training(self, history: pd.Series):
        """Rows are (week, cut) pairs; the target is the remaining-days total."""
        values = history.to_numpy(dtype=float)
        index = history.index
        feats = _features(index, values)

        X, y = [], []
        mondays = index - pd.to_timedelta(index.dayofweek, unit="D")
        for monday in pd.unique(mondays):
            mask = mondays == monday
            week_vals = values[mask]
            if len(week_vals) != 7:
                continue
            week_pos = np.where(mask)[0]
            for cut in range(-1, 6):
                remaining = week_vals[cut + 1:].sum()
                # Feature row taken at the cut: the last observed day of the
                # week, or the day before the week starts when cut == -1.
                anchor = week_pos[cut] if cut >= 0 else week_pos[0] - 1
                if anchor < 0 or np.isnan(feats[anchor]).any():
                    continue
                observed_sum = week_vals[: cut + 1].sum() if cut >= 0 else 0.0
                X.append(np.concatenate([feats[anchor], [cut, observed_sum, 6 - cut]]))
                y.append(remaining)
        return (np.array(X), np.array(y)) if X else (np.empty((0, 0)), np.empty(0))

    def fit(self, history: pd.Series) -> "QuantileGBM":
        import lightgbm as lgb

        X, y = self._build_training(history)
        self.fallback = history.to_numpy(dtype=float)
        self.trained = False
        if len(y) < 80:
            return self

        self.models = {}
        for q in GBM_QUANTILES:
            try:
                m = lgb.LGBMRegressor(
                    objective="quantile", alpha=q, n_estimators=150, learning_rate=0.06,
                    num_leaves=7, min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                    verbose=-1, random_state=0,
                )
                m.fit(X, y)
                self.models[q] = m
            except Exception:
                pass
        self.trained = len(self.models) == len(GBM_QUANTILES)
        self._last_feats = _features(history.index, history.to_numpy(dtype=float))
        self._last_index = history.index
        self._last_values = history.to_numpy(dtype=float)
        return self

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        if not self.trained:
            pool = self.fallback if self.fallback is not None else np.array([0.0])
            return np.random.choice(pool, size=n, replace=True) * len(dows)

        row = np.concatenate([
            self._last_feats[-1], [task.cut_dow, task.observed_total, 6 - task.cut_dow]
        ]).reshape(1, -1)

        qs, preds = [], []
        for q, m in sorted(self.models.items()):
            try:
                preds.append(float(m.predict(row)[0]))
                qs.append(q)
            except Exception:
                pass
        if not preds:
            return np.zeros(n)

        # Enforce monotone quantiles (boosting fits them independently, so they
        # can cross), then invert by interpolation to get samples.
        preds = np.maximum.accumulate(np.maximum(preds, 0.0))
        u = np.random.uniform(qs[0], qs[-1], size=n)
        return np.interp(u, qs, preds)


class LinearPoolEnsemble(Model):
    """Mix the member distributions by pooling their samples.

    A linear opinion pool averages the *densities*, which makes the result at
    least as wide as its members — it is honest about disagreement rather than
    averaging it away. Contrast `VincentizedEnsemble`, which averages quantiles
    and stays sharp.
    """

    name = "ensemble-pool"
    family = "ensemble"
    description = "Linear opinion pool over member models (mixture of predictive densities)."
    assumptions = "Members are individually reasonable; disagreement is real uncertainty."
    fails_when = "One member is badly wrong — the pool inherits its tail."

    def __init__(self, members: list[Model]):
        self.members = members

    def fit(self, history: pd.Series) -> "LinearPoolEnsemble":
        for m in self.members:
            m.fit(history)
        return self

    def sample_week_total(self, task: ForecastTask, n: int = 20000) -> np.ndarray:
        per = max(n // max(len(self.members), 1), 1)
        return np.concatenate([m.sample_week_total(task, per) for m in self.members])


class VincentizedEnsemble(LinearPoolEnsemble):
    """Quantile averaging — averages the *positions*, not the densities.

    Sharper than the linear pool, and usually better calibrated when the members
    agree on shape but disagree on level.
    """

    name = "ensemble-vincent"
    family = "ensemble"
    description = "Quantile-averaged (Vincentized) combination of member models."
    assumptions = "Members share a shape; their disagreement is mostly about level."
    fails_when = "Members are genuinely bimodal — averaging quantiles invents a middle."

    def sample_week_total(self, task: ForecastTask, n: int = 20000) -> np.ndarray:
        grid = np.linspace(0.001, 0.999, 512)
        qs = [np.quantile(m.sample_week_total(task, 4000), grid) for m in self.members]
        averaged = np.mean(qs, axis=0)
        u = np.random.uniform(0.001, 0.999, size=n)
        return np.interp(u, grid, averaged)
