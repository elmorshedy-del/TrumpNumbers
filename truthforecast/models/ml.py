"""Gradient boosting on quantiles, and ensembles.

The GBM optimizes pinball loss directly, so it is scored on exactly the thing it
was trained on. That is not a small detail: train a model against an improper
loss and it learns to distort, quietly and effectively.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import CONVENTION
from .base import ForecastTask, Model

warnings.filterwarnings("ignore")

GBM_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _week_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First day of each day's week, under the counting convention."""
    dow = index.dayofweek.to_numpy()
    into = (dow + 1) % 7 if CONVENTION.week_anchor == "SUN" else dow
    return index - pd.to_timedelta(into, unit="D")


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
        """Rows are (week, cut) pairs; the target is the remaining-days total.

        The weeks are cut where the *convention* cuts them. Anchoring training
        on Monday while every prediction is made against a Sunday-anchored week
        makes `cut`, `observed_sum` and `days_left` mean one thing in training
        and a different thing at prediction time: cut 0 taught the model
        "Monday is banked, six days to go" and then asked it about a Sunday.
        """
        values = history.to_numpy(dtype=float)
        index = history.index
        feats = _features(index, values)

        X, y = [], []
        starts = _week_starts(index)
        for monday in pd.unique(starts):
            mask = starts == monday
            week_vals = values[mask]
            if len(week_vals) != 7:
                continue
            week_pos = np.where(mask)[0]
            for cut in range(-1, 6):
                remaining = week_vals[cut + 1:].sum()
                # Feature row taken at the cut: the last observed day of the
                # week, or the day before the week starts when cut == -1.
                anchor = week_pos[cut] if cut >= 0 else week_pos[0] - 1
                if anchor < 0:
                    continue
                # NaN lag features are kept rather than dropped. LightGBM treats
                # missing values natively, and the prediction row genuinely has
                # one — the embargoed day sits in lag-1 at the earliest cuts. A
                # training set scrubbed of every NaN teaches the model nothing
                # about what to do when one arrives, which is the whole reason
                # its predictions used to fall back to a crude pool.
                observed_sum = week_vals[: cut + 1].sum() if cut >= 0 else 0.0
                X.append(np.concatenate([feats[anchor], [cut, observed_sum, 6 - cut]]))
                y.append(remaining)
        return (np.array(X), np.array(y)) if X else (np.empty((0, 0)), np.empty(0))

    def fit(self, history: pd.Series) -> "QuantileGBM":
        import lightgbm as lgb

        X, y = self._build_training(history)
        self.fallback = history.to_numpy(dtype=float)
        self._history = history.astype(float)
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
        self._history = history.astype(float)
        return self

    def _anchor_row(self, task: ForecastTask) -> np.ndarray:
        """Features as of the cut — the same row shape training was built from.

        Training anchors each row on the last observed day of that week (or the
        day before the week starts, at cut -1). Predicting from the last day of
        *history* instead would hand the model a Saturday row every time and
        lags that stop before the week began, while training taught it rows
        anchored on every weekday. The model then reads a feature vector from a
        distribution it never saw, which is the ordinary way a gradient booster
        is made to look worthless by its plumbing rather than by its merits.
        """
        # Features come from the history attached to the TASK, not from the copy
        # kept at fit time. This model refits every fourth week, so the fitted
        # copy can be three weeks stale — and lag-1 through lag-14 on a
        # three-week-old series are not lags, they are old news. The fitted
        # trees are what refitting is saving; the input row costs nothing to
        # build fresh.
        source = task.history if len(task.history) else self._history
        series = source[source.index < task.week_start].astype(float)
        if series.empty:
            series = self._history
        if task.cut_dow >= 0 and len(task.observed):
            # Extend the series with the days of the target week already known,
            # so lag and rolling features are anchored where training put them.
            series = pd.concat([
                series,
                pd.Series(np.asarray(task.observed, dtype=float), index=task.observed_dates),
            ])

        # Reindex onto a gapless calendar, running right up to the anchor day
        # training would have used: the last observed day of the week, or the
        # day before the week starts at cut -1. Two things break without it.
        # `_features` computes lags by POSITION, so the embargoed day between
        # history and the target week silently turns "yesterday" into two days
        # ago for every lag on the row, while training — built on a contiguous
        # index — meant yesterday. And at cut -1 the row would be anchored on
        # the last day of history, which the embargo puts a day earlier than the
        # day training anchored on, so the calendar features were for the wrong
        # weekday. Unknown days become NaN: a value LightGBM handles natively,
        # and an honest statement that the day was withheld.
        anchor_day = (
            task.week_start + pd.Timedelta(days=task.cut_dow)
            if task.cut_dow >= 0
            else task.week_start - pd.Timedelta(days=1)
        )
        last = max(anchor_day, series.index[-1])
        series = series.reindex(pd.date_range(series.index[0], last, freq="D"))

        index, values = series.index, series.to_numpy(dtype=float)
        feats = _features(index, values)
        pos = int(index.get_loc(anchor_day))
        return np.concatenate(
            [feats[pos], [task.cut_dow, task.observed_total, 6 - task.cut_dow]]
        ).reshape(1, -1)

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        if not self.trained:
            pool = self.fallback if self.fallback is not None else np.array([0.0])
            return np.random.choice(pool, size=n, replace=True) * len(dows)

        row = self._anchor_row(task)
        # NaN is a legitimate feature value here (see `_build_training`); an
        # infinity is not, and neither is an all-missing row.
        if np.isinf(row).any() or np.isnan(row).all():
            pool = self.fallback if self.fallback is not None else np.array([0.0])
            return np.random.choice(pool, size=n, replace=True) * len(dows)

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
        per = max(n // max(len(self.members), 1), 2000)
        qs = [np.quantile(m.sample_week_total(task, per), grid) for m in self.members]
        averaged = np.mean(qs, axis=0)
        u = np.random.uniform(0.001, 0.999, size=n)
        return np.interp(u, grid, averaged)
