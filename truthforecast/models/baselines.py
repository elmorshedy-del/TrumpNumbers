"""Baselines and resampling models.

These exist to be beaten. A leaderboard without them is unreadable: "CRPS 31.4"
means nothing until you know that simply reusing last week's total scores 38.
Several of them are also genuinely hard to beat, which is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DailyModel, ForecastTask, Model


class EmpiricalDayModel(DailyModel):
    """Climatology. Resample each remaining day from that weekday's history.

    Assumes nothing about shape — no Poisson, no Negative Binomial, no
    distributional form at all. With heavy-tailed counts that is a real
    advantage, and it makes this the reference model for the skill score.
    """

    name = "empirical-dow"
    family = "baseline"
    description = "Resamples each remaining weekday from its own historical values."
    assumptions = "Days are exchangeable within a weekday; no trend, no memory."
    fails_when = "The level shifts, or activity clusters across days within a week."

    def __init__(self, lookback_days: int | None = None):
        self.lookback_days = lookback_days
        self.pools: dict[int, np.ndarray] = {}
        self.fallback = np.array([0.0])

    def fit(self, history: pd.Series) -> "EmpiricalDayModel":
        h = history if self.lookback_days is None else history.iloc[-self.lookback_days:]
        vals = h.to_numpy(dtype=float)
        dows = h.index.dayofweek.to_numpy()
        self.pools = {d: vals[dows == d] for d in range(7)}
        self.fallback = vals if len(vals) else np.array([0.0])
        return self

    def sample_days(self, dows: np.ndarray, n: int) -> np.ndarray:
        out = np.empty((n, len(dows)))
        for j, d in enumerate(dows):
            pool = self.pools.get(int(d))
            if pool is None or len(pool) == 0:
                pool = self.fallback
            out[:, j] = np.random.choice(pool, size=n, replace=True)
        return out


class RecentEmpiricalDayModel(EmpiricalDayModel):
    """Climatology with a short memory — adapts to regime shifts, at the cost of noise."""

    name = "empirical-dow-8w"
    family = "baseline"
    description = "As empirical-dow, but only the last 8 weeks."
    assumptions = "Recent behaviour is more relevant than old behaviour."
    fails_when = "Eight weeks is too thin to see the tail; tail estimates get soft."

    def __init__(self):
        super().__init__(lookback_days=56)


class LastWeekModel(Model):
    """Seasonal naive: this week will look like last week.

    Spread comes from the historical distribution of week-over-week changes, so
    it is a genuine distribution rather than a point forecast with error bars
    bolted on.
    """

    name = "last-week"
    family = "baseline"
    description = "Last week's total, with the historical week-over-week change distribution."
    assumptions = "Consecutive weeks are similar; changes are stationary."
    fails_when = "Volume is trending, or last week was itself an outlier."

    def __init__(self):
        self.deltas = np.array([0.0])
        self.last_total = 0.0
        self.dow_share = np.full(7, 1 / 7)

    def fit(self, history: pd.Series) -> "LastWeekModel":
        weekly = history.resample("W-SUN").sum()
        weekly = weekly.iloc[1:-1] if len(weekly) > 2 else weekly
        vals = weekly.to_numpy(dtype=float)
        self.deltas = np.diff(vals) if len(vals) > 1 else np.array([0.0])
        self.last_total = float(vals[-1]) if len(vals) else 0.0

        by_dow = history.groupby(history.index.dayofweek).mean()
        by_dow = by_dow.reindex(range(7)).fillna(by_dow.mean() if len(by_dow) else 0.0)
        total = by_dow.sum()
        self.dow_share = (by_dow / total).to_numpy() if total > 0 else np.full(7, 1 / 7)
        return self

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        week_samples = self.last_total + np.random.choice(self.deltas, size=n, replace=True)
        share = self.dow_share[dows].sum()
        return np.maximum(week_samples * share, 0.0)


class TrailingMedianModel(Model):
    """Trailing 4-week median week total, scaled to the remaining days.

    The median rather than the mean, deliberately: one 168-post day drags a mean
    somewhere no week actually lives.
    """

    name = "trailing-median-4w"
    family = "baseline"
    description = "Median of the last 4 complete weeks, apportioned over remaining days."
    assumptions = "The recent level is the best guess; weekday shares are stable."
    fails_when = "A regime change happened inside the 4-week window."

    def __init__(self, weeks: int = 4):
        self.weeks = weeks
        self.pool = np.array([0.0])
        self.dow_share = np.full(7, 1 / 7)

    def fit(self, history: pd.Series) -> "TrailingMedianModel":
        weekly = history.resample("W-SUN").sum()
        weekly = weekly.iloc[1:-1] if len(weekly) > 2 else weekly
        vals = weekly.to_numpy(dtype=float)
        self.pool = vals[-self.weeks:] if len(vals) else np.array([0.0])

        by_dow = history.groupby(history.index.dayofweek).mean()
        by_dow = by_dow.reindex(range(7)).fillna(by_dow.mean() if len(by_dow) else 0.0)
        total = by_dow.sum()
        self.dow_share = (by_dow / total).to_numpy() if total > 0 else np.full(7, 1 / 7)
        # Residual spread around the trailing median, for the predictive width.
        med = float(np.median(self.pool)) if len(self.pool) else 0.0
        self.resid = vals - med if len(vals) else np.array([0.0])
        return self

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        centre = float(np.median(self.pool))
        draws = centre + np.random.choice(self.resid, size=n, replace=True)
        share = self.dow_share[dows].sum()
        return np.maximum(draws * share, 0.0)


class BlockBootstrapModel(Model):
    """Resample contiguous week-fragments, not individual days.

    The ordinary bootstrap assumes order does not matter. It does: if heavy days
    cluster, shuffling days independently understates how extreme a week can
    get, and the tail is the whole question. So the resampling unit is the
    intact remainder-of-week — Wednesday-to-Sunday is drawn as one piece from
    some historical Wednesday-to-Sunday, preserving whatever clustering lives
    inside it.

    `IndependentDayBootstrapModel` is the same thing with the blocks broken up,
    kept as the control: if the two agree, days genuinely do not cluster much,
    and that agreement is evidence rather than decoration.
    """

    name = "block-bootstrap"
    family = "resampling"
    description = "Resamples the intact remainder-of-week as a single block."
    assumptions = "Within-week clustering is a stable feature of the process."
    fails_when = "The level shifts; blocks from an old regime mislead."
    blockwise = True

    def __init__(self, lookback_weeks: int | None = None):
        self.lookback_weeks = lookback_weeks
        self.history: pd.Series = pd.Series(dtype=float)

    def fit(self, history: pd.Series) -> "BlockBootstrapModel":
        if self.lookback_weeks:
            history = history.iloc[-self.lookback_weeks * 7:]
        self.history = history.astype(float)
        return self

    def _blocks(self, dows: np.ndarray) -> np.ndarray:
        """Historical totals for the same set of weekdays, kept as whole weeks."""
        h = self.history
        if h.empty:
            return np.array([0.0])
        idx_dow = h.index.dayofweek.to_numpy()
        vals = h.to_numpy(dtype=float)
        # Group by the Monday of each day's week, then sum the requested dows.
        mondays = h.index - pd.to_timedelta(idx_dow, unit="D")
        frame = pd.DataFrame({"v": vals, "dow": idx_dow, "monday": mondays})
        wanted = frame[frame["dow"].isin(set(int(d) for d in dows))]
        by_week = wanted.groupby("monday").agg(total=("v", "sum"), k=("v", "size"))
        complete = by_week[by_week["k"] == len(dows)]["total"].to_numpy(dtype=float)
        return complete if len(complete) else np.array([float(vals.mean() * len(dows))])

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        return np.random.choice(self._blocks(dows), size=n, replace=True)


class IndependentDayBootstrapModel(BlockBootstrapModel):
    """The control for `block-bootstrap`: same data, blocks destroyed."""

    name = "iid-day-bootstrap"
    family = "resampling"
    description = "Resamples each remaining day independently — the no-clustering control."
    assumptions = "Days within a week are independent given the weekday."
    fails_when = "Bursts span days, which is precisely what it cannot represent."
    blockwise = False

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        h = self.history
        if h.empty:
            return np.zeros(n)
        vals = h.to_numpy(dtype=float)
        idx_dow = h.index.dayofweek.to_numpy()
        total = np.zeros(n)
        for d in dows:
            pool = vals[idx_dow == int(d)]
            if len(pool) == 0:
                pool = vals
            total += np.random.choice(pool, size=n, replace=True)
        return total
