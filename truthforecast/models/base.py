"""The contract every model in the zoo obeys.

One rule drives the design: **a forecast is a distribution, not a number.**
Every model returns samples of the week total, and scoring consumes only those
samples. That is what makes a Poisson GLM and a Hawkes process and a gradient
boosting machine directly comparable, and it is what lets the leaderboard be
graded with proper scoring rules instead of accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import N_PREDICTIVE_SAMPLES


@dataclass
class ForecastTask:
    """Predict one week's total, from a specific moment inside that week.

    `history` contains only days strictly before the week's unobserved portion —
    the walk-forward harness is responsible for that, so a model physically
    cannot see the future it is being asked about.

    `cut_dow` is the last *completed* POSITION in the target week (0 = the
    week's first day .. 6 = its last), or -1 when the week has not started.
    `observed` holds that week's counts for positions 0..cut_dow.

    Position is not the same as weekday once the week can start on a Sunday:
    under the Kalshi convention position 0 is Sunday (pandas weekday 6), and a
    model looking up "weekday 0" for the week's opening day would be reading
    Monday's history to forecast a Sunday. `remaining_dows` does that mapping so
    no model has to know about it.
    """

    history: pd.Series
    week_start: pd.Timestamp
    cut_dow: int
    observed: np.ndarray

    @property
    def remaining_dows(self) -> np.ndarray:
        """Pandas weekday indices (Mon=0..Sun=6) still to be predicted."""
        start = int(self.week_start.dayofweek)
        return np.array([(start + p) % 7 for p in range(self.cut_dow + 1, 7)], dtype=int)

    @property
    def remaining_dates(self) -> pd.DatetimeIndex:
        """The actual calendar days still to be predicted.

        A weekday index is not enough for every model. Anything that forecasts
        *forward from the end of history* — a SARIMA, a state-space filter —
        needs to know how far ahead the first unobserved day is, and that gap is
        not constant: the backtest embargoes a day before the week starts, the
        live path does not, and at a mid-week cut the observed days sit in
        between. Forecasting `len(remaining)` steps from the end of history and
        calling it the rest of the week is off by that gap, and for a
        seasonal model it is off by a phase as well.
        """
        n = 6 - self.cut_dow
        if n <= 0:
            return pd.DatetimeIndex([])
        first = self.week_start + pd.Timedelta(days=self.cut_dow + 1)
        return pd.date_range(first, periods=n, freq="D")

    @property
    def observed_dates(self) -> pd.DatetimeIndex:
        """The days of the target week already banked."""
        if not len(self.observed):
            return pd.DatetimeIndex([])
        return pd.date_range(self.week_start, periods=len(self.observed), freq="D")

    @property
    def gap_days(self) -> int:
        """Unobserved days between the end of history and the target week.

        Zero on the live path, `embargo_days` in the backtest. A filter that
        propagates state forward has to step over these blind rather than
        pretend the week starts the day after history ends.
        """
        if not len(self.history):
            return 0
        return max(int((self.week_start - self.history.index[-1]).days) - 1, 0)

    @property
    def observed_total(self) -> int:
        return int(self.observed.sum()) if len(self.observed) else 0


class Model:
    """Base class. Subclasses implement `fit` and one of the sampling hooks."""

    name: str = "base"
    family: str = "misc"
    description: str = ""
    assumptions: str = ""
    fails_when: str = ""
    # Expensive models can be refit less often during the backtest.
    refit_every_weeks: int = 1
    # Set by models that need post-level timestamps rather than daily counts.
    # The harness calls `fit_events` for these, with events strictly before the
    # forecast origin.
    uses_events: bool = False

    def fit(self, history: pd.Series) -> "Model":
        raise NotImplementedError

    def fit_events(self, times_days: np.ndarray, hours: np.ndarray, span_days: float) -> "Model":
        """Optional hook for models fitted on individual post times."""
        return self

    def sample_week_total(self, task: ForecastTask, n: int = N_PREDICTIVE_SAMPLES) -> np.ndarray:
        """Samples of the *week* total, including the days already observed."""
        remaining = self.sample_remaining(task, n)
        return task.observed_total + remaining

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        """Samples of the sum of the week's not-yet-observed days."""
        raise NotImplementedError

    def sample_one_day(self, date: pd.Timestamp, n: int) -> np.ndarray | None:
        """This model's own distribution for a single whole day, or None.

        Used for the day in progress. A model that can say what an ordinary
        Saturday looks like can have that answer conditioned on how *this*
        Saturday is going (`partial.DayConditioner`), which is strictly better
        than substituting a different estimator for it — and it is what keeps
        the projection continuous when the date rolls over, since at midnight
        the conditioning is the identity and the model is drawing exactly the
        day it was drawing a minute earlier.

        Returning None is allowed: models that only ever produce a *sum* over
        several days (last-week, the bootstraps, the ensembles) genuinely have
        no per-day distribution to offer, and fall back to the empirical one.
        """
        return None

    def meta(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "assumptions": self.assumptions,
            "fails_when": self.fails_when,
        }


class DailyModel(Model):
    """For models that generate a distribution over each remaining day.

    Days are sampled independently *given the fitted parameters*. That is not a
    claim that days are independent — models that carry day-to-day memory
    (INGARCH, the HMM, Hawkes) override `sample_remaining` and propagate their
    own state forward instead.
    """

    def sample_days(
        self, dows: np.ndarray, n: int, dates: pd.DatetimeIndex | None = None
    ) -> np.ndarray:
        """Return an (n, len(dows)) array of sampled daily counts.

        `dates` carries the calendar days those weekdays correspond to, for the
        models whose answer depends on *when* and not only on *which weekday* —
        a linear trend has to be evaluated at the right week, a SARIMA has to be
        stepped the right number of days forward. Models that genuinely only
        need the weekday ignore it.
        """
        raise NotImplementedError

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        return self.sample_days(dows, n, task.remaining_dates).sum(axis=1)

    def sample_one_day(self, date: pd.Timestamp, n: int) -> np.ndarray | None:
        """A daily model always has a view of a single day: ask it for one."""
        try:
            out = self.sample_days(
                np.array([int(date.dayofweek)]), n, pd.DatetimeIndex([date])
            )
        except Exception:
            return None
        return np.asarray(out, dtype=float).ravel()


@dataclass
class Forecast:
    """A model's answer, reduced to what the site and the scorer need."""

    model: str
    samples: np.ndarray = field(repr=False)

    def quantiles(self, levels) -> dict:
        qs = np.quantile(self.samples, list(levels))
        return {str(l): float(q) for l, q in zip(levels, qs)}

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    def prob_at_least(self, threshold: float) -> float:
        return float((self.samples >= threshold).mean())

    def interval(self, coverage: float = 0.80) -> tuple[float, float]:
        a = (1 - coverage) / 2
        lo, hi = np.quantile(self.samples, [a, 1 - a])
        return float(lo), float(hi)


def dow_matrix(index: pd.DatetimeIndex) -> np.ndarray:
    """One-hot day-of-week design matrix (7 columns, no intercept)."""
    dows = index.dayofweek.to_numpy()
    out = np.zeros((len(dows), 7))
    out[np.arange(len(dows)), dows] = 1.0
    return out


def safe_positive(x: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), floor)
