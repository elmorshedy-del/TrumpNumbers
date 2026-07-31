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

    `cut_dow` is the last *completed* day of the target week (0=Mon..6=Sun),
    or -1 when the week has not started. `observed` holds that week's counts for
    days 0..cut_dow.
    """

    history: pd.Series
    week_start: pd.Timestamp
    cut_dow: int
    observed: np.ndarray

    @property
    def remaining_dows(self) -> np.ndarray:
        """Day-of-week indices still to be predicted."""
        return np.arange(self.cut_dow + 1, 7)

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

    def sample_days(self, dows: np.ndarray, n: int) -> np.ndarray:
        """Return an (n, len(dows)) array of sampled daily counts."""
        raise NotImplementedError

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        return self.sample_days(dows, n).sum(axis=1)


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
