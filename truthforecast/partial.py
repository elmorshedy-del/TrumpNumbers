"""Today, half-finished — and the fact that "half" is a continuous quantity.

The week forecast has to say something about a day that is under way. Two
obvious answers are both wrong, and measurably so:

* **Ignore it** and draw a whole fresh day from history. Never biased, never
  informative: CRPS sits at 8.53 from midnight to midnight, so at 9pm — when
  the day is 92% decided — it is still guessing.
* **Scale it up**, `posts_so_far / fraction_of_day_elapsed`. Far worse where it
  matters most: CRPS 31.50 at 6am against 8.53 for ignoring it, because
  dividing by a small fraction multiplies the morning's noise. Its failure case
  is stark — nothing posted by noon predicts a total of zero, on a record of
  556 days containing no zero days at all.

The Poisson-Gamma conjugate update is the textbook third answer and it also
loses to ignoring the day until noon (13.77 at 6am). It assumes posts arrive on
a fixed daily clock, so an empty morning is strong evidence of a quiet day. His
mornings are not evidence of that: on the 44 days with nothing posted by noon
the median finish was 6 and the maximum was 69. He starts when he starts.

What works is to assume nothing about the clock: find days that looked like
today looks now, and use what they went on to do.

## Why this file was rewritten a second time

The estimator above was right and its *plumbing* was not. Three defects, all of
which showed up as the projection moving when nothing had happened:

**It was a step function in time.** Conditioning ran on `hour <= now.hour`, so
00:01 and 00:59 returned identical answers and 01:01 jumped by 1.5 posts. Worse,
today's partial hour was compared against history's *complete* one: at 14:01
today has had one minute of hour 14 and was being matched against days that had
had sixty. Time is continuous, so the conditioning is now continuous — elapsed
seconds into the local day, on both sides of the comparison.

**Early in the day the conditioning was degenerate.** With nothing yet posted,
408 of 509 historical days tied at distance zero, and `argsort` kept an
arbitrary 60 of them. The answer depended on numpy's tie-breaking. Weights now
fall off smoothly with distance, so when the conditioning carries no
information every day counts and nothing is discarded for being 61st.

**The projection jumped at midnight.** This is the one worth understanding. At
23:59 tomorrow was a *model's* whole-day draw; at 00:01 the same day became
*this* estimator's problem — two different objects, disagreeing by about two
posts, swapped at the stroke of a clock with no information arriving. The fix is
not to reconcile the two numbers but to stop having two: this file no longer
produces a day distribution of its own. It produces the **conditional map** —
the monotone transform taking a day's unconditional distribution to its
distribution given what today has done so far — and each model's own day
distribution is pushed through it. At elapsed time zero the map is the identity,
so the day a model was drawing at 23:59 is exactly the day it draws at 00:01,
and the projection is continuous across midnight by construction rather than by
coincidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .combine import GRID
from .config import LOCAL_TZ

# How much a day differing by one "normalised day's worth" of posts-so-far is
# discounted. Small enough to stay conditional on what today looks like, wide
# enough that the pool is never one lucky day.
BANDWIDTH = 0.35

# Days on a different weekday still count, at this weight. A Saturday's shape
# is mostly told by Saturdays, but 80 Saturdays is a thin sample for a tail.
OFF_WEEKDAY_WEIGHT = 0.35

# Below this many usable history days, conditioning is noise; fall back to
# drawing a whole day, which is at least never biased.
MIN_HISTORY_DAYS = 60

SECONDS_PER_DAY = 86400.0


def elapsed_seconds(now: pd.Timestamp) -> float:
    """Seconds into the local day. Continuous, unlike an hour index."""
    return float((now - now.normalize()).total_seconds())


def _into_day_seconds(events: pd.DataFrame) -> pd.Series:
    """Seconds into the local day for each post.

    Uses the post's own timestamp when there is one. The `hour`-only fallback
    exists for callers holding a reduced frame; it is hour-quantised, which is
    the very thing this module was rewritten to stop doing, so it is a fallback
    and not a path anything real should be taking.
    """
    if "local" in events:
        local = events["local"]
        if getattr(local.dt, "tz", None) is not None:
            local = local.dt.tz_localize(None)
        return (local - events["local_date"]).dt.total_seconds()
    return events["hour"].astype(float) * 3600.0


def _posts_before(events: pd.DataFrame, cutoff_s: float) -> pd.Series:
    """Per-day count of posts landing at or before `cutoff_s` seconds into the day."""
    if events.empty:
        return pd.Series(dtype="int64")
    return events[_into_day_seconds(events) <= cutoff_s].groupby("local_date").size()


def _day_frame(events: pd.DataFrame, cutoff_s: float) -> pd.DataFrame:
    """Per-day totals and posts-so-far-at-this-time, zero-filled.

    Zero-filling is load-bearing: grouping the event frame alone drops every day
    that produced nothing, which are exactly the days that tell the estimator a
    quiet day is possible.
    """
    counts = events.groupby("local_date").size().sort_index()
    if counts.empty:
        return pd.DataFrame(columns=["total", "upto"])
    span = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    total = counts.reindex(span, fill_value=0).astype(float)
    upto = _posts_before(events, cutoff_s).reindex(span, fill_value=0).astype(float)
    return pd.DataFrame({"total": total, "upto": upto}, index=span)


def _weekday_levels(frame: pd.DataFrame) -> np.ndarray:
    """Mean day total per weekday, with a floor so division is always safe."""
    dow = frame.index.dayofweek.to_numpy()
    overall = max(float(frame["total"].mean()), 1e-6)
    out = np.array([
        float(frame["total"].to_numpy()[dow == d].mean()) if (dow == d).any() else overall
        for d in range(7)
    ])
    return np.where(out > 0, out, overall)


def _slope(run: float, rise: float) -> float:
    """Slope of the conditional map at an edge, safe when the pool is degenerate.

    A zero run means every day in the pool had the same total, so there is no
    rank information to extrapolate along and the honest extension is flat.
    """
    return float(rise / run) if run > 1e-9 else 0.0


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantiles of a weighted empirical distribution."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0:
        return np.full(len(levels), float(np.median(values)) if len(values) else 0.0)
    # Midpoint convention: each atom spans its own share of the mass.
    cum = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(levels, cum, v)


class DayConditioner:
    """The map from "a whole day, unconditionally" to "what is LEFT of this one".

    Holds two quantile functions. `baseline` is the distribution of a day's
    *total*, with weekday weighting and nothing else. `conditional` is the
    distribution of the *remainder* — what days like today still had left at
    this moment — over the same pool, reweighted toward days whose progress
    looked like today's.

    Mapping totals to remainders rather than totals to totals is what makes the
    estimator behave at both ends of the day, and neither end is an accident:

    * At elapsed time zero nothing has happened on any day, so every remainder
      *is* that day's total and every conditioning weight is exactly one. The
      two curves coincide and the map is the identity — a model's day comes
      back unchanged, which is precisely what that same model was drawing for
      that same day one minute before midnight.
    * At the end of the day every matched day has essentially nothing left, so
      the conditional collapses to zero whatever the bandwidth. Going through
      totals instead made this fragile: it had to subtract the posts already
      banked, and any looseness in the matching came back as phantom posts in
      the last minutes of the day.
    """

    def __init__(self, baseline: np.ndarray, conditional: np.ndarray, posts_so_far: int,
                 elapsed: float, n_days: int, effective_n: float):
        self.baseline = np.maximum.accumulate(np.asarray(baseline, dtype=float))
        self.conditional = np.maximum.accumulate(np.asarray(conditional, dtype=float))
        self.posts_so_far = int(posts_so_far)
        self.elapsed = float(elapsed)
        self.n_days = int(n_days)
        self.effective_n = float(effective_n)

    @property
    def is_identity(self) -> bool:
        """True when the conditioning is doing nothing — i.e. at the day's start."""
        return bool(np.allclose(self.baseline, self.conditional, rtol=1e-3, atol=1e-6))

    def remainder(self, day_samples: np.ndarray) -> np.ndarray:
        """What is left of today, given a model's own view of the whole day.

        A rank-preserving map: a model that thinks today is a busy day is told
        what busy days still had left at this hour, not what an average one did.
        """
        x = np.asarray(day_samples, dtype=float)
        y = np.interp(x, self.baseline, self.conditional)
        # `np.interp` clamps outside its range, which would flatten both tails
        # of a heavy-tailed count onto a single point. Extend along the map's
        # own slope at each end instead.
        #
        # The slope matters, and a shift will not do: late in the day the
        # conditional is compressed toward zero, so translating a model's
        # extreme day out past the end of the pool invents posts in a day that
        # has no time left in it. Carrying the slope means a compressed map
        # compresses the tail too, and at elapsed zero — where the map is the
        # identity and the slope is one — the tail passes through untouched.
        edge = max(len(GRID) // 10, 1)
        lo = _slope(self.baseline[edge] - self.baseline[0],
                    self.conditional[edge] - self.conditional[0])
        hi = _slope(self.baseline[-1] - self.baseline[-1 - edge],
                    self.conditional[-1] - self.conditional[-1 - edge])
        y = np.where(x < self.baseline[0],
                     self.conditional[0] + (x - self.baseline[0]) * lo, y)
        y = np.where(x > self.baseline[-1],
                     self.conditional[-1] + (x - self.baseline[-1]) * hi, y)
        return np.maximum(y, 0.0)

    def sample_remainder(self, n: int) -> np.ndarray:
        """Draw the remainder directly, for callers with no day distribution."""
        u = np.random.uniform(GRID[0], GRID[-1], size=n)
        return np.maximum(np.interp(u, GRID, self.conditional), 0.0)


def build_conditioner(
    events: pd.DataFrame,
    now: pd.Timestamp,
    window_start: str | None = None,
    bandwidth: float = BANDWIDTH,
) -> DayConditioner:
    """Build today's conditional map from the days that came before it.

    Days are matched on posts-so-far normalised by that weekday's own level, so
    a quiet Saturday morning is compared against other quiet mornings rather
    than against Mondays, and their day totals are rescaled to today's weekday
    level on the way back out.
    """
    today = now.normalize()
    cutoff = elapsed_seconds(now)
    dow = int(now.dayofweek)

    past = events[events["local_date"] < today]
    # The same modelling window as everything else. Reaching back through a
    # different regime to decide what today looks like was matching a second-term
    # Friday morning against the campaign.
    if window_start:
        past = past[past["local_date"] >= pd.Timestamp(window_start)]

    so_far = int(
        _posts_before(events[events["local_date"] == today], cutoff).sum()
    )

    frame = _day_frame(past, cutoff)
    if len(frame) < MIN_HISTORY_DAYS:
        pool = frame["total"].to_numpy(dtype=float) if len(frame) else np.array([0.0])
        flat = np.quantile(pool, GRID)
        return DayConditioner(flat, flat, so_far, cutoff / SECONDS_PER_DAY, len(frame), len(frame))

    levels = _weekday_levels(frame)
    hist_dow = frame.index.dayofweek.to_numpy()
    hist_level = levels[hist_dow]
    mu_today = float(levels[dow])

    # Rescaled to today's weekday level: a busy weekday's leftovers must not
    # import its volume along with its shape.
    rescale = mu_today / hist_level
    scaled_total = frame["total"].to_numpy(dtype=float) * rescale
    scaled_rest = (
        frame["total"].to_numpy(dtype=float) - frame["upto"].to_numpy(dtype=float)
    ) * rescale

    # Weekday affinity applies at every hour, including the very first second —
    # so the identity at elapsed zero is "a typical day OF THIS WEEKDAY", which
    # is the same thing the models mean by a fresh day.
    w_dow = np.where(hist_dow == dow, 1.0, OFF_WEEKDAY_WEIGHT)

    # Conditioning weight: how alike is this day's progress to today's? At
    # elapsed zero every `upto` is zero and so is `so_far`, so every weight is
    # exactly 1 and the conditional collapses onto the baseline. That identity
    # is what makes the midnight handover continuous, and it is a property of
    # the arithmetic rather than a tuning choice.
    hist_norm = frame["upto"].to_numpy(dtype=float) / hist_level
    today_norm = so_far / max(mu_today, 1e-9)
    w_cond = np.exp(-0.5 * ((hist_norm - today_norm) / max(bandwidth, 1e-6)) ** 2)

    baseline = _weighted_quantiles(scaled_total, w_dow, GRID)
    weights = w_dow * w_cond
    conditional = _weighted_quantiles(scaled_rest, weights, GRID)
    effective_n = float(weights.sum() ** 2 / max((weights**2).sum(), 1e-12))
    return DayConditioner(
        baseline, conditional, so_far, cutoff / SECONDS_PER_DAY, len(frame), effective_n
    )


def sample_rest_of_today(
    events: pd.DataFrame,
    now: pd.Timestamp,
    n: int = 20_000,
    window_start: str | None = None,
) -> np.ndarray:
    """Draw how many more posts today produces, given what it has produced.

    Kept for callers that have no day distribution of their own — the models
    that do get `DayConditioner.remainder` applied to theirs instead, which is
    what makes the projection continuous across midnight.
    """
    return build_conditioner(events, now, window_start=window_start).sample_remainder(n)


def elapsed_share_of_posts(events: pd.DataFrame, now: pd.Timestamp, window_start=None) -> float:
    """Share of a typical day of this weekday's posts that have landed by now.

    Reported for the site rather than used for scaling — the scaling estimator
    is the one that fails. It is here so a reader can see how much of the day is
    already decided when a number is quoted.
    """
    sub = events[events["local_date"].dt.dayofweek == now.dayofweek]
    if window_start:
        sub = sub[sub["local_date"] >= pd.Timestamp(window_start)]
    sub = sub[sub["local_date"] < now.normalize()]
    if sub.empty:
        return 0.0
    return float((_into_day_seconds(sub) <= elapsed_seconds(now)).mean())


def describe(events: pd.DataFrame, now: pd.Timestamp, window_start: str | None = None) -> dict:
    """What the site shows about the day in progress."""
    cond = build_conditioner(events, now, window_start=window_start)
    samples = cond.sample_remainder(8000)
    return {
        "posts_so_far": cond.posts_so_far,
        "share_of_day_elapsed": round(cond.elapsed, 4),
        "share_of_posts_elapsed": round(
            elapsed_share_of_posts(events, now, window_start), 3
        ),
        "rest_of_today_median": round(float(np.median(samples)), 1),
        "rest_of_today_80": [
            round(float(np.quantile(samples, 0.1)), 1),
            round(float(np.quantile(samples, 0.9)), 1),
        ],
        "day_total_median": round(cond.posts_so_far + float(np.median(samples)), 1),
        "comparable_days": round(cond.effective_n, 1),
        "method": (
            "Each model's own distribution for today is conditioned on how today has actually "
            "gone, using the days whose posting looked most like it at this exact moment. "
            "Neither ignores today nor scales it up — both of those score worse, in opposite "
            "halves of the day. At midnight the conditioning is the identity, so the projection "
            "does not jump when the date changes."
        ),
    }
