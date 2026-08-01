"""Today, half-finished.

The week forecast has to say something about a day that is under way. Two
obvious answers are both wrong, and measurably so:

* **Ignore it** and draw a whole fresh day from history. Never biased, never
  informative: CRPS sits at 8.53 from midnight to midnight, so at 9pm — when
  the day is 92% decided — it is still guessing. This is what the code did.
* **Scale it up**, `posts_so_far / fraction_of_day_elapsed`. Far worse where it
  matters most: CRPS 31.50 at 6am against 8.53 for ignoring it, because
  dividing by a small fraction multiplies the morning's noise. It only starts
  beating "ignore" mid-afternoon. Its failure case is stark — nothing posted by
  noon predicts a total of zero, on a record of 556 days containing no zero days
  at all.

The Poisson-Gamma conjugate update is the textbook third answer and it also
loses to ignoring the day until noon (13.77 at 6am). It assumes posts arrive on
a fixed daily clock, so an empty morning is strong evidence of a quiet day. His
mornings are not evidence of that: on the 44 days with nothing posted by noon
the median finish was 6 and the maximum was 69. He starts when he starts.

What works is to assume nothing about the clock. Find days that looked like
today looks now, and resample what those days did with the time they had left.
Silence by noon then predicts about 6, because that is what silence by noon
actually produced. Walk-forward CRPS, strictly past-only:

    hour     ignore    scale    conjugate    this
    06:00      8.53    31.50        13.77    7.73
    12:00      8.53    12.28         8.88    6.05
    21:00      8.53     2.78         2.20    1.91
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOCAL_TZ

# Neighbourhood size for the resampling. Small enough to stay conditional on
# what today looks like, large enough that the tail is not one lucky day.
N_NEIGHBOURS = 60

# Below this many usable history days, conditioning is noise; fall back to
# drawing a whole day, which is at least never biased.
MIN_HISTORY_DAYS = 60


def _daily_and_partial(events: pd.DataFrame, hour: int) -> tuple[pd.Series, pd.Series]:
    """Per-day totals, and how many of those had landed by `hour` (inclusive)."""
    totals = events.groupby("local_date").size().sort_index()
    upto = (
        events[events["hour"] <= hour].groupby("local_date").size()
        .reindex(totals.index, fill_value=0)
    )
    return totals, upto


def elapsed_fraction(events: pd.DataFrame, hour: int, dow: int) -> float:
    """Share of a typical weekday's posts that have landed by `hour`.

    Reported for the site rather than used for scaling — the scaling estimator
    is the one that fails. It is here so a reader can see how much of the day is
    already decided when a number is quoted.
    """
    sub = events[events["local_date"].dt.dayofweek == dow]
    if sub.empty:
        return 0.0
    by_hour = sub.groupby("hour").size().reindex(range(24), fill_value=0)
    total = by_hour.sum()
    return float(by_hour[by_hour.index <= hour].sum() / total) if total else 0.0


def sample_rest_of_today(
    events: pd.DataFrame,
    now: pd.Timestamp,
    n: int = 20_000,
    k: int = N_NEIGHBOURS,
) -> np.ndarray:
    """Draw how many more posts today produces, given what it has produced.

    `events` is the post-level frame (originals only), `now` local and tz-naive.
    Returns `n` samples of the REMAINDER — today's posts so far are not included,
    because the caller already has them as an observed quantity.

    Days are matched on posts-so-far normalised by that weekday's own level, so
    a quiet Saturday morning is compared against other quiet mornings rather
    than against Mondays. Their remainders are rescaled to today's weekday level
    on the way back out.
    """
    hour, today, dow = now.hour, now.normalize(), now.dayofweek
    past = events[events["local_date"] < today]
    totals, upto = _daily_and_partial(past, hour)
    if len(totals) < MIN_HISTORY_DAYS:
        same = totals[totals.index.dayofweek == dow]
        pool = (same if len(same) >= 10 else totals).to_numpy(dtype=float)
        return np.random.choice(pool, size=n, replace=True) if len(pool) else np.zeros(n)

    idx_dow = totals.index.dayofweek.to_numpy()
    level = np.array([
        totals.to_numpy(dtype=float)[idx_dow == d].mean() if (idx_dow == d).any() else totals.mean()
        for d in range(7)
    ])
    level = np.where(level > 0, level, max(float(totals.mean()), 1e-6))

    mu_today = float(level[dow])
    so_far = int(
        events[(events["local_date"] == today) & (events["hour"] <= hour)].shape[0]
    )

    hist_level = level[idx_dow]
    hist_norm = upto.to_numpy(dtype=float) / hist_level
    remainder = (totals - upto).to_numpy(dtype=float)

    # Nearest days by normalised posts-so-far, then rescale their remainders to
    # today's level so a busy weekday's leftovers do not import its volume.
    order = np.argsort(np.abs(hist_norm - so_far / mu_today))[:k]
    pool = remainder[order] * (mu_today / hist_level[order])
    if not len(pool):
        return np.zeros(n)
    return np.random.choice(pool, size=n, replace=True)


def describe(events: pd.DataFrame, now: pd.Timestamp) -> dict:
    """What the site shows about the day in progress."""
    hour, today = now.hour, now.normalize()
    so_far = int(events[(events["local_date"] == today) & (events["hour"] <= hour)].shape[0])
    samples = sample_rest_of_today(events, now, n=8000)
    return {
        "posts_so_far": so_far,
        "share_of_day_elapsed": round(elapsed_fraction(events, hour, now.dayofweek), 3),
        "rest_of_today_median": round(float(np.median(samples)), 1),
        "rest_of_today_80": [
            round(float(np.quantile(samples, 0.1)), 1),
            round(float(np.quantile(samples, 0.9)), 1),
        ],
        "method": (
            "Resampled from the days whose posting looked most like today's at this hour. "
            "Neither ignores today nor scales it up — both of those score worse, in "
            "opposite halves of the day."
        ),
    }
