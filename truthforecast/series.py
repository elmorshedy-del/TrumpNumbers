"""Posts -> the count series the models actually see.

Two decisions are made here and both are load-bearing:

**Local days, not UTC days.** The day-of-week structure (busy Mondays, quiet
Thursdays) is a fact about Trump's day, so days are bucketed in US Eastern. In
UTC his late-evening posts spill into tomorrow and smear the weekly profile.

**ReTruths are re-dated, not discarded.** The mirror stores a ReTruth under the
ORIGINAL author's timestamp, so counting them as filed would place activity on
days Trump did nothing — which is why they used to be dropped entirely. But the
mirror's own ingest order brackets the moment each one actually appeared, so the
reshare time is recoverable to within hours (`redate_retruths`). They now count,
on the right day.

**The convention is a setting, not an assumption.** Which days open a week, and
which post types count, decide what the headline number *means* — and the
defaults now match the Kalshi weekly market so the number is comparable to the
one people quote. See `config.CountConvention`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONVENTION, LOCAL_TZ, WINDOW
from .ingest.store import connect, load_posts

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def redate_retruths(df: pd.DataFrame) -> pd.DataFrame:
    """Place ReTruths on the day Trump reshared, not the day the original ran.

    The mirror stores a ReTruth under the ORIGINAL author's timestamp, which is
    why they were excluded from the modelled series entirely: counting them as
    posted would credit activity to days on which nothing happened, sometimes
    over a year earlier.

    They can be recovered. `trumpstruth_id` is the mirror's own arrival order,
    so the ids either side of a ReTruth bracket the moment it was actually
    seen — regardless of how old its content is. Interpolating the neighbouring
    originals' timestamps against id puts the reshare within hours of the truth
    for the bulk of cases. Measured against the stored dates: 72% agree to
    within a day (he mostly reshares fresh material), and the remaining 28% are
    exactly the ones the stored date gets wrong, one of them by 420 days.

    Adds `effective_utc` — the reshare time for ReTruths, the post time for
    everything else — and leaves `created_utc` untouched so the original record
    is still inspectable.
    """
    if df.empty or "is_retruth" not in df:
        return df

    out = df.sort_values("trumpstruth_id").reset_index(drop=True)
    out["effective_utc"] = out["created_utc"]

    rt = out["is_retruth"] == 1
    originals = out[~rt]
    if rt.any() and len(originals) >= 2:
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = np.interp(
            out.loc[rt, "trumpstruth_id"].to_numpy(dtype=float),
            originals["trumpstruth_id"].to_numpy(dtype=float),
            (originals["created_utc"] - epoch).dt.total_seconds().to_numpy(),
        )
        # Whole microseconds: interpolating between two timestamps lands on
        # fractions the column's resolution cannot hold, and pandas refuses a
        # lossy cast rather than silently truncating. Daily buckets do not care
        # about sub-second precision anyway.
        micros = np.rint(secs * 1e6).astype("int64")
        out.loc[rt, "effective_utc"] = pd.to_datetime(micros, unit="us", utc=True).as_unit(
            out["effective_utc"].dt.unit
        )

    return out


@dataclass
class CountSeries:
    """Daily counts plus the weekly aggregation built from them."""

    daily: pd.Series  # DatetimeIndex (local dates, tz-naive), int counts
    label: str

    @property
    def weekly(self) -> pd.Series:
        """Weekly totals, indexed by the week's LAST day.

        The anchor follows `CONVENTION`: Saturday-ending (Sunday–Saturday, the
        Kalshi week) by default. The old docstring claimed Monday-anchored while
        the code resampled W-SUN, which is Sunday-ending — the label was wrong,
        the arithmetic was not.
        """
        return self.daily.resample(CONVENTION.resample_rule).sum().rename("week_total")

    def complete_weeks(self, through: pd.Timestamp | None = None) -> pd.Series:
        """Weekly totals excluding any partial week at either end.

        A partial final week would read as a sudden collapse in volume and
        poison both the fit and the backtest.
        """
        weekly = self.weekly
        if weekly.empty:
            return weekly
        first_day, last_day = self.daily.index[0], self.daily.index[-1]
        keep = [
            wk for wk in weekly.index
            if (wk - pd.Timedelta(days=6)) >= first_day and wk <= last_day
        ]
        out = weekly.loc[keep]
        if through is not None:
            out = out[out.index <= through]
        return out

    def restrict(self, start: str | None = None, end: str | None = None) -> "CountSeries":
        d = self.daily
        if start:
            d = d[d.index >= pd.Timestamp(start)]
        if end:
            d = d[d.index <= pd.Timestamp(end)]
        return CountSeries(daily=d, label=self.label)


def load_frame(db_path=None, include_deleted: bool = True) -> pd.DataFrame:
    """All stored posts as a DataFrame with a local-time column."""
    with connect(db_path) as conn:
        rows = load_posts(conn, include_deleted=include_deleted)
    if not rows:
        return pd.DataFrame(
            columns=["trumpstruth_id", "created_utc", "is_retruth", "text", "local"]
        )

    df = pd.DataFrame([dict(r) for r in rows])
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, format="mixed")
    # Bucket on when the post ENTERED the timeline, which for a ReTruth is the
    # reshare rather than the original's date. See `redate_retruths`.
    df = redate_retruths(df)
    df["local"] = df["effective_utc"].dt.tz_convert(LOCAL_TZ)
    df["local_date"] = df["local"].dt.tz_localize(None).dt.normalize()
    df["hour"] = df["local"].dt.hour
    df["dow"] = df["local_date"].dt.dayofweek
    return df.sort_values("effective_utc").reset_index(drop=True)


def daily_counts(
    df: pd.DataFrame,
    kind: str = "originals",
    start: str | None = None,
    end: str | None = None,
) -> CountSeries:
    """Build a zero-filled daily count series.

    `kind`: "convention" (what CONVENTION says counts — the default),
    "originals", "retruths", or "all".

    Zero-filling matters: days with no posts are real observations, not missing
    data. Dropping them would delete exactly the low tail the models need in
    order to know that a quiet day is possible.
    """
    if kind == "convention":
        sub = df if CONVENTION.include_retruths else df[df["is_retruth"] == 0]
        if not CONVENTION.include_deleted and "is_deleted" in sub:
            sub = sub[sub["is_deleted"] == 0]
    elif kind == "originals":
        sub = df[df["is_retruth"] == 0]
    elif kind == "retruths":
        sub = df[df["is_retruth"] == 1]
    elif kind == "all":
        sub = df
    else:
        raise ValueError(f"unknown kind: {kind}")

    if sub.empty:
        return CountSeries(daily=pd.Series(dtype="int64"), label=kind)

    counts = sub.groupby("local_date").size()
    lo = pd.Timestamp(start) if start else counts.index.min()
    hi = pd.Timestamp(end) if end else counts.index.max()
    full = pd.date_range(lo, hi, freq="D")
    daily = counts.reindex(full, fill_value=0).astype("int64")
    daily.index.name = "date"
    return CountSeries(daily=daily, label=kind)


def convention_events(df: pd.DataFrame) -> pd.DataFrame:
    """Post-level rows matching what CONVENTION counts.

    The event-level consumers — the Hawkes fit and the partial-day sampler —
    must see the same posts the daily series is built from, or they will model
    one population and be scored against another.
    """
    sub = df if CONVENTION.include_retruths else df[df["is_retruth"] == 0]
    if not CONVENTION.include_deleted and "is_deleted" in sub:
        sub = sub[sub["is_deleted"] == 0]
    return sub


def modeling_series(df: pd.DataFrame | None = None, kind: str = "convention") -> CountSeries:
    """The series the models are fit on, restricted to the configured window."""
    df = load_frame() if df is None else df
    return daily_counts(df, kind=kind, start=WINDOW.start)


def days_into_week(day: pd.Timestamp) -> int:
    """How many days into the current week `day` falls, under CONVENTION."""
    if CONVENTION.week_anchor == "SUN":
        return (int(day.dayofweek) + 1) % 7      # Sunday opens the week
    return int(day.dayofweek)                    # Monday opens the week


def current_week_bounds(now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day (local, tz-naive) of the week containing `now`."""
    now = (now or pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)).normalize()
    start = now - pd.Timedelta(days=days_into_week(now))
    return start, start + pd.Timedelta(days=6)


def week_progress(series: CountSeries, now: pd.Timestamp | None = None) -> dict:
    """What we know about the in-flight week.

    `days_observed` counts only days that are fully in the past. Today is
    excluded from the observed total because it is still accumulating, and
    treating a half-finished day as a finished one biases every projection
    downwards.
    """
    now = now or pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)
    today = now.normalize()
    start, end = current_week_bounds(now)
    into = days_into_week(today)

    daily = series.daily
    observed = daily[(daily.index >= start) & (daily.index < today)]
    today_count = int(daily.get(today, 0))

    return {
        "week_start": start.date().isoformat(),
        "week_end": end.date().isoformat(),
        "convention": CONVENTION.label,
        "days_observed": int(len(observed)),
        "observed_total": int(observed.sum()),
        "today": today.date().isoformat(),
        "today_partial_count": today_count,
        "today_dow": int(today.dayofweek),
        "days_into_week": into,
        "days_remaining": int(6 - into),  # excludes today
        "week_to_date_including_today": int(observed.sum()) + today_count,
    }


def day_of_week_table(series: CountSeries) -> pd.DataFrame:
    """Per-weekday summary. Median alongside mean, deliberately.

    With a right-skewed count distribution the mean describes a day that
    essentially never occurs, so the site leads with the median.
    """
    d = series.daily
    g = pd.DataFrame({"count": d.values, "dow": d.index.dayofweek})
    out = g.groupby("dow")["count"].agg(
        n="size", mean="mean", median="median", std="std",
        q10=lambda s: s.quantile(0.10), q90=lambda s: s.quantile(0.90), max="max",
    )
    out.index = [DAY_NAMES[i] for i in out.index]
    return out.round(2)


def to_arrays(series: CountSeries) -> tuple[np.ndarray, np.ndarray]:
    """(counts, day-of-week) as plain arrays for the model layer."""
    return series.daily.to_numpy(dtype=float), series.daily.index.dayofweek.to_numpy()
