"""Posts -> the count series the models actually see.

Two decisions are made here and both are load-bearing:

**Local days, not UTC days.** The day-of-week structure (busy Mondays, quiet
Thursdays) is a fact about Trump's day, so days are bucketed in US Eastern. In
UTC his late-evening posts spill into tomorrow and smear the weekly profile.

**Originals only, by default.** A ReTruth is stored with the ORIGINAL author's
timestamp, not the moment Trump reshared it (see `ingest/parse.Post`). Counting
ReTruths would place activity on days Trump did nothing — a June 26 ReTruth of
someone else's post would add a post to June 26, months after the fact. So the
modeled series is originals; ReTruths are available separately, clearly labeled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LOCAL_TZ, WINDOW
from .ingest.store import connect, load_posts

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class CountSeries:
    """Daily counts plus the weekly aggregation built from them."""

    daily: pd.Series  # DatetimeIndex (local dates, tz-naive), int counts
    label: str

    @property
    def weekly(self) -> pd.Series:
        """Monday-anchored weekly totals, indexed by the week's Monday."""
        return self.daily.resample("W-SUN").sum().rename("week_total")

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
    df["local"] = df["created_utc"].dt.tz_convert(LOCAL_TZ)
    df["local_date"] = df["local"].dt.tz_localize(None).dt.normalize()
    df["hour"] = df["local"].dt.hour
    df["dow"] = df["local_date"].dt.dayofweek
    return df.sort_values("created_utc").reset_index(drop=True)


def daily_counts(
    df: pd.DataFrame,
    kind: str = "originals",
    start: str | None = None,
    end: str | None = None,
) -> CountSeries:
    """Build a zero-filled daily count series.

    `kind`: "originals" (default, the modeled series), "retruths", or "all".

    Zero-filling matters: days with no posts are real observations, not missing
    data. Dropping them would delete exactly the low tail the models need in
    order to know that a quiet day is possible.
    """
    if kind == "originals":
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


def modeling_series(df: pd.DataFrame | None = None, kind: str = "originals") -> CountSeries:
    """The series the models are fit on, restricted to the configured window."""
    df = load_frame() if df is None else df
    return daily_counts(df, kind=kind, start=WINDOW.start)


def current_week_bounds(now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Monday 00:00 and Sunday 00:00 (local, tz-naive) of the week containing `now`."""
    now = (now or pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)).normalize()
    monday = now - pd.Timedelta(days=int(now.dayofweek))
    return monday, monday + pd.Timedelta(days=6)


def week_progress(series: CountSeries, now: pd.Timestamp | None = None) -> dict:
    """What we know about the in-flight week.

    `days_observed` counts only days that are fully in the past. Today is
    excluded from the observed total because it is still accumulating, and
    treating a half-finished day as a finished one biases every projection
    downwards.
    """
    now = now or pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)
    today = now.normalize()
    monday, sunday = current_week_bounds(now)

    daily = series.daily
    observed = daily[(daily.index >= monday) & (daily.index < today)]
    today_count = int(daily.get(today, 0))

    return {
        "week_start": monday.date().isoformat(),
        "week_end": sunday.date().isoformat(),
        "days_observed": int(len(observed)),
        "observed_total": int(observed.sum()),
        "today": today.date().isoformat(),
        "today_partial_count": today_count,
        "today_dow": int(today.dayofweek),
        "days_remaining": int(6 - today.dayofweek),  # excludes today
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
