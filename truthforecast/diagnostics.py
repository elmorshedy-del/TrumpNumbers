"""Diagnostics that the site reports as findings, not just internals.

The order here follows the reasoning the project is built on: find out what kind
of thing the data is before choosing a model for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CONVENTION, WINDOW
from .series import DAY_NAMES, CountSeries, convention_events


def conditional_dispersion(series: CountSeries) -> dict:
    """Is it still not Poisson once the obvious structure is removed?

    The raw dispersion ratio answers a question nobody was asking. *Homogeneous*
    Poisson — one fixed rate for every day of every week — is refuted by the
    weekly cycle alone, and a busy Monday against a quiet Thursday inflates the
    ratio all by itself. Beating that strawman says nothing about whether a
    Poisson GLM, which is what the model zoo actually contains, is adequate.

    So the same test is run against rates that already know what a Poisson GLM
    knows. Under a correctly specified Poisson model the Pearson statistic
    divided by its degrees of freedom sits near 1:

      * `by_weekday`: rate = that weekday's mean. What `poisson-glm` assumes.
      * `by_weekday_and_level`: rate = weekday factor x trailing 28-day level,
        computed causally (yesterday backwards), so a drifting level cannot be
        mistaken for scatter.

    If the ratio stays far above 1 after both, the overdispersion is a property
    of the process rather than of the model's ignorance about calendars — which
    is the claim the Negative Binomial family is here to answer.
    """
    d = series.daily.astype(float)
    x = d.to_numpy(dtype=float)
    n = len(x)
    if n < 30 or x.mean() <= 0:
        return {"available": False}

    dow = d.index.dayofweek.to_numpy()
    dow_mean = np.array([x[dow == k].mean() if (dow == k).any() else x.mean() for k in range(7)])

    def pearson(mu: np.ndarray, params: int) -> float | None:
        ok = np.isfinite(mu) & (mu > 0)
        if ok.sum() <= params:
            return None
        stat = float((((x[ok] - mu[ok]) ** 2) / mu[ok]).sum())
        return round(stat / (int(ok.sum()) - params), 2)

    mu_dow = dow_mean[dow]

    # Causal trailing level: yesterday and earlier only, so today never helps
    # explain itself. Scaled by the weekday factor so the two effects compose.
    level = d.shift(1).rolling(28, min_periods=7).mean().to_numpy()
    factors = dow_mean / x.mean()
    mu_both = level * factors[dow]

    return {
        "available": True,
        "raw": round(float(x.var(ddof=1) / x.mean()), 2),
        "by_weekday": pearson(mu_dow, 7),
        "by_weekday_and_level": pearson(mu_both, 8),
        "note": (
            "Pearson chi-square per degree of freedom. A well-specified Poisson model sits "
            "near 1. The raw ratio compares against a single fixed rate, which the weekly "
            "cycle refutes on its own; these two compare against what a Poisson GLM actually "
            "assumes."
        ),
    }


def dispersion_test(series: CountSeries) -> dict:
    """Is this Poisson? Almost certainly not — quantify how badly.

    A Poisson process has variance == mean, and that is forced rather than
    conventional: if the rate is fixed and events are independent, arrival
    timing is the only randomness left, so one number sets both the level and
    the scatter.

    The test statistic is the standard dispersion test: sum((x - xbar)^2)/xbar
    is approximately chi-square with n-1 df under Poisson.
    """
    x = series.daily.to_numpy(dtype=float)
    n = len(x)
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    if mean <= 0:
        return {"n": n, "mean": mean, "variance": var, "poisson_plausible": False}

    stat = float(((x - mean) ** 2).sum() / mean)
    p = float(stats.chi2.sf(stat, df=n - 1))
    poisson_sd = float(np.sqrt(mean))

    return {
        "n": n,
        "mean": round(mean, 3),
        "median": float(np.median(x)),
        "variance": round(var, 3),
        "sd": round(float(np.sqrt(var)), 3),
        "dispersion_ratio": round(var / mean, 2),
        "chi2": round(stat, 1),
        "p_value": p,
        "poisson_plausible": bool(p > 0.01),
        # What a Poisson world would have predicted, for the site to contrast.
        "poisson_sd": round(poisson_sd, 2),
        "poisson_range_95": [
            max(0, round(mean - 2 * poisson_sd)),
            round(mean + 2 * poisson_sd),
        ],
        "actual_range": [int(x.min()), int(x.max())],
        "actual_range_95": [
            float(np.quantile(x, 0.025)),
            float(np.quantile(x, 0.975)),
        ],
    }


def ingest_lag(df: pd.DataFrame) -> dict:
    """How long after a post exists do we know about it?

    Every alert this project can send is bounded by this number, so it is worth
    measuring rather than assuming. `first_seen_utc` minus the post's own
    timestamp is the whole chain: Truth Social publishes, trumpstruth.org
    crawls, we poll. Only the last link is ours.

    Two exclusions, both load-bearing. Backfilled rows were all "first seen" in
    one burst hours or years after the fact, so they would report the age of the
    archive rather than the latency of the pipeline. And ReTruths carry the
    ORIGINAL author's timestamp, so a reshare of a three-day-old post reads as
    three days of lag — measured before that filter went in, the median came out
    at 4,728 minutes, which is a fact about ReTruths and nothing else.

    What is left is originals observed by a live poll, so this fills in as the
    deployment runs and says so when it has too little to be worth reading.
    """
    from .ingest.store import connect, get_state

    if df.empty or "first_seen_utc" not in df:
        return {"available": False}

    with connect() as conn:
        backfill_end = get_state(conn, "last_backfill_utc")
    if not backfill_end:
        return {"available": False, "note": "no backfill recorded"}

    seen = pd.to_datetime(df["first_seen_utc"], utc=True, format="mixed", errors="coerce")
    live = df[(seen > pd.Timestamp(backfill_end)) & (df["is_retruth"] == 0)]
    if len(live) < 5:
        return {
            "available": False,
            "n": int(len(live)),
            "note": "not enough posts observed live yet — this fills in as the poller runs",
        }

    lag_min = (
        (seen[live.index] - live["created_utc"]).dt.total_seconds() / 60.0
    ).clip(lower=0)

    return {
        "available": True,
        "n": int(len(lag_min)),
        "median_minutes": round(float(lag_min.median()), 1),
        "p90_minutes": round(float(lag_min.quantile(0.90)), 1),
        "max_minutes": round(float(lag_min.max()), 1),
        "note": (
            "Post timestamp to the moment this pipeline stored it: Truth Social publishing, "
            "trumpstruth.org crawling, and our poll interval combined. Only the poll interval "
            "is ours to shorten; the rest is the floor."
        ),
    }


def tail_summary(series: CountSeries) -> dict:
    """How heavy is the right tail? Decides whether means are usable at all."""
    x = series.daily.to_numpy(dtype=float)
    mean, median = float(x.mean()), float(np.median(x))
    top = np.sort(x)[::-1]
    share_top1 = float(top[: max(1, len(x) // 100)].sum() / max(x.sum(), 1))

    return {
        "mean_over_median": round(mean / median, 2) if median else None,
        "max_over_median": round(float(x.max()) / median, 1) if median else None,
        "skewness": round(float(stats.skew(x)), 2),
        "top_1pct_share_of_all_posts": round(share_top1, 3),
        "zero_days": int((x == 0).sum()),
        "zero_share": round(float((x == 0).mean()), 3),
        "quantiles": {
            str(q): float(np.quantile(x, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
    }


def deseasonalize(series: CountSeries) -> pd.Series:
    """Remove the weekly cycle multiplicatively, before any memory analysis.

    Skipping this is the classic trap: the raw ACF of a series with a strong
    weekly rhythm shows a beautiful lag-7 spike, and that spike is the calendar
    reflected back at you, not evidence that the process remembers a week ago.
    """
    d = series.daily.astype(float)
    dow = d.index.dayofweek
    overall = d.mean()
    if overall <= 0:
        return d
    factors = d.groupby(dow).mean() / overall
    factors = factors.replace(0, np.nan).fillna(1.0)
    return d / factors.reindex(dow).to_numpy()


def autocorrelation(series: CountSeries, max_lag: int = 21) -> dict:
    """ACF and PACF on the deseasonalized series.

    ACF answers "how much does today resemble N days ago"; PACF strips out what
    the intervening days already explain, separating direct influence from
    inherited influence.
    """
    from statsmodels.tsa.stattools import acf, pacf

    x = deseasonalize(series).to_numpy(dtype=float)
    max_lag = int(min(max_lag, len(x) // 3))
    if max_lag < 2:
        return {"lags": [], "acf": [], "pacf": [], "conf": None}

    a = acf(x, nlags=max_lag, fft=True)
    p = pacf(x, nlags=max_lag, method="ywm")
    conf = 1.96 / np.sqrt(len(x))  # white-noise band

    raw = acf(series.daily.to_numpy(dtype=float), nlags=max_lag, fft=True)
    return {
        "lags": list(range(max_lag + 1)),
        "acf": [round(float(v), 4) for v in a],
        "pacf": [round(float(v), 4) for v in p],
        # Kept so the site can show the difference the correction makes.
        "acf_raw_with_weekly_cycle": [round(float(v), 4) for v in raw],
        "conf": round(float(conf), 4),
        "note": "ACF/PACF computed after removing the weekly cycle; the raw series is shown for contrast.",
    }


def day_of_week_profile(series: CountSeries) -> dict:
    d = series.daily.astype(float)
    g = pd.DataFrame({"count": d.values, "dow": d.index.dayofweek})
    stats_by_dow = g.groupby("dow")["count"].agg(["size", "mean", "median", "std"])
    overall = float(d.mean())
    return {
        "days": [
            {
                "day": DAY_NAMES[int(i)],
                "n": int(r["size"]),
                "mean": round(float(r["mean"]), 2),
                "median": round(float(r["median"]), 1),
                "sd": round(float(r["std"]), 2) if pd.notna(r["std"]) else None,
                "index_vs_overall": round(float(r["mean"]) / overall, 3) if overall else None,
            }
            for i, r in stats_by_dow.iterrows()
        ],
        "overall_mean": round(overall, 2),
    }


def structural_breaks(series: CountSeries, max_breaks: int = 4) -> dict:
    """Where did the process change?

    Fitting across a structural break describes neither regime. Trump has
    plausibly been several different posting processes (private citizen,
    candidate, president), and choosing the window is a bigger lever on the
    answer than choosing the model — so it is surfaced rather than buried.
    """
    try:
        import ruptures as rpt
    except ImportError:
        return {"available": False, "breaks": []}

    x = series.daily.to_numpy(dtype=float)
    if len(x) < 60:
        return {"available": True, "breaks": [], "note": "series too short"}

    # Smooth to weekly means first: daily counts are so noisy that a daily-level
    # changepoint search finds bursts rather than regimes. The weeks are the
    # convention's weeks, so a break lands on a boundary the rest of the project
    # recognises.
    resampled = series.daily.resample(CONVENTION.resample_rule).mean()
    weekly = resampled.to_numpy(dtype=float)
    weekly_index = resampled.index

    signal = np.log1p(weekly).reshape(-1, 1)
    algo = rpt.Binseg(model="l2").fit(signal)
    try:
        idx = algo.predict(n_bkps=min(max_breaks, max(1, len(weekly) // 20)))
    except Exception:
        return {"available": True, "breaks": []}

    breaks = []
    prev = 0
    for cut in idx:
        seg = weekly[prev:cut]
        if len(seg) == 0:
            continue
        breaks.append(
            {
                "start": weekly_index[prev].date().isoformat(),
                "end": weekly_index[min(cut, len(weekly_index) - 1)].date().isoformat(),
                "weeks": int(len(seg)),
                "mean_posts_per_day": round(float(np.mean(seg)), 2),
            }
        )
        prev = cut

    out = {"available": True, "segments": breaks, "breaks": [b["start"] for b in breaks[1:]]}
    out.update(_penalized_break_count(signal, len(weekly)))
    out["note"] = (
        "The segment list comes from a search told in advance how many breaks to find, so its "
        "count is a setting, not a result. `n_breaks_penalized` lets the data choose instead: "
        "if it is 0, the segmentation above is a partition of noise."
    )
    return out


def _penalized_break_count(signal: np.ndarray, n_weeks: int) -> dict:
    """Let the data choose the number of breaks, instead of being told.

    `Binseg(n_bkps=k)` returns exactly k breakpoints whether or not any exist —
    it partitions a straight line just as willingly as a step function. Read
    naively, "four regime changes" is then a restatement of `max_breaks=4`. A
    penalized search has to pay for each break it claims, so zero is an
    available answer, and that makes the finding falsifiable.
    """
    try:
        import ruptures as rpt
    except ImportError:  # pragma: no cover - handled by the caller
        return {}
    if n_weeks < 12:
        return {}
    try:
        var = float(np.var(signal))
        # BIC-flavoured penalty: one break must buy more than log(n) * sigma^2.
        penalty = max(var, 1e-6) * float(np.log(n_weeks)) * 2.0
        found = rpt.Pelt(model="l2", min_size=4).fit(signal).predict(pen=penalty)
        return {"n_breaks_penalized": max(0, len(found) - 1), "penalty": round(penalty, 4)}
    except Exception:
        return {}


def window_vs_breaks(breaks: dict) -> dict:
    """Does the configured modelling window contain the breaks it was chosen to exclude?

    The project's stated reason for having a window is that fitting across a
    structural break describes neither regime. The changepoint search then runs
    *inside* that window — so if it still finds breaks there, the argument has
    not been carried through to its conclusion, and the models are doing the
    thing the window was meant to prevent.

    Reported rather than acted on: moving the window is a judgement about which
    regime you want described, not a fact the data settles.
    """
    inside = [b for b in (breaks.get("breaks") or []) if b > WINDOW.start]
    penalized = breaks.get("n_breaks_penalized")
    most_recent = inside[-1] if inside else None

    weeks_since = None
    if most_recent:
        weeks_since = int(
            (pd.Timestamp.today().normalize() - pd.Timestamp(most_recent)).days // 7
        )

    return {
        "window_start": WINDOW.start,
        "window_label": WINDOW.label,
        "breaks_inside_window": inside,
        "most_recent_break": most_recent,
        "weeks_since_most_recent_break": weeks_since,
        "n_breaks_penalized": penalized,
        "note": (
            "Every model here is fitted from the window start, across any break listed above. "
            "That is the practice the window exists to avoid. Whether it matters depends on "
            "whether the breaks are real — see n_breaks_penalized, which is what the data says "
            "when it is not told how many to find."
        ),
    }


def detect_bursts(df: pd.DataFrame, series: CountSeries, top_n: int = 25) -> list[dict]:
    """Flag the biggest days and describe what was in them.

    This is the "trigger" view, and it is deliberately DESCRIPTIVE. Showing what
    Trump posted about on a heavy day is not evidence that the topic caused the
    heavy day: a self-exciting process produces bursts with no external trigger
    at all, and whatever he happens to be posting about will always be there to
    blame. The site labels this accordingly.
    """
    d = series.daily
    if d.empty:
        return []

    thresh = float(d.quantile(0.90))
    heavy = d[d >= max(thresh, 1)].sort_values(ascending=False).head(top_n)

    # The same posts the counts are built from. Describing a heavy day using
    # only originals, when the count that made it heavy includes ReTruths,
    # explains the day with a subset of what happened in it.
    counted = convention_events(df)
    out = []
    for date, count in heavy.items():
        day_posts = counted[counted["local_date"] == date]
        out.append(
            {
                "date": date.date().isoformat(),
                "dow": DAY_NAMES[int(date.dayofweek)],
                "count": int(count),
                "vs_median": round(float(count) / max(float(d.median()), 1), 1),
                "peak_hour": (
                    int(day_posts["hour"].mode().iloc[0]) if len(day_posts) else None
                ),
                "keywords": _keywords(day_posts["text"].tolist()),
                "sample_posts": [
                    {"text": t[:280], "id": int(i)}
                    for t, i in zip(
                        day_posts["text"].head(3).tolist(),
                        day_posts["trumpstruth_id"].head(3).tolist(),
                    )
                ],
            }
        )
    return out


_STOPWORDS = set(
    """the a an and or but if then than that this these those of to in on for with at by from
    is are was were be been being do does did doing have has had having i me my we our you your
    he him his she her it its they them their what which who whom will would can could should
    not no nor so very just about into over under again more most other some such only own same
    too s t don now they're it's i'm we're rt https http com www amp""".split()
)


def _keywords(texts: list[str], top: int = 8) -> list[str]:
    from collections import Counter
    import re

    words = Counter()
    for t in texts:
        for w in re.findall(r"[A-Za-z']{3,}", (t or "").lower()):
            if w not in _STOPWORDS:
                words[w] += 1
    return [w for w, _ in words.most_common(top)]


def hourly_profile(df: pd.DataFrame) -> list[int]:
    """Posts by local hour — the circadian background the Hawkes model needs.

    Built from the posts the convention counts, which is what the Hawkes fit is
    actually given. A profile drawn from a different population than the model
    consumes describes neither.
    """
    counts = convention_events(df).groupby("hour").size().reindex(range(24), fill_value=0)
    return [int(v) for v in counts.to_numpy()]


def run_all(df: pd.DataFrame, series: CountSeries) -> dict:
    breaks = structural_breaks(series)
    return {
        "window": {
            "start": series.daily.index[0].date().isoformat() if len(series.daily) else None,
            "end": series.daily.index[-1].date().isoformat() if len(series.daily) else None,
            "days": int(len(series.daily)),
        },
        "ingest_lag": ingest_lag(df),
        "dispersion": dispersion_test(series),
        "conditional_dispersion": conditional_dispersion(series),
        "tail": tail_summary(series),
        "day_of_week": day_of_week_profile(series),
        "autocorrelation": autocorrelation(series),
        "structural_breaks": breaks,
        "window_vs_breaks": window_vs_breaks(breaks),
        "bursts": detect_bursts(df, series),
        "hourly_profile": hourly_profile(df),
    }
