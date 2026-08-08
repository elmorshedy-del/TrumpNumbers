"""The live forecast: where does the current week finish?

Reported as a median with asymmetric intervals, never as "157 ± 10". The
distribution is genuinely lopsided — the downside is floored (he cannot post a
negative number of times, and there is a base level he rarely drops below) while
the upside is open, because one burst weekend rewrites the total. A symmetric
error bar would be a claim about the shape of the world that is simply false.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import combine, partial
from .config import LOCAL_TZ, N_PREDICTIVE_SAMPLES, QUANTILE_LEVELS, WINDOW
from .models import ForecastTask
from .models.registry import build_all_models
from .series import CountSeries, days_into_week, week_progress

log = logging.getLogger(__name__)


def _fit_events(model, events: pd.DataFrame, origin: pd.Timestamp, lookback_days: int = 180):
    sub = events[events["local_date"] < origin]
    sub = sub[sub["local_date"] >= origin - pd.Timedelta(days=lookback_days)]
    if len(sub) < 200:
        return
    t0 = sub["local"].min()
    times = (sub["local"] - t0).dt.total_seconds().to_numpy() / 86400.0
    model.fit_events(times, sub["hour"].to_numpy(), float(times.max()))


def current_week_forecast(
    series: CountSeries,
    events: pd.DataFrame | None = None,
    model_names: list[str] | None = None,
    now: pd.Timestamp | None = None,
    n_samples: int = N_PREDICTIVE_SAMPLES,
) -> dict:
    """Forecast the in-flight week from every model, plus a headline number."""
    now = now or pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)
    prog = week_progress(series, now)
    monday = pd.Timestamp(prog["week_start"])
    today = pd.Timestamp(prog["today"])

    daily = series.daily.astype(float)
    # History excludes the target week entirely; days of this week that are
    # already complete enter through `observed`, not through the training data.
    history = daily[daily.index < monday]

    # Today counts. It used to be dropped and replaced by a whole fresh day
    # drawn from history, which never learned anything from a day already 92%
    # decided by 9pm. Today's posts now enter `observed` like any other day, the
    # models are asked only for the days after it, and what remains of today is
    # sampled separately — conditioned on how today has actually gone. See
    # `partial.py` for why neither ignoring nor rescaling it is good enough.
    # Reindexed over the whole elapsed part of the week, so every position from
    # the week's opening day to today has a row even when it produced nothing.
    # Without this the array is shorter than `cut` claims, and its last element
    # is not today but the most recent day that happened to post — which is a
    # trap for anything that indexes from the end.
    observed = (
        daily[(daily.index >= monday) & (daily.index <= today)]
        .reindex(pd.date_range(monday, today, freq="D"), fill_value=0.0)
        .to_numpy()
    )
    # Position within the week, not pandas weekday: under the Sunday-anchored
    # convention those differ, and the cut indexes position.
    cut = days_into_week(today)

    # The day in progress. Rather than substituting one estimator's view of
    # today for the models' own — which is what made the projection step at
    # midnight, when the same calendar day changed hands between them — build
    # the conditional MAP once and push each model's own day distribution
    # through it. At elapsed time zero that map is the identity, so the day a
    # model drew at 23:59 is the day it draws at 00:01.
    conditioner = None
    if events is not None and not events.empty:
        try:
            conditioner = partial.build_conditioner(events, now, window_start=WINDOW.start)
        except Exception as exc:  # noqa: BLE001 - a bad draw must not lose the forecast
            log.warning("partial-day conditioning failed, treating today as finished: %s", exc)

    if conditioner is not None and len(observed):
        # Today's banked posts come from the same clock the conditioning uses:
        # posts whose own timestamp is at or before `now`. The daily series
        # counts whatever is *stored* under today's date, which live is the same
        # thing and otherwise is not — a post timestamped a few minutes ahead
        # (source clock skew) would be banked here and unseen by the
        # conditioning, so the day would be counted once and forecast again.
        # Pinning both to one definition also makes the forecast a function of
        # `now`, which is what lets it be replayed at all.
        so_far = conditioner.posts_so_far
        observed = observed.copy()
        observed[-1] = float(so_far)          # the last row IS today; see the reindex above
        prog["today_partial_count"] = int(so_far)
        prog["week_to_date_including_today"] = int(prog["observed_total"]) + int(so_far)

    task = ForecastTask(history=history, week_start=monday, cut_dow=cut, observed=observed)

    results = []
    for model in build_all_models():
        if model_names and model.name not in model_names:
            continue
        try:
            model.fit(history)
            if getattr(model, "uses_events", False) and events is not None:
                _fit_events(model, events, monday)
            samples = model.sample_week_total(task, n_samples)
            if samples is None or not len(samples):
                continue
            # Add what today has left, conditioned on how today has gone. A
            # model that can describe an ordinary Saturday gets *its own*
            # Saturday conditioned; one that only ever produces multi-day sums
            # falls back to the empirical day, which is the honest answer for a
            # model that has no per-day view to condition.
            if conditioner is not None:
                own_day = model.sample_one_day(today, len(samples))
                rest = (
                    conditioner.remainder(own_day)
                    if own_day is not None and len(own_day) == len(samples)
                    else conditioner.sample_remainder(len(samples))
                )
                samples = samples + rest
            entry = {
                "model": model.name,
                "family": model.family,
                "median": round(float(np.median(samples)), 1),
                "mean": round(float(np.mean(samples)), 1),
                "quantiles": {
                    str(q): round(float(np.quantile(samples, q)), 1) for q in QUANTILE_LEVELS
                },
                # The fine grid the combination arithmetic runs on. Not for
                # display — it exists so the headline and the threshold table
                # are the same distribution, and so the tails do not have to be
                # invented back from nine published points.
                "_curve": combine.dense_quantiles(samples),
            }
            extra = getattr(model, "describe", None)
            if callable(extra):
                d = extra()
                if d:
                    entry["detail"] = d
            results.append(entry)
        except Exception as exc:
            log.warning("forecast failed for %s: %s", model.name, exc)

    today_detail = {}
    if events is not None and not events.empty:
        try:
            today_detail = partial.describe(events, now, window_start=WINDOW.start)
        except Exception as exc:  # noqa: BLE001
            log.warning("partial-day description failed: %s", exc)

    return {
        "generated_at": pd.Timestamp.now(tz=LOCAL_TZ).isoformat(),
        "week": prog,
        "today": today_detail,
        "per_model": results,
    }


def headline_curve(forecast: dict, ranking: list[str] | None = None, top_k: int = 3):
    """The combined quantile function the site leads with, and its members.

    This is the deployed artifact. `backtest.walkforward` scores exactly this
    rule week by week under the name `headline-top3`, so the number on the front
    page has a CRPS and a coverage figure of its own instead of inheriting the
    reputation of whichever models went into it.
    """
    per = {r["model"]: r for r in forecast["per_model"]}
    chosen = [m for m in (ranking or []) if m in per][:top_k] or list(per)[:top_k]
    if not chosen:
        return None, []
    curves = [
        per[m].get("_curve")
        if per[m].get("_curve") is not None
        else np.interp(
            combine.GRID,
            [float(q) for q in QUANTILE_LEVELS],
            [per[m]["quantiles"][str(q)] for q in QUANTILE_LEVELS],
        )
        for m in chosen
    ]
    return combine.vincentize(curves), chosen


def headline(
    forecast: dict,
    ranking: list[str] | None = None,
    top_k: int = 3,
) -> dict:
    """Combine the best-scoring models into the number the site leads with.

    `ranking` comes from the backtest leaderboard (best CRPS first). Averaging
    the top few quantile-wise is more robust than trusting a single winner,
    whose margin over the runner-up is usually well inside the noise of ~50
    scored weeks.
    """
    curve, chosen = headline_curve(forecast, ranking, top_k)
    if curve is None:
        return {}

    q = {k: round(v, 1) for k, v in combine.quantiles_at(curve, QUANTILE_LEVELS).items()}

    observed = forecast["week"]["week_to_date_including_today"]
    median = q["0.5"]
    lo80, hi80 = q["0.1"], q["0.9"]

    return {
        "models_used": chosen,
        "median": median,
        "interval_50": [q["0.25"], q["0.75"]],
        "interval_80": [lo80, hi80],
        "interval_95": [q["0.025"], q["0.975"]],
        "quantiles": q,
        # Deliberately surfaced: the interval is not symmetric, and the gap
        # between the two halves is itself the message.
        "asymmetry": {
            "downside": round(median - lo80, 1),
            "upside": round(hi80 - median, 1),
            "note": "Upside exceeding downside is the heavy right tail, not a rounding artefact.",
        },
        "week_to_date": observed,
        "implied_remaining": round(median - observed, 1),
    }


def threshold_probabilities(forecast: dict, thresholds: list[float], ranking=None, top_k=3) -> list[dict]:
    """P(week total >= T), read off the same distribution as the interval.

    Two corrections over what this used to do. It now inverts the *headline*
    curve rather than averaging each model's probability separately: those are
    two different combinations, so "P(at least 140)" was not the probability
    implied by the interval printed directly above it. And it no longer clips.
    Interpolating over nine published quantiles with the ends pinned to 0 and 1
    made every threshold past the 97.5th percentile read as an exact 0%, and
    every one below the 2.5th as an exact 100% — which is what the site was
    printing. A week total is unbounded above; certainty is not available.

    Tail probabilities also need far more data than central estimates, because
    the events are by definition rare, so they stay flagged as soft rather than
    presented at the same confidence as the median.
    """
    curve, chosen = headline_curve(forecast, ranking, top_k)
    if curve is None:
        return []

    out = []
    for t in thresholds:
        p, bounded = combine.prob_at_least(curve, t)
        soft = bool(bounded or p < 0.10 or p > 0.90)
        note = ""
        if bounded:
            note = (
                f"outside the sampled range — the honest statement is "
                f"{'>' if p > 0.5 else '<'}{'99.9' if p > 0.5 else '0.1'}%"
            )
        elif soft:
            note = "estimated in the tail — treat as approximate"
        out.append(
            {
                "threshold": round(float(t), 1),
                "probability": round(p, 3),
                "soft": soft,
                "bounded": bounded,
                "note": note,
                "models_used": chosen,
            }
        )
    return out
