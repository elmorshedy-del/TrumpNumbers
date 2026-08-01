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

from . import partial
from .config import LOCAL_TZ, N_PREDICTIVE_SAMPLES, QUANTILE_LEVELS
from .models import ForecastTask
from .models.registry import build_all_models
from .series import CountSeries, week_progress

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
    observed = daily[(daily.index >= monday) & (daily.index <= today)].to_numpy()
    cut = int(today.dayofweek)

    rest_of_today = np.zeros(n_samples)
    if events is not None and not events.empty:
        try:
            rest_of_today = partial.sample_rest_of_today(events, now, n=n_samples)
        except Exception as exc:  # noqa: BLE001 - a bad draw must not lose the forecast
            log.warning("partial-day sampling failed, treating today as finished: %s", exc)

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
            # Add what today has left. Drawn independently of the model's own
            # days, which is the same conditional-independence the daily models
            # already assume between days.
            if len(rest_of_today):
                extra_today = (
                    rest_of_today if len(rest_of_today) == len(samples)
                    else np.random.choice(rest_of_today, size=len(samples), replace=True)
                )
                samples = samples + extra_today
            entry = {
                "model": model.name,
                "family": model.family,
                "median": round(float(np.median(samples)), 1),
                "mean": round(float(np.mean(samples)), 1),
                "quantiles": {
                    str(q): round(float(np.quantile(samples, q)), 1) for q in QUANTILE_LEVELS
                },
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
            today_detail = partial.describe(events, now)
        except Exception as exc:  # noqa: BLE001
            log.warning("partial-day description failed: %s", exc)

    return {
        "generated_at": pd.Timestamp.now(tz=LOCAL_TZ).isoformat(),
        "week": prog,
        "today": today_detail,
        "per_model": results,
    }


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
    per = {r["model"]: r for r in forecast["per_model"]}
    chosen = [m for m in (ranking or []) if m in per][:top_k] or list(per)[:top_k]
    if not chosen:
        return {}

    levels = [str(q) for q in QUANTILE_LEVELS]
    qmat = np.array([[per[m]["quantiles"][l] for l in levels] for m in chosen])
    avg = qmat.mean(axis=0)
    q = dict(zip(levels, [round(float(v), 1) for v in avg]))

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
    """P(week total >= T), with an explicit softness flag.

    Tail probabilities need far more data than central estimates, because the
    events are by definition rare. A probability estimated from ~50 scored weeks
    out beyond the 90th percentile is a coarse instrument and is marked as such
    rather than quietly presented at the same confidence as the median.
    """
    per = {r["model"]: r for r in forecast["per_model"]}
    chosen = [m for m in (ranking or []) if m in per][:top_k] or list(per)[:top_k]
    levels = [float(q) for q in QUANTILE_LEVELS]

    out = []
    for t in thresholds:
        probs = []
        for m in chosen:
            qs = [per[m]["quantiles"][str(q)] for q in levels]
            # Invert the quantile function: P(X >= t) = 1 - F(t).
            p = 1.0 - float(np.interp(t, qs, levels, left=0.0, right=1.0))
            probs.append(p)
        p = float(np.mean(probs))
        out.append(
            {
                "threshold": round(float(t), 1),
                "probability": round(p, 3),
                "soft": bool(p < 0.10 or p > 0.90),
                "note": "estimated in the tail — treat as approximate" if (p < 0.10 or p > 0.90) else "",
            }
        )
    return out
