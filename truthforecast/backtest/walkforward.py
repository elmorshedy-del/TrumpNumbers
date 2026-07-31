"""Walk-forward evaluation.

Judging a model on the data it was fitted to measures its ability to look up the
answer key. So every forecast here is made from a model that saw only the past.

The specifically time-series trap this avoids: a random train/test split would
let a model see Tuesday and Thursday while being scored on Wednesday. That is
not a mild optimism, it is reading tomorrow's newspaper. Training is therefore
strictly-before, expanding-origin, with a one-day embargo at the seam so rolling
features cannot leak across it.

For every historical week and every weekday cut, each model forecasts that
week's total from what was knowable at that moment — which answers the question
you actually have on a Wednesday: where does this week finish?
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BACKTEST, QUANTILE_LEVELS, THRESHOLD_PERCENTILES
from ..models import ForecastTask
from ..models.registry import REFERENCE_MODEL, build_all_models
from .scoring import score_forecast

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    rows: pd.DataFrame
    thresholds: list[float]
    holdout_start: pd.Timestamp | None


def _week_mondays(daily: pd.Series) -> list[pd.Timestamp]:
    idx = daily.index
    mondays = pd.unique(idx - pd.to_timedelta(idx.dayofweek, unit="D"))
    out = []
    for m in sorted(pd.to_datetime(mondays)):
        week = daily[(daily.index >= m) & (daily.index < m + pd.Timedelta(days=7))]
        if len(week) == 7:  # complete weeks only
            out.append(m)
    return out


def run_backtest(
    daily: pd.Series,
    events: pd.DataFrame | None = None,
    n_samples: int = 4000,
    min_train_weeks: int = BACKTEST.min_train_weeks,
    embargo_days: int = BACKTEST.embargo_days,
    max_weeks: int | None = None,
    progress: bool = True,
) -> BacktestResult:
    """Score every model on every (week, cut) pair.

    `events` carries post-level timestamps for models that need them (Hawkes).
    """
    daily = daily.astype(float)
    mondays = _week_mondays(daily)
    if len(mondays) <= min_train_weeks:
        raise ValueError(f"need more than {min_train_weeks} complete weeks, have {len(mondays)}")

    test_mondays = mondays[min_train_weeks:]
    if max_weeks:
        test_mondays = test_mondays[-max_weeks:]

    weekly_totals = np.array(
        [daily[(daily.index >= m) & (daily.index < m + pd.Timedelta(days=7))].sum()
         for m in mondays[:min_train_weeks]]
    )
    thresholds = [float(np.quantile(weekly_totals, p)) for p in THRESHOLD_PERCENTILES]

    models = build_all_models()
    fitted_at: dict[str, int] = {}
    rows = []
    t_start = time.time()

    for wi, monday in enumerate(test_mondays):
        week = daily[(daily.index >= monday) & (daily.index < monday + pd.Timedelta(days=7))]
        actual_total = float(week.sum())

        # Training data ends `embargo_days` before the target week starts.
        train_end = monday - pd.Timedelta(days=embargo_days)
        history = daily[daily.index < train_end]
        if len(history) < 60:
            continue

        for model in models:
            last = fitted_at.get(model.name)
            if last is None or (wi - last) >= model.refit_every_weeks:
                try:
                    model.fit(history)
                    if getattr(model, "uses_events", False) and events is not None:
                        _fit_events(model, events, train_end)
                    fitted_at[model.name] = wi
                except Exception as exc:
                    log.warning("fit failed for %s at %s: %s", model.name, monday.date(), exc)

            for cut in BACKTEST.cut_days:
                # cut = last completed day-of-week index. Observed days come
                # from the target week itself, which is legitimate: on Thursday
                # you really do know Monday through Wednesday.
                observed = week.to_numpy()[: cut + 1] if cut >= 0 else np.array([])
                task = ForecastTask(
                    history=history, week_start=monday, cut_dow=cut, observed=observed
                )
                try:
                    samples = model.sample_week_total(task, n_samples)
                except Exception as exc:
                    log.warning("predict failed for %s: %s", model.name, exc)
                    continue
                if samples is None or len(samples) == 0 or not np.isfinite(samples).any():
                    continue

                s = score_forecast(samples, actual_total, QUANTILE_LEVELS, thresholds)
                rows.append(
                    {
                        "week": monday,
                        "cut_dow": cut,
                        "model": model.name,
                        "family": model.family,
                        "actual": actual_total,
                        "observed_so_far": float(observed.sum()) if len(observed) else 0.0,
                        **{k: v for k, v in s.items() if k != "brier"},
                        **{f"brier_{k}": v for k, v in s["brier"].items()},
                    }
                )

        if progress and (wi + 1) % 10 == 0:
            log.info(
                "backtest %s/%s weeks (%.0fs elapsed)",
                wi + 1, len(test_mondays), time.time() - t_start,
            )

    df = pd.DataFrame(rows)
    holdout_start = (
        test_mondays[-BACKTEST.holdout_weeks] if len(test_mondays) > BACKTEST.holdout_weeks else None
    )
    return BacktestResult(rows=df, thresholds=thresholds, holdout_start=holdout_start)


def _fit_events(model, events: pd.DataFrame, train_end: pd.Timestamp, lookback_days: int = 180):
    """Feed post-level timestamps (strictly before the origin) to a Hawkes fit."""
    sub = events[(events["local_date"] < train_end)]
    sub = sub[sub["local_date"] >= train_end - pd.Timedelta(days=lookback_days)]
    if len(sub) < 200:
        return
    t0 = sub["local"].min()
    times = (sub["local"] - t0).dt.total_seconds().to_numpy() / 86400.0
    model.fit_events(times, sub["hour"].to_numpy(), float(times.max()))
