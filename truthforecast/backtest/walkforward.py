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

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import combine
from ..config import BACKTEST, QUANTILE_LEVELS, THRESHOLD_PERCENTILES
from ..models import ForecastTask
from ..models.registry import REFERENCE_MODEL, build_all_models
from ..series import days_into_week
from .scoring import score_forecast

log = logging.getLogger(__name__)


def _stable_seed(*parts) -> int:
    """A seed derived from what the forecast IS, not from when it was made.

    Seeding once at the top of the run would make the whole thing reproducible
    but order-dependent: one model failing to fit shifts every later draw, so an
    unrelated change reshuffles the board. Keying on (model, week, cut) instead
    means each forecast is independently reproducible, and adding or removing a
    model leaves every other model's numbers untouched — which is what makes a
    before/after comparison of a fix mean anything.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


@dataclass
class BacktestResult:
    rows: pd.DataFrame
    thresholds: list[float]
    holdout_start: pd.Timestamp | None
    # (week, cut) -> {model: dense quantile curve}. Kept in memory rather than
    # exported: it is what lets the deployed selection rule be replayed and
    # scored after the fact, without re-running every model.
    curves: dict = field(default_factory=dict)


def _week_starts(daily: pd.Series) -> list[pd.Timestamp]:
    """First day of every complete week in the series, under CONVENTION.

    Anchored on whatever day the convention opens a week — Sunday for the
    Kalshi week, Monday for the original one — rather than assuming Monday.
    """
    idx = daily.index
    offsets = np.array([days_into_week(d) for d in idx])
    starts = pd.unique(idx - pd.to_timedelta(offsets, unit="D"))
    out = []
    for m in sorted(pd.to_datetime(starts)):
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
    seed: int = BACKTEST.seed,
) -> BacktestResult:
    """Score every model on every (week, cut) pair.

    `events` carries post-level timestamps for models that need them (Hawkes).

    Every draw runs from a stream seeded on (model, week, cut), so re-running on
    unchanged data reproduces the leaderboard exactly. That is not tidiness: the
    top twelve models sit inside half a CRPS point of each other, which is well
    under the Monte-Carlo spread of 4,000 samples, so an unseeded board reorders
    itself between runs and the ordering reads as a result. Vary `seed` to
    measure that spread deliberately instead of inheriting it.
    """
    daily = daily.astype(float)
    mondays = _week_starts(daily)
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
    curves: dict = {}
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
                    np.random.seed(_stable_seed(seed, model.name, monday, "fit"))
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
                    np.random.seed(_stable_seed(seed, model.name, monday, cut))
                    samples = model.sample_week_total(task, n_samples)
                except Exception as exc:
                    log.warning("predict failed for %s: %s", model.name, exc)
                    continue
                if samples is None or len(samples) == 0 or not np.isfinite(samples).any():
                    continue

                curves.setdefault((monday, cut), {})[model.name] = combine.dense_quantiles(samples)
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
    result = BacktestResult(
        rows=df, thresholds=thresholds, holdout_start=holdout_start, curves=curves
    )
    headline_rows = score_headline_rule(result, QUANTILE_LEVELS, thresholds)
    if not headline_rows.empty:
        result.rows = pd.concat([df, headline_rows], ignore_index=True)
    return result


# The candidate selection rules, scored side by side. "Top three, averaged
# quantile-wise" was the deployed rule and was never measured against the
# obvious alternatives — including the simplest one, which is to use the single
# best model and not combine at all. Naming them here is the same
# pre-registration the model list gets: the winner is chosen from a fixed list,
# in the open, on development weeks.
HEADLINE_RULES: dict[str, tuple[int, str]] = {
    "headline-top1": (1, "vincent"),
    "headline-top3": (3, "vincent"),
    "headline-top3-pool": (3, "pool"),
    "headline-top5": (5, "vincent"),
}


def score_headline_rule(
    result: BacktestResult,
    quantiles=QUANTILE_LEVELS,
    thresholds=(),
    min_weeks: int = BACKTEST.headline_min_weeks,
) -> pd.DataFrame:
    """Grade the rules that could sit on the front page, week by week.

    The front page does not show a model. It shows "average the top three on the
    leaderboard, quantile-wise" — and that composite appeared on no row of the
    leaderboard, so the one number anybody reads was the one number nobody
    scored. Its members' good scores are not its score: a combination can be
    worse than its best member, and the only way to know is to run the rule.

    Selection here is strictly retrospective. At each week the ranking is
    recomputed from the weeks *before* it, so a rule never picks its models
    using the week it is about to be graded on. That is more conservative than
    the live path, which selects from a leaderboard covering all history —
    including the week in flight, though only its earlier cuts. The gap is
    small and points the safe way: the scored version knows less than the
    deployed one.
    """
    rows = result.rows
    if rows.empty or not result.curves:
        return pd.DataFrame()

    weeks = sorted(rows["week"].unique())
    actual = rows.groupby("week")["actual"].first()
    observed_by = (
        rows.groupby(["week", "cut_dow"])["observed_so_far"].first().to_dict()
    )
    out = []
    for i, week in enumerate(weeks):
        if i < min_weeks:
            continue
        prior = rows[rows["week"] < week]
        ranking = prior.groupby("model")["crps"].mean().sort_values().index.tolist()
        for cut in sorted(rows.loc[rows["week"] == week, "cut_dow"].unique()):
            available = result.curves.get((week, cut), {})
            y = float(actual.loc[week])
            for name, (top_k, how) in HEADLINE_RULES.items():
                chosen = [m for m in ranking if m in available][:top_k]
                if len(chosen) < top_k:
                    continue
                curves = [available[m] for m in chosen]
                curve = (
                    combine.vincentize(curves) if how == "vincent"
                    else combine.linear_pool(curves)
                )
                np.random.seed(_stable_seed(name, week, cut))
                samples = combine.samples_from(curve, 4000)
                s = score_forecast(samples, y, quantiles, thresholds)
                out.append(
                    {
                        "week": week,
                        "cut_dow": cut,
                        "model": name,
                        "family": "deployed",
                        "actual": y,
                        "observed_so_far": float(observed_by.get((week, cut), 0.0)),
                        **{k: v for k, v in s.items() if k != "brier"},
                        **{f"brier_{k}": v for k, v in s["brier"].items()},
                    }
                )
    return pd.DataFrame(out)


def _fit_events(model, events: pd.DataFrame, train_end: pd.Timestamp, lookback_days: int = 180):
    """Feed post-level timestamps (strictly before the origin) to a Hawkes fit."""
    sub = events[(events["local_date"] < train_end)]
    sub = sub[sub["local_date"] >= train_end - pd.Timedelta(days=lookback_days)]
    if len(sub) < 200:
        return
    t0 = sub["local"].min()
    times = (sub["local"] - t0).dt.total_seconds().to_numpy() / 86400.0
    model.fit_events(times, sub["hour"].to_numpy(), float(times.max()))
