"""One test per defect found in the second review.

Each of these failed before the corresponding fix. They are written to pin the
*behaviour*, not the implementation: a rewrite that keeps the property passes,
and a rewrite that quietly reintroduces the bug does not. The bugs they cover
share a shape — none of them raised, none of them showed up in the output as
anything but a slightly different number, and several of them were contradicted
by a docstring sitting directly above the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from truthforecast import combine
from truthforecast.config import CONVENTION
from truthforecast.models import ForecastTask
from truthforecast.models.baselines import (
    BlockBootstrapModel,
    LastWeekModel,
    TrailingMedianModel,
)
from truthforecast.models.counts import (
    PoissonGLM,
    ZeroInflatedNegBinModel,
    ZeroInflatedPoissonModel,
)
from truthforecast.models.dynamics import INGARCHModel, LogSarimaModel, PoissonHMM
from truthforecast.models.ml import QuantileGBM


def series(values, start="2025-01-05"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def backtest_task(hist, week_start, week_vals, cut):
    """A task shaped exactly like the walk-forward harness builds one.

    Including the embargo: history stops two days before the week opens, which
    is the gap several models were stepping straight over.
    """
    return ForecastTask(
        history=hist[hist.index < week_start - pd.Timedelta(days=1)],
        week_start=pd.Timestamp(week_start),
        cut_dow=cut,
        observed=np.asarray(week_vals[: cut + 1], dtype=float) if cut >= 0 else np.array([]),
    )


# --------------------------------------------------- the week's own boundary

def test_task_knows_which_calendar_days_it_must_predict():
    """`remaining_dows` alone cannot tell a model how far ahead it is forecasting."""
    hist = series(np.full(200, 15.0))
    week_start = pd.Timestamp("2025-06-01")  # a Sunday
    t = backtest_task(hist, week_start, [1, 2, 3, 4, 5, 6, 7], cut=5)

    assert list(t.remaining_dates) == [pd.Timestamp("2025-06-07")]
    assert t.gap_days == 1                       # the embargoed day
    assert (t.remaining_dates[0] - t.history.index[-1]).days == 8


def test_baselines_cut_the_week_where_the_convention_does():
    """A Sunday-anchored week must not be scored against Monday-anchored history."""
    # 20 posts on the week's opening day, 1 on every other day. Under the right
    # anchor every complete week totals 26; under a Monday anchor it does not.
    idx = pd.date_range("2025-01-05", periods=210, freq="D")   # starts Sunday
    opening = CONVENTION.start_dow
    vals = np.where(idx.dayofweek == opening, 20.0, 1.0)
    hist = pd.Series(vals, index=idx)

    assert LastWeekModel().fit(hist).last_total == pytest.approx(26.0)
    assert float(np.median(TrailingMedianModel().fit(hist).pool)) == pytest.approx(26.0)


def test_block_bootstrap_blocks_are_whole_convention_weeks():
    idx = pd.date_range("2025-01-05", periods=210, freq="D")
    opening = CONVENTION.start_dow
    hist = pd.Series(np.where(idx.dayofweek == opening, 20.0, 1.0), index=idx)

    m = BlockBootstrapModel().fit(hist)
    t = backtest_task(hist, pd.Timestamp("2025-06-01"), [0] * 7, cut=-1)
    blocks = m._blocks(t.remaining_dows)
    # Every whole week of this series totals 26; a misanchored block would mix
    # two weeks and produce something else.
    assert set(np.unique(blocks)) == {26.0}


def test_gbm_training_rows_use_the_convention_week():
    """cut 0 must mean 'the week's opening day is banked', in training too."""
    idx = pd.date_range("2025-01-05", periods=210, freq="D")
    opening = CONVENTION.start_dow
    hist = pd.Series(np.where(idx.dayofweek == opening, 20.0, 1.0), index=idx)

    X, y = QuantileGBM()._build_training(hist)
    assert len(y) > 100
    cuts = X[:, -3]
    observed_sum = X[:, -2]
    # At cut 0 exactly the opening day is banked, so observed_sum is its value.
    assert set(np.unique(observed_sum[cuts == 0])) == {20.0}
    # And the remainder is the other six days.
    assert set(np.unique(y[cuts == 0])) == {6.0}


# ------------------------------------------------- forecasting the right day

def test_sarima_forecasts_the_day_it_was_asked_about():
    """A seasonal model stepped the wrong number of days is in the wrong phase.

    Built so the answer is unambiguous: a spike on one weekday and nothing else.
    Asked for that weekday, the forecast has to be high — whatever the gap
    between the end of history and the day in question.
    """
    idx = pd.date_range("2025-01-05", periods=400, freq="D")
    hist = pd.Series(np.where(idx.dayofweek == 2, 60.0, 3.0), index=idx)  # Wednesdays

    m = LogSarimaModel().fit(hist)
    week_start = pd.Timestamp("2026-02-08")            # a Sunday
    t = backtest_task(hist, week_start, [3, 3, 3, 60, 3, 3, 3], cut=2)
    # cut 2 banks Sun/Mon/Tue, so the first remaining day is the Wednesday.
    assert t.remaining_dates[0].dayofweek == 2

    np.random.seed(0)
    draws = m.sample_days(t.remaining_dows, 4000, t.remaining_dates)
    assert np.median(draws[:, 0]) > 25, "the Wednesday spike was forecast onto another day"


def test_ingarch_state_and_observation_come_from_the_same_moment():
    """The persistence term must be advanced through the week, not left stale.

    Built to isolate exactly the defect. Both weeks end on the same 90-post day,
    so the old code — which read `prev_y` from the week's last observed day but
    left `prev_lambda` back at the end of history — produced *identical*
    forecasts for them. They are not the same situation: five loud days in a row
    have driven the conditional rate somewhere one loud day has not, and
    `b * log(lambda)` is where this model keeps that.
    """
    hist = series(np.random.default_rng(0).poisson(12, 300).astype(float))
    m = INGARCHModel().fit(hist)
    # Known, well-behaved persistence rather than whatever the optimiser found
    # on memoryless data: w=1.0, a=0.4, b=0.3, flat weekdays.
    m.params = np.array([1.0, 0.4, 0.3, 0, 0, 0, 0, 0, 0, np.log(0.3)])
    m.dow_effect = np.zeros(7)
    m.alpha_nb, m.last_lambda, m.last_y = 0.3, 15.0, 12.0

    week_start = hist.index[-1] + pd.Timedelta(days=8)
    one_loud_day = backtest_task(hist, week_start, [0, 0, 0, 0, 90, 0, 0], cut=4)
    five_loud_days = backtest_task(hist, week_start, [90, 90, 90, 90, 90, 0, 0], cut=4)

    np.random.seed(1)
    lo = np.median(m.sample_remaining(one_loud_day, 8000))
    np.random.seed(1)
    hi = np.median(m.sample_remaining(five_loud_days, 8000))
    assert hi > 1.3 * lo, (
        f"a sustained burst and a single loud day forecast the same thing ({hi} vs {lo}) — "
        "the conditional rate was never advanced through the week"
    )


def test_hmm_belief_updates_on_the_weeks_own_days():
    """Stickiness is the model's whole claim; it has to be applied at the cut."""
    rng = np.random.default_rng(2)
    vals, state = [], 0
    for _ in range(500):
        state = state if rng.random() < 0.95 else 1 - state
        vals.append(rng.poisson(5 if state == 0 else 60))
    hist = series(np.asarray(vals, dtype=float))

    m = PoissonHMM(n_states=2).fit(hist)
    rates = np.sort(np.asarray(m.model.lambdas_).ravel())

    quiet = m._filter_forward(m.state_probs, 1, np.array([4.0, 5.0, 6.0, 5.0]))
    loud = m._filter_forward(m.state_probs, 1, np.array([58.0, 62.0, 59.0, 61.0]))
    order = np.argsort(np.asarray(m.model.lambdas_).ravel())
    assert quiet[order[0]] > 0.9, "four quiet days did not move the belief to the quiet state"
    assert loud[order[-1]] > 0.9, "four 60-post days did not move the belief to the loud state"
    assert rates[0] < rates[-1]


def test_hmm_filter_survives_a_count_that_would_underflow():
    """A 168-post day against a rate of 8 has a Poisson pmf of ~0 in float."""
    rng = np.random.default_rng(3)
    hist = series(rng.poisson(8, 300).astype(float))
    m = PoissonHMM(n_states=2).fit(hist)
    out = m._filter_forward(m.state_probs, 0, np.array([168.0]))
    assert np.isfinite(out).all() and out.sum() == pytest.approx(1.0)


# ----------------------------------------------- models that are what they say

def test_zip_actually_puts_mass_on_a_structural_zero():
    """`which="mean"` returns (1-pi)*mu; drawn through one Poisson that is not a ZIP.

    Simulated with a heavy inflation so the difference is unmistakable: a third
    of days are switched off entirely.
    """
    rng = np.random.default_rng(4)
    on = rng.random(600) > 0.35
    hist = series((rng.poisson(30, 600) * on).astype(float))

    z = ZeroInflatedPoissonModel().fit(hist)
    p = PoissonGLM().fit(hist)
    np.random.seed(5)
    zs = z.sample_days(np.arange(7), 20_000)
    ps = p.sample_days(np.arange(7), 20_000)

    assert (zs == 0).mean() > 0.25, "zero-inflated model produced almost no structural zeros"
    assert (zs == 0).mean() > 5 * (ps == 0).mean()
    assert zs.var() > 1.5 * ps.var(), "the zero-inflated predictive was not wider than Poisson"


def test_zinb_is_not_just_a_negative_binomial_in_disguise():
    rng = np.random.default_rng(6)
    on = rng.random(600) > 0.35
    hist = series((rng.negative_binomial(3, 3 / (3 + 25), 600) * on).astype(float))

    m = ZeroInflatedNegBinModel().fit(hist)
    np.random.seed(7)
    draws = m.sample_days(np.arange(7), 20_000)
    assert (draws == 0).mean() > 0.25
    assert m.alpha > 0


# ------------------------------------------------------- combining and tails

def test_threshold_probabilities_never_print_certainty():
    """An unbounded count has no threshold with probability exactly 0 or 1."""
    curve = combine.dense_quantiles(np.random.default_rng(8).normal(150, 20, 50_000))
    p_hi, bounded_hi = combine.prob_at_least(curve, 10_000)
    p_lo, bounded_lo = combine.prob_at_least(curve, -10_000)
    assert bounded_hi and 0.0 < p_hi < 0.01
    assert bounded_lo and 0.99 < p_lo < 1.0


def test_headline_interval_and_threshold_agree():
    """The probability under the interval must be the one the interval implies."""
    from truthforecast.forecast import headline, headline_curve, threshold_probabilities

    rng = np.random.default_rng(9)
    per_model = []
    for i, mu in enumerate((140, 155, 170)):
        s = rng.normal(mu, 25, 40_000)
        per_model.append(
            {
                "model": f"m{i}",
                "family": "test",
                "quantiles": {},
                "_curve": combine.dense_quantiles(s),
            }
        )
    fc = {"per_model": per_model, "week": {"week_to_date_including_today": 100}}

    head = headline(fc, ranking=["m0", "m1", "m2"])
    median = head["median"]
    probs = threshold_probabilities(fc, [median], ranking=["m0", "m1", "m2"])
    # By definition P(X >= median) is 0.5, whatever the members disagreed about.
    assert probs[0]["probability"] == pytest.approx(0.5, abs=0.01)
    curve, chosen = headline_curve(fc, ranking=["m0", "m1", "m2"])
    assert chosen == ["m0", "m1", "m2"] and np.all(np.diff(curve) >= 0)


def test_vincentized_curve_is_monotone():
    rng = np.random.default_rng(10)
    curves = [combine.dense_quantiles(rng.gamma(2, 30, 20_000)) for _ in range(4)]
    out = combine.vincentize(curves)
    assert np.all(np.diff(out) >= 0)


# ---------------------------------------------------------- reproducibility

def test_the_backtest_reproduces_itself():
    """Unseeded, the top of this board reorders between identical runs."""
    from truthforecast.backtest.walkforward import run_backtest

    rng = np.random.default_rng(11)
    daily = series(rng.negative_binomial(2, 2 / (2 + 18), 420).astype(float))

    a = run_backtest(daily, n_samples=300, min_train_weeks=26, max_weeks=4, progress=False)
    b = run_backtest(daily, n_samples=300, min_train_weeks=26, max_weeks=4, progress=False)
    ka = a.rows.groupby("model")["crps"].mean()
    kb = b.rows.groupby("model")["crps"].mean()
    pd.testing.assert_series_equal(ka, kb)


# ---------------------------------------------------------------- the ingest

def test_the_feed_cannot_undelete_a_post():
    """RSS carries no deletion information, so it must not clear the flag."""
    import sqlite3
    from datetime import datetime, timezone

    from truthforecast.ingest.parse import Post
    from truthforecast.ingest.store import SCHEMA, load_posts, upsert_posts

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    when = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

    # The listing sees it, and sees that it is deleted.
    upsert_posts(conn, [Post(trumpstruth_id=1, created_utc=when, text="x",
                             is_deleted=True, time_precision="minute")])
    # The feed re-reads it on the next poll and knows nothing about deletion.
    upsert_posts(conn, [Post(trumpstruth_id=1, created_utc=when, text="x",
                             is_deleted=False, time_precision="second")])

    assert load_posts(conn)[0]["is_deleted"] == 1, "the feed resurrected a deleted post"

    # A later listing pass that no longer shows the badge still clears it.
    upsert_posts(conn, [Post(trumpstruth_id=1, created_utc=when, text="x",
                             is_deleted=False, time_precision="minute")])
    assert load_posts(conn)[0]["is_deleted"] == 0


def test_recent_posts_are_ordered_by_when_they_appeared():
    """A ReTruth of an old post is recent news, filed under an old timestamp."""
    from truthforecast.series import redate_retruths

    df = pd.DataFrame(
        {
            "trumpstruth_id": [1, 2, 3],
            "created_utc": pd.to_datetime(
                ["2026-07-30T10:00Z", "2026-07-30T11:00Z", "2020-01-01T00:00Z"], utc=True
            ),
            "is_retruth": [0, 0, 1],
        }
    )
    out = redate_retruths(df).sort_values(
        ["effective_utc", "trumpstruth_id"], ascending=False
    )
    assert out.iloc[0]["trumpstruth_id"] == 3, "the reshare sorted six years into the past"
    # Re-dated to the mirror's arrival order, not to 2020.
    assert out.iloc[0]["effective_utc"].year == 2026


# ------------------------------------------------- the day in progress, in time

def _day_events(days, times):
    """Post-level frame with real timestamps: `times` are (hour, minute) pairs."""
    rows = []
    for d in days:
        day = pd.Timestamp(d)
        for h, m in times:
            rows.append({"local_date": day, "hour": h,
                         "local": day + pd.Timedelta(hours=h, minutes=m)})
    out = pd.DataFrame(rows)
    out["local_date"] = pd.to_datetime(out["local_date"])
    out["local"] = pd.to_datetime(out["local"])
    return out


def _flat_events(n_days=200, per_day=12, spread=0):
    """Days of `per_day` posts spread over 06:00-20:00.

    `spread` adds day-to-day variation in the count. Zero is a deliberate
    stress case — every day identical, so the pool carries no rank information
    at all and the conditioning has nothing to preserve.
    """
    rng = np.random.default_rng(99)
    days = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in days:
        k = per_day if not spread else max(1, int(rng.normal(per_day, spread)))
        rows.extend(
            _day_events([d], [(int(6 + 14 * i / max(k, 1)), (i * 7) % 60) for i in range(k)])
            .to_dict("records")
        )
    out = pd.DataFrame(rows)
    out["local_date"] = pd.to_datetime(out["local_date"])
    out["local"] = pd.to_datetime(out["local"])
    return out


def test_the_projection_does_not_jump_at_midnight():
    """Time is continuous; the date rolling over is not information.

    The day in progress used to be owned by a different estimator from the one
    that owned tomorrow, so at the stroke of midnight the same calendar day
    changed hands between two objects that disagreed — and the projection moved
    with nothing having happened. The conditioning is now the identity at
    elapsed time zero, so a model's day survives the handover unchanged.
    """
    from truthforecast import partial

    ev = _flat_events()
    boundary = pd.Timestamp("2025-07-20")
    rng = np.random.default_rng(0)
    model_day = rng.negative_binomial(3, 3 / (3 + 12), 40_000).astype(float)

    before = partial.build_conditioner(ev, boundary - pd.Timedelta(seconds=1))
    after = partial.build_conditioner(ev, boundary)

    # The last second of yesterday has essentially nothing left in it...
    assert before.remainder(model_day).mean() < 0.5
    # ...and the first instant of today hands the model back its own day.
    assert after.is_identity, "conditioning at elapsed zero must be the identity"
    assert after.remainder(model_day).mean() == pytest.approx(model_day.mean(), rel=0.02)


def test_the_projection_does_not_step_at_an_hour_boundary():
    """Conditioning on `hour <= now.hour` made this a step function in time.

    14:01 and 14:59 returned identical answers and 15:01 jumped, because
    today's partial hour was being compared against history's complete one.
    """
    from truthforecast import partial

    ev = _flat_events()
    day = pd.Timestamp("2025-07-20")
    vals = [
        partial.build_conditioner(ev, day + pd.Timedelta(hours=h, minutes=m))
        .sample_remainder(4000).mean()
        for h, m in [(13, 58), (13, 59), (14, 0), (14, 1), (14, 2)]
    ]
    steps = np.abs(np.diff(vals))
    assert steps.max() < 0.6, f"the estimator steps across the hour: {np.round(vals, 2)}"
    # And it genuinely moves through the hour rather than holding flat.
    inside = [
        partial.build_conditioner(ev, day + pd.Timedelta(hours=14, minutes=m))
        .sample_remainder(4000).mean()
        for m in (1, 59)
    ]
    assert inside[0] != inside[1], "nothing changed across 58 minutes of the day"


def test_what_is_left_of_the_day_decays_to_nothing():
    """Monotone in elapsed time, and zero once the day is over."""
    from truthforecast import partial

    ev = _flat_events()
    day = pd.Timestamp("2025-07-20")
    means = [
        partial.build_conditioner(ev, day + pd.Timedelta(hours=h)).sample_remainder(4000).mean()
        for h in (0, 6, 12, 18, 23)
    ]
    assert means[0] > means[-1]
    # Allowing a little sampling wobble, the sequence has to fall.
    assert all(b <= a + 0.75 for a, b in zip(means, means[1:])), np.round(means, 2)
    end = partial.build_conditioner(ev, day + pd.Timedelta(hours=23, minutes=59, seconds=59))
    assert end.sample_remainder(4000).mean() < 0.5


def test_a_busy_model_is_told_what_busy_days_had_left():
    """The map is rank-preserving, not a replacement of the model's opinion."""
    from truthforecast import partial

    cond = partial.build_conditioner(_flat_events(spread=5), pd.Timestamp("2025-07-20 12:00"))
    quiet = cond.remainder(np.full(5000, 4.0))
    busy = cond.remainder(np.full(5000, 40.0))
    assert busy.mean() > quiet.mean(), "the conditioning threw away the model's view"


def test_an_identical_pool_carries_no_rank_information():
    """The degenerate case has to stay finite rather than invent a tail.

    Every day the same, so there is nothing to tell a busy model from a quiet
    one — and extrapolating a model's extreme day past the end of a pool with
    no spread must not manufacture posts in a day that has no time left.
    """
    from truthforecast import partial

    ev = _flat_events(spread=0)
    late = partial.build_conditioner(ev, pd.Timestamp("2025-07-20 23:59:59"))
    out = late.remainder(np.full(5000, 400.0))
    assert np.isfinite(out).all() and out.max() < 0.5


def test_the_projection_can_never_fall_below_what_has_already_happened():
    """A week total cannot be less than the posts already in it.

    The failure this pins down: on a day that has not posted yet, the daily
    series has no row for today at all, so the observed array was one short and
    its last element was *yesterday*. Overwriting "today's count so far" onto
    the end of it therefore erased a completed day — a quiet Saturday deleted
    Friday's seven posts and the headline came out below the banked total. An
    impossible number is the one kind this project must never print.
    """
    from truthforecast.forecast import current_week_forecast, headline
    from truthforecast.series import daily_counts

    days = pd.date_range("2025-01-05", periods=180, freq="D")   # a Sunday start
    ev = _day_events(days, [(10, 0), (10, 30), (18, 0), (18, 15)])
    df = ev.copy()
    df["is_retruth"] = 0
    series = daily_counts(df, "all")
    # Cut the series short so "today" has produced nothing and has no row.
    quiet_today = pd.Timestamp("2025-07-05 16:00")               # a Saturday
    series = series.restrict(end="2025-07-04")

    fc = current_week_forecast(series, events=ev, now=quiet_today, n_samples=4000)
    head = headline(fc, ranking=[r["model"] for r in fc["per_model"]])
    banked = fc["week"]["week_to_date_including_today"]
    assert head["median"] >= banked, f"median {head['median']} below the banked {banked}"
    assert head["interval_80"][0] >= banked
    assert fc["week"]["today_partial_count"] == 0


def test_every_daily_model_can_describe_a_single_day():
    """The conditioning needs a per-day distribution; daily models must supply one."""
    from truthforecast.models.base import DailyModel
    from truthforecast.models.registry import build_all_models

    rng = np.random.default_rng(1)
    s = series(rng.negative_binomial(2, 2 / (2 + 18), 420).astype(float))
    date = s.index[-1] + pd.Timedelta(days=2)
    for model in build_all_models():
        model.fit(s)
        out = model.sample_one_day(date, 500)
        if isinstance(model, DailyModel):
            assert out is not None and len(out) == 500, model.name
            assert np.isfinite(out).all() and (out >= 0).all(), model.name


# ----------------------------------------------- a week is over, or it is not

def test_a_week_is_not_complete_until_it_has_ended():
    """Seven rows is not the same as seven finished days.

    Found by reading the deployed site rather than the code: the prospective
    record claimed to have scored a week that was still running. The daily
    series is zero-filled up to the last day carrying a post, so on the week's
    own final day the seventh row exists and holds a *partial* count — and the
    week read as complete, with the day-in-progress total standing in for the
    finished one.
    """
    from truthforecast.series import CountSeries

    # Two full weeks then a Saturday still in progress (Sun-anchored weeks).
    idx = pd.date_range("2025-01-05", periods=21, freq="D")   # Sun 5 Jan ...
    s = CountSeries(daily=pd.Series(np.full(21, 10.0), index=idx), label="t")
    last_day = idx[-1]                                        # Sat 25 Jan
    assert last_day.dayofweek == 5

    during = s.complete_weeks(now=last_day + pd.Timedelta(hours=14))
    after = s.complete_weeks(now=last_day + pd.Timedelta(days=1))
    assert last_day not in during.index, "an in-flight week was reported complete"
    assert last_day in after.index, "a finished week must be scored"
    assert len(during) == len(after) - 1


def test_the_backtest_will_not_score_the_week_in_progress():
    from truthforecast.backtest.walkforward import _week_starts

    idx = pd.date_range("2025-01-05", periods=21, freq="D")
    daily = pd.Series(np.full(21, 10.0), index=idx)
    saturday = idx[-1]
    during = _week_starts(daily, today=saturday)
    after = _week_starts(daily, today=saturday + pd.Timedelta(days=1))
    assert len(during) == 2 and len(after) == 3
    assert (saturday - pd.Timedelta(days=6)) not in during


# -------------------------------------------------------------- the live log

def test_the_live_record_is_append_only_and_scores_closed_weeks(tmp_path):
    from truthforecast import live

    path = tmp_path / "log.jsonl"
    curve = combine.dense_quantiles(np.random.default_rng(12).normal(150, 20, 20_000))
    fc = {"week": {"week_start": "2026-07-19", "week_end": "2026-07-25",
                   "days_into_week": 3, "week_to_date_including_today": 80}}
    head = {"median": 150.0, "models_used": ["a"], "quantiles": {"0.5": 150.0}}

    assert live.record(fc, head, curve=curve, path=path)
    # The cut is already registered: the first commitment stands, and a later
    # pass on the same day must not quietly improve on it.
    assert live.record(fc, head, curve=curve, path=path) is False
    assert len(live.load(path)) == 1

    # A different cut is a new commitment.
    fc2 = {"week": dict(fc["week"], days_into_week=4)}
    assert live.record(fc2, head, curve=curve, path=path)
    assert len(live.load(path)) == 2, "the log must never overwrite a forecast"

    # Nothing closed yet.
    assert live.score(pd.Series(dtype=float), path=path)["available"] is False
    weekly = pd.Series([148.0], index=[pd.Timestamp("2026-07-25")])
    scored = live.score(weekly, path=path)
    assert scored["available"] and scored["n_weeks"] == 1
    assert scored["n_forecasts"] == 2      # two cuts of the same week
    assert scored["enough_to_conclude"] is False
    # One week is not a calibration claim, and the record has to say so.
    assert "Too short" in scored["note"]
