"""Scoring-rule and model-recovery tests.

The scoring tests check the estimators against closed forms. The recovery tests
simulate from a process with known parameters and confirm the fitter finds them
again — which is what makes the leaderboard trustworthy. A model zoo nobody has
verified is a row of plausible numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from truthforecast.backtest.scoring import (
    crps_from_samples,
    log_score,
    pinball_loss,
    pit_uniformity,
    pit_value,
    skill_score,
)
from truthforecast.models import ForecastTask
from truthforecast.models.counts import HierarchicalDOWModel, NegBinGLM, PoissonGLM
from truthforecast.models.dynamics import HawkesModel, PoissonHMM
from truthforecast.models.registry import build_all_models


def daily_series(values, start="2025-01-06"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


# ---------------------------------------------------------------- scoring

def test_crps_of_point_forecast_is_absolute_error():
    samples = np.full(500, 10.0)
    assert crps_from_samples(samples, 12.0) == pytest.approx(2.0, abs=1e-9)


def test_crps_matches_closed_form_for_a_normal():
    mu, sd, y = 10.0, 3.0, 12.0
    z = (y - mu) / sd
    exact = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    rng = np.random.default_rng(1)
    got = crps_from_samples(rng.normal(mu, sd, 200_000), y)
    assert got == pytest.approx(exact, rel=0.02)


def test_crps_prefers_the_sharper_of_two_honest_forecasts():
    """Sharpness is rewarded, but only when the forecast is right."""
    rng = np.random.default_rng(2)
    tight = rng.normal(10, 1, 50_000)
    wide = rng.normal(10, 8, 50_000)
    assert crps_from_samples(tight, 10.0) < crps_from_samples(wide, 10.0)


def test_crps_punishes_confident_and_wrong():
    rng = np.random.default_rng(3)
    confident_wrong = rng.normal(10, 1, 50_000)
    humble = rng.normal(10, 8, 50_000)
    assert crps_from_samples(confident_wrong, 40.0) > crps_from_samples(humble, 40.0)


def test_log_score_matches_normal_density():
    mu, sd, y = 10.0, 3.0, 12.0
    rng = np.random.default_rng(4)
    got = log_score(rng.normal(mu, sd, 200_000), y)
    assert got == pytest.approx(-np.log(stats.norm.pdf(y, mu, sd)), rel=0.03)


def test_pinball_is_asymmetric():
    samples = np.arange(0, 100, dtype=float)
    low = pinball_loss(samples, 90.0, [0.1])["0.1"]
    high = pinball_loss(samples, 10.0, [0.1])["0.1"]
    assert low > high  # under-predicting hurts more at a low quantile


def test_pit_is_uniform_for_a_calibrated_forecaster():
    rng = np.random.default_rng(5)
    pits = [pit_value(rng.normal(0, 1, 4000), rng.normal(0, 1)) for _ in range(400)]
    assert pit_uniformity(np.array(pits))["calibrated"] is True


def test_pit_rejects_an_overconfident_forecaster():
    """Too-narrow intervals pile the PIT up at both ends — the exact failure
    the leaderboard's calibration column exists to catch."""
    rng = np.random.default_rng(6)
    pits = [pit_value(rng.normal(0, 0.3, 4000), rng.normal(0, 1)) for _ in range(400)]
    assert pit_uniformity(np.array(pits))["calibrated"] is False


def test_skill_score_sign():
    assert skill_score(8.0, 10.0) == pytest.approx(0.2)
    assert skill_score(12.0, 10.0) < 0


# ------------------------------------------------------- model recovery

def test_poisson_glm_recovers_a_flat_rate():
    rng = np.random.default_rng(7)
    s = daily_series(rng.poisson(20, 400))
    m = PoissonGLM().fit(s)
    mu = m._mu(np.arange(7))
    assert np.all(np.abs(mu - 20) < 3)


def test_negbin_recovers_known_overdispersion():
    """Simulate NB2 with a known alpha and check the fit finds it."""
    rng = np.random.default_rng(8)
    mu, alpha = 20.0, 0.5
    r = 1 / alpha
    s = daily_series(rng.negative_binomial(r, r / (r + mu), 700))
    m = NegBinGLM().fit(s)
    assert m.alpha == pytest.approx(alpha, rel=0.45)


def test_negbin_is_wider_than_poisson_on_overdispersed_data():
    """The whole reason the Negative Binomial family exists."""
    rng = np.random.default_rng(9)
    mu, r = 20.0, 2.0
    s = daily_series(rng.negative_binomial(r, r / (r + mu), 500))
    task = ForecastTask(history=s, week_start=s.index[-1], cut_dow=-1, observed=np.array([]))

    pois = PoissonGLM().fit(s).sample_week_total(task, 20_000)
    nb = NegBinGLM().fit(s).sample_week_total(task, 20_000)
    width = lambda x: np.quantile(x, 0.9) - np.quantile(x, 0.1)
    assert width(nb) > 1.5 * width(pois)


def test_hierarchical_model_shrinks_a_thin_weekday():
    """A weekday with an extreme mean should be pulled toward the others."""
    rng = np.random.default_rng(10)
    vals, idx = [], pd.date_range("2025-01-06", periods=280, freq="D")
    for d in idx:
        vals.append(rng.poisson(80 if d.dayofweek == 2 else 20))
    s = pd.Series(np.array(vals, dtype=float), index=idx)

    m = HierarchicalDOWModel().fit(s)
    raw_wed = s[s.index.dayofweek == 2].mean()
    assert m.mu[2] < raw_wed          # shrunk toward the pool
    assert m.mu[2] > s.mean()          # but still clearly the busy day
    assert 0.0 <= m.shrinkage_weight <= 1.0


def test_hierarchical_predictive_is_overdispersed():
    """Shrinking only the mean would leave a Poisson-width predictive.

    That was a real bug: with ~80 observations per weekday the posterior on the
    rate is tight, so the forecast collapsed back to variance == mean.
    """
    rng = np.random.default_rng(11)
    mu, r = 20.0, 1.0
    s = daily_series(rng.negative_binomial(r, r / (r + mu), 500))
    m = HierarchicalDOWModel().fit(s)
    draws = m.sample_days(np.arange(7), 20_000)
    assert draws.var() > 2.5 * draws.mean()


def test_hmm_separates_two_regimes():
    rng = np.random.default_rng(12)
    vals, state = [], 0
    for _ in range(600):
        state = state if rng.random() < 0.9 else 1 - state
        vals.append(rng.poisson(8 if state == 0 else 55))
    m = PoissonHMM(n_states=2).fit(daily_series(vals))
    d = m.describe()
    assert d, "HMM should fit"
    lo, hi = d["state_rates"]
    assert lo < 20 < hi
    assert min(d["stickiness"]) > 0.5  # both states persist


def test_hawkes_recovers_the_branching_ratio():
    """Simulate a Hawkes process with known eta, then refit it."""
    rng = np.random.default_rng(13)
    eta, beta, mu, T = 0.5, 20.0, 6.0, 240.0

    times = list(rng.uniform(0, T, rng.poisson(mu * T)))
    frontier = list(times)
    while frontier:
        parent = frontier.pop()
        for _ in range(rng.poisson(eta)):
            child = parent + rng.exponential(1 / beta)
            if child < T:
                times.append(child)
                frontier.append(child)

    t = np.sort(np.array(times))
    hours = ((t % 1.0) * 24).astype(int)
    m = HawkesModel().fit_events(t, hours, T)
    assert m.fitted
    assert m.eta == pytest.approx(eta, abs=0.2)


def test_hawkes_simulation_obeys_the_stationary_identity():
    """A Hawkes process runs at mu / (1 - eta), not at mu.

    Every immigrant drags a cascade of descendants behind it. Seeding the
    simulation with the full observed rate and then adding offspring on top
    counts that cascade twice — at eta = 0.7 it inflates the forecast more than
    threefold, which is exactly the bug this pins down.

    Here mu = 6/day and eta = 0.7, so the process should run at
    6 / 0.3 = 20 posts/day, i.e. 140 over a week.
    """
    rng = np.random.default_rng(14)
    s = daily_series(rng.poisson(20, 400))
    m = HawkesModel().fit(s)
    m.mu_per_day, m.eta, m.beta, m.fitted = 6.0, 0.7, 50.0, True

    task = ForecastTask(history=s, week_start=s.index[-1], cut_dow=-1, observed=np.array([]))
    got = np.median(m.sample_week_total(task, 3000))
    assert got == pytest.approx(140, rel=0.25)


# --------------------------------------------------------------- contract

def test_every_registered_model_produces_a_distribution():
    """The contract the leaderboard depends on: fit, then emit samples."""
    rng = np.random.default_rng(15)
    s = daily_series(rng.negative_binomial(2, 2 / (2 + 20), 420))
    task = ForecastTask(history=s, week_start=s.index[-1], cut_dow=2,
                        observed=np.array([10.0, 12.0, 9.0]))

    for model in build_all_models():
        model.fit(s)
        out = model.sample_week_total(task, 800)
        assert out is not None and len(out) > 0, f"{model.name} returned nothing"
        assert np.isfinite(out).all(), f"{model.name} returned non-finite values"
        assert (out >= 0).all(), f"{model.name} returned negative counts"
        # Observed days are already banked, so the total cannot fall below them.
        assert np.median(out) >= 30, f"{model.name} ignored the observed days"


def test_models_expose_their_metadata():
    for model in build_all_models():
        meta = model.meta()
        assert meta["description"] and meta["assumptions"] and meta["fails_when"], model.name


# --- the day in progress ----------------------------------------------------


def _events(days, per_hour):
    """Synthetic post-level frame: `days` dates, posts at the given hours."""
    import pandas as pd
    rows = []
    for d in days:
        for h, n in per_hour.items():
            for _ in range(n):
                rows.append({"local_date": pd.Timestamp(d), "hour": h})
    df = pd.DataFrame(rows)
    df["local_date"] = pd.to_datetime(df["local_date"])
    return df


def test_a_silent_morning_does_not_predict_a_silent_day():
    """The failure that rules out naive scaling: 0 by noon is not 0 by midnight."""
    import numpy as np
    import pandas as pd
    from truthforecast.partial import sample_rest_of_today

    # 200 history days that post nothing before noon and 10 posts after it.
    days = pd.date_range("2025-01-01", periods=200, freq="D")
    ev = _events(days, {14: 5, 20: 5})
    # Today contributes no rows at all — that IS the silent morning.
    today = pd.Timestamp("2025-07-25 12:00")

    rest = sample_rest_of_today(ev, today, n=2000)
    # Scaling by elapsed share would say 0/0.0 -> 0. History says ~10.
    assert np.median(rest) > 5, f"predicted {np.median(rest)} more posts after a silent morning"


def test_a_finished_day_has_little_left():
    import numpy as np
    import pandas as pd
    from truthforecast.partial import sample_rest_of_today

    days = pd.date_range("2025-01-01", periods=200, freq="D")
    ev = _events(days, {9: 10, 11: 10})          # all activity before noon
    today = pd.Timestamp("2025-07-25 23:00")
    ev = pd.concat([ev, _events([today.normalize()], {9: 10, 11: 10})], ignore_index=True)

    rest = sample_rest_of_today(ev, today, n=2000)
    assert np.quantile(rest, 0.9) < 2, "a day whose hours are gone should have nothing left"


def test_today_is_counted_in_the_week_total():
    """Regression: today's posts used to be dropped and re-simulated."""
    import numpy as np
    import pandas as pd
    from truthforecast.series import daily_counts
    from truthforecast.forecast import current_week_forecast

    days = pd.date_range("2025-01-06", periods=180, freq="D")   # a Monday start
    ev = _events(days, {10: 8, 18: 8})
    df = ev.copy()
    df["is_retruth"] = 0
    series = daily_counts(df, "all")

    now = pd.Timestamp("2025-07-04 23:00")                       # a Friday, nearly over
    fc = current_week_forecast(series, events=ev, now=now, n_samples=3000)
    assert fc["today"]["posts_so_far"] == 16
    assert fc["week"]["week_to_date_including_today"] == fc["week"]["observed_total"] + 16
