"""Models with memory: INGARCH, log-SARIMA, regime-switching HMM, Hawkes.

Everything in `counts.py` treats days as independent draws given the weekday.
These four do not. They are the models that can represent "he posted a lot
yesterday, so today is different", which is the whole reason bursts exist.

Two rival explanations for clumping run through this file, and they are kept
side by side on purpose:

  * **Something outside caused it** — a storm knocks down forty trees at once.
    Regime-switching (`PoissonHMM`) is that story: an unobserved state, driven
    by whatever is going on in the world, sets today's rate.
  * **The events cause each other** — an earthquake and its aftershocks, each
    with aftershocks of their own. `HawkesModel` is that story: every post
    raises the chance of the next one, and the echo decays.

These two generate remarkably similar daily counts, and separating them from
daily totals alone is not a settled exercise. The project therefore reports
both fits and their scores, and asserts neither mechanism.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import DailyModel, ForecastTask, Model, safe_positive

warnings.filterwarnings("ignore")


class INGARCHModel(Model):
    """Observation-driven count autoregression — GARCH's count cousin.

    log(lambda_t) = w + a*log(1+y_{t-1}) + b*log(lambda_{t-1}) + weekday effect

    The lambda_{t-1} term is what gives it persistence: a busy day raises the
    rate, and the elevation decays slowly rather than resetting overnight. That
    is a compact way to represent clustering without committing to a mechanism.
    """

    name = "ingarch"
    family = "dynamics"
    description = "Log-linear count autoregression with a persistent conditional rate."
    assumptions = "Today's rate depends on yesterday's count and yesterday's rate."
    fails_when = "Bursts are near-instantaneous within a day rather than across days."

    def __init__(self):
        self.params = None
        self.dow_effect = np.zeros(7)
        self.last_lambda = 1.0
        self.last_y = 0.0
        self.alpha_nb = 0.5

    @staticmethod
    def _filter(y, dow, w, a, b, dow_eff):
        n = len(y)
        loglam = np.empty(n)
        prev_loglam = np.log(max(y.mean(), 0.5))
        for t in range(n):
            prev_y = y[t - 1] if t > 0 else y.mean()
            prev_loglam = w + a * np.log1p(prev_y) + b * prev_loglam + dow_eff[dow[t]]
            prev_loglam = np.clip(prev_loglam, -5, 8)
            loglam[t] = prev_loglam
        return np.exp(loglam)

    def fit(self, history: pd.Series) -> "INGARCHModel":
        from scipy.optimize import minimize
        from scipy.special import gammaln

        y = history.to_numpy(dtype=float)
        dow = history.index.dayofweek.to_numpy()
        if len(y) < 30:
            self.params = None
            self.last_lambda = float(max(y.mean(), 0.5)) if len(y) else 1.0
            return self

        def nll(theta):
            w, a, b = theta[0], theta[1], theta[2]
            dow_eff = np.concatenate([[0.0], theta[3:9]])
            alpha = np.exp(theta[9])
            if not (-1 < a < 1 and -1 < b < 1) or a + b > 0.999:
                return 1e9
            lam = self._filter(y, dow, w, a, b, dow_eff)
            r = 1.0 / alpha
            # Negative Binomial likelihood: the conditional rate captures the
            # clustering, but the counts are still overdispersed around it.
            ll = (
                gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                + r * np.log(r / (r + lam)) + y * np.log(lam / (r + lam))
            )
            return -np.sum(ll) if np.isfinite(ll).all() else 1e9

        x0 = np.array([np.log(max(y.mean(), 0.5)) * 0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0, np.log(0.5)])
        try:
            res = minimize(nll, x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "fatol": 1e-3, "xatol": 1e-3})
            self.params = res.x
            w, a, b = res.x[0], res.x[1], res.x[2]
            self.dow_effect = np.concatenate([[0.0], res.x[3:9]])
            self.alpha_nb = float(np.exp(res.x[9]))
            lam = self._filter(y, dow, w, a, b, self.dow_effect)
            self.last_lambda = float(lam[-1])
        except Exception:
            self.params = None
            self.last_lambda = float(max(y.mean(), 0.5))
        self.last_y = float(y[-1])
        return self

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        if self.params is None:
            return np.random.poisson(self.last_lambda, size=(n, len(dows))).sum(axis=1).astype(float)

        w, a, b = self.params[0], self.params[1], self.params[2]
        r = 1.0 / max(self.alpha_nb, 1e-9)

        # Run the filter forward from the end of history to the cut, so the
        # conditional rate and the last count come from the same moment. Seeding
        # `prev_y` from the week's most recent observed day while leaving
        # `prev_loglam` at the end of history mixed a state from up to nine days
        # earlier with an observation from yesterday — and the persistence term
        # b*log(lambda) is most of what this model is.
        loglam = float(np.log(max(self.last_lambda, 1e-6)))
        prev_y_scalar = float(self.last_y)
        last_hist = task.history.index[-1] if len(task.history) else task.week_start

        # Days the embargo hides come first and are stepped through blind: the
        # model's own expectation stands in for the count it was not allowed to
        # see. Then the days of the target week that really are known.
        for k in range(task.gap_days):
            dow = int((last_hist + pd.Timedelta(days=k + 1)).dayofweek)
            loglam = float(np.clip(
                w + a * np.log1p(prev_y_scalar) + b * loglam + self.dow_effect[dow], -5, 8
            ))
            prev_y_scalar = float(np.exp(loglam))
        for date, y_obs in zip(task.observed_dates, np.asarray(task.observed, dtype=float)):
            loglam = float(np.clip(
                w + a * np.log1p(prev_y_scalar) + b * loglam + self.dow_effect[int(date.dayofweek)],
                -5, 8,
            ))
            prev_y_scalar = float(y_obs)

        prev_y = np.full(n, prev_y_scalar, dtype=float)
        prev_loglam = np.full(n, loglam, dtype=float)

        total = np.zeros(n)
        for d in dows:
            loglam = w + a * np.log1p(prev_y) + b * prev_loglam + self.dow_effect[int(d)]
            loglam = np.clip(loglam, -5, 8)
            lam = np.exp(loglam)
            p = r / (r + lam)
            draw = np.random.negative_binomial(r, p).astype(float)
            total += draw
            prev_y, prev_loglam = draw, loglam
        return total


class LogSarimaModel(DailyModel):
    """SARIMA on log1p counts — the standard time-series answer, for contrast.

    Working in log space makes effects multiplicative, which travels far better
    across quiet and loud days than an additive rule: "a rally multiplies
    posting by 1.4" survives where "a rally adds 15 posts" does not.
    """

    name = "log-sarima"
    family = "dynamics"
    description = "SARIMA(1,0,1)x(1,0,1,7) on log1p(counts)."
    assumptions = "Log counts are approximately Gaussian with weekly seasonality."
    fails_when = "Zeros and extreme spikes — log1p tames them but does not make them Gaussian."
    refit_every_weeks = 2

    def __init__(self):
        self.res = None
        self.fallback = 1.0
        self.resid_sd = 0.5

    def fit(self, history: pd.Series) -> "LogSarimaModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.log1p(history.to_numpy(dtype=float))
        self.fallback = float(history.mean()) if len(history) else 1.0
        self.last_index = history.index[-1] if len(history) else None
        if len(y) < 60:
            self.res = None
            return self
        try:
            self.res = SARIMAX(
                y, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7),
                trend="c", enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            self.resid_sd = float(np.std(self.res.resid[10:]))
        except Exception:
            self.res = None
        return self

    def sample_days(self, dows: np.ndarray, n: int, dates=None) -> np.ndarray:
        """Forecast the days actually being asked about.

        A state-space model forecasts *forward from the end of its training
        data*, so `get_forecast(steps=len(remaining))` returns the next few days
        after history — which are not the remaining days of the target week.
        The gap is the embargo plus however much of the week is already
        observed: at a Friday cut in the backtest, the one remaining day sits
        eight steps past the end of history, and the model was being asked for
        step 1. For a seasonal model that is not merely a shift, it is the wrong
        phase of the weekly cycle — a Saturday forecast made with Sunday's
        seasonal term.
        """
        h = len(dows)
        if self.res is None:
            return np.random.poisson(self.fallback, size=(n, h)).astype(float)

        offsets = np.arange(1, h + 1)
        if dates is not None and len(dates) == h and self.last_index is not None:
            offsets = np.array([int((d - self.last_index).days) for d in dates])
            if (offsets < 1).any():
                offsets = np.arange(1, h + 1)
        try:
            fc = self.res.get_forecast(steps=int(offsets.max()))
            mu_all = np.asarray(fc.predicted_mean, dtype=float)
            sd_all = np.sqrt(np.asarray(fc.var_pred_mean, dtype=float))
            mu, sd = mu_all[offsets - 1], sd_all[offsets - 1]
        except Exception:
            return np.random.poisson(self.fallback, size=(n, h)).astype(float)
        draws = np.random.normal(mu, np.maximum(sd, 1e-3), size=(n, h))
        return np.maximum(np.expm1(draws), 0.0)


class PoissonHMM(DailyModel):
    """Hidden regimes: quiet days and manic days, and how sticky each is.

    You cannot see which mood he is in; you only see the counts, the way a
    windowless office worker infers the weather from whether colleagues carry
    umbrellas. What makes it work is that states persist — so a single quiet day
    does not convince the model the burst is over.

    The stickiness (the diagonal of the transition matrix) is the number that
    answers "he's had two quiet days — does that mean anything for the
    weekend?", which a raw base rate cannot.
    """

    name = "poisson-hmm"
    family = "regime"
    description = "Hidden Markov Model with Poisson emissions; 2-3 latent activity states."
    assumptions = "A small number of discrete, persistent activity regimes."
    fails_when = "Activity varies continuously rather than in discrete modes."
    refit_every_weeks = 2

    n_states = 2

    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.model = None
        self.state_probs = None
        self.fallback = 1.0

    def fit(self, history: pd.Series) -> "PoissonHMM":
        from hmmlearn import hmm

        y = history.to_numpy(dtype=float)
        self.fallback = float(y.mean()) if len(y) else 1.0
        if len(y) < 60:
            self.model = None
            return self
        try:
            m = hmm.PoissonHMM(n_components=self.n_states, n_iter=200, random_state=0)
            m.fit(y.reshape(-1, 1))
            self.model = m
            post = m.predict_proba(y.reshape(-1, 1))
            self.state_probs = post[-1]
        except Exception:
            self.model = None
        return self

    def describe(self) -> dict:
        """Reported on the site: what the states are and how sticky they are."""
        if self.model is None:
            return {}
        rates = np.asarray(self.model.lambdas_).ravel()
        order = np.argsort(rates)
        trans = np.asarray(self.model.transmat_)
        return {
            "state_rates": [round(float(rates[i]), 2) for i in order],
            "stickiness": [round(float(trans[i, i]), 3) for i in order],
            "expected_run_length_days": [
                round(float(1.0 / max(1 - trans[i, i], 1e-6)), 1) for i in order
            ],
        }

    def _emit(self, rates: np.ndarray) -> np.ndarray:
        """Draw counts given each sample's current state rate."""
        return np.random.poisson(rates).astype(float)

    def _emission_logpmf(self, y: float, rates: np.ndarray) -> np.ndarray:
        """log P(count | state). Must match what `_emit` draws from.

        Filtering with a tighter distribution than the model samples from makes
        the state belief far more confident than the model is entitled to be —
        a 60-post day against rates of 5/15/45 is essentially decisive under a
        Poisson likelihood and merely suggestive under an overdispersed one. The
        subclass that widened its emissions has to widen its filter with them.
        """
        from scipy.stats import poisson

        return poisson.logpmf(y, np.maximum(rates, 1e-9))

    def _filter_forward(self, probs: np.ndarray, gap_days: int, observed) -> np.ndarray:
        """Advance the state belief from the end of history to the cut.

        Stickiness is the whole claim this model makes — "he has had two quiet
        days, does that mean anything for the weekend?" — and it was being
        thrown away at exactly the moment it pays. The belief came from the last
        day of *history*, which at a Friday cut is more than a week stale, and
        the six days of the target week the forecaster genuinely knows about
        never entered it. Blind days (the embargo) advance by the transition
        matrix alone; observed days additionally get a Bayes update on their
        count.
        """
        from scipy.special import logsumexp

        rates = np.asarray(self.model.lambdas_).ravel()
        trans = np.asarray(self.model.transmat_)
        p = np.asarray(probs, dtype=float)
        p = p / max(p.sum(), 1e-12)

        for _ in range(int(gap_days)):
            p = p @ trans
        for y in np.asarray(observed, dtype=float):
            prior = p @ trans
            # In logs: a 168-post day against a rate of 8 underflows outright,
            # and the resulting 0/0 would silently reset the belief to uniform.
            logpost = np.log(np.maximum(prior, 1e-300)) + self._emission_logpmf(y, rates)
            p = np.exp(logpost - logsumexp(logpost))
        return p / max(p.sum(), 1e-12)

    def sample_remaining(self, task: ForecastTask, n: int) -> np.ndarray:
        dows = task.remaining_dows
        if len(dows) == 0:
            return np.zeros(n)
        probs = self.state_probs
        if self.model is not None and probs is not None:
            try:
                probs = self._filter_forward(probs, task.gap_days, task.observed)
            except Exception:
                probs = self.state_probs
        return self.sample_days(dows, n, task.remaining_dates, state_probs=probs).sum(axis=1)

    def sample_days(self, dows: np.ndarray, n: int, dates=None, state_probs=None) -> np.ndarray:
        h = len(dows)
        if self.model is None:
            return np.random.poisson(self.fallback, size=(n, h)).astype(float)

        rates = np.asarray(self.model.lambdas_).ravel()
        trans = np.asarray(self.model.transmat_)
        k = len(rates)

        # Start from the filtered state distribution as of the cut and propagate
        # the chain forward, so persistence is carried into the future.
        start = self.state_probs if state_probs is None else state_probs
        start = np.asarray(start, dtype=float)
        states = np.random.choice(k, size=n, p=start / start.sum())
        out = np.empty((n, h))
        for j in range(h):
            nxt = states.copy()
            for s in range(k):
                mask = states == s
                if mask.any():
                    nxt[mask] = np.random.choice(k, size=int(mask.sum()), p=trans[s])
            states = nxt
            out[:, j] = self._emit(rates[states])
        return out


class NegBinHMM(PoissonHMM):
    """HMM with overdispersed emissions — states plus within-state wobble."""

    name = "negbin-hmm"
    family = "regime"
    description = "Three-state HMM; Poisson emissions with an overdispersion correction."
    assumptions = "As poisson-hmm, with extra scatter inside each regime."
    fails_when = "States are poorly separated; the extra freedom just absorbs noise."

    def __init__(self):
        super().__init__(n_states=3)
        self.alpha = 0.3

    def fit(self, history: pd.Series) -> "NegBinHMM":
        super().fit(history)
        # Estimate within-state overdispersion from the scatter of the observed
        # counts around their own most-likely state mean. A pure Poisson HMM
        # forces variance == mean inside each regime, which is the same
        # constraint that broke the plain Poisson model in the first place.
        self.alpha = 0.3
        if self.model is not None:
            try:
                y = history.to_numpy(dtype=float)
                rates = np.asarray(self.model.lambdas_).ravel()
                assign = self.model.predict(y.reshape(-1, 1))
                mu = rates[assign]
                resid_var = float(np.mean((y - mu) ** 2))
                mean_mu = float(np.mean(mu))
                # var = mu + alpha*mu^2  =>  alpha = (var - mu) / mu^2
                self.alpha = float(np.clip((resid_var - mean_mu) / max(mean_mu**2, 1e-9), 1e-4, 10))
            except Exception:
                self.alpha = 0.3
        return self

    def _emit(self, rates: np.ndarray) -> np.ndarray:
        mu = np.maximum(rates, 1e-6)
        r = 1.0 / max(self.alpha, 1e-9)
        p = r / (r + mu)
        return np.random.negative_binomial(r, p).astype(float)

    def _emission_logpmf(self, y: float, rates: np.ndarray) -> np.ndarray:
        """The same overdispersed emission the sampler uses, in the filter."""
        from scipy.stats import nbinom

        mu = np.maximum(rates, 1e-9)
        r = 1.0 / max(self.alpha, 1e-9)
        return nbinom.logpmf(y, r, r / (r + mu))


class HawkesModel(Model):
    """Self-exciting process: each post makes the next post more likely.

    Fitted on post-level timestamps, not daily totals, because the echo decays
    over hours and daily aggregation destroys it. The kernel is exponential:
    intensity(t) = mu(hour) + sum over past posts of eta * beta * exp(-beta*(t - t_i)).

    The quantity worth reading is the **branching ratio** eta: the average
    number of further posts one post spawns. It is the same number as R0 in an
    epidemic. Below 1, bursts die out; approaching 1, a single post can set off
    an avalanche of unpredictable length.

    Forecasting is by simulation to the end of the week, which propagates both
    the circadian background and the self-excitation.
    """

    name = "hawkes"
    family = "self-exciting"
    description = "Hawkes process with exponential kernel and a circadian background."
    assumptions = "Posts trigger further posts; the effect decays exponentially."
    fails_when = "Bursts are actually driven by outside events — indistinguishable here."
    refit_every_weeks = 4
    uses_events = True

    def __init__(self):
        self.mu_hourly = np.full(24, 0.5)
        self.mu_per_day = 12.0
        self.eta = 0.5
        self.beta = 2.0
        self.fitted = False

    def fit_events(self, times_days: np.ndarray, hours: np.ndarray, span_days: float) -> "HawkesModel":
        """`times_days` are event times in days since the window start."""
        from scipy.optimize import minimize

        if len(times_days) < 200 or span_days <= 0:
            self.fitted = False
            return self

        t = np.sort(np.asarray(times_days, dtype=float))
        observed_rate = len(t) / max(span_days, 1e-9)  # events per day, total

        # Circadian shape, normalized to sum to 1. It describes WHEN within a day
        # activity happens; the overall level is `mu_per_day`, fitted below.
        counts = np.bincount(hours.astype(int), minlength=24).astype(float)
        self.hour_shape = counts / max(counts.sum(), 1.0)
        self.mu_hourly = counts / max(span_days, 1.0)

        def nll(theta):
            eta = 1 / (1 + np.exp(-theta[0]))
            beta = np.exp(theta[1])
            mu_rate = np.exp(theta[2])
            if not (0 < eta < 0.999) or not (0.05 < beta < 5000) or not (1e-6 < mu_rate < 1e6):
                return 1e12
            # Recursive excitation term A_i = sum_{j<i} exp(-beta (t_i - t_j)).
            A = 0.0
            loglik = 0.0
            prev = t[0]
            for i in range(len(t)):
                if i > 0:
                    A = (A + 1.0) * np.exp(-beta * (t[i] - prev))
                    prev = t[i]
                lam = mu_rate + eta * beta * A
                loglik += np.log(max(lam, 1e-12))
            # Compensator: background mass plus the mass of every decayed kernel.
            comp = mu_rate * span_days + eta * np.sum(1 - np.exp(-beta * (span_days - t)))
            return -(loglik - comp)

        # mu is a FREE parameter, not the empirical rate. Pinning it to the
        # observed rate leaves the likelihood no background to work with, so
        # every cluster has to be explained by self-excitation and eta is driven
        # toward 1 regardless of the truth. The stationary identity is
        # observed_rate = mu / (1 - eta), so the sensible start is a background
        # carrying about half the observed activity.
        x0 = np.array([0.0, np.log(20.0), np.log(max(observed_rate * 0.5, 1e-3))])
        try:
            res = minimize(nll, x0, method="Nelder-Mead",
                           options={"maxiter": 1200, "fatol": 1e-3, "xatol": 1e-3})
            self.eta = float(1 / (1 + np.exp(-res.x[0])))
            self.beta = float(np.exp(res.x[1]))
            self.mu_per_day = float(np.exp(res.x[2]))
            self.fitted = True
        except Exception:
            self.fitted = False
        return self

    def fit(self, history: pd.Series) -> "HawkesModel":
        # Daily history alone cannot identify an intra-day kernel; the harness
        # calls `fit_events` with post-level timestamps. This keeps a usable
        # fallback so the model still produces a distribution either way.
        self.daily_mean = float(history.mean()) if len(history) else 1.0
        self.dow_factor = np.ones(7)
        if len(history):
            by = history.groupby(history.index.dayofweek).mean()
            overall = history.mean()
            if overall > 0:
                self.dow_factor = (by.reindex(range(7)).fillna(overall) / overall).to_numpy()
        return self

    def describe(self) -> dict:
        if not self.fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "branching_ratio": round(self.eta, 3),
            "decay_beta_per_day": round(self.beta, 2),
            "kernel_half_life_hours": round(float(np.log(2) / self.beta * 24), 2),
            "interpretation": (
                f"each post triggers on average {self.eta:.2f} further posts; "
                f"a burst dies out below 1.0"
            ),
        }

    def sample_remaining(self, task: ForecastTask, n: int, max_events: int = 4000) -> np.ndarray:
        dows = task.remaining_dows
        h = len(dows)
        if h == 0:
            return np.zeros(n)

        if not self.fitted:
            base = self.daily_mean * self.dow_factor[dows].sum()
            return np.random.poisson(max(base, 0.1), size=n).astype(float)

        # Immigrant-and-offspring simulation: draw the background events, then
        # let each event spawn a Poisson(eta) cascade of descendants. Only
        # descendants landing inside the horizon are counted.
        #
        # The background is the IMMIGRANT rate, not the observed rate. A
        # stationary Hawkes process runs at mu/(1 - eta), because every
        # immigrant drags a cascade of descendants along behind it. Seeding the
        # simulation with the full observed rate and then adding cascades on top
        # counts the offspring twice and inflates the forecast by 1/(1 - eta) —
        # at eta = 0.73 that is a factor of nearly four.
        mu_per_day = float(getattr(self, "mu_per_day", self.mu_hourly.sum() * (1 - self.eta)))

        # The immigrant rate is modulated by the weekday. Every other model in
        # the zoo knows that Saturday is not Tuesday; this one was flat across
        # the week, which on a series whose weekday means run 14 to 24 posts is
        # a large amount of structure to leave on the table — and it showed, in
        # the worst interval coverage on the board. The factors are normalised
        # to average 1 so the process keeps its fitted overall level.
        factor = np.ones(h)
        dow_factor = np.asarray(getattr(self, "dow_factor", np.ones(7)), dtype=float)
        if np.isfinite(dow_factor).all() and dow_factor.mean() > 0:
            factor = dow_factor[np.asarray(dows, dtype=int)] / dow_factor.mean()

        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        horizon = float(h)

        # Vectorized generation-by-generation cascade across all n paths at once.
        # Every live event is a row; `owner` says which simulation path it
        # belongs to. Each generation spawns Poisson(eta) children per event,
        # displaced by an Exponential(beta) delay, and children past the horizon
        # are dropped. Doing this one sample at a time in Python made the
        # backtest (thousands of forecasts) impractical.
        per_day = mu_per_day * factor                      # immigrants per day
        n_bg_day = rng.poisson(np.tile(per_day, (n, 1)))    # (n, h)
        n_bg = n_bg_day.sum(axis=1)
        owner = np.repeat(np.arange(n), n_bg)
        # Immigrant times land inside the day they were drawn for, so a busy
        # weekday's cascades start on that weekday.
        day_of = np.repeat(np.tile(np.arange(h), n), n_bg_day.ravel())
        times = day_of + rng.uniform(0.0, 1.0, size=owner.size)
        totals = np.bincount(owner, minlength=n).astype(float)

        for _ in range(60):  # generation cap; at eta<1 the cascade dies long before
            if owner.size == 0:
                break
            k = rng.poisson(self.eta, size=owner.size)
            if k.sum() == 0:
                break
            parent = np.repeat(np.arange(owner.size), k)
            child_times = times[parent] + rng.exponential(1.0 / self.beta, size=parent.size)
            child_owner = owner[parent]
            keep = child_times < horizon
            owner, times = child_owner[keep], child_times[keep]
            if owner.size == 0:
                break
            totals += np.bincount(owner, minlength=n)
            # Guard against a near-critical fit blowing up memory.
            if owner.size > max_events * n:
                break
        return totals
