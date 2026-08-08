"""Count regression models: Poisson -> Negative Binomial -> zero-inflated.

The progression is the argument. Poisson is included knowing it will lose,
because *how badly* it loses is the finding: it has one parameter doing two
jobs, since a fixed rate with independent events forces variance == mean. When
that fails, either the rate is not fixed (Negative Binomial: a hidden daily
"mood" is redrawn each day) or the zeros are of two different kinds
(zero-inflation: a switch fires before the count is even drawn).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import DailyModel, dow_matrix, safe_positive

warnings.filterwarnings("ignore")


def _design(index: pd.DatetimeIndex, with_trend: bool, t0: pd.Timestamp | None = None):
    X = dow_matrix(index)
    if with_trend:
        t0 = t0 or index[0]
        weeks = ((index - t0).days / 7.0).to_numpy().reshape(-1, 1)
        X = np.hstack([X, weeks])
    return X


class PoissonGLM(DailyModel):
    """Day-of-week Poisson. The control that shows why the others exist."""

    name = "poisson-glm"
    family = "count-glm"
    description = "Log-linear Poisson regression on day-of-week."
    assumptions = "Fixed rate per weekday; posts independent; variance == mean."
    fails_when = "Always here — the data is wildly overdispersed. Kept as the reference failure."

    with_trend = False
    zero_inflated = False

    def __init__(self, with_trend: bool = False):
        self.with_trend = with_trend
        self.res = None
        self.t0 = None
        self.fallback_mu = 1.0

    def _fit_model(self, y, X):
        import statsmodels.api as sm

        return sm.GLM(y, X, family=sm.families.Poisson()).fit()

    def fit(self, history: pd.Series) -> "PoissonGLM":
        y = history.to_numpy(dtype=float)
        self.t0 = history.index[0]
        X = _design(history.index, self.with_trend, self.t0)
        self.fallback_mu = float(y.mean()) if len(y) else 1.0
        try:
            self.res = self._fit_model(y, X)
        except Exception:
            self.res = None
        self.last_index = history.index[-1]
        return self

    def _design_row(self, dows: np.ndarray, dates=None) -> np.ndarray:
        X = np.zeros((len(dows), 7))
        X[np.arange(len(dows)), dows] = 1.0
        if self.with_trend:
            # Evaluated at each forecast day's own position on the trend line.
            # A single "one week past the end of history" value for every
            # remaining day was both wrong for the day and wrong by the gap:
            # at a Friday cut the last remaining day is nine days past the end
            # of the backtest's history, not seven.
            if dates is not None and len(dates) == len(dows):
                weeks = np.array([(d - self.t0).days / 7.0 for d in dates], dtype=float)
            else:
                weeks = np.full(len(dows), ((self.last_index - self.t0).days / 7.0) + 1.0)
            X = np.hstack([X, weeks.reshape(-1, 1)])
        return X

    def _mu(self, dows: np.ndarray, dates=None) -> np.ndarray:
        if self.res is None:
            return np.full(len(dows), self.fallback_mu)
        try:
            return safe_positive(self.res.predict(self._design_row(dows, dates)))
        except Exception:
            return np.full(len(dows), self.fallback_mu)

    def sample_days(self, dows: np.ndarray, n: int, dates=None) -> np.ndarray:
        mu = self._mu(dows, dates)
        return np.random.poisson(mu, size=(n, len(dows))).astype(float)


class NegBinGLM(PoissonGLM):
    """Negative Binomial: a two-stage lottery.

    Draw today's rate from a Gamma (some days quiet, some manic), then draw
    today's count from a Poisson at that rate. Variance becomes
    mu + mu^2/k, so the model can hold both 0-post days and 168-post days.

    Note what has actually been claimed: that there is a hidden daily state,
    never directly observed, that swings hard. That is a theory about a person,
    not a technical patch — the HMM in `regime.py` takes the same idea and gives
    the state memory.
    """

    name = "negbin-glm"
    family = "count-glm"
    description = "Negative Binomial (NB2) regression on day-of-week."
    assumptions = "Rate varies day to day around a weekday mean; days independent."
    fails_when = "Bursts run across consecutive days — it has no memory."

    def __init__(self, with_trend: bool = False):
        super().__init__(with_trend=with_trend)
        self.alpha = 1.0

    def _fit_model(self, y, X):
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import NegativeBinomial

        try:
            res = NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=200)
            self.alpha = float(max(res.params[-1], 1e-6))
            # The alpha column is not part of the mean model; refit the mean at
            # the estimated dispersion so `predict` returns mu, not something
            # that includes the dispersion parameter.
            return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        except Exception:
            mean, var = float(np.mean(y)), float(np.var(y))
            self.alpha = float(max((var - mean) / max(mean**2, 1e-9), 1e-6))
            return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()

    def sample_days(self, dows: np.ndarray, n: int, dates=None) -> np.ndarray:
        mu = self._mu(dows, dates)
        r = 1.0 / max(self.alpha, 1e-9)          # NB2: var = mu + alpha*mu^2
        p = r / (r + mu)
        return np.random.negative_binomial(r, p, size=(n, len(dows))).astype(float)


class NegBinTrendGLM(NegBinGLM):
    name = "negbin-glm-trend"
    family = "count-glm"
    description = "Negative Binomial on day-of-week plus a linear weekly trend."
    assumptions = "As negbin-glm, plus a steady drift in level."
    fails_when = "The trend is extrapolated past where it holds."

    def __init__(self):
        super().__init__(with_trend=True)


class ZeroInflatedPoissonModel(PoissonGLM):
    """Two kinds of zero: a quiet day, versus a day posting was never on.

    Same number on the page, different mechanism — a courtroom or a long flight
    is not a low draw from the usual process, it is a different process.
    """

    name = "zip-glm"
    family = "count-glm"
    description = "Zero-inflated Poisson: a structural-zero switch, then a count."
    assumptions = "A fraction of days are 'off' entirely; the rest are Poisson."
    fails_when = "Overdispersion comes from bursts rather than from zeros."

    def __init__(self):
        super().__init__(with_trend=False)
        self.inflate = 0.0

    def _fit_model(self, y, X):
        from statsmodels.discrete.count_model import ZeroInflatedPoisson

        res = ZeroInflatedPoisson(y, X, inflation="logit").fit(disp=0, maxiter=200)
        self._zi_res = res
        return res

    def _components(self, dows: np.ndarray, dates=None) -> tuple[np.ndarray, np.ndarray]:
        """(rate of the count component, probability the switch is ON).

        Both halves are needed to *sample* a zero-inflated model, and asking for
        `which="mean"` gets neither: that returns the marginal mean
        (1 - pi) * mu, which drawn through a single Poisson is an ordinary
        Poisson at a deflated rate — narrower than the plain Poisson this model
        exists to widen, and with no structural zeros in it at all. The model
        was scoring identically to `poisson-glm` to two decimal places because
        it *was* `poisson-glm`.
        """
        if self.res is None:
            return np.full(len(dows), self.fallback_mu), np.ones(len(dows))
        X = self._design_row(dows, dates)
        infl = np.ones((len(dows), 1))
        try:
            mu = safe_positive(self.res.predict(X, exog_infl=infl, which="mean-main"))
            p_main = np.clip(self.res.predict(X, exog_infl=infl, which="prob-main"), 0.0, 1.0)
            return mu, p_main
        except Exception:
            try:  # older statsmodels: recover pi from the inflation parameters
                mu = safe_positive(self.res.predict(X, exog_infl=infl, which="mean"))
                logit = float(np.asarray(self.res.params)[0])
                pi = 1.0 / (1.0 + np.exp(-logit))
                return mu / max(1.0 - pi, 1e-6), np.full(len(dows), 1.0 - pi)
            except Exception:
                return np.full(len(dows), self.fallback_mu), np.ones(len(dows))

    def _mu(self, dows: np.ndarray, dates=None) -> np.ndarray:
        """Marginal mean, for callers that want a point rate."""
        mu, p_main = self._components(dows, dates)
        return safe_positive(mu * p_main)

    def _count_draw(self, mu: np.ndarray, n: int) -> np.ndarray:
        return np.random.poisson(mu, size=(n, len(mu))).astype(float)

    def sample_days(self, dows: np.ndarray, n: int, dates=None) -> np.ndarray:
        mu, p_main = self._components(dows, dates)
        on = np.random.random((n, len(dows))) < p_main
        return self._count_draw(mu, n) * on


class ZeroInflatedNegBinModel(ZeroInflatedPoissonModel):
    """Both mechanisms at once: a switch that can turn the day off, and a rate
    that wanders when it is on.

    It used to be neither. The fit ran, `alpha` was read off it, and then the
    whole zero-inflated result was thrown away and replaced with a plain
    Negative Binomial GLM refitted at that alpha — so `zinb-glm` was
    `negbin-glm` under another name, which is exactly what its score said
    (18.12 against 18.10 over 52 weeks). The inflation is now carried into the
    sampler, where it belongs.
    """

    name = "zinb-glm"
    family = "count-glm"
    description = "Zero-inflated Negative Binomial: structural zeros plus overdispersion."
    assumptions = "Both mechanisms operate — off-days and a wandering rate."
    fails_when = "Thin data; the two mechanisms compete to explain the same zeros."

    def __init__(self):
        super().__init__()
        self.alpha = 1.0

    def _fit_model(self, y, X):
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

        res = ZeroInflatedNegativeBinomialP(y, X, inflation="logit").fit(disp=0, maxiter=200)
        # Parameter order is [inflation..., exog..., alpha].
        self.alpha = float(np.clip(np.asarray(res.params)[-1], 1e-6, 50.0))
        return res

    def fit(self, history: pd.Series) -> "ZeroInflatedNegBinModel":
        super().fit(history)
        if self.res is None:  # the ZI fit failed; fall back to moment-matched NB2
            y = history.to_numpy(dtype=float)
            mean, var = float(np.mean(y)) if len(y) else 1.0, float(np.var(y)) if len(y) else 1.0
            self.alpha = float(max((var - mean) / max(mean**2, 1e-9), 1e-4))
        return self

    def _count_draw(self, mu: np.ndarray, n: int) -> np.ndarray:
        r = 1.0 / max(self.alpha, 1e-9)
        p = r / (r + mu)
        return np.random.negative_binomial(r, p, size=(n, len(mu))).astype(float)


class HierarchicalDOWModel(DailyModel):
    """Empirical-Bayes day-of-week rates — all seven days borrow strength.

    A weekday with few observations and a wild mean should not be believed at
    face value, the way a restaurant with three five-star reviews is not the
    best in the city. Each weekday's rate is shrunk toward the overall rate by
    an amount set by how much the weekdays genuinely differ relative to their
    own noise.

    This is not merely prudent: with three or more related quantities to
    estimate, shrinking every one of them toward the group mean provably beats
    estimating each on its own (James-Stein). With ~7 weekday cells and heavy
    variance, it is close to the difference between an estimate and noise.

    The predictive distribution is the Gamma-Poisson mixture, i.e. Negative
    Binomial, so the posterior uncertainty in the rate is carried through into
    the forecast instead of being dropped.
    """

    name = "hier-dow-eb"
    family = "shrinkage"
    description = "Empirical-Bayes Gamma-Poisson day-of-week rates with shrinkage."
    assumptions = "Weekday rates are draws from a common population."
    fails_when = "A weekday is genuinely unlike the others; shrinkage over-corrects."

    def __init__(self):
        self.mu = np.ones(7)
        self.alpha = np.full(7, 0.5)

    def fit(self, history: pd.Series) -> "HierarchicalDOWModel":
        y = history.to_numpy(dtype=float)
        dows = history.index.dayofweek.to_numpy()
        grand = float(y.mean()) if len(y) else 1.0

        # Per-weekday mean and overdispersion, by method of moments.
        # Both are shrunk, and BOTH matter. Shrinking only the mean was the
        # first version of this model and it was wrong: posterior uncertainty in
        # a weekday's rate is tiny once you have ~80 observations of it, so the
        # predictive collapsed to a Poisson and inherited exactly the
        # variance == mean straitjacket this whole family exists to escape.
        # The day-to-day spread within a weekday is the dominant uncertainty and
        # has to be modelled, not just the uncertainty about its average.
        raw_mu = np.array([y[dows == d].mean() if (dows == d).any() else grand for d in range(7)])
        raw_var = np.array([
            y[dows == d].var(ddof=1) if (dows == d).sum() > 1 else float(y.var(ddof=1) or 1.0)
            for d in range(7)
        ])
        n_d = np.array([max(int((dows == d).sum()), 1) for d in range(7)])

        # James-Stein style shrinkage weight for the weekday means: trust a
        # weekday's own average in proportion to how much the weekdays genuinely
        # differ, relative to the noise in each weekday's estimate.
        between = float(raw_mu.var(ddof=1)) if len(raw_mu) > 1 else 0.0
        within_se2 = float(np.mean(raw_var / n_d))
        true_between = max(between - within_se2, 1e-9)
        w = true_between / (true_between + within_se2)  # 1 = trust the day, 0 = trust the pool
        self.mu = np.maximum(w * raw_mu + (1 - w) * grand, 1e-6)

        # NB2 dispersion per weekday, shrunk toward the pooled value.
        global_var = float(y.var(ddof=1)) if len(y) > 1 else grand
        global_alpha = max((global_var - grand) / max(grand**2, 1e-9), 1e-4)
        raw_alpha = np.array([
            max((raw_var[d] - raw_mu[d]) / max(raw_mu[d] ** 2, 1e-9), 1e-4) for d in range(7)
        ])
        wa = n_d / (n_d + 30.0)  # a weekday needs plenty of data to move its own dispersion
        self.alpha = np.clip(wa * raw_alpha + (1 - wa) * global_alpha, 1e-4, 20.0)
        self.shrinkage_weight = float(w)
        return self

    def sample_days(self, dows: np.ndarray, n: int, dates=None) -> np.ndarray:
        out = np.empty((n, len(dows)))
        for j, d in enumerate(dows):
            d = int(d)
            mu, alpha = self.mu[d], self.alpha[d]
            r = 1.0 / max(alpha, 1e-9)
            out[:, j] = np.random.negative_binomial(r, r / (r + mu), size=n)
        return out
