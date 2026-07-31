"""Proper scoring rules.

A scoring rule is *proper* when the best expected score comes from reporting
what you actually believe. That is a mathematical property, not a stylistic
preference, and it matters here for a concrete reason: grade forecasts the wrong
way and you do not merely measure badly, you teach the forecaster to lie. A
desert forecaster who says "0% rain" every day is right 98% of the time and has
told you nothing.

Everything below is proper. Accuracy, hit rate, and "was the point forecast
close" are not, and are deliberately absent.
"""

from __future__ import annotations

import numpy as np


def crps_from_samples(samples: np.ndarray, observed: float) -> float:
    """Continuous Ranked Probability Score — the primary metric.

    Compares an entire predictive distribution against the single value that
    happened. Uses the energy form

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the forecast. The first term rewards
    being close; the second penalizes being needlessly vague. Lower is better,
    and it is in the units of the forecast (posts), so it is directly readable.

    The second term is computed in O(n log n) from the sorted samples rather
    than by forming the O(n^2) pairwise matrix.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")

    term1 = np.mean(np.abs(x - observed))
    # E|X - X'| = (2 / n^2) * sum_i (2i - n + 1) * x_(i)   for sorted x
    i = np.arange(n)
    term2 = 2.0 * np.sum((2 * i - n + 1) * x) / (n * n)
    return float(term1 - 0.5 * term2)


def log_score(samples: np.ndarray, observed: float, bandwidth: float | None = None) -> float:
    """Negative log predictive density, estimated by a Gaussian kernel.

    Harsher than CRPS on confident errors: assign near-zero density to something
    that then happens and it hurts enormously. This is the same quantity as
    likelihood, so maximum-likelihood fitting is already scoring itself
    properly — a tidy unification.

    Reported as a *penalized* score: a forecast that puts no mass anywhere near
    the outcome is floored rather than allowed to return infinity, which would
    make the leaderboard mean undefined.
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan")
    sd = float(np.std(x))
    if bandwidth is None:
        # Silverman's rule, with a floor so a degenerate forecast is merely bad
        # rather than infinitely bad.
        bandwidth = max(1.06 * sd * n ** (-1 / 5), 0.5)
    z = (observed - x) / bandwidth
    dens = np.mean(np.exp(-0.5 * z**2)) / (bandwidth * np.sqrt(2 * np.pi))
    return float(-np.log(max(dens, 1e-12)))


def pinball_loss(samples: np.ndarray, observed: float, quantiles) -> dict:
    """Quantile (pinball) loss — what quantile regression optimizes."""
    qs = np.quantile(samples, list(quantiles))
    out = {}
    for level, q in zip(quantiles, qs):
        diff = observed - q
        out[str(level)] = float(max(level * diff, (level - 1) * diff))
    out["mean"] = float(np.mean([v for k, v in out.items() if k != "mean"]))
    return out


def brier_score(samples: np.ndarray, observed: float, threshold: float) -> float:
    """Brier score for the binary event "week total >= threshold"."""
    p = float((np.asarray(samples) >= threshold).mean())
    y = 1.0 if observed >= threshold else 0.0
    return float((p - y) ** 2)


def pit_value(samples: np.ndarray, observed: float) -> float:
    """Probability Integral Transform: which percentile did the outcome land in?

    This is the calibration check. If the forecasts are honest, PIT values
    across many forecasts should be spread evenly between 0 and 1 — sometimes
    low, sometimes high, no pattern. Piling up at the extremes means the
    intervals are too narrow and the forecaster is kidding itself; piling up in
    the middle means they are too wide.

    Randomized in the ties, because a discrete count forecast otherwise produces
    a lumpy PIT even when perfectly calibrated.
    """
    x = np.asarray(samples, dtype=float)
    below = float((x < observed).mean())
    equal = float((x == observed).mean())
    return float(below + np.random.uniform() * equal)


def interval_width(samples: np.ndarray, coverage: float = 0.80) -> float:
    """Sharpness. Only meaningful once calibration has been checked."""
    a = (1 - coverage) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return float(hi - lo)


def interval_covers(samples: np.ndarray, observed: float, coverage: float = 0.80) -> bool:
    a = (1 - coverage) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return bool(lo <= observed <= hi)


def score_forecast(samples: np.ndarray, observed: float, quantiles, thresholds) -> dict:
    """Every rule at once, for one forecast."""
    return {
        "crps": crps_from_samples(samples, observed),
        "log_score": log_score(samples, observed),
        "pinball": pinball_loss(samples, observed, quantiles)["mean"],
        "pit": pit_value(samples, observed),
        "width80": interval_width(samples, 0.80),
        "covered80": interval_covers(samples, observed, 0.80),
        "covered95": interval_covers(samples, observed, 0.95),
        "median": float(np.median(samples)),
        "brier": {str(t): brier_score(samples, observed, t) for t in thresholds},
    }


def pit_uniformity(pits: np.ndarray, bins: int = 10) -> dict:
    """Chi-square test that the PIT values are uniform, plus the histogram."""
    from scipy import stats

    p = np.asarray(pits, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) < 10:
        return {"n": int(len(p)), "histogram": [], "p_value": None}

    counts, _ = np.histogram(p, bins=bins, range=(0, 1))
    expected = len(p) / bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    pval = float(stats.chi2.sf(chi2, df=bins - 1))
    return {
        "n": int(len(p)),
        "histogram": [int(c) for c in counts],
        "expected_per_bin": round(expected, 1),
        "chi2": round(chi2, 2),
        "p_value": pval,
        # A low p-value means the forecasts are NOT calibrated.
        "calibrated": bool(pval > 0.05),
    }


def skill_score(model_crps: float, reference_crps: float) -> float:
    """Fraction of the reference model's CRPS removed. Positive = better.

    Raw CRPS is unreadable on its own; against climatology it becomes "this
    model removes 12% of the error the do-nothing model makes".
    """
    if not reference_crps:
        return float("nan")
    return float(1.0 - model_crps / reference_crps)
