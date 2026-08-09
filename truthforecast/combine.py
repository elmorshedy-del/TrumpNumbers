"""Combining several predictive distributions into the one on the front page.

The site does not lead with a model. It leads with a *rule*: take the models
that scored best, average them, print the result. That rule is the artifact
people actually read, and until now it was the only thing here that was never
scored — the leaderboard graded twenty models and then the page showed a
twenty-first number that appeared on no row of it.

So the combination lives in one place, and both callers use it: the live
forecast, and the walk-forward evaluation that gives that live number its own
CRPS and its own coverage. If the rule is worse than its own best member, that
is a finding, and it should be visible rather than assumed away.

Two smaller things are fixed by having a single implementation:

* **The headline and the threshold table were different distributions.** The
  median and interval came from averaging quantiles; "P(at least 140)" came
  from averaging probabilities. Both are defensible combinations and they are
  not the same one, so the probability printed under an interval was not the
  probability implied by it.
* **Tails were being clipped to 0 and 1.** Interpolating a quantile function
  over nine published levels leaves nothing beyond 2.5% and 97.5%, so any
  threshold past those read as an exact 0% or 100%. A weekly post count has no
  upper bound; a printed 100% is never right, and it was being printed.
"""

from __future__ import annotations

import numpy as np

from .config import COMBINATION_GRID

GRID = np.asarray(COMBINATION_GRID, dtype=float)


def dense_quantiles(samples: np.ndarray) -> np.ndarray:
    """A distribution reduced to a fine quantile grid.

    Fine enough that inverting it back to samples loses nothing that matters,
    and small enough (999 floats) to carry through a backtest row or a JSON
    export without thought.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.zeros(len(GRID))
    return np.quantile(x, GRID)


def vincentize(curves) -> np.ndarray:
    """Average several quantile functions — the combination the site uses.

    Quantile averaging keeps the members' shape and averages their *level*,
    which is the right way round when models mostly disagree about how busy the
    week is rather than about what a week looks like. The alternative, averaging
    densities (`linear_pool`), is wider by construction because it treats
    disagreement as additional uncertainty. Both are scored; neither is
    asserted.
    """
    mat = np.asarray([np.asarray(c, dtype=float) for c in curves], dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros(len(GRID))
    out = mat.mean(axis=0)
    # Averaging can leave a hair of non-monotonicity from floating point; a
    # quantile function that goes backwards would invert to negative density.
    return np.maximum.accumulate(out)


def linear_pool(curves, n: int = 20_000) -> np.ndarray:
    """Mixture of the member distributions, expressed on the same grid."""
    mat = np.asarray([np.asarray(c, dtype=float) for c in curves], dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros(len(GRID))
    pooled = np.concatenate([samples_from(c, n // max(len(mat), 1)) for c in mat])
    return dense_quantiles(pooled)


def samples_from(curve: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Invert a quantile function back into samples.

    Draws u uniformly over the grid's own span rather than over (0, 1): the
    grid stops at 0.1% and 99.9%, and pretending otherwise would extrapolate a
    tail the curve does not contain.
    """
    draw = (rng.uniform if rng is not None else np.random.uniform)(GRID[0], GRID[-1], size=n)
    return np.interp(draw, GRID, np.asarray(curve, dtype=float))


def quantiles_at(curve: np.ndarray, levels) -> dict:
    """Read specific levels off a dense curve, for publication."""
    vals = np.interp(np.asarray(levels, dtype=float), GRID, np.asarray(curve, dtype=float))
    return {str(l): float(v) for l, v in zip(levels, vals)}


def prob_at_least(curve: np.ndarray, threshold: float) -> tuple[float, bool]:
    """P(X >= threshold) from a quantile function, and whether it is a bound.

    Beyond the ends of the grid the honest answer is an inequality, not a
    number: all the curve knows is that the probability is under 0.1% or over
    99.9%. Returning the bound with a flag lets the site say "<0.1%" instead of
    the flat 0% it used to print.
    """
    q = np.asarray(curve, dtype=float)
    t = float(threshold)
    if t <= q[0]:
        return float(1.0 - GRID[0]), True
    if t >= q[-1]:
        return float(1.0 - GRID[-1]), True
    return float(1.0 - np.interp(t, q, GRID)), False
