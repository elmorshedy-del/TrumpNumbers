"""The pre-registered model list.

Fixed here, before any scoring runs, and the leaderboard reports every entry.
That is the defence against the quiet version of the multiple-comparisons
problem: not testing sixty things and admitting it, but testing sixty things
without noticing — every abandoned variant, every "let me try it the other way"
is an unrecorded test, and reporting only the survivor paints the target around
the bullet hole.

If you add a model, add it here and re-run the whole backtest. Do not add a
model, look at its score, and then decide whether to keep it.
"""

from __future__ import annotations

from .baselines import (
    BlockBootstrapModel,
    EmpiricalDayModel,
    IndependentDayBootstrapModel,
    LastWeekModel,
    RecentEmpiricalDayModel,
    TrailingMedianModel,
)
from .counts import (
    HierarchicalDOWModel,
    NegBinGLM,
    NegBinTrendGLM,
    PoissonGLM,
    ZeroInflatedNegBinModel,
    ZeroInflatedPoissonModel,
)
from .dynamics import HawkesModel, INGARCHModel, LogSarimaModel, NegBinHMM, PoissonHMM
from .ml import LinearPoolEnsemble, QuantileGBM, VincentizedEnsemble

# The reference model for skill scores. Climatology: no trend, no memory, no
# distributional assumption. "CRPS 31" is meaningless until you know this one.
REFERENCE_MODEL = "empirical-dow"


def build_base_models() -> list:
    """Every model except the ensembles, which are built from these."""
    return [
        EmpiricalDayModel(),
        RecentEmpiricalDayModel(),
        LastWeekModel(),
        TrailingMedianModel(),
        BlockBootstrapModel(),
        IndependentDayBootstrapModel(),
        PoissonGLM(),
        NegBinGLM(),
        NegBinTrendGLM(),
        ZeroInflatedPoissonModel(),
        ZeroInflatedNegBinModel(),
        HierarchicalDOWModel(),
        INGARCHModel(),
        LogSarimaModel(),
        PoissonHMM(),
        NegBinHMM(),
        HawkesModel(),
        QuantileGBM(),
    ]


ENSEMBLE_MEMBERS = ("negbin-glm", "hier-dow-eb", "block-bootstrap", "empirical-dow")


def build_all_models() -> list:
    """Base models plus the two ensembles built over a fixed member set."""
    base = build_base_models()
    by_name = {m.name: m for m in base}

    def members():
        # Fresh instances so ensembles do not share fitted state with the
        # standalone entries they are scored against.
        lookup = {m.name: m for m in build_base_models()}
        return [lookup[n] for n in ENSEMBLE_MEMBERS if n in lookup]

    return base + [LinearPoolEnsemble(members()), VincentizedEnsemble(members())]


def model_catalog() -> list[dict]:
    """Metadata for the site's model page."""
    return [m.meta() for m in build_all_models()]
