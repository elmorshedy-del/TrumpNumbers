"""Project-wide configuration.

Everything that is a *choice* rather than a fact lives here, so the choices are
visible and reviewable. The training window in particular deserves that
prominence: which slice of history you model is a bigger lever on the answer
than which model you pick, and it is usually made by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TTF_DATA_DIR", ROOT / "data"))
# Overridable on its own, so a process that only wants the ingest path — the
# notifier, say — can keep a small database of its own instead of sharing the
# full archive and racing whoever else is writing it.
DB_PATH = Path(os.environ.get("TTF_DB_PATH", DATA_DIR / "posts.db"))
EXPORT_DIR = DATA_DIR / "exports"
SITE_DIR = ROOT / "site"

# trumpstruth.org mirrors @realDonaldTrump in full and its robots.txt permits
# crawling ("Disallow:" with an empty value). Truth Social's own API returns 403
# to datacenter IPs, which is why the mirror is the ingest path.
SOURCE_BASE = "https://trumpstruth.org"
USER_AGENT = os.environ.get(
    "TTF_USER_AGENT",
    "trump-truth-forecast/0.1 (open-source posting-volume research; +https://trumpstruth.org)",
)

# The site renders listing timestamps in US Eastern with no timezone label.
# Verified empirically against the RSS feed's explicit UTC pubDate: the offset
# came out at exactly -4h during EDT. tests/test_parse.py pins this.
SITE_TZ = ZoneInfo("America/New_York")

# Trump's local day is the unit that produces the day-of-week structure, so the
# daily series is bucketed in Eastern time rather than UTC.
LOCAL_TZ = ZoneInfo("America/New_York")

# Politeness. The archive backfill is ~400 requests total; there is no reason to
# go faster than this.
REQUEST_DELAY_S = float(os.environ.get("TTF_REQUEST_DELAY", "1.0"))
REQUEST_TIMEOUT_S = 30.0
MAX_RETRIES = 5

ARCHIVE_START = "2022-02-01"  # Trump's first Truth Social posts


@dataclass(frozen=True)
class ModelingWindow:
    """Which slice of history the models are fit on.

    `start` is deliberately a config value and is surfaced on the site. The
    posting process has changed regime at least twice (private citizen ->
    candidate -> president); fitting across a structural break describes no
    regime at all, the way an average of a man's height across his lifetime
    describes him at no age.
    """

    start: str = os.environ.get("TTF_WINDOW_START", "2025-01-20")
    label: str = os.environ.get("TTF_WINDOW_LABEL", "Second term (from inauguration)")


WINDOW = ModelingWindow()

@dataclass(frozen=True)
class CountConvention:
    """What counts as a post, and where a week starts.

    These were hardcoded, and each of them silently decided what the headline
    number means. The default now matches the Kalshi weekly market, because a
    number nobody can compare to the market everyone is quoting is a number
    that invites exactly the wrong comparison — the site read 97 while the
    market's week stood at 164, and both were correct about different things.

    Kalshi resolves on: Sunday 00:00 ET through Saturday 23:59 ET, counting
    Truths, ReTruths and Quote Truths, excluding deleted posts.

    Set TTF_WEEK_ANCHOR=MON / TTF_INCLUDE_RETRUTHS=0 / TTF_INCLUDE_DELETED=1 to
    recover the original convention. It is not wrong — it is a cleaner measure
    of *original authorship*. It just answers a different question.
    """

    # "SUN" -> Sunday..Saturday (Kalshi). "MON" -> Monday..Sunday (original).
    week_anchor: str = os.environ.get("TTF_WEEK_ANCHOR", "SUN").upper()
    include_retruths: bool = os.environ.get("TTF_INCLUDE_RETRUTHS", "1") == "1"
    include_deleted: bool = os.environ.get("TTF_INCLUDE_DELETED", "0") == "1"

    @property
    def resample_rule(self) -> str:
        """Pandas anchor for the week's LAST day."""
        return "W-SAT" if self.week_anchor == "SUN" else "W-SUN"

    @property
    def start_dow(self) -> int:
        """Weekday index the week opens on, in pandas terms (Mon=0 .. Sun=6)."""
        return 6 if self.week_anchor == "SUN" else 0

    @property
    def label(self) -> str:
        span = "Sunday–Saturday" if self.week_anchor == "SUN" else "Monday–Sunday"
        kinds = "all post types" if self.include_retruths else "original posts only"
        dele = "including deleted" if self.include_deleted else "excluding deleted"
        return f"{span}, {kinds}, {dele}"


CONVENTION = CountConvention()

# Retained for callers that want the raw index; prefer CONVENTION.start_dow.
WEEK_START_DOW = CONVENTION.start_dow

# Quantiles carried end to end: models emit them, scoring consumes them, the
# site draws them. Deliberately asymmetric coverage bands (50/80/95).
QUANTILE_LEVELS: tuple[float, ...] = (
    0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975,
)

# The grid the *combination* arithmetic runs on. Averaging quantiles across
# models over the nine published levels alone throws away the shape between
# them, and — worse — leaves nothing outside 2.5%/97.5% to interpolate against,
# so any threshold past the outer published quantile reads as an exact 0% or
# 100%. A week total has no upper bound; a printed 100% is never right.
COMBINATION_GRID: tuple[float, ...] = tuple(
    round(v, 5) for v in [i / 1000 for i in range(1, 1000)]
)

# Threshold events for Brier scoring and the site's "P(at least N)" readouts.
# Resolved against the empirical weekly distribution at runtime.
THRESHOLD_PERCENTILES: tuple[float, ...] = (0.50, 0.75, 0.90)

N_PREDICTIVE_SAMPLES = 20_000


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward protocol.

    Fixed before any scoring happens. The model list is pre-registered in
    `models.registry` for the same reason: a leaderboard that only shows the
    winner is a bullseye painted after the shot.
    """

    min_train_weeks: int = 26
    embargo_days: int = 1  # keeps rolling features from straddling the seam
    holdout_weeks: int = 8  # untouched until the very end

    # The cuts the live site actually faces, and only those.
    #
    # -1 is Monday: the week has produced nothing yet, which is the hardest
    # forecast of the seven and the one on display for a seventh of all time.
    # Scoring it is not optional — a leaderboard that omits the hardest case is
    # not measuring the thing being shown.
    #
    # 6 is excluded: by the end of Sunday the week is fully observed, so every
    # model returns the answer exactly. That row scores 0 CRPS, covers its own
    # interval by construction, reports zero width, and contributes a uniform
    # random PIT value. Averaged in, it flatters coverage (a poisson-glm at a
    # true 0.29 reads as 0.39), shrinks reported interval widths by a seventh,
    # dilutes CRPS by a seventh, and weakens the calibration test — all of it
    # uniformly across models, so it never shows up as a ranking anomaly.
    cut_days: tuple[int, ...] = field(default=(-1, 0, 1, 2, 3, 4, 5))

    # How many past weeks the live selection rule is allowed to look at before
    # it is willing to pick. Scored walk-forward as `headline-top3`, so the
    # number on the front page has a row on its own leaderboard.
    headline_top_k: int = 3
    headline_min_weeks: int = 12

    # Every forecast draws from a stream seeded from (model, week, cut), so a
    # re-run on unchanged data reproduces the leaderboard exactly. Without it
    # the top twelve models — which span 0.5 CRPS over 52 weeks — reorder
    # between runs from Monte-Carlo noise alone, and the ordering looks like a
    # finding. Change the value to measure that noise rather than inherit it.
    seed: int = int(os.environ.get("TTF_SEED", "20260101"))


BACKTEST = BacktestConfig()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
