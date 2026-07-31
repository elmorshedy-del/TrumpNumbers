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

# Week convention for the headline forecast: Monday 00:00 -> Sunday 23:59 local.
WEEK_START_DOW = 0  # Monday, matching pandas' Monday=0

# Quantiles carried end to end: models emit them, scoring consumes them, the
# site draws them. Deliberately asymmetric coverage bands (50/80/95).
QUANTILE_LEVELS: tuple[float, ...] = (
    0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975,
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


BACKTEST = BacktestConfig()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
