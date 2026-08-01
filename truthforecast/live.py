"""The prospective record: what was forecast, before the week closed.

Everything else in this project is retrospective. The models are graded on
history by code written after that history happened, and no amount of care about
embargoes and expanding origins fully answers the objection — the *choices*
(which models, which window, which convention) were all made with the whole
series visible. A backtest is an upper bound on live performance and cannot be
anything else.

This is the part that can be. Every time the pipeline builds a forecast it
appends one immutable line here: the week, the cut, the moment, and the
distribution. When the week closes, the same lines are scored against what
actually happened. Nothing is ever rewritten — a corrected count changes the
outcome a forecast is graded against, never the forecast.

After a year that is a forecasting record nobody can accuse of hindsight, and
it is the only version of this project whose calibration claim cannot be
doubted. Until then it says how many weeks it has, and declines to draw
conclusions from four.

The file is deliberately absent from the repository until a deployment writes
it. A register seeded with entries made during development — by code that was
still being changed, against a database being rebuilt — would be a record of
nothing, and its first rows would be exactly the ones nobody could check. It
starts empty, on the first scheduled run, and only grows.

    python -m truthforecast.live --score     # score whatever has closed
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.scoring import crps_from_samples, interval_covers, pit_value
from .combine import GRID, samples_from
from .config import DATA_DIR, EXPORT_DIR, QUANTILE_LEVELS, ensure_dirs

log = logging.getLogger(__name__)

LOG_PATH = Path(DATA_DIR) / "live_forecasts.jsonl"

# One line per (week, cut, model-set) — a later forecast at the same cut
# supersedes an earlier one for scoring, because it is the one that was on the
# page longest. Both stay in the file: the record is append-only.
SCHEMA_VERSION = 1


def record(forecast: dict, headline: dict, curve=None, path: Path | None = None) -> bool:
    """Append one live forecast. Returns True if a line was written.

    One line per (week, cut), and it is the FIRST forecast made at that cut that
    stands. That is deliberately the hardest and least gameable version: the
    forecast for a Wednesday is the one made as Wednesday began, not the one
    refined at eleven at night when the day is nearly decided. A register you
    are allowed to keep amending until the answer arrives is not a register.

    Safe to call on every pipeline pass — it becomes a no-op once the cut has
    been recorded, so the file changes about once a day.
    """
    if not headline or not headline.get("median"):
        return False
    week = forecast.get("week") or {}
    path = path or LOG_PATH
    ensure_dirs()
    key = (week.get("week_start"), week.get("days_into_week"))
    if key[0] is None or key[1] is None:
        return False
    if any((e.get("week_start"), e.get("cut")) == key for e in load(path)):
        return False
    entry = {
        "v": SCHEMA_VERSION,
        "made_at": datetime.now(timezone.utc).isoformat(),
        "week_start": week.get("week_start"),
        "week_end": week.get("week_end"),
        "cut": week.get("days_into_week"),
        "convention": week.get("convention"),
        "observed_so_far": week.get("week_to_date_including_today"),
        "models_used": headline.get("models_used", []),
        "median": headline.get("median"),
        "quantiles": headline.get("quantiles", {}),
        # The dense curve, so the record can be scored with a proper rule later
        # rather than only checked against its own interval. Rounded to keep the
        # file readable; a tenth of a post is far below anything that matters.
        "curve": [round(float(v), 2) for v in curve] if curve is not None else None,
    }
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return True


def load(path: Path | None = None) -> list[dict]:
    path = path or LOG_PATH
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping unreadable line in %s", path)
    return out


def score(weekly_actual: pd.Series, path: Path | None = None) -> dict:
    """Grade every logged forecast whose week has since closed.

    `weekly_actual` is indexed by the week's LAST day, which is what
    `CountSeries.weekly` produces. A week is scored only once it is complete in
    the series; the week in flight is left alone.
    """
    entries = load(path)
    if not entries:
        return {"available": False, "n_forecasts": 0, "note": "no live forecasts logged yet"}

    closed = {pd.Timestamp(d).date(): float(v) for d, v in weekly_actual.items()}

    # One forecast per (week, cut): the FIRST one made, which is the one this
    # project committed to. A file that somehow holds several — an old log, a
    # concurrent writer — is graded on the earliest, never the luckiest.
    latest: dict[tuple, dict] = {}
    for e in entries:
        if e.get("week_start") is None or e.get("cut") is None:
            continue
        key = (e["week_start"], int(e["cut"]))
        prev = latest.get(key)
        if prev is None or e.get("made_at", "") < prev.get("made_at", ""):
            latest[key] = e

    rows = []
    for (week_start, cut), e in sorted(latest.items()):
        end = pd.Timestamp(e["week_end"]).date() if e.get("week_end") else None
        if end not in closed:
            continue
        actual = closed[end]
        curve = e.get("curve")
        if curve:
            samples = samples_from(np.asarray(curve, dtype=float), 20_000)
        elif e.get("quantiles"):
            levels = sorted(float(k) for k in e["quantiles"])
            vals = [e["quantiles"][str(l)] for l in levels]
            samples = np.interp(np.random.uniform(levels[0], levels[-1], 20_000), levels, vals)
        else:
            continue
        rows.append(
            {
                "week_start": week_start,
                "week_end": e["week_end"],
                "cut": cut,
                "made_at": e.get("made_at"),
                "actual": actual,
                "median": e.get("median"),
                "crps": crps_from_samples(samples, actual),
                "covered80": interval_covers(samples, actual, 0.80),
                "covered95": interval_covers(samples, actual, 0.95),
                "pit": pit_value(samples, actual),
                "error": float(actual - (e.get("median") or 0.0)),
            }
        )

    if not rows:
        return {
            "available": False,
            "n_forecasts": len(latest),
            "n_scored": 0,
            "note": "forecasts are logged but no logged week has closed yet",
        }

    df = pd.DataFrame(rows)
    n_weeks = int(df["week_start"].nunique())
    by_cut = (
        df.groupby("cut")
        .agg(n=("crps", "size"), crps=("crps", "mean"), coverage80=("covered80", "mean"))
        .round(3)
    )
    return {
        "available": True,
        "n_forecasts": int(len(df)),
        "n_weeks": n_weeks,
        "first_week": str(df["week_start"].min()),
        "last_week": str(df["week_start"].max()),
        "crps": round(float(df["crps"].mean()), 2),
        "coverage80": round(float(df["covered80"].mean()), 3),
        "coverage95": round(float(df["covered95"].mean()), 3),
        "median_abs_error": round(float(df["error"].abs().median()), 1),
        "by_cut": {
            str(int(c)): {k: (None if pd.isna(v) else float(v)) for k, v in r.items()}
            for c, r in by_cut.iterrows()
        },
        # A prospective record is worth exactly as much as its length. Saying so
        # is the point: the alternative is a confident calibration claim built
        # on six weeks, which is the failure this whole file exists to avoid.
        "enough_to_conclude": bool(n_weeks >= 30),
        "note": (
            f"Scored prospectively: {n_weeks} week(s) forecast before the fact and graded after. "
            + (
                "Long enough to read the calibration."
                if n_weeks >= 30
                else "Too short to conclude anything — this fills in as the deployment runs."
            )
        ),
    }


def main(argv=None) -> int:
    import argparse

    from .series import load_frame, modeling_series

    p = argparse.ArgumentParser(description="Prospective live-forecast record")
    p.add_argument("--score", action="store_true", help="score the closed weeks and print")
    p.add_argument("--list", action="store_true", help="print the raw log")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.list:
        for e in load():
            print(json.dumps({k: v for k, v in e.items() if k != "curve"}))
        return 0

    series = modeling_series(load_frame())
    print(json.dumps(score(series.complete_weeks()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
