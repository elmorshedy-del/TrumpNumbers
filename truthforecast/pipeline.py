"""Orchestration: ingest -> series -> diagnostics -> forecast -> JSON for the site.

Two tiers, because they cost very different amounts:

* `refresh_forecast` is cheap (seconds). The daemon runs it every few minutes so
  the live numbers move as posts arrive.
* `refresh_backtest` is expensive (minutes). It runs nightly. Its output is what
  ranks the models, and that ranking is what the headline forecast leans on.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import diagnostics as diag
from .config import EXPORT_DIR, WINDOW, ensure_dirs
from .forecast import current_week_forecast, headline, threshold_probabilities
from .ingest.run import backfill, poll
from .models.registry import model_catalog
from .series import daily_counts, load_frame, modeling_series

log = logging.getLogger(__name__)


def _write(name: str, payload: dict) -> Path:
    ensure_dirs()
    path = EXPORT_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s", path)
    return path


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def load_ranking() -> list[str]:
    """Model order from the last backtest, best CRPS first."""
    path = EXPORT_DIR / "backtest.json"
    if not path.exists():
        return []
    try:
        rep = json.loads(path.read_text())
        board = rep.get("leaderboard_holdout") or rep.get("leaderboard") or []
        return [r["model"] for r in board]
    except Exception:
        return []


def refresh_forecast(df=None) -> dict:
    """Recompute the live numbers and rewrite the site's JSON."""
    df = load_frame() if df is None else df
    series = modeling_series(df)
    if series.daily.empty:
        raise RuntimeError("no data — run a backfill first")

    events = df[df["is_retruth"] == 0]
    fc = current_week_forecast(series, events=events)
    ranking = load_ranking()

    thresholds = []
    weekly = series.complete_weeks()
    if len(weekly) >= 10:
        thresholds = [float(np.quantile(weekly, p)) for p in (0.5, 0.75, 0.9)]

    head = headline(fc, ranking=ranking)
    probs = threshold_probabilities(fc, thresholds, ranking=ranking) if thresholds else []

    recent = series.daily.tail(70)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": WINDOW.start, "label": WINDOW.label},
        "headline": head,
        "week": fc["week"],
        "threshold_probabilities": probs,
        "per_model": fc["per_model"],
        "ranking": ranking,
        "daily_recent": [
            {"date": d.date().isoformat(), "count": int(v), "dow": int(d.dayofweek)}
            for d, v in recent.items()
        ],
        "weekly_history": [
            {"week_ending": d.date().isoformat(), "total": int(v)}
            for d, v in series.complete_weeks().tail(60).items()
        ],
    }
    _write("forecast.json", _json_safe(payload))
    return payload


def refresh_diagnostics(df=None) -> dict:
    df = load_frame() if df is None else df
    series = modeling_series(df)
    payload = diag.run_all(df, series)

    retruths = daily_counts(df, "retruths", start=WINDOW.start)
    payload["retruths"] = {
        "total": int(retruths.daily.sum()),
        "share_of_all_posts": round(
            float(retruths.daily.sum()) / max(float(series.daily.sum() + retruths.daily.sum()), 1), 3
        ),
        "note": (
            "ReTruths are excluded from the modeled series. The mirror stores them with the "
            "ORIGINAL author's timestamp, not the moment Trump reshared, so counting them would "
            "place activity on days he did nothing."
        ),
    }
    payload["source"] = {
        "name": "trumpstruth.org",
        "account": "@realDonaldTrump on Truth Social",
        "note": "Truth Social's own API refuses datacenter IPs; this mirror is the ingest path.",
    }
    _write("diagnostics.json", _json_safe(payload))
    return payload


def refresh_posts(df=None, limit: int = 60) -> dict:
    """The most recent posts, for the live feed on the site."""
    df = load_frame() if df is None else df
    recent = df.sort_values("created_utc", ascending=False).head(limit)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posts": [
            {
                "id": int(r.trumpstruth_id),
                "at": r.local.isoformat(),
                "text": (r.text or "")[:600],
                "is_retruth": bool(r.is_retruth),
                "source_handle": r.source_handle,
                "has_media": bool(r.has_media),
                "url": r.truth_url or f"https://trumpstruth.org/statuses/{int(r.trumpstruth_id)}",
            }
            for r in recent.itertuples()
        ],
    }
    _write("posts.json", _json_safe(payload))
    return payload


def refresh_backtest(df=None, n_samples: int = 4000, max_weeks: int | None = None) -> dict:
    """The expensive one. Re-ranks every model. Run nightly."""
    from .backtest import build_report, run_backtest

    df = load_frame() if df is None else df
    series = modeling_series(df)
    result = run_backtest(
        series.daily, events=df[df["is_retruth"] == 0],
        n_samples=n_samples, max_weeks=max_weeks,
    )
    report = build_report(result)
    report["catalog"] = model_catalog()
    _write("backtest.json", _json_safe(report))
    return report


def refresh_all(include_backtest: bool = False) -> None:
    df = load_frame()
    refresh_diagnostics(df)
    refresh_posts(df)
    if include_backtest:
        refresh_backtest(df)
    refresh_forecast(df)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Trump Truth Social forecast pipeline")
    p.add_argument("--backfill", metavar="YYYY-MM-DD", nargs="?", const="2022-02-01",
                   help="walk the archive back to this date")
    p.add_argument("--poll", action="store_true", help="fetch the newest posts")
    p.add_argument("--diagnose", action="store_true", help="rebuild diagnostics.json")
    p.add_argument("--forecast", action="store_true", help="rebuild forecast.json")
    p.add_argument("--backtest", action="store_true", help="rerun the walk-forward backtest")
    p.add_argument("--all", action="store_true", help="everything except the backtest")
    p.add_argument("--max-weeks", type=int, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.backfill:
        backfill(since=args.backfill)
    if args.poll:
        poll()
    if args.diagnose:
        refresh_diagnostics()
    if args.backtest:
        refresh_backtest(max_weeks=args.max_weeks)
    if args.forecast:
        refresh_forecast()
    if args.all:
        refresh_all()
    if not any([args.backfill, args.poll, args.diagnose, args.forecast, args.backtest, args.all]):
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
