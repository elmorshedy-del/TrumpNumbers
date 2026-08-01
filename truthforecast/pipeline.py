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
from . import live
from .config import CONVENTION, EXPORT_DIR, WINDOW, ensure_dirs
from .forecast import current_week_forecast, headline, headline_curve, threshold_probabilities
from .ingest.run import backfill, poll, reconcile
from .ingest.run import coverage as ingest_coverage
from .models.registry import model_catalog
from .series import convention_events, daily_counts, load_frame, modeling_series

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
    """Model order from the last backtest, best CRPS first.

    Deliberately the **development** leaderboard, not the holdout one. The
    holdout exists to answer "how well does the chosen procedure do on weeks
    nobody looked at" — and it can only answer that while nothing is chosen with
    it. Ranking the live headline on eight weeks would spend the one clean
    sample on a selection, leave nothing to check the selection with, and report
    the resulting score as if it were still out-of-sample.

    It is also the noisier board: eight weeks reorders models whose CRPS differs
    by less than the sampling noise, so selecting on it mostly chases noise. The
    holdout is still computed, still shown, and still the honest number — it is
    just not allowed to pick.
    """
    path = EXPORT_DIR / "backtest.json"
    if not path.exists():
        return []
    try:
        rep = json.loads(path.read_text())
        return [r["model"] for r in rep.get("leaderboard") or []]
    except Exception:
        return []


def refresh_forecast(df=None) -> dict:
    """Recompute the live numbers and rewrite the site's JSON."""
    df = load_frame() if df is None else df
    series = modeling_series(df)
    if series.daily.empty:
        raise RuntimeError("no data — run a backfill first")

    events = convention_events(df)
    fc = current_week_forecast(series, events=events)
    ranking = load_ranking()

    thresholds = []
    weekly = series.complete_weeks()
    if len(weekly) >= 10:
        thresholds = [float(np.quantile(weekly, p)) for p in (0.5, 0.75, 0.9)]

    head = headline(fc, ranking=ranking)
    probs = threshold_probabilities(fc, thresholds, ranking=ranking) if thresholds else []

    # Log the forecast before it can be revised, then report how the ones
    # already logged have done. This is the only number here that is not
    # retrospective.
    curve, _ = headline_curve(fc, ranking=ranking)
    try:
        live.record(fc, head, curve=curve)
        live_record = live.score(series.complete_weeks())
    except Exception as exc:  # noqa: BLE001 - the record must never break the build
        log.warning("live forecast record failed: %s", exc)
        live_record = {"available": False, "note": "live record unavailable"}

    recent = series.daily.tail(70)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": ingest_coverage(),
        "window": {"start": WINDOW.start, "label": WINDOW.label},
        "ranking_basis": (
            "development leaderboard (the holdout is reported, never used to select)"
            if ranking else "no backtest yet — models are in registry order"
        ),
        "headline": head,
        "week": fc["week"],
        # How much of today is already decided, and what is left of it. The
        # headline leans on this, so it travels with it rather than being an
        # invisible internal.
        "today": fc.get("today", {}),
        "threshold_probabilities": probs,
        # What this project has forecast before the fact, and how it did.
        "live_record": live_record,
        "per_model": [
            {k: v for k, v in m.items() if not k.startswith("_")} for m in fc["per_model"]
        ],
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
    everything = daily_counts(df, "all", start=WINDOW.start)
    payload["retruths"] = {
        "total": int(retruths.daily.sum()),
        # Against all posts, not against the modelled series plus themselves.
        # Under the current convention the modelled series already contains the
        # ReTruths, so the old denominator counted them twice and understated
        # the share it was reporting.
        "share_of_all_posts": round(
            float(retruths.daily.sum()) / max(float(everything.daily.sum()), 1), 3
        ),
        "counted_in_series": bool(CONVENTION.include_retruths),
        "note": (
            "ReTruths are counted, on the day Trump reshared rather than the day the original "
            "ran. The mirror files them under the ORIGINAL author's timestamp, so counting them "
            "as filed would credit activity to days on which nothing happened; the mirror's own "
            "ingest order brackets the real reshare moment and recovers it to within hours."
            if CONVENTION.include_retruths else
            "ReTruths are excluded from the modelled series by configuration "
            "(TTF_INCLUDE_RETRUTHS=0), which measures original authorship rather than the "
            "market's definition."
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
    """The most recent posts, for the live feed on the site.

    Ordered by when each post entered the timeline, not by the timestamp the
    mirror files it under. Those differ for a ReTruth — its stored time is the
    *original author's* — so sorting on `created_utc` buried a reshare of a
    month-old post a month down the list. The feed exists to show what just
    happened, and a ReTruth is something that just happened.
    """
    df = load_frame() if df is None else df
    order = ["effective_utc"] if "effective_utc" in df else ["created_utc"]
    # Tie-break on arrival order. A ReTruth newer than every original in the
    # frame has its reshare time clamped to the newest original's, so the two
    # can land on the same instant; the id says which came first.
    recent = df.sort_values(order + ["trumpstruth_id"], ascending=False).head(limit)
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
    # The same posts the daily series is built from. Passing originals-only here
    # while the target counts ReTruths meant the event-level models — Hawkes —
    # were fitted on about three quarters of the process and then graded against
    # all of it, so both the background rate and the branching ratio were
    # estimates of something other than what was being forecast. The live path
    # already used the convention; the backtest did not, which is the worse way
    # round: the model was scored under one population and deployed under
    # another.
    result = run_backtest(
        series.daily, events=convention_events(df),
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
    p.add_argument("--poll", action="store_true", help="fetch everything since the last poll")
    p.add_argument("--reconcile", metavar="DAYS", nargs="?", type=int, const=14,
                   help="re-read the last N days so deletions and corrections land")
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
    if args.reconcile:
        reconcile(days=args.reconcile)
    if args.diagnose:
        refresh_diagnostics()
    if args.backtest:
        refresh_backtest(max_weeks=args.max_weeks)
    if args.forecast:
        refresh_forecast()
    if args.all:
        refresh_all()
    if not any([args.backfill, args.poll, args.reconcile, args.diagnose,
                args.forecast, args.backtest, args.all]):
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
