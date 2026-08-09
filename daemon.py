#!/usr/bin/env python3
"""Self-updating local runner.

Four cadences, because the work costs four very different amounts:

  * poll + forecast   every ~15 min   (seconds — keeps the live numbers moving)
  * diagnostics       hourly          (a few seconds)
  * reconcile         daily           (~15s — re-reads the last two weeks so
                                       deletions and corrections land)
  * full backtest     nightly         (minutes — re-ranks every model)

SQLite holds all the state, so a crash or a restart resumes cleanly: the poll is
idempotent, and its coverage watermark only advances when a run actually closes
the gap, so an interrupted catch-up is retried rather than skipped.

    python daemon.py                 # run forever, serving on :8000
    python daemon.py --once          # one pass, no server
    python daemon.py --backtest      # force a backtest now, then exit
    python daemon.py --serve-only    # just serve what is already exported

The GitHub Actions workflows in .github/workflows run the same functions on the
same cadences; this is the local equivalent, not a second implementation.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import os
import posixpath
import random
import socketserver
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from truthforecast.config import DB_PATH, EXPORT_DIR, SITE_DIR, ensure_dirs
from truthforecast.ingest.run import backfill, poll, reconcile
from truthforecast.ingest.store import connect, post_count
from truthforecast.notify import configured as notify_channels
from truthforecast.notify import notify
from truthforecast.pipeline import (
    refresh_backtest,
    refresh_diagnostics,
    refresh_forecast,
    refresh_posts,
)
from truthforecast.series import load_frame

log = logging.getLogger("daemon")

# Drop this to 60 if you are running it for the notifications rather than the
# forecast: the poll costs two requests, and the loop's latency is the floor on
# how fast an alert can reach you.
POLL_SECONDS = int(os.environ.get("TTF_POLL_SECONDS", 15 * 60))
DIAGNOSTICS_SECONDS = 60 * 60
RECONCILE_SECONDS = 24 * 60 * 60
BACKTEST_SECONDS = 24 * 60 * 60


class SiteHandler(http.server.SimpleHTTPRequestHandler):
    """Serve `site/` at the root, with the exports mounted at /data/exports/.

    Which is exactly the layout the deployed bundle has. Local and hosted then
    resolve data through the same relative path, so "works on my machine"
    cannot mean a different URL shape than the one users get.
    """

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/data/exports/"):
            # basename() so a crafted path cannot climb out of the export dir.
            return str(EXPORT_DIR / posixpath.basename(clean))
        return super().translate_path(path)


def serve(port: int = 8000) -> None:
    handler = functools.partial(SiteHandler, directory=str(SITE_DIR))

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

    def run():
        with Quiet(("", port), handler) as httpd:
            log.info("serving http://localhost:%s/", port)
            httpd.serve_forever()

    threading.Thread(target=run, daemon=True).start()


def ensure_data() -> None:
    """First run on a clean machine: fetch the archive before anything else."""
    ensure_dirs()
    if not DB_PATH.exists():
        log.info("no database yet — running the initial archive backfill (a few minutes)")
        backfill()
        return
    with connect() as conn:
        n = post_count(conn)
    if n < 500:
        log.info("only %s posts stored — completing the backfill", n)
        backfill()


def push_notifications() -> None:
    """Never let a notifier failure take the pipeline down with it."""
    if not notify_channels():
        return
    try:
        notify()
    except Exception:
        log.exception("notification pass failed")


def best_effort(what: str, fn, *args) -> None:
    """Run a derived export, and never let it take the pass down with it.

    The order of work in a pass is not the order of its consequences. The feed,
    the diagnostics and the leaderboard are all rebuilt from scratch next pass,
    so failing one costs half an hour of staleness. `refresh_forecast` runs
    LAST and is the only step that writes anything irreplaceable: the register
    keeps the FIRST forecast per (week, cut) and is append-only, so a cut that
    no pass ever reaches is a permanent hole in the prospective record — the
    one number here that is not retrospective.

    Letting a cosmetic export abort the pass inverts that. It nearly happened:
    a single unserialisable value in the posts feed raised before the forecast
    was ever built, and the condition that caused it persists for as long as
    the offending post sits in the 60-post window — about a day, which is
    exactly one cut. So the cheap steps are isolated and the expensive one is
    not: if `refresh_forecast` fails the job SHOULD go red, leaving the last
    good deploy up and the reason in the run log.
    """
    try:
        fn(*args)
    except Exception:
        log.exception("%s failed; continuing so the forecast still runs", what)


def one_pass(do_backtest: bool = False, do_reconcile: bool = False) -> None:
    poll()
    push_notifications()
    if do_reconcile:
        reconcile()
    df = load_frame()
    best_effort("posts export", refresh_posts, df)
    best_effort("diagnostics export", refresh_diagnostics, df)
    if do_backtest:
        best_effort("backtest", refresh_backtest, df)
    # Last, and deliberately unguarded — see best_effort.
    refresh_forecast(df)


def loop() -> None:
    ensure_data()
    last_diag = last_backtest = last_reconcile = datetime.min.replace(tzinfo=timezone.utc)

    while True:
        now = datetime.now(timezone.utc)
        try:
            poll()
            push_notifications()

            if now - last_reconcile > timedelta(seconds=RECONCILE_SECONDS):
                reconcile()
                last_reconcile = now

            df = load_frame()
            best_effort("posts export", refresh_posts, df)

            if now - last_diag > timedelta(seconds=DIAGNOSTICS_SECONDS):
                best_effort("diagnostics export", refresh_diagnostics, df)
                last_diag = now

            if now - last_backtest > timedelta(seconds=BACKTEST_SECONDS):
                log.info("running nightly backtest — this takes a few minutes")
                best_effort("backtest", refresh_backtest, df)
                last_backtest = now

            # Last, so it picks up any fresh ranking the backtest just wrote.
            refresh_forecast(df)
            log.info("cycle complete")
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("cycle failed; will retry")

        # Jitter so a restarted daemon does not hit the source on a fixed beat.
        time.sleep(POLL_SECONDS + random.uniform(-60, 60))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true", help="single pass, then exit")
    p.add_argument("--backtest", action="store_true", help="force a backtest this pass")
    p.add_argument("--reconcile", action="store_true",
                   help="re-read the recent past this pass (deletions, corrections)")
    p.add_argument("--serve-only", action="store_true", help="serve existing exports, no work")
    p.add_argument("--no-serve", action="store_true", help="do not start the web server")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    if args.serve_only:
        serve(args.port)
        log.info("serve-only; Ctrl-C to stop")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0

    if args.once or args.backtest or args.reconcile:
        ensure_data()
        one_pass(do_backtest=args.backtest, do_reconcile=args.reconcile)
        log.info("done — exports in %s", EXPORT_DIR)
        return 0

    if not args.no_serve:
        serve(args.port)
    try:
        loop()
    except KeyboardInterrupt:
        log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
