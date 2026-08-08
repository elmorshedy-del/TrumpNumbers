"""Ingest entry points: full backfill, incremental poll, and reconciliation.

The poll is the part that has to be right, because its failure mode is silent.
Reading only the newest page and trusting it means that any interruption longer
than one page of posts — a stalled runner, a delayed cron, an outage at the
mirror — writes *quiet days* into the series rather than gaps. Every model
downstream then learns from the lie, and nothing in the output looks wrong.

So the poll is defined by coverage, not by cadence: it walks back until it
reaches the point the last **completed** poll got to, and it only advances that
watermark when the walk actually closed the gap. A poll that gives up early
leaves the watermark where it was, and the next run picks the gap back up.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..config import ARCHIVE_START
from .client import TruthStruthClient, utc_now
from .store import connect, date_range, get_state, post_count, set_state, upsert_posts

log = logging.getLogger(__name__)

# Re-read this far past the last covered point on every poll. Cheap insurance
# against clock skew and against a post landing in the mirror out of order.
POLL_OVERLAP = timedelta(hours=2)

# Ceiling on one catch-up walk: ~24k posts, about four minutes at 1 req/s. A
# longer outage is closed over several runs rather than in one very long one.
MAX_CATCHUP_PAGES = 240

# How far back a reconciliation pass re-reads. Deletions and late corrections
# only reach us by looking again at days we have already seen.
RECONCILE_DAYS = 14

# Cold start with no watermark: cover a day rather than assume the archive is
# complete up to now.
COLD_START_LOOKBACK = timedelta(days=1)


def _parse_day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _parse_state_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def backfill(since: str = ARCHIVE_START, client: TruthStruthClient | None = None) -> int:
    """Walk the archive backwards to `since`, persisting each page as it lands.

    Safe to interrupt and re-run: the store is keyed on the post id, so a resumed
    run simply re-writes rows it already has.
    """
    client = client or TruthStruthClient()
    until = _parse_day(since)
    total_new = 0

    with connect() as conn:
        for page in client.walk_back(start=utc_now() + timedelta(days=1), until=until):
            total_new += upsert_posts(conn, page.posts)
            conn.commit()
        set_state(conn, "last_backfill_utc", utc_now().isoformat())
        set_state(conn, "backfill_since", since)
        log.info(
            "backfill complete: +%s new, %s total, range %s",
            total_new, post_count(conn), date_range(conn),
        )
    return total_new


def poll(
    client: TruthStruthClient | None = None,
    *,
    catch_up: bool = True,
    max_pages: int = MAX_CATCHUP_PAGES,
) -> int:
    """Fetch everything posted since the last completed poll.

    Two sources, because they disagree about what "newest" means — the feed
    orders by ingest, the listing by content time — and the union is what keeps
    a ReTruth of older content from being missed:

    * the RSS feed: the 100 newest posts, with exact UTC timestamps;
    * the listing, walked back to the coverage watermark.

    On a normal cadence the walk is one or two pages. After an outage it is
    however many pages the gap needs, which is the entire point: the cost of an
    extra page is one second, and the cost of a missed one is a permanently
    wrong day in the series that no later run will notice.
    """
    client = client or TruthStruthClient()
    started = utc_now()
    new = 0

    with connect() as conn:
        new += upsert_posts(conn, client.fetch_feed())
        conn.commit()

        watermark = _parse_state_time(get_state(conn, "last_poll_utc"))
        target = (watermark - POLL_OVERLAP) if watermark else (started - COLD_START_LOOKBACK)
        if not catch_up:
            target = started - POLL_OVERLAP

        pages = seen = 0
        reached = None
        exhausted = False
        for page in client.walk_back(
            start=started + timedelta(days=1), until=target, max_pages=max_pages
        ):
            new += upsert_posts(conn, page.posts)
            conn.commit()
            pages += 1
            seen += len(page.posts)
            reached = page.reached or reached
            exhausted = exhausted or page.exhausted

        # Only a walk that actually got past the target closed the gap. Anything
        # else leaves the watermark alone so the next run retries it.
        complete = bool(seen) and (exhausted or (reached is not None and reached <= target))
        if complete:
            set_state(conn, "last_poll_utc", started.isoformat())
        else:
            log.warning(
                "poll did not close the gap (target %s, reached %s, %s page(s)) — "
                "watermark left at %s for the next run",
                target, reached, pages, watermark,
            )
        set_state(conn, "last_poll_attempt_utc", started.isoformat())
        set_state(conn, "last_poll_complete", "1" if complete else "0")
        set_state(conn, "last_poll_pages", str(pages))

    log.info(
        "poll: %s new of %s seen over %s page(s), back to %s (%s)",
        new, seen, pages, target, "complete" if complete else "INCOMPLETE",
    )
    return new


def reconcile(days: int = RECONCILE_DAYS, client: TruthStruthClient | None = None) -> int:
    """Re-read the recent past so corrections at the source reach the series.

    A poll only ever looks at what is new. Posts get deleted and the mirror
    marks them; timestamps get sharpened from minute to second precision when a
    post reaches the feed after the listing. Neither ever arrives on its own.
    Without this pass the stored history drifts away from what the source
    currently says, and the drift is invisible because nothing re-checks.
    """
    client = client or TruthStruthClient()
    started = utc_now()
    until = started - timedelta(days=days)
    changed = 0

    with connect() as conn:
        for page in client.walk_back(start=started + timedelta(days=1), until=until):
            changed += upsert_posts(conn, page.posts)
            conn.commit()
        set_state(conn, "last_reconcile_utc", started.isoformat())
        set_state(conn, "reconcile_days", str(days))

    log.info("reconcile: re-read the last %s days, %s row(s) new", days, changed)
    return changed


def coverage() -> dict:
    """What the ingest layer currently knows — for the site's freshness banner.

    A static page cannot tell whether its numbers are five minutes or five days
    old, and a stale forecast presented as a live one is the most likely way
    this project misleads anybody. So the freshness facts travel with the data.
    """
    with connect() as conn:
        first, last = date_range(conn)
        return {
            "posts_stored": post_count(conn),
            "earliest_post_utc": first,
            "latest_post_utc": last,
            "last_poll_utc": get_state(conn, "last_poll_utc"),
            "last_poll_attempt_utc": get_state(conn, "last_poll_attempt_utc"),
            "last_poll_complete": get_state(conn, "last_poll_complete") == "1",
            "last_reconcile_utc": get_state(conn, "last_reconcile_utc"),
            "last_backfill_utc": get_state(conn, "last_backfill_utc"),
        }
