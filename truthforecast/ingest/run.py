"""Ingest entry points: full backfill and incremental poll."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..config import ARCHIVE_START
from .client import TruthStruthClient, utc_now
from .store import connect, date_range, post_count, set_state, upsert_posts

log = logging.getLogger(__name__)


def _parse_day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def backfill(since: str = ARCHIVE_START, client: TruthStruthClient | None = None) -> int:
    """Walk the archive backwards to `since`, persisting each page as it lands.

    Safe to interrupt and re-run: the store is keyed on the post id, so a resumed
    run simply re-writes rows it already has.
    """
    client = client or TruthStruthClient()
    until = _parse_day(since)
    total_new = 0

    with connect() as conn:
        for batch in client.walk_back(start=utc_now() + timedelta(days=1), until=until):
            total_new += upsert_posts(conn, batch)
            conn.commit()
        set_state(conn, "last_backfill_utc", utc_now().isoformat())
        set_state(conn, "backfill_since", since)
        log.info(
            "backfill complete: +%s new, %s total, range %s",
            total_new, post_count(conn), date_range(conn),
        )
    return total_new


def poll(client: TruthStruthClient | None = None) -> int:
    """Fetch the newest posts. Cheap enough to run every few minutes.

    Reads both the RSS feed (exact UTC, 100 newest) and the first listing page,
    because the two disagree slightly about which posts are "newest" — the feed
    orders by ingest, the listing by content time — and the union is what keeps
    ReTruths of older content from being missed.
    """
    client = client or TruthStruthClient()
    posts = list(client.fetch_feed())
    page, _ = client.fetch_page()
    posts.extend(page)

    with connect() as conn:
        new = upsert_posts(conn, posts)
        set_state(conn, "last_poll_utc", utc_now().isoformat())
    log.info("poll: %s new of %s seen", new, len(posts))
    return new
