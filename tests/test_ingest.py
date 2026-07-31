"""Ingest tests.

These run against saved fixtures of the real site, so they fail loudly if
trumpstruth.org changes shape. That matters more than usual here: a parser that
quietly returns nothing would write a run of zero-post days into the series, and
every model downstream would faithfully learn from the lie.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from truthforecast.ingest.client import cursor_for, decode_cursor
from truthforecast.ingest.parse import (
    ParseError,
    parse_listing,
    parse_listing_timestamp,
    parse_rss,
)
from truthforecast.ingest.store import connect, post_count, upsert_posts

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def rss_text():
    return (FIXTURES / "feed.xml").read_text()


@pytest.fixture(scope="module")
def listing_html():
    return (FIXTURES / "listing.html").read_text()


def test_rss_parses_a_full_page(rss_text):
    posts = parse_rss(rss_text)
    assert len(posts) == 100
    assert all(p.trumpstruth_id > 0 for p in posts)
    assert all(p.created_utc.tzinfo is not None for p in posts)
    # The feed is the only source with second-level precision.
    assert all(p.time_precision == "second" for p in posts)


def test_rss_detects_retruths(rss_text):
    posts = parse_rss(rss_text)
    rts = [p for p in posts if p.is_retruth]
    assert rts, "fixture should contain at least one ReTruth"
    assert all(p.source_handle and p.source_handle.startswith("@") for p in rts)


def test_listing_counts_only_top_level_posts(listing_html):
    """A ReTruth embeds the original as a nested .status block.

    On a typical page 159 .status nodes correspond to 100 real posts. Counting
    the nested ones would both inflate daily totals and inject the original's
    (often far older) timestamp into the series.
    """
    from selectolax.parser import HTMLParser

    tree = HTMLParser(listing_html)
    all_nodes = tree.css(".status")
    top_level = [n for n in all_nodes if n.attributes.get("data-status-url")]
    assert len(all_nodes) > len(top_level), "fixture should contain nested quoted posts"

    posts, _ = parse_listing(listing_html)
    assert len(posts) <= len(top_level)
    assert len({p.trumpstruth_id for p in posts}) == len(posts), "ids must be unique"


def test_listing_timestamps_are_eastern_not_utc():
    """Pins the timezone, verified against the feed's explicit UTC pubDate.

    The site prints Eastern wall-clock with no zone label. If this assertion
    ever fails, every timestamp in the archive has silently shifted by hours and
    the day buckets are wrong.
    """
    # 2026-07-30 18:24 Eastern (EDT, UTC-4) == 2026-07-30 22:24 UTC
    got = parse_listing_timestamp("July 30, 2026, 6:24 PM")
    assert got == datetime(2026, 7, 30, 22, 24, tzinfo=timezone.utc)

    # And in winter the offset must be 5 hours, not 4 — a fixed offset would
    # silently corrupt half the archive.
    winter = parse_listing_timestamp("January 15, 2026, 10:00 AM")
    assert winter == datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)


def test_listing_and_rss_agree_on_shared_posts(rss_text, listing_html):
    """The two sources must not disagree about when a post happened."""
    rss = {p.trumpstruth_id: p for p in parse_rss(rss_text)}
    listing, _ = parse_listing(listing_html)
    shared = [p for p in listing if p.trumpstruth_id in rss]
    assert len(shared) >= 20, "fixtures should overlap substantially"

    for p in shared:
        delta = abs((p.created_utc - rss[p.trumpstruth_id].created_utc).total_seconds())
        # Listing labels drop seconds, so allow up to a minute.
        assert delta <= 60, f"post {p.trumpstruth_id} disagrees by {delta}s"


def test_bad_timestamp_raises():
    with pytest.raises(ParseError):
        parse_listing_timestamp("sometime last Tuesday")


def test_garbage_listing_raises():
    with pytest.raises(ParseError):
        parse_listing("<html><body><p>nothing here</p></body></html>")


def test_cursor_roundtrip():
    when = datetime(2026, 3, 2, 12, 30, 0, tzinfo=timezone.utc)
    decoded = decode_cursor(cursor_for(when))
    assert decoded is not None
    # Cursors are expressed in the site's display timezone.
    assert decoded.year == 2026 and decoded.month == 3


def test_upsert_is_idempotent(tmp_path, rss_text):
    """Overlapping polls must not inflate the counts the models learn from."""
    db = tmp_path / "t.db"
    posts = parse_rss(rss_text)

    with connect(db) as conn:
        first = upsert_posts(conn, posts)
        assert first == len(posts)

    with connect(db) as conn:
        second = upsert_posts(conn, posts)
        assert second == 0
        assert post_count(conn) == len(posts)


def test_second_precision_supersedes_minute(tmp_path, rss_text, listing_html):
    """A listing row (minute) must be upgraded by the feed row (seconds)."""
    db = tmp_path / "t.db"
    listing, _ = parse_listing(listing_html)
    rss = parse_rss(rss_text)
    shared = {p.trumpstruth_id for p in listing} & {p.trumpstruth_id for p in rss}
    assert shared

    with connect(db) as conn:
        upsert_posts(conn, listing)
        upsert_posts(conn, rss)
        sid = next(iter(shared))
        row = conn.execute(
            "SELECT time_precision FROM posts WHERE trumpstruth_id = ?", (sid,)
        ).fetchone()
        assert row["time_precision"] == "second"


# --- polling coverage -------------------------------------------------------
#
# The poll's failure mode is silent: posts missed during an outage do not show
# up as a gap, they show up as *quiet days*, which every model downstream then
# fits as real. These tests pin the two behaviours that prevent it — walk back
# far enough to cover the gap, and refuse to record a gap as covered when the
# walk stopped early.


class FakeClient:
    """A client whose walk_back stops after `pages` pages, recording its target."""

    def __init__(self, posts, pages=3, exhausted=False, reach_target=True):
        self._posts = posts
        self._pages = pages
        self._exhausted = exhausted
        self._reach_target = reach_target
        self.targets = []

    def fetch_feed(self):
        return self._posts[:1]

    def walk_back(self, start=None, until=None, max_pages=None):
        from truthforecast.ingest.client import WalkPage

        self.targets.append(until)
        for i in range(self._pages):
            last = i == self._pages - 1
            reached = until - timedelta(hours=1) if (last and self._reach_target) else start
            yield WalkPage(self._posts, reached, exhausted=self._exhausted and last)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    from truthforecast.ingest import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "poll.db")
    return tmp_path / "poll.db"


def test_poll_walks_back_to_the_last_covered_point(isolated_db, rss_text):
    """The catch-up target is the previous watermark, not "the newest page"."""
    from truthforecast.ingest.run import POLL_OVERLAP, poll
    from truthforecast.ingest.store import get_state, set_state

    posts = parse_rss(rss_text)
    watermark = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    with connect(isolated_db) as conn:
        set_state(conn, "last_poll_utc", watermark.isoformat())

    client = FakeClient(posts)
    poll(client=client)

    assert client.targets == [watermark - POLL_OVERLAP]
    with connect(isolated_db) as conn:
        assert get_state(conn, "last_poll_complete") == "1"
        assert get_state(conn, "last_poll_utc") != watermark.isoformat()


def test_an_incomplete_walk_leaves_the_watermark_alone(isolated_db, rss_text):
    """A walk that gave up early must not be recorded as coverage."""
    from truthforecast.ingest.run import poll
    from truthforecast.ingest.store import get_state, set_state

    posts = parse_rss(rss_text)
    watermark = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    with connect(isolated_db) as conn:
        set_state(conn, "last_poll_utc", watermark.isoformat())

    poll(client=FakeClient(posts, pages=2, reach_target=False))

    with connect(isolated_db) as conn:
        # Unchanged, so the next run retries the same gap rather than skipping it.
        assert get_state(conn, "last_poll_utc") == watermark.isoformat()
        assert get_state(conn, "last_poll_complete") == "0"


# --- notifications ----------------------------------------------------------


def test_notifier_adopts_the_present_on_first_run(isolated_db, rss_text):
    """A fresh database must not page you about the whole archive."""
    from truthforecast.ingest.store import get_state
    from truthforecast.notify import notify

    posts = parse_rss(rss_text)
    with connect(isolated_db) as conn:
        upsert_posts(conn, posts)

    assert notify() == 0
    with connect(isolated_db) as conn:
        assert get_state(conn, "notified_max_id") == str(max(p.trumpstruth_id for p in posts))


def test_a_failed_send_does_not_advance_the_watermark(isolated_db, monkeypatch, rss_text):
    """Better to notify twice than to drop one silently."""
    from truthforecast import notify as notify_mod
    from truthforecast.ingest.store import get_state, set_state

    posts = sorted(parse_rss(rss_text), key=lambda p: p.trumpstruth_id)
    with connect(isolated_db) as conn:
        upsert_posts(conn, posts)
        set_state(conn, "notified_max_id", str(posts[-3].trumpstruth_id))

    monkeypatch.setenv("TTF_WEBHOOK_URL", "http://127.0.0.1:9/never")
    monkeypatch.setattr(notify_mod, "_post", lambda *a, **k: False)

    assert notify_mod.notify() == 2
    with connect(isolated_db) as conn:
        assert get_state(conn, "notified_max_id") == str(posts[-3].trumpstruth_id)
