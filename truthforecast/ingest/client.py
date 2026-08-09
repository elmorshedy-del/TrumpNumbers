"""HTTP client for trumpstruth.org.

The site paginates with an opaque-looking `cursor` query parameter that is in
fact base64-encoded JSON:

    {"status_created_at": "2026-02-27 15:38:47", "_pointsToNextItems": true}

Because it is just a timestamp, a cursor can be *synthesized* for any date
rather than discovered by walking from the present. That turns a full-archive
backfill into ~400 requests that can start anywhere and resume anywhere.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

import requests

from ..config import (
    MAX_RETRIES,
    REQUEST_DELAY_S,
    REQUEST_TIMEOUT_S,
    SITE_TZ,
    SOURCE_BASE,
    USER_AGENT,
)
from .parse import Post, parse_listing, parse_rss

log = logging.getLogger(__name__)

PER_PAGE = 100


def cursor_for(when: datetime) -> str:
    """Build a 'next items' cursor pointing just before `when`.

    The site's cursor timestamps are in its own display timezone (Eastern), the
    same clock the listing labels use, so an aware datetime is converted before
    being formatted.
    """
    if when.tzinfo is not None:
        when = when.astimezone(SITE_TZ)
    payload = {
        "status_created_at": when.strftime("%Y-%m-%d %H:%M:%S"),
        "_pointsToNextItems": True,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


def decode_cursor(cursor: str) -> datetime | None:
    """Inverse of `cursor_for` — used to detect a stalled walk."""
    try:
        payload = json.loads(base64.b64decode(cursor + "==="[: (-len(cursor)) % 4]))
        stamp = payload.get("status_created_at")
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") if stamp else None
    except Exception:
        return None


@dataclass
class WalkPage:
    """One page of a backwards walk, plus how far back the walk has reached.

    `reached` is the cursor timestamp for the *next* page, in UTC — the point in
    the archive the walk has now covered down to. The caller needs it to know
    whether a catch-up actually closed the gap it was aiming at, because a walk
    that stops early (page cap, dead cursor) leaves a hole that must not be
    recorded as covered. `exhausted` means the archive ended.
    """

    posts: list[Post]
    reached: datetime | None
    exhausted: bool = False


class TruthStruthClient:
    def __init__(self, session: requests.Session | None = None, delay: float = REQUEST_DELAY_S):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.delay = delay
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, params: dict | None = None) -> str:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                res = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
                if res.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"status {res.status_code}")
                res.raise_for_status()
                return res.text
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                backoff = min(2**attempt, 30)
                log.warning("GET %s failed (%s); retry in %ss", url, exc, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} attempts: {last_error}")

    def fetch_feed(self) -> list[Post]:
        """The 100 newest posts, with exact UTC timestamps. Used for polling."""
        return parse_rss(self._get(f"{SOURCE_BASE}/feed"))

    def fetch_page(self, cursor: str | None = None) -> tuple[list[Post], str | None]:
        params = {"sort": "desc", "per_page": str(PER_PAGE), "removed": "include"}
        if cursor:
            params["cursor"] = cursor
        return parse_listing(self._get(f"{SOURCE_BASE}/", params=params))

    def walk_back(
        self,
        start: datetime | None = None,
        until: datetime | None = None,
        max_pages: int = 2000,
    ) -> Iterator[WalkPage]:
        """Page backwards through the archive, newest first.

        Yields one page at a time so a caller can persist incrementally and
        resume after an interruption. Stops when `until` is passed, when the
        site stops returning posts, or when the walk stops making progress.

        Every stop is reported through the final `WalkPage`: `reached` says how
        far back the walk got and `exhausted` distinguishes "archive ended" from
        "gave up". A caller that treats an early stop as success would record a
        gap as covered, and the missing days would enter the series as quiet
        days rather than as unknowns.
        """
        cursor = cursor_for(start) if start else None
        # The cursor timestamp is the site's own sort key, so it is the only
        # value that decreases monotonically as we page back. Two things that
        # look like they should work do not:
        #   - the oldest *post* date on a page, because a ReTruth carries the
        #     original author's date and can be years older than its neighbours;
        #   - the smallest *post id*, because ids are ingest order, which the
        #     listing does not sort by.
        # Both were tried; both halted the walk early on real data.
        prev_cursor_dt = decode_cursor(cursor) if cursor else None
        until_naive = until.astimezone(SITE_TZ).replace(tzinfo=None) if until else None

        reached = site_naive_to_utc(prev_cursor_dt)

        for page in range(max_pages):
            posts, next_cursor = self.fetch_page(cursor)
            if not posts:
                log.info("walk_back: empty page at %s, stopping", cursor)
                yield WalkPage([], reached, exhausted=True)
                return

            if not next_cursor:
                log.info("walk_back: no further cursor, archive exhausted")
                yield WalkPage(posts, reached, exhausted=True)
                return

            cursor_dt = decode_cursor(next_cursor)
            if cursor_dt is None:
                log.warning("walk_back: undecodable cursor, stopping")
                yield WalkPage(posts, reached)
                return

            reached = site_naive_to_utc(cursor_dt)
            yield WalkPage(posts, reached)

            if until_naive is not None and cursor_dt <= until_naive:
                log.info("walk_back: reached %s, stopping", until_naive.date())
                return
            if prev_cursor_dt is not None and cursor_dt >= prev_cursor_dt:
                log.warning("walk_back: cursor stopped advancing at %s, stopping", cursor_dt)
                return

            prev_cursor_dt = cursor_dt
            cursor = next_cursor
            if page and page % 40 == 0:
                log.info("walk_back: page %s, back to %s", page, cursor_dt.date())

        log.warning("walk_back: hit max_pages=%s at %s", max_pages, reached)


def site_naive_to_utc(when: datetime | None) -> datetime | None:
    """Read a bare cursor timestamp as the site's own clock and return UTC."""
    if when is None:
        return None
    return when.replace(tzinfo=SITE_TZ).astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
