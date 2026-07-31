"""SQLite store — the durable time series.

`trumpstruth_id` is the natural key and the upsert is idempotent, so polling
windows can overlap freely and a re-run of the backfill changes nothing. That
matters more than it sounds: the daemon re-reads the same 100 newest posts every
cycle, and any non-idempotent write would inflate the counts the models learn
from.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from ..config import DB_PATH, ensure_dirs
from .parse import Post

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    trumpstruth_id  INTEGER PRIMARY KEY,
    truth_id        TEXT,
    truth_url       TEXT,
    created_utc     TEXT NOT NULL,
    text            TEXT NOT NULL DEFAULT '',
    is_retruth      INTEGER NOT NULL DEFAULT 0,
    source_handle   TEXT,
    is_deleted      INTEGER NOT NULL DEFAULT 0,
    has_media       INTEGER NOT NULL DEFAULT 0,
    media_kinds     TEXT NOT NULL DEFAULT '',
    links           TEXT NOT NULL DEFAULT '',
    time_precision  TEXT NOT NULL DEFAULT 'minute',
    first_seen_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc);

CREATE TABLE IF NOT EXISTS ingest_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_posts(conn: sqlite3.Connection, posts: Iterable[Post]) -> int:
    """Insert new posts; refresh mutable fields on ones already stored.

    `created_utc` is refreshed too, because a listing-sourced row (minute
    precision) is later superseded by the RSS-sourced row for the same post
    (second precision) — but only in that direction, never the reverse.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = [dict(p.as_row(), first_seen_utc=now) for p in posts]
    if not rows:
        return 0

    before = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    conn.executemany(
        """
        INSERT INTO posts (
            trumpstruth_id, truth_id, truth_url, created_utc, text,
            is_retruth, source_handle, is_deleted, has_media, media_kinds, links,
            time_precision, first_seen_utc
        ) VALUES (
            :trumpstruth_id, :truth_id, :truth_url, :created_utc, :text,
            :is_retruth, :source_handle, :is_deleted, :has_media, :media_kinds, :links,
            :time_precision, :first_seen_utc
        )
        ON CONFLICT(trumpstruth_id) DO UPDATE SET
            truth_id   = COALESCE(excluded.truth_id, posts.truth_id),
            truth_url  = COALESCE(excluded.truth_url, posts.truth_url),
            source_handle = COALESCE(excluded.source_handle, posts.source_handle),
            -- either source spotting a ReTruth is enough; the listing marks it
            -- by author attribution, the feed by an "RT @" prefix
            is_retruth = MAX(excluded.is_retruth, posts.is_retruth),
            is_deleted = excluded.is_deleted,
            text       = CASE WHEN excluded.text != '' THEN excluded.text ELSE posts.text END,
            created_utc = CASE
                WHEN excluded.time_precision = 'second' AND posts.time_precision = 'minute'
                THEN excluded.created_utc ELSE posts.created_utc END,
            time_precision = CASE
                WHEN excluded.time_precision = 'second' THEN 'second' ELSE posts.time_precision END
        """,
        rows,
    )
    after = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    return after - before


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO ingest_state (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value, datetime.now(timezone.utc).isoformat()),
    )


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM ingest_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def post_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]


def date_range(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
    row = conn.execute("SELECT MIN(created_utc) a, MAX(created_utc) b FROM posts").fetchone()
    return row["a"], row["b"]


def load_posts(
    conn: sqlite3.Connection,
    since: str | None = None,
    include_deleted: bool = True,
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM posts"
    clauses, params = [], []
    if since:
        clauses.append("created_utc >= ?")
        params.append(since)
    if not include_deleted:
        clauses.append("is_deleted = 0")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_utc"
    return conn.execute(sql, params).fetchall()
