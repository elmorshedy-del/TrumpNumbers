"""The market record: everything about this week that cannot be re-fetched later.

The backtest can be rebuilt from the mirror at any time. What can never be
rebuilt is what the *markets* believed while the week was still open — and that
is the only data that can ever answer whether the markets were wrong. This
module records it as it happens, under one rule:

    record whatever cannot be reconstructed later, at the finest
    granularity that is cheap, and decide what it is for afterwards.

What that rule selects, and why:

* **Kalshi order books** (`kalshi_book_*.jsonl`). Prices are reconstructible —
  Kalshi serves 1-minute candlesticks even for settled markets — but resting
  depth is not: the book at 11pm during a burst exists once. Snapshotted
  hourly for every open market in the tracked series (the weekly post count,
  and the deleted-post market, whose subject is this project's own exclusion
  rule).
* **Kalshi candlesticks and trades** (`kalshi_candles_YYYY-MM.jsonl`,
  `kalshi_trades_YYYY-MM.jsonl`). Reconstructible today, but retention is
  Kalshi's promise to keep, not ours; a daily incremental pull caps the loss
  if that promise changes. Candles at 1-minute resolution are the event-study
  record — what the price did in the minutes after a burst began. They are the
  one place verbatim storage lost the argument to arithmetic: see
  `_slim_candle` for exactly what is kept and why. Trades are kept whole.
* **Polymarket books and prices** (`polymarket_book_*.jsonl`,
  `polymarket_prices.jsonl`). Polymarket runs parallel post-count markets on
  *offset* windows, and its public price history for closed markets came back
  empty when checked — for this venue the snapshot IS the record. Two venues
  pricing one process is also the cheapest possible test of market
  rationality: when they disagree, at least one is wrong.
* **The public schedule** (`calendar.jsonl`). Factba.se revises entries as
  pool reports land, so the archive as downloaded later is not what was known
  in advance. Each entry is recorded the first time it is seen, so "what did
  the schedule say before the day happened" stays answerable. The first run
  seeds the whole archive and marks it retrospective, because entries adopted
  in bulk were *not* observed in advance and must not be mistaken for it.
* **Our own forecast** (`forecast_history.jsonl`). The live register keeps the
  first forecast per (week, cut) — the right rule for a register, and too
  coarse to line up against a moving price. Every pipeline pass appends its
  headline here, so model and market can be compared on the same clock.
* **Post provenance** (`posts_ledger.jsonl`). The archive database lives in a
  cache and is rebuilt from the mirror after eviction — but a rebuild loses
  `first_seen_utc` and the deletion history, which is exactly the evidence the
  ReTruth re-dating and the deleted-post exclusion rest on. One compact line
  per post, and a new line when a post's mutable fields change.

Everything is append-only JSONL under `data/record/`, one envelope per line
with `at` (UTC), `venue`/`kind`, and the payload as the source returned it.
Trimming is limited to oversized prose fields (contract rules, market
descriptions) and the measured candlestick exception above; numbers are never
rounded, because the whole point of a record kept in advance is not having
decided yet what it is for. Closed days' book files and closed months' candle
and trade files rotate to gzip; nothing is ever rewritten.

The snapshot path imports nothing outside the standard library, so the hourly
workflow needs no dependency install at all.

    python -m truthforecast.record --snapshot   # order books, both venues
    python -m truthforecast.record --daily      # candles, trades, calendar, rotate
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DATA_DIR, USER_AGENT

log = logging.getLogger(__name__)

RECORD_DIR = Path(os.environ.get("TTF_RECORD_DIR", DATA_DIR / "record"))

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# The weekly post-count market, and the deleted-post market. The second is not
# a curiosity: the counting convention excludes deleted posts, so that market
# prices the exact quantity this pipeline subtracts.
KALSHI_SERIES = ("KXTRUTHSOCIAL", "KXTRUMPDELETE")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
# Polymarket has no series concept; its weekly events are found by search and
# kept when the title says they count Trump's Truth Social posts.
POLYMARKET_QUERY = "truth social posts"
POLYMARKET_TITLE_WORDS = ("truth social posts",)

CALENDAR_URL = "https://media-cdn.factba.se/rss/json/trump/calendar-full.json"

REQUEST_DELAY_S = 0.25
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3

# Candlestick requests are capped by the API at a few thousand rows; two days
# of minutes stays comfortably under it.
CANDLE_CHUNK_MINUTES = 2 * 24 * 60

# Prose fields longer than this are dropped from *snapshot* lines (they repeat
# every hour); metadata lines keep everything.
SLIM_MAX_CHARS = 400

_last_request = 0.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _utc_now().isoformat()


def _http_get_json(url: str, params: dict | None = None):
    """GET a JSON document with the project's UA, politeness delay and retries."""
    global _last_request
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        elapsed = time.monotonic() - _last_request
        if elapsed < REQUEST_DELAY_S:
            time.sleep(REQUEST_DELAY_S - elapsed)
        _last_request = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as res:
                return json.load(res)
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            # A 404 is an answer (no book for a dead bracket), not an outage.
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                return None
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} attempts: {last_error}")


# Tests replace this; everything below fetches through it.
fetch_json = _http_get_json


def _append(path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
    return len(rows)


def _scan(path: Path, key) -> dict:
    """Latest value per key over an existing JSONL file. Missing file -> {}."""
    if not path.exists():
        return {}
    out: dict = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = key(row)
            if k is not None:
                out[k] = row
    return out


def _slim(obj, max_chars: int = SLIM_MAX_CHARS):
    """Drop oversized prose so hourly snapshots stay lean.

    Only long strings are removed — never numbers, never short fields, never
    keys whose values fit. An unknown field that fits is kept, because the
    schema of someone else's API is not ours to prune.
    """
    if isinstance(obj, dict):
        return {k: _slim(v, max_chars) for k, v in obj.items()
                if not (isinstance(v, str) and len(v) > max_chars)}
    if isinstance(obj, list):
        return [_slim(v, max_chars) for v in obj]
    return obj


def _sha(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]


def _iter_jsonl(paths) -> "iter":
    """Rows across plain and rotated (gzipped) JSONL files alike."""
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _month_of(ts: int | None) -> str:
    if ts is None:
        return "unknown"
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m")


def _load_state(out_dir: Path) -> dict:
    path = out_dir / "state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("unreadable %s; starting over (files are the record, this is a cursor)", path)
        return {}


def _save_state(out_dir: Path, state: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "state.json").write_text(json.dumps(state, indent=1, sort_keys=True))


# --------------------------------------------------------------- Kalshi

def _kalshi_open_markets() -> list[dict]:
    """Every open market in the tracked series, with its event attached."""
    out = []
    for series in KALSHI_SERIES:
        payload = fetch_json(
            f"{KALSHI_BASE}/events",
            {"series_ticker": series, "status": "open", "with_nested_markets": "true"},
        ) or {}
        for ev in payload.get("events") or []:
            for m in ev.get("markets") or []:
                out.append({"series": series, "event_ticker": ev.get("event_ticker"), "market": m})
    return out


def snapshot_kalshi(out_dir: Path) -> int:
    """One line per open market: its summary and its full order book."""
    at = _now_iso()
    rows = []
    for item in _kalshi_open_markets():
        ticker = item["market"].get("ticker")
        if not ticker:
            continue
        book = fetch_json(f"{KALSHI_BASE}/markets/{ticker}/orderbook")
        rows.append(
            {
                "at": at,
                "venue": "kalshi",
                "kind": "book",
                "series": item["series"],
                "event": item["event_ticker"],
                "ticker": ticker,
                "market": _slim(item["market"]),
                "orderbook": (book or {}).get("orderbook_fp")
                or (book or {}).get("orderbook")
                or book,
            }
        )
    day = _utc_now().date().isoformat()
    return _append(out_dir / f"kalshi_book_{day}.jsonl", rows)


# The fields that define what a market IS and how it resolved. Volume, open
# interest and last price change every day and are captured elsewhere (books
# hourly, candles daily); hashing them here would re-record every market's
# full metadata every single day for no information at all.
_KALSHI_STABLE = (
    "ticker", "event_ticker", "market_type", "title", "yes_sub_title", "subtitle",
    "strike_type", "floor_strike", "cap_strike", "open_time", "close_time",
    "expected_expiration_time", "expiration_time", "status", "result",
    "settlement_value", "settlement_value_dollars", "can_close_early",
)


def record_kalshi_markets(out_dir: Path) -> list[dict]:
    """Market metadata, appended when anything *stable* about a market changes —
    which in practice means lifecycle: created, closed, settled, result known.

    Returns every market seen (changed or not) so the candle and trade pulls
    can work from one listing instead of re-fetching it.
    """
    path = out_dir / "kalshi_markets.jsonl"
    known = _scan(path, lambda r: r.get("ticker"))
    seen, rows = [], []
    at = _now_iso()
    for series in KALSHI_SERIES:
        cursor = None
        for _ in range(20):  # cursor pagination, defensively bounded
            params = {"series_ticker": series, "limit": "1000"}
            if cursor:
                params["cursor"] = cursor
            payload = fetch_json(f"{KALSHI_BASE}/markets", params) or {}
            for m in payload.get("markets") or []:
                m["_series"] = series
                seen.append(m)
                fp = _sha({k: m.get(k) for k in _KALSHI_STABLE})
                prev = known.get(m.get("ticker"))
                if prev is None or prev.get("fp") != fp:
                    rows.append(
                        {"at": at, "venue": "kalshi", "kind": "market",
                         "ticker": m.get("ticker"), "fp": fp, "market": m}
                    )
            cursor = payload.get("cursor")
            if not cursor:
                break
    _append(path, rows)
    log.info("kalshi markets: %d tracked, %d changed", len(seen), len(rows))
    return seen


def _epoch(stamp: str | None) -> int | None:
    if not stamp:
        return None
    try:
        return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _slim_candle(ticker: str, c: dict) -> dict:
    """A 1-minute candle reduced to the fields that carry information.

    Verbatim storage was measured first and rejected on the numbers: ten weeks
    of this series is ~94 MB, and nearly all of it is the OHLC *of the bid and
    ask quotes*, which wiggle every minute a market maker breathes. What is
    kept — trade OHLC, closing bid/ask, volume, open interest — is the entire
    price path; what is dropped is reconstructible microstructure noise whose
    fine detail lives in the trade tape and the hourly book snapshots anyway.
    """
    out = {
        "ticker": ticker,
        "ts": c.get("end_period_ts"),
        "bid": (c.get("yes_bid") or {}).get("close_dollars"),
        "ask": (c.get("yes_ask") or {}).get("close_dollars"),
        "vol": c.get("volume_fp", c.get("volume")),
        "oi": c.get("open_interest_fp", c.get("open_interest")),
    }
    price = c.get("price") or {}
    for src, dst in (("open_dollars", "o"), ("high_dollars", "h"), ("low_dollars", "l"),
                     ("close_dollars", "c"), ("mean_dollars", "m")):
        if src in price:
            out[dst] = price[src]
    return out


def record_kalshi_candles(out_dir: Path, markets: list[dict], state: dict) -> int:
    """1-minute candlesticks, pulled incrementally per market.

    Written to monthly files (rotated to gzip once the month closes) and
    deduped on (ticker, minute). `state` holds how far each ticker is covered,
    so settled markets stop costing requests once their history is complete.
    Losing the state file is harmless — the pull restarts from each market's
    open and the dedupe keeps the files exact.
    """
    seen = set()
    for row in _iter_jsonl(sorted(out_dir.glob("kalshi_candles_*.jsonl*"))):
        seen.add((row.get("ticker"), row.get("ts")))
    covered = state.setdefault("kalshi_candles_covered_to", {})
    now_ts = int(_utc_now().timestamp())
    written = 0
    for m in markets:
        ticker = m.get("ticker")
        series = m.get("_series") or KALSHI_SERIES[0]
        open_ts = _epoch(m.get("open_time")) or now_ts
        close_ts = _epoch(m.get("close_time")) or now_ts
        stop = min(close_ts + 3600, now_ts)  # an hour past close catches settlement prints
        start = int(covered.get(ticker) or open_ts)
        while start < stop:
            end = min(start + CANDLE_CHUNK_MINUTES * 60, stop)
            payload = fetch_json(
                f"{KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks",
                {"start_ts": str(start), "end_ts": str(end), "period_interval": "1"},
            ) or {}
            by_month: dict[str, list] = {}
            for c in payload.get("candlesticks") or []:
                key = (ticker, c.get("end_period_ts"))
                if key in seen:
                    continue
                seen.add(key)
                slim = _slim_candle(ticker, c)
                by_month.setdefault(_month_of(slim["ts"]), []).append(slim)
            for month, rows in by_month.items():
                written += _append(out_dir / f"kalshi_candles_{month}.jsonl", rows)
            covered[ticker] = end
            start = end
    log.info("kalshi candles: %d new rows", written)
    return written


def record_kalshi_trades(out_dir: Path, markets: list[dict], state: dict) -> int:
    """The public trade tape, verbatim, deduped on trade id.

    Written to monthly files by trade time (rotated to gzip once the month
    closes). Trades are kept whole — each one is an actual event, and at a few
    hundred bytes each the tape earns its size.
    """
    seen = set()
    for row in _iter_jsonl(sorted(out_dir.glob("kalshi_trades_*.jsonl*"))):
        seen.add(row.get("trade_id"))
    done = state.setdefault("kalshi_trades_done", {})
    now_iso = _now_iso()
    written = 0
    for m in markets:
        ticker = m.get("ticker")
        if done.get(ticker):
            continue
        cursor, new_rows, hit_known = None, [], False
        for _ in range(50):  # newest first; bounded walk
            params = {"ticker": ticker, "limit": "1000"}
            if cursor:
                params["cursor"] = cursor
            payload = fetch_json(f"{KALSHI_BASE}/markets/trades", params) or {}
            trades = payload.get("trades") or []
            for t in trades:
                if t.get("trade_id") in seen:
                    hit_known = True
                    continue
                seen.add(t.get("trade_id"))
                new_rows.append({"venue": "kalshi", "kind": "trade", **t})
            cursor = payload.get("cursor")
            if not cursor or hit_known:
                break
        by_month: dict[str, list] = {}
        for t in new_rows:
            by_month.setdefault(_month_of(_epoch(t.get("created_time"))), []).append(t)
        for month, rows in by_month.items():
            written += _append(out_dir / f"kalshi_trades_{month}.jsonl", rows)
        # A settled market's tape is final: mark it done and stop asking.
        if (m.get("status") or "").lower() in ("settled", "finalized"):
            done[ticker] = now_iso
    log.info("kalshi trades: %d new rows", written)
    return written


# ----------------------------------------------------------- Polymarket

def _polymarket_events(closed: bool | None = False) -> list[dict]:
    """Post-count events found by search, fetched in full by slug."""
    found = fetch_json(
        f"{GAMMA_BASE}/public-search",
        {"q": POLYMARKET_QUERY, "limit_per_type": "50"},
    ) or {}
    out = []
    for ev in found.get("events") or []:
        title = (ev.get("title") or "").lower()
        if not any(w in title for w in POLYMARKET_TITLE_WORDS):
            continue
        if closed is not None and bool(ev.get("closed")) != closed:
            continue
        full = fetch_json(f"{GAMMA_BASE}/events", {"slug": ev.get("slug")})
        if isinstance(full, list) and full:
            out.append(full[0])
    return out


_PM_MARKET_FIELDS = (
    "id", "question", "groupItemTitle", "slug", "outcomes", "outcomePrices",
    "bestBid", "bestAsk", "lastTradePrice", "volume", "volumeNum", "liquidity",
    "liquidityNum", "closed", "endDate", "clobTokenIds", "umaResolutionStatus",
)


def snapshot_polymarket(out_dir: Path) -> int:
    """One line per open event, plus book depth for every live bracket."""
    at = _now_iso()
    rows = []
    for ev in _polymarket_events(closed=False):
        brackets = [
            {k: m.get(k) for k in _PM_MARKET_FIELDS if k in m}
            for m in ev.get("markets") or []
        ]
        rows.append(
            {
                "at": at, "venue": "polymarket", "kind": "book",
                "slug": ev.get("slug"), "event_id": ev.get("id"),
                "title": ev.get("title"), "end_date": ev.get("endDate"),
                "markets": brackets,
            }
        )
        for m in brackets:
            ask = m.get("bestAsk")
            try:
                live = ask is not None and 0.01 <= float(ask) <= 0.99
            except (TypeError, ValueError):
                live = False
            if not live:
                continue
            try:
                tokens = json.loads(m.get("clobTokenIds") or "[]")
            except json.JSONDecodeError:
                tokens = []
            if not tokens:
                continue
            book = fetch_json(f"{CLOB_BASE}/book", {"token_id": tokens[0]})
            if book and not book.get("error"):
                rows.append(
                    {
                        "at": at, "venue": "polymarket", "kind": "depth",
                        "slug": ev.get("slug"), "market_id": m.get("id"),
                        "question": m.get("question"), "token_id": tokens[0],
                        "book": book,
                    }
                )
    day = _utc_now().date().isoformat()
    return _append(out_dir / f"polymarket_book_{day}.jsonl", rows)


# What defines a Polymarket event and how it resolved — never its prices or
# volumes, which the hourly snapshot already captures and which would
# otherwise re-record every event every day.
_PM_EVENT_STABLE = ("id", "slug", "title", "startDate", "endDate", "closed", "closedTime")
_PM_BRACKET_STABLE = (
    "id", "question", "groupItemTitle", "slug", "outcomes", "clobTokenIds",
    "conditionId", "closed", "closedTime", "endDate", "umaResolutionStatus",
)


def _pm_stable(ev: dict) -> dict:
    out = {k: ev.get(k) for k in _PM_EVENT_STABLE}
    out["markets"] = [
        {k: m.get(k) for k in _PM_BRACKET_STABLE} for m in ev.get("markets") or []
    ]
    return out


def record_polymarket_markets(out_dir: Path) -> list[dict]:
    """Event and bracket metadata, open and closed, appended on lifecycle change."""
    path = out_dir / "polymarket_markets.jsonl"
    known = _scan(path, lambda r: r.get("event_id"))
    at = _now_iso()
    events = _polymarket_events(closed=None)
    rows = []
    for ev in events:
        stable = _pm_stable(ev)
        fp = _sha(stable)
        prev = known.get(ev.get("id"))
        if prev is None or prev.get("fp") != fp:
            rows.append(
                {"at": at, "venue": "polymarket", "kind": "event",
                 "event_id": ev.get("id"), "slug": ev.get("slug"), "fp": fp,
                 "event": stable}
            )
    _append(path, rows)
    log.info("polymarket events: %d tracked, %d changed", len(events), len(rows))
    return events


def record_polymarket_prices(out_dir: Path, events: list[dict]) -> int:
    """Price history per bracket token while the market is alive.

    Polymarket's history endpoint answered empty for a long-closed market when
    this was built, so waiting until settlement to collect it would collect
    nothing. Pulled daily for open events; rows dedupe on (token, t).
    """
    path = out_dir / "polymarket_prices.jsonl"
    seen = set()
    for row in _iter_jsonl([path] if path.exists() else []):
        tok = row.get("token_id")
        for t, _ in row.get("points") or []:
            seen.add((tok, t))
    written = 0
    at = _now_iso()
    for ev in events:
        if ev.get("closed"):
            continue
        for m in ev.get("markets") or []:
            try:
                tokens = json.loads(m.get("clobTokenIds") or "[]")
            except json.JSONDecodeError:
                tokens = []
            if not tokens:
                continue
            payload = fetch_json(
                f"{CLOB_BASE}/prices-history",
                # interval=max with minute fidelity: the endpoint 400s on some
                # interval/fidelity combinations, and this one is verified.
                {"market": tokens[0], "interval": "max", "fidelity": "1"},
            ) or {}
            # One line per pull, points batched as [t, p] pairs: the 80-char
            # token id belongs to the line, not to every minute of it.
            points = []
            for pt in payload.get("history") or []:
                key = (tokens[0], pt.get("t"))
                if key in seen:
                    continue
                seen.add(key)
                points.append([pt.get("t"), pt.get("p")])
            if points:
                _append(path, [{
                    "at": at, "venue": "polymarket", "kind": "prices",
                    "token_id": tokens[0], "market_id": m.get("id"),
                    "slug": ev.get("slug"), "points": points,
                }])
                written += len(points)
    log.info("polymarket prices: %d new points", written)
    return written


# ------------------------------------------------------------- schedule

# Presentation-only fields that change without the entry meaning anything new;
# hashing them would re-record the whole feed every time the layout shifts.
_CALENDAR_NOISE = ("newmonth", "daycount", "lastdaily")


def record_calendar(out_dir: Path) -> int:
    """Point-in-time record of the public schedule.

    Every entry is appended the first time it is seen. On the first run the
    whole archive arrives at once and is marked `retrospective`: those entries
    were downloaded today, not observed in advance, and a schedule feature fed
    to a backtest must know the difference or it is reading tomorrow's
    newspaper. From then on, daily runs append only what changed, stamped with
    the moment it was first seen — which for a "President Schedule" entry is
    normally the evening before the day it describes.
    """
    path = out_dir / "calendar.jsonl"
    first_run = not path.exists()
    known = _scan(path, lambda r: r.get("hash"))
    feed = fetch_json(CALENDAR_URL)
    if not isinstance(feed, list):
        raise RuntimeError("calendar feed did not return a list")
    at = _now_iso()
    rows = []
    for entry in feed:
        if not isinstance(entry, dict):
            continue
        stable = {k: v for k, v in entry.items() if k not in _CALENDAR_NOISE}
        h = _sha(stable)
        if h in known:
            continue
        known[h] = True
        rows.append(
            {
                "at": at, "venue": "factbase", "kind": "calendar", "hash": h,
                "as_of_mode": "retrospective" if first_run else "live",
                "entry": stable,
            }
        )
    n = _append(path, rows)
    log.info("calendar: %d new entries%s", n, " (retrospective seed)" if first_run else "")
    return n


# ------------------------------------------------- our own side of the tape

def append_forecast_history(payload: dict, out_dir: Path | None = None) -> bool:
    """One line per pipeline pass: the forecast that stood while prices moved.

    The live register (`live.py`) keeps the FIRST forecast per (week, cut) —
    the right rule for a register, and far too coarse to line up against a
    market that reprices all day. This is the fine-grained companion: every
    rebuild appends its headline and threshold table, so any recorded market
    snapshot can be matched to the forecast that was on the page at that
    moment. Append-only, like everything here; it supersedes nothing.
    """
    head = payload.get("headline") or {}
    if not head.get("median"):
        return False
    out_dir = out_dir or RECORD_DIR
    week = payload.get("week") or {}
    line = {
        "at": payload.get("generated_at") or _now_iso(),
        "kind": "forecast",
        "week_start": week.get("week_start"),
        "cut": week.get("days_into_week"),
        "observed_so_far": week.get("week_to_date_including_today"),
        "today_so_far": (payload.get("today") or {}).get("posts_so_far"),
        "median": head.get("median"),
        "quantiles": head.get("quantiles"),
        "models_used": head.get("models_used"),
        "thresholds": [
            {"t": p.get("threshold"), "p": p.get("probability")}
            for p in payload.get("threshold_probabilities") or []
        ],
    }
    return _append(out_dir / "forecast_history.jsonl", [line]) == 1


def append_posts_ledger(df, out_dir: Path | None = None, recent_days: int = 21) -> int:
    """Post provenance the cached database cannot durably hold.

    The archive is rebuilt from the mirror after any cache eviction, and the
    rebuild loses two things the mirror cannot return: when each post was
    first seen (the evidence behind ReTruth re-dating and the ingest-lag
    measurement) and the history of deletions (the convention excludes deleted
    posts, so a deletion changes the count a market settles on). One line per
    post; a fresh line when its mutable fields change. Only the recent window
    is re-fingerprinted — older posts do not change, and new ids are caught
    regardless of the date they carry, which for a ReTruth is the original's.
    """
    out_dir = out_dir or RECORD_DIR
    path = out_dir / "posts_ledger.jsonl"
    known = {r.get("id"): r.get("fp") for r in _scan(path, lambda r: r.get("id")).values()}
    first_run = not known

    cutoff = _utc_now() - timedelta(days=recent_days)
    rows = []
    at = _now_iso()
    for r in df.itertuples():
        pid = int(r.trumpstruth_id)
        created = r.created_utc.isoformat()
        if not first_run and pid in known and r.created_utc.to_pydatetime() < cutoff:
            continue
        text = r.text or ""
        text_sha = hashlib.sha1(text.encode()).hexdigest()[:12]
        deleted = int(getattr(r, "is_deleted", 0) or 0)
        fp = _sha([created, int(r.is_retruth), deleted, text_sha])
        if known.get(pid) == fp:
            continue
        known[pid] = fp
        rows.append(
            {
                "at": at, "kind": "post", "id": pid, "fp": fp,
                "created_utc": created,
                "first_seen_utc": getattr(r, "first_seen_utc", None),
                "is_retruth": int(r.is_retruth),
                "is_deleted": deleted,
                "source_handle": getattr(r, "source_handle", None),
                "has_media": int(getattr(r, "has_media", 0) or 0),
                "text_sha1": text_sha,
                "text_len": len(text),
            }
        )
    n = _append(path, rows)
    if n:
        log.info("posts ledger: %d line(s)%s", n, " (initial seed)" if first_run else "")
    return n


# ------------------------------------------------------------- housekeeping

def rotate(out_dir: Path) -> int:
    """Gzip closed periods: book files from past days, candle and trade files
    from past months. The current period stays appendable; nothing is ever
    rewritten, only compressed once it can no longer grow."""
    today = _utc_now().date().isoformat()
    month = today[:7]
    n = 0
    for path in sorted(out_dir.glob("*_book_*.jsonl")):
        if path.stem.rsplit("_", 1)[-1] >= today:
            continue
        n += _gzip_in_place(path)
    for pattern in ("kalshi_candles_*.jsonl", "kalshi_trades_*.jsonl"):
        for path in sorted(out_dir.glob(pattern)):
            stamp = path.stem.rsplit("_", 1)[-1]
            if stamp >= month or stamp == "unknown":
                continue
            n += _gzip_in_place(path)
    if n:
        log.info("rotated %d file(s)", n)
    return n


def _gzip_in_place(path: Path) -> int:
    with path.open("rb") as src, gzip.open(f"{path}.gz", "wb", compresslevel=9) as dst:
        dst.write(src.read())
    path.unlink()
    return 1


# --------------------------------------------------------------------- CLI

def snapshot(out_dir: Path | None = None) -> None:
    """The hourly pass: order books on both venues. Each venue independently —
    one exchange's outage must not lose the other's book."""
    out_dir = out_dir or RECORD_DIR
    for step in (snapshot_kalshi, snapshot_polymarket):
        try:
            n = step(out_dir)
            log.info("%s: %d line(s)", step.__name__, n)
        except Exception as exc:  # noqa: BLE001 - the other venue must still record
            log.error("%s failed: %s", step.__name__, exc)


def daily(out_dir: Path | None = None) -> None:
    """The daily pass: everything reconstructible-today-but-maybe-not-forever,
    the schedule, and rotation. On a fresh checkout this IS the backfill —
    every incremental pull starts from each market's open when it has no state."""
    out_dir = out_dir or RECORD_DIR
    state = _load_state(out_dir)
    kalshi_markets: list[dict] = []
    try:
        kalshi_markets = record_kalshi_markets(out_dir)
    except Exception as exc:  # noqa: BLE001
        log.error("kalshi market metadata failed: %s", exc)
    try:
        record_kalshi_candles(out_dir, kalshi_markets, state)
    except Exception as exc:  # noqa: BLE001
        log.error("kalshi candles failed: %s", exc)
    try:
        record_kalshi_trades(out_dir, kalshi_markets, state)
    except Exception as exc:  # noqa: BLE001
        log.error("kalshi trades failed: %s", exc)
    pm_events: list[dict] = []
    try:
        pm_events = record_polymarket_markets(out_dir)
    except Exception as exc:  # noqa: BLE001
        log.error("polymarket metadata failed: %s", exc)
    try:
        record_polymarket_prices(out_dir, pm_events)
    except Exception as exc:  # noqa: BLE001
        log.error("polymarket prices failed: %s", exc)
    try:
        record_calendar(out_dir)
    except Exception as exc:  # noqa: BLE001
        log.error("calendar failed: %s", exc)
    try:
        rotate(out_dir)
    except Exception as exc:  # noqa: BLE001
        log.error("rotation failed: %s", exc)
    _save_state(out_dir, state)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Append-only record of markets, schedule and forecasts")
    p.add_argument("--snapshot", action="store_true", help="order books on both venues (hourly)")
    p.add_argument("--daily", action="store_true", help="candles, trades, metadata, calendar, rotation")
    p.add_argument("--dir", type=Path, default=None, help="record directory (default data/record)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.snapshot:
        snapshot(args.dir)
    if args.daily:
        daily(args.dir)
    if not (args.snapshot or args.daily):
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
