"""The market record: the properties that make it a record.

Everything here pins behaviour that, if silently lost, would corrupt the
dataset in ways nothing downstream could detect: a snapshot that overwrites,
a dedupe that misses, a calendar seed that claims to have been known in
advance, a ledger that fails to notice a deletion. No test touches the
network — `record.fetch_json` is swapped for a fake.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from truthforecast import record


def read_lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def fake_fetch(routes):
    """A fetch_json stand-in: first route whose substring matches the URL wins."""

    def fetch(url, params=None):
        for needle, payload in routes:
            if needle in url:
                return payload(url, params) if callable(payload) else payload
        raise AssertionError(f"unexpected URL in test: {url}")

    return fetch


# ------------------------------------------------------------------ kalshi

KALSHI_EVENTS = {
    "events": [
        {
            "event_ticker": "KXTRUTHSOCIAL-26AUG08",
            "markets": [
                {
                    "ticker": "KXTRUTHSOCIAL-26AUG08-B149",
                    "yes_sub_title": "140-159",
                    "rules_primary": "x" * 5000,  # prose the snapshot must drop
                    "volume": 123,
                }
            ],
        }
    ]
}
KALSHI_BOOK = {"orderbook_fp": {"no_dollars": [["0.0100", "5138.99"]], "yes_dollars": []}}


def test_kalshi_snapshot_appends_book_lines_and_drops_prose(tmp_path, monkeypatch):
    monkeypatch.setattr(
        record, "fetch_json",
        fake_fetch([("orderbook", KALSHI_BOOK), ("/events", KALSHI_EVENTS)]),
    )
    n = record.snapshot_kalshi(tmp_path)
    files = list(tmp_path.glob("kalshi_book_*.jsonl"))
    assert n == len(record.KALSHI_SERIES) and len(files) == 1
    line = read_lines(files[0])[0]
    assert line["venue"] == "kalshi" and line["kind"] == "book"
    assert line["orderbook"] == KALSHI_BOOK["orderbook_fp"]
    assert "rules_primary" not in line["market"]      # oversized prose gone
    assert line["market"]["volume"] == 123            # numbers never trimmed
    # a second snapshot appends rather than replacing: it is a record
    record.snapshot_kalshi(tmp_path)
    assert len(read_lines(files[0])) == 2 * len(record.KALSHI_SERIES)


def _candles(url, params):
    # One candle per requested chunk, keyed to the chunk start so different
    # chunks yield different rows but a re-pull of the same chunk repeats.
    start = int(params["start_ts"])
    return {"candlesticks": [{
        "end_period_ts": start + 60,
        "price": {"close_dollars": "0.1700", "open_dollars": "0.1600"},
        "yes_bid": {"close_dollars": "0.1600", "high_dollars": "0.9900"},
        "yes_ask": {"close_dollars": "0.1800", "low_dollars": "0.0100"},
        "volume_fp": "816.88", "open_interest_fp": "656.88",
    }]}


def test_candle_pull_is_incremental_deduped_and_slimmed(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "fetch_json", fake_fetch([("candlesticks", _candles)]))
    now = datetime.now(timezone.utc)
    market = {
        "ticker": "M1", "_series": "KXTRUTHSOCIAL",
        "open_time": (now - timedelta(hours=6)).isoformat(),
        "close_time": (now + timedelta(days=1)).isoformat(),
    }
    state = {}
    n1 = record.record_kalshi_candles(tmp_path, [market], state)
    assert n1 == 1  # six hours fits one chunk
    files = list(tmp_path.glob("kalshi_candles_*.jsonl"))
    assert len(files) == 1 and files[0].stem.endswith(now.strftime("%Y-%m"))
    row = read_lines(files[0])[0]
    # the slimming contract: price path, closing quotes, volume, OI — kept;
    # quote OHLC noise — gone; nothing rounded
    assert row["c"] == "0.1700" and row["bid"] == "0.1600" and row["ask"] == "0.1800"
    assert row["vol"] == "816.88" and row["oi"] == "656.88"
    assert "high_dollars" not in json.dumps(row)
    covered = state["kalshi_candles_covered_to"]["M1"]
    # second run: nothing new has elapsed beyond the covered point, and the
    # dedupe must hold even if the same chunk were re-fetched
    n2 = record.record_kalshi_candles(tmp_path, [market], state)
    assert n2 == 0
    assert state["kalshi_candles_covered_to"]["M1"] >= covered
    # losing the state must not duplicate rows, only re-fetch them
    n3 = record.record_kalshi_candles(tmp_path, [market], {})
    assert n3 == 0


TRADES_PAGE = {
    "trades": [
        {"trade_id": "t2", "created_time": "2026-08-05T04:00:00Z"},
        {"trade_id": "t1", "created_time": "2026-08-05T03:00:00Z"},
    ],
    "cursor": None,
}


def test_trades_dedupe_and_settled_markets_stop_costing_requests(tmp_path, monkeypatch):
    calls = []

    def counting(url, params=None):
        calls.append(url)
        return TRADES_PAGE

    monkeypatch.setattr(record, "fetch_json", counting)
    market = {"ticker": "M1", "status": "settled"}
    state = {}
    assert record.record_kalshi_trades(tmp_path, [market], state) == 2
    assert (tmp_path / "kalshi_trades_2026-08.jsonl").exists()  # filed by trade month
    assert record.record_kalshi_trades(tmp_path, [market], state) == 0
    # the settled market was marked done: the second run made no request
    assert len(calls) == 1


# -------------------------------------------------------------- polymarket

PM_SEARCH = {"events": [
    {"slug": "trump-truth-social-posts-aug", "title": "Donald Trump # Truth Social posts Aug 4-11?", "closed": False},
    {"slug": "unrelated", "title": "Fed decision in September?", "closed": False},
]}
PM_EVENT = [{
    "id": "9001", "slug": "trump-truth-social-posts-aug",
    "title": "Donald Trump # Truth Social posts Aug 4-11?", "closed": False,
    "markets": [
        {"id": "1", "question": "100-124?", "bestBid": 0.3, "bestAsk": "0.35",
         "clobTokenIds": json.dumps(["tokLIVE", "tokNO"])},
        {"id": "2", "question": "0-19?", "bestBid": None, "bestAsk": "0.001",
         "clobTokenIds": json.dumps(["tokDEAD", "tokNO2"])},
    ],
}]
PM_BOOK = {"bids": [{"price": "0.30", "size": "100"}], "asks": [{"price": "0.35", "size": "50"}]}


def test_polymarket_snapshot_records_event_and_depth_for_live_brackets_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        record, "fetch_json",
        fake_fetch([("public-search", PM_SEARCH), ("/events", PM_EVENT), ("/book", PM_BOOK)]),
    )
    record.snapshot_polymarket(tmp_path)
    lines = read_lines(next(tmp_path.glob("polymarket_book_*.jsonl")))
    kinds = [x["kind"] for x in lines]
    assert kinds == ["book", "depth"]                 # unrelated event filtered out
    assert lines[1]["token_id"] == "tokLIVE"          # dead bracket got no depth call
    assert lines[0]["markets"][0]["bestAsk"] == "0.35"


# ---------------------------------------------------------------- calendar

CAL_V1 = [
    {"date": "2026-08-05", "type": "President Schedule", "details": "Meeting",
     "newmonth": True, "daycount": 11},
    {"date": "2026-08-04", "type": "Pool Report Schedule", "details": "Lid called",
     "newmonth": False, "daycount": 10},
]


def test_calendar_first_run_is_retrospective_then_point_in_time(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "fetch_json", fake_fetch([("calendar", CAL_V1)]))
    assert record.record_calendar(tmp_path) == 2
    lines = read_lines(tmp_path / "calendar.jsonl")
    assert all(x["as_of_mode"] == "retrospective" for x in lines)

    # same feed with only presentation noise changed: nothing new to record
    noisy = [dict(e, daycount=99) for e in CAL_V1]
    monkeypatch.setattr(record, "fetch_json", fake_fetch([("calendar", noisy)]))
    assert record.record_calendar(tmp_path) == 0

    # a real revision arrives: exactly one line, and it is marked live
    revised = CAL_V1 + [{"date": "2026-08-06", "type": "President Schedule", "details": "Travel"}]
    monkeypatch.setattr(record, "fetch_json", fake_fetch([("calendar", revised)]))
    assert record.record_calendar(tmp_path) == 1
    assert read_lines(tmp_path / "calendar.jsonl")[-1]["as_of_mode"] == "live"


# ----------------------------------------------------- forecasts and posts

def test_forecast_history_appends_every_pass_and_skips_empty(tmp_path):
    payload = {
        "generated_at": "2026-08-05T06:00:00+00:00",
        "headline": {"median": 148.6, "quantiles": {"0.5": 148.6}, "models_used": ["m"]},
        "week": {"week_start": "2026-08-02", "days_into_week": 3,
                 "week_to_date_including_today": 58},
        "today": {"posts_so_far": 4},
        "threshold_probabilities": [{"threshold": 201.4, "probability": 0.086}],
    }
    assert record.append_forecast_history(payload, tmp_path)
    assert record.append_forecast_history(payload, tmp_path)  # every pass, no dedupe
    lines = read_lines(tmp_path / "forecast_history.jsonl")
    assert len(lines) == 2
    assert lines[0]["thresholds"] == [{"t": 201.4, "p": 0.086}]
    assert lines[0]["observed_so_far"] == 58
    assert not record.append_forecast_history({"headline": {}}, tmp_path)


def _frame(rows):
    df = pd.DataFrame(rows)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df


def test_posts_ledger_seeds_then_records_only_changes(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    df = _frame([
        {"trumpstruth_id": 1, "created_utc": old, "text": "a", "is_retruth": 0,
         "is_deleted": 0, "first_seen_utc": old, "source_handle": None, "has_media": 0},
        {"trumpstruth_id": 2, "created_utc": recent, "text": "b", "is_retruth": 0,
         "is_deleted": 0, "first_seen_utc": recent, "source_handle": None, "has_media": 0},
    ])
    assert record.append_posts_ledger(df, tmp_path) == 2       # initial seed: everything
    assert record.append_posts_ledger(df, tmp_path) == 0       # unchanged: nothing

    # a deletion inside the recent window is one new line for that post only
    df.loc[df["trumpstruth_id"] == 2, "is_deleted"] = 1
    assert record.append_posts_ledger(df, tmp_path) == 1
    last = read_lines(tmp_path / "posts_ledger.jsonl")[-1]
    assert last["id"] == 2 and last["is_deleted"] == 1

    # a new ReTruth carries the ORIGINAL's old date; it must be caught by id,
    # not by falling inside the recent window
    df2 = pd.concat([df, _frame([
        {"trumpstruth_id": 3, "created_utc": old, "text": "rt", "is_retruth": 1,
         "is_deleted": 0, "first_seen_utc": recent, "source_handle": "x", "has_media": 0},
    ])], ignore_index=True)
    assert record.append_posts_ledger(df2, tmp_path) == 1
    assert read_lines(tmp_path / "posts_ledger.jsonl")[-1]["id"] == 3


# ------------------------------------------------------------ housekeeping

def test_polymarket_prices_batch_points_and_dedupe_across_pulls(tmp_path, monkeypatch):
    history = {"history": [{"t": 100, "p": 0.3}, {"t": 160, "p": 0.31}]}
    monkeypatch.setattr(record, "fetch_json", fake_fetch([("prices-history", history)]))
    assert record.record_polymarket_prices(tmp_path, PM_EVENT) == 2 * len(PM_EVENT[0]["markets"])
    lines = read_lines(tmp_path / "polymarket_prices.jsonl")
    assert lines[0]["points"] == [[100, 0.3], [160, 0.31]]
    # the same history again: every point already recorded, nothing appended
    assert record.record_polymarket_prices(tmp_path, PM_EVENT) == 0
    # a new point arrives: only it is appended
    history["history"].append({"t": 220, "p": 0.4})
    assert record.record_polymarket_prices(tmp_path, PM_EVENT) == 1 * len(PM_EVENT[0]["markets"])


def test_metadata_ignores_price_churn_but_records_lifecycle(tmp_path, monkeypatch):
    def routes(ev):
        return fake_fetch([("public-search", {"events": [
            {"slug": ev["slug"], "title": ev["title"], "closed": ev["closed"]}]}),
            ("/events", [ev])])

    ev = dict(PM_EVENT[0])
    monkeypatch.setattr(record, "fetch_json", routes(ev))
    record.record_polymarket_markets(tmp_path)
    assert len(read_lines(tmp_path / "polymarket_markets.jsonl")) == 1

    # volume and prices churn daily; that is the snapshot's job, not this file's
    churned = dict(ev)
    churned["markets"] = [dict(m, bestBid=0.9, volume="99999") for m in ev["markets"]]
    monkeypatch.setattr(record, "fetch_json", routes(churned))
    record.record_polymarket_markets(tmp_path)
    assert len(read_lines(tmp_path / "polymarket_markets.jsonl")) == 1

    # resolution is lifecycle: exactly one new line
    settled = dict(churned, closed=True)
    monkeypatch.setattr(record, "fetch_json", routes(settled))
    record.record_polymarket_markets(tmp_path)
    lines = read_lines(tmp_path / "polymarket_markets.jsonl")
    assert len(lines) == 2 and lines[-1]["event"]["closed"] is True


def test_rotation_gzips_closed_days_and_months_only(tmp_path):
    now = datetime.now(timezone.utc)
    today, month = now.date().isoformat(), now.strftime("%Y-%m")
    old_book = tmp_path / "kalshi_book_2026-01-01.jsonl"
    cur_book = tmp_path / f"kalshi_book_{today}.jsonl"
    old_candles = tmp_path / "kalshi_candles_2026-01.jsonl"
    cur_candles = tmp_path / f"kalshi_candles_{month}.jsonl"
    for p in (old_book, cur_book, old_candles, cur_candles):
        p.write_text('{"a":1}\n')
    assert record.rotate(tmp_path) == 2
    assert not old_book.exists() and cur_book.exists()
    assert not old_candles.exists() and cur_candles.exists()
    with gzip.open(f"{old_book}.gz") as fh:
        assert json.loads(fh.read()) == {"a": 1}


def test_candle_dedupe_survives_rotation(tmp_path, monkeypatch):
    """A row inside a gzipped month must still stop its own re-recording."""
    monkeypatch.setattr(record, "fetch_json", fake_fetch([("candlesticks", _candles)]))
    now = datetime.now(timezone.utc)
    market = {
        "ticker": "M1", "_series": "KXTRUTHSOCIAL",
        "open_time": (now - timedelta(hours=6)).isoformat(),
        "close_time": (now + timedelta(days=1)).isoformat(),
    }
    assert record.record_kalshi_candles(tmp_path, [market], {}) == 1
    path = next(tmp_path.glob("kalshi_candles_*.jsonl"))
    with path.open("rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
        dst.write(src.read())
    path.unlink()
    assert record.record_kalshi_candles(tmp_path, [market], {}) == 0


def test_snapshot_survives_one_venue_failing(tmp_path, monkeypatch):
    def broken_kalshi(url, params=None):
        if "kalshi" in url:
            raise RuntimeError("exchange down")
        return fake_fetch([("public-search", PM_SEARCH), ("/events", PM_EVENT),
                           ("/book", PM_BOOK)])(url, params)

    monkeypatch.setattr(record, "fetch_json", broken_kalshi)
    record.snapshot(tmp_path)  # must not raise
    assert list(tmp_path.glob("polymarket_book_*.jsonl"))     # the healthy venue recorded
    assert not list(tmp_path.glob("kalshi_book_*.jsonl"))
