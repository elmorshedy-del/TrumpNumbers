"""The weather alerter: the properties that keep it a fact-reader.

The failure modes worth pinning are the ones that would make it either lie
(alert on a forecast dressed as a fact, or on a rounding artefact) or spam
(re-alert on the same stale book every five minutes). No network: the fetch
layer is swapped for fakes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from truthforecast import weather


# ------------------------------------------------------------- conversion

def test_fahrenheit_is_floored_so_rounding_cannot_invent_a_fact():
    assert weather._floor_f(32.8) == 91    # 91.04 -> 91: the Philadelphia reading
    assert weather._floor_f(32.2) == 89    # 89.96 -> 89, NOT 90: understate, never overstate
    assert weather._floor_f(30.0) == 86    # exact 86.0


# ------------------------------------------------------------- certainty

BETWEEN = {"strike_type": "between", "floor_strike": 89, "cap_strike": 90}
LESS = {"strike_type": "less", "cap_strike": 82}
GREATER = {"strike_type": "greater", "floor_strike": 89}


def test_certainty_flows_only_upward_from_the_recorded_max():
    # between 89-90: dead only once 91 is on the books
    assert weather.certain_side(BETWEEN, 91) == "no"
    assert weather.certain_side(BETWEEN, 90) is None   # still reachable: not a fact
    assert weather.certain_side(BETWEEN, 88) is None   # below the bracket is NEVER certain
    # "81 or below" (cap 82): dead the moment 82 is recorded
    assert weather.certain_side(LESS, 82) == "no"
    assert weather.certain_side(LESS, 81) is None
    # "90 or above" (floor 89): guaranteed once 90 is recorded
    assert weather.certain_side(GREATER, 90) == "yes"
    assert weather.certain_side(GREATER, 89) is None


def test_executable_prices_come_from_the_correct_side_of_the_book():
    m = dict(BETWEEN, yes_bid_dollars="0.6700", yes_ask_dollars="0.6900")
    assert weather.executable(m, "no") == 0.33         # NO fills against the YES bid
    assert weather.executable(m, "yes") == 0.69
    assert weather.executable(dict(BETWEEN, yes_bid_dollars=None), "no") is None


# ------------------------------------------------------------ observation

def _obs_payload(day_iso, entries):
    feats = []
    for hour, temp_c, max6 in entries:
        feats.append({"properties": {
            "timestamp": f"{day_iso}T{hour:02d}:54:00+00:00",
            "temperature": {"value": temp_c},
            "maxTemperatureLast6Hours": {"value": max6},
        }})
    return {"features": feats}


def test_station_max_uses_six_hour_groups_but_not_windows_crossing_midnight(monkeypatch):
    day = datetime.now(timezone.utc).date()
    # 04:54 six-hour max reaches back before local midnight (UTC station) and
    # must be ignored; the 18:54 one is inside the day and must win over the
    # ambient readings — that gap IS the Philadelphia signal.
    payload = _obs_payload(day.isoformat(), [
        (4, 20.0, 25.0),      # 6h window starts 22:54 yesterday: excluded
        (16, 31.1, None),     # 88 F ambient
        (18, 30.0, 32.8),     # 6h max 91 F: the recorded fact
    ])
    monkeypatch.setattr(weather, "fetch_json", lambda url, params=None: payload)
    best = weather.station_max("KPHL", day, "UTC")
    assert best["f"] == 91 and best["kind"] == "6h-max"

    without_6h = _obs_payload(day.isoformat(), [(16, 31.1, None)])
    monkeypatch.setattr(weather, "fetch_json", lambda url, params=None: without_6h)
    assert weather.station_max("KPHL", day, "UTC")["f"] == 87  # 31.1C -> 87.98 -> 87


# ------------------------------------------------------------------ scan

def _routes(day):
    stamp = day.strftime("%y%b%d").upper()
    series = {"series": [{
        "ticker": "KXHIGHPHIL", "title": "Highest temperature in Philadelphia",
        "settlement_sources": [{"url":
            "https://forecast.weather.gov/product.php?site=PHI&product=CLI&issuedby=PHL"}],
    }]}
    events = {"events": [{
        "event_ticker": f"KXHIGHPHIL-{stamp}",
        "markets": [
            {"ticker": f"KXHIGHPHIL-{stamp}-B89.5", "strike_type": "between",
             "floor_strike": 89, "cap_strike": 90, "yes_sub_title": "89° to 90°",
             "yes_bid_dollars": "0.6700", "yes_ask_dollars": "0.6900"},
            {"ticker": f"KXHIGHPHIL-{stamp}-B91.5", "strike_type": "between",
             "floor_strike": 91, "cap_strike": 92, "yes_sub_title": "91° to 92°",
             "yes_bid_dollars": "0.4000", "yes_ask_dollars": "0.4500"},
        ],
    }]}
    obs = _obs_payload(day.isoformat(), [(18, 30.0, 32.8)])   # max-so-far 91 F
    station = {"properties": {"timeZone": "UTC"}}

    def fetch(url, params=None):
        if url.endswith("/series"):
            return series
        if url.endswith("/events"):
            return events
        if "/observations" in url:
            return obs
        if "/stations/" in url:
            return station
        raise AssertionError(f"unexpected URL: {url}")

    return fetch


def test_scan_alerts_once_on_the_dead_bracket_and_never_on_the_live_one(monkeypatch):
    day = datetime.now(timezone.utc).date()
    monkeypatch.setattr(weather, "fetch_json", _routes(day))
    state = {}
    alerts = weather.scan(state)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["ticker"].endswith("B89.5") and a["side"] == "no"
    assert a["cost"] == 0.33 and a["max_f"] == 91 and a["margin_f"] == 0
    # 91-92 is where the max SITS: still live, never alerted — that would be a forecast

    # once sent, the same fact must not page again
    state.setdefault("alerted", {})[a["key"]] = "sent"
    assert weather.scan(state) == []


def test_scan_respects_the_price_ceiling(monkeypatch):
    day = datetime.now(timezone.utc).date()
    monkeypatch.setattr(weather, "fetch_json", _routes(day))
    # NO costs 0.33; a ceiling below that means no alert, however dead the bracket
    assert weather.scan({}, max_price=0.20) == []


def test_scan_ignores_events_for_other_days(monkeypatch):
    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
    monkeypatch.setattr(weather, "fetch_json", _routes(tomorrow))
    assert weather.scan({}) == []


def test_alert_message_carries_the_evidence_and_flags_tight_margins(monkeypatch):
    day = datetime.now(timezone.utc).date()
    monkeypatch.setattr(weather, "fetch_json", _routes(day))
    alerts = weather.scan({})
    title, body = weather.format_alerts(alerts)
    assert "settled-fact" in title
    assert "buy NO @ 0.33" in body
    assert "max-so-far 91°F" in body and "32.8°C" in body   # raw reading, checkable
    assert "1°F margin" in body                             # margin 0 flagged tight
    assert "kalshi.com/markets/kxhighphil" in body


def test_event_day_parser():
    assert weather._event_day("KXHIGHPHIL-26AUG05") == datetime(2026, 8, 5).date()
    assert weather._event_day("KXHIGHNY-27JAN31") == datetime(2027, 1, 31).date()
    assert weather._event_day("garbage") is None
