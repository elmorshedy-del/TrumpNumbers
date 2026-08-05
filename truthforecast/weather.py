"""Settled-fact alerts for Kalshi's daily high-temperature markets.

The trade this watches for is not a forecast. A daily maximum can never go
back down, so the moment a settlement station's observed max-so-far passes a
bracket's ceiling, that bracket is dead — the only question left is whether
the order book has noticed. When it hasn't, the dead side still trades at a
price, and buying the certain side is not a bet but a fact the market hasn't
read yet. Observed in the wild: a Philadelphia 89–90° bracket still offering
NO at 33 cents more than an hour after the station had recorded 91°F.

Three rules keep this honest:

* **Only irreversible facts trigger.** "Max-so-far is past the ceiling" can
  never un-happen; "it's 88° and climbing" is a forecast wearing a fact's
  clothes and does not alert, however tempting the extrapolation.
* **The trigger is conservative, the message is complete.** NWS reports
  Celsius; settlement is whole Fahrenheit. Fahrenheit is *floored* before
  comparing, so rounding can only suppress an alert, never invent one — and
  the alert carries the raw reading and its timestamp so the human can check
  the margin before trading. A one-degree margin is flagged as such.
* **Alert once per fact.** The state file remembers (market, side); a stale
  book that stays stale does not page you every five minutes.

Everything configures itself from the exchanges' own metadata: the series
list comes from Kalshi's climate category, the settlement station is parsed
from each series' NWS Climatological Report URL (``issuedby=PHL`` -> KPHL),
and the station's timezone comes from the NWS API. A city Kalshi adds next
month is covered without a code change.

Alerts go to whatever channels `notify.py` has configured — Telegram, ntfy,
Discord, Slack, or a plain webhook. Nothing here places orders; the alert is
evidence, the decision stays yours.

    python -m truthforecast.weather --scan --dry-run   # look, print, send nothing
    python -m truthforecast.weather --scan             # one pass, alert, remember
    python -m truthforecast.weather --watch            # poll every TTF_POLL_SECONDS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import DATA_DIR
from .record import KALSHI_BASE, fetch_json

log = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
STATE_PATH = Path(os.environ.get("TTF_WX_STATE", DATA_DIR / "weather_state.json"))

# The certain side must cost no more than this to be worth an alert. At 0.90
# the gross edge is ten cents against a fee of about half a cent out at that
# price; anything tighter is real money only at size this book does not have.
MAX_PRICE = float(os.environ.get("TTF_WX_MAX_PRICE", "0.90"))

_MONTHS = {m: i + 1 for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


# ------------------------------------------------------------ discovery

def discover_series() -> list[dict]:
    """Every Kalshi daily-high series that settles on an NWS climate report.

    The station is parsed from the settlement source URL itself — the same
    document the market resolves on — rather than from a hand-kept city map
    that would rot the day Kalshi adds a city.
    """
    payload = fetch_json(f"{KALSHI_BASE}/series", {"category": "Climate and Weather"}) or {}
    out = []
    for s in payload.get("series") or []:
        if not (s.get("ticker") or "").startswith("KXHIGH"):
            continue
        for src in s.get("settlement_sources") or []:
            m = re.search(r"product=CLI.*?issuedby=([A-Z0-9]{3,4})", src.get("url") or "")
            if m:
                out.append({
                    "series": s["ticker"],
                    "title": s.get("title"),
                    "station": f"K{m.group(1)}",
                    "cli_url": src.get("url"),
                })
                break
    return out


def station_timezone(station: str, state: dict) -> str | None:
    cache = state.setdefault("station_tz", {})
    if station not in cache:
        meta = fetch_json(f"{NWS_BASE}/stations/{station}") or {}
        cache[station] = (meta.get("properties") or {}).get("timeZone")
    return cache[station]


# ---------------------------------------------------------- observations

def _floor_f(celsius: float) -> int:
    """Whole Fahrenheit, floored — the conservative side of every rounding.

    32.2 C is 89.96 F: calling it 90 would let a conversion artefact claim a
    bracket was eliminated when the climate report may yet say 89. Flooring
    means the trigger can only understate the max, so a false alert would
    need the station itself to be wrong, not our arithmetic.
    """
    return int(celsius * 9 / 5 + 32 + 1e-9)


def station_max(station: str, day: date, tz: str) -> dict | None:
    """The station's observed maximum so far for `day`, in its own clock.

    Uses every temperature reading since local midnight plus any six-hour
    maximum groups whose window sits entirely inside the day — those catch a
    spike the five-minute feed missed, which is exactly what happened in the
    Philadelphia case (max-so-far 91°F against an ambient reading of 88°F).
    """
    zone = ZoneInfo(tz)
    midnight = datetime(day.year, day.month, day.day, tzinfo=zone)
    payload = fetch_json(
        f"{NWS_BASE}/stations/{station}/observations",
        {"start": midnight.isoformat(), "limit": "500"},
    ) or {}
    best: dict | None = None
    for obs in payload.get("features") or []:
        p = obs.get("properties") or {}
        stamp = p.get("timestamp")
        try:
            at = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(zone)
        except (AttributeError, ValueError):
            continue
        if at.date() != day:
            continue
        candidates = []
        t = (p.get("temperature") or {}).get("value")
        if t is not None:
            candidates.append((float(t), "reading"))
        m6 = (p.get("maxTemperatureLast6Hours") or {}).get("value")
        if m6 is not None and at - timedelta(hours=6) >= midnight:
            candidates.append((float(m6), "6h-max"))
        for c, kind in candidates:
            if best is None or c > best["c"]:
                best = {"c": c, "f": _floor_f(c), "at": at.isoformat(), "kind": kind}
    return best


# ------------------------------------------------------------ evaluation

def certain_side(market: dict, max_f: int) -> str | None:
    """Which side is already settled fact, given the observed max so far.

    A maximum only rises, so certainty flows one way: brackets *below* the
    observed max are dead (NO certain), thresholds *at or below* it are
    guaranteed (YES certain). Nothing above the observed max is ever certain
    — that would be forecasting, which this module refuses to do.
    """
    kind = market.get("strike_type")
    cap = market.get("cap_strike")
    floor = market.get("floor_strike")
    if kind == "less" and cap is not None and max_f >= float(cap):
        return "no"            # "X° or below" needs max < cap; it already isn't
    if kind == "between" and cap is not None and max_f >= float(cap) + 1:
        return "no"            # bracket ceiling already exceeded
    if kind == "greater" and floor is not None and max_f >= float(floor) + 1:
        return "yes"           # "above X°" already recorded
    return None


def executable(market: dict, side: str) -> float | None:
    """What the certain side costs right now, from the quotes on the market.

    Buying NO fills against the best YES bid, so its cost is 1 minus that
    bid; buying YES lifts the YES ask. No quote on the relevant side means
    no executable edge, whatever theoretical value is going unclaimed.
    """
    try:
        if side == "no":
            bid = float(market.get("yes_bid_dollars") or 0)
            return round(1.0 - bid, 4) if bid > 0 else None
        ask = float(market.get("yes_ask_dollars") or 0)
        return ask if ask > 0 else None
    except (TypeError, ValueError):
        return None


def _event_day(event_ticker: str) -> date | None:
    """KXHIGHPHIL-26AUG05 -> 2026-08-05."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


# ------------------------------------------------------------------ scan

def scan(state: dict, max_price: float = MAX_PRICE, cities: str = "") -> list[dict]:
    """One pass over every city: observations, then markets, then facts."""
    alerts = []
    for entry in discover_series():
        if cities and cities.lower() not in entry["series"].lower():
            continue
        try:
            tz = station_timezone(entry["station"], state)
            if not tz:
                log.warning("%s: no timezone for %s, skipped", entry["series"], entry["station"])
                continue
            today = datetime.now(ZoneInfo(tz)).date()
            events = fetch_json(
                f"{KALSHI_BASE}/events",
                {"series_ticker": entry["series"], "status": "open",
                 "with_nested_markets": "true"},
            ) or {}
            todays = [
                ev for ev in events.get("events") or []
                if _event_day(ev.get("event_ticker")) == today
            ]
            if not todays:
                continue
            obs = station_max(entry["station"], today, tz)
            if obs is None:
                continue
            for ev in todays:
                for m in ev.get("markets") or []:
                    side = certain_side(m, obs["f"])
                    if side is None:
                        continue
                    cost = executable(m, side)
                    if cost is None or cost > max_price:
                        continue
                    key = f"{m.get('ticker')}:{side}"
                    if key in (state.get("alerted") or {}):
                        continue
                    margin = obs["f"] - _needed_f(m, side)
                    alerts.append({
                        "key": key,
                        "series": entry["series"],
                        "station": entry["station"],
                        "event": ev.get("event_ticker"),
                        "ticker": m.get("ticker"),
                        "bracket": m.get("yes_sub_title") or m.get("subtitle"),
                        "side": side,
                        "cost": cost,
                        "edge": round(1.0 - cost, 4),
                        "max_f": obs["f"],
                        "obs_c": obs["c"],
                        "obs_at": obs["at"],
                        "obs_kind": obs["kind"],
                        "margin_f": margin,
                        "url": f"https://kalshi.com/markets/{entry['series'].lower()}",
                    })
        except Exception as exc:  # noqa: BLE001 - one city must not lose the rest
            log.error("%s failed: %s", entry["series"], exc)
    return alerts


def _needed_f(market: dict, side: str) -> int:
    """The observed max at which this side became certain — for the margin."""
    kind = market.get("strike_type")
    if kind == "less":
        return int(float(market.get("cap_strike")))
    if kind == "between":
        return int(float(market.get("cap_strike"))) + 1
    return int(float(market.get("floor_strike"))) + 1


# ------------------------------------------------------------------ alert

def format_alerts(alerts: list[dict]) -> tuple[str, str]:
    n = len(alerts)
    title = "Kalshi temp: settled-fact mispricing" + ("" if n == 1 else f" — {n} found")
    lines = []
    for a in alerts:
        tight = "  ⚠ 1°F margin — verify before trading" if a["margin_f"] <= 0 else ""
        lines.append(
            f"{a['series'].removeprefix('KXHIGH')} {a['bracket']}: "
            f"buy {a['side'].upper()} @ {a['cost']:.2f} (edge {a['edge']:.2f})\n"
            f"{a['station']} max-so-far {a['max_f']}°F "
            f"({a['obs_c']:.1f}°C {a['obs_kind']} at {a['obs_at']}), "
            f"margin {a['margin_f']}°F{tight}\n{a['url']}"
        )
    return title, "\n\n".join(lines)


def send(alerts: list[dict]) -> bool:
    from . import notify  # imported here so --dry-run needs no requests install

    title, body = format_alerts(alerts)
    payload = {"kind": "weather-edge", "count": len(alerts), "alerts": alerts}
    results = [fn(title, body, payload) for fn in notify.CHANNELS]
    sent = [r for r in results if r is not None]
    if not sent:
        log.warning("weather: %d alert(s) but no channel configured", len(alerts))
        return False
    return all(sent)


# ------------------------------------------------------------------ state

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            log.warning("weather: unreadable state, starting fresh (worst case: one duplicate alert)")
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1, sort_keys=True))


def run_once(dry_run: bool = False, max_price: float = MAX_PRICE, cities: str = "") -> int:
    state = load_state()
    alerts = scan(state, max_price=max_price, cities=cities)
    if not alerts:
        log.info("weather: no settled-fact edges right now")
        save_state(state)  # keeps the timezone cache warm even on quiet passes
        return 0
    if dry_run:
        title, body = format_alerts(alerts)
        print(f"{title}\n\n{body}")
        save_state(state)
        return len(alerts)
    ok = send(alerts)
    if ok:
        marked = state.setdefault("alerted", {})
        for a in alerts:
            marked[a["key"]] = datetime.now().isoformat()
        log.info("weather: sent %d alert(s)", len(alerts))
    else:
        # State stays put: the next pass retries, and a duplicate on the
        # channel that worked is the acceptable half of that trade.
        log.warning("weather: send failed, will retry next pass")
    save_state(state)
    return len(alerts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Settled-fact alerts for Kalshi temperature markets")
    p.add_argument("--scan", action="store_true", help="one pass over every city")
    p.add_argument("--watch", action="store_true", help="loop forever (TTF_POLL_SECONDS, default 120)")
    p.add_argument("--dry-run", action="store_true", help="print instead of sending")
    p.add_argument("--max-price", type=float, default=MAX_PRICE,
                   help="alert only when the certain side costs at most this (default %(default)s)")
    p.add_argument("--cities", default="", help="substring filter on the series ticker, e.g. PHIL")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.watch:
        every = int(os.environ.get("TTF_POLL_SECONDS", "120"))
        while True:
            run_once(dry_run=args.dry_run, max_price=args.max_price, cities=args.cities)
            time.sleep(every)
    else:
        run_once(dry_run=args.dry_run, max_price=args.max_price, cities=args.cities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
