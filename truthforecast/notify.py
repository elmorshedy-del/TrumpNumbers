"""Push new posts to wherever you actually look.

Keyed on `trumpstruth_id`, which is the mirror's own ingest order, not the post
timestamp. That matters here specifically: a ReTruth is stored under the
*original* author's date, so a timestamp watermark would either miss it or
replay months of history. Ingest order is monotonic and says exactly what this
needs to know — "new to the mirror since last time".

Two deliberate choices about failure:

* **Cold start sends nothing.** On the first run the watermark is set to
  whatever is already stored and no notifications go out. Otherwise a fresh
  database would page you a hundred times about last week.
* **A failed send does not advance the watermark.** The next run retries, which
  can duplicate a notification on a channel that did work. For a notifier that
  is the right bias: twice is annoying, never is useless.

Configure by environment; anything unset is simply skipped.

    TTF_NTFY_TOPIC        ntfy.sh topic (or a full URL to a self-hosted server)
    TTF_TELEGRAM_TOKEN    bot token, with TTF_TELEGRAM_CHAT_ID
    TTF_DISCORD_WEBHOOK   Discord webhook URL
    TTF_SLACK_WEBHOOK     Slack incoming-webhook URL
    TTF_WEBHOOK_URL       anything else: receives the raw JSON payload
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

from .config import LOCAL_TZ, REQUEST_TIMEOUT_S
from .ingest.store import connect, get_state, set_state

log = logging.getLogger(__name__)

# A burst is the thing this project exists to measure, and it is also the thing
# that will empty your phone battery. Past this many, one summary line instead.
MAX_INDIVIDUAL = int(os.environ.get("TTF_NOTIFY_MAX_PER_RUN", "5"))

STATE_KEY = "notified_max_id"


def _pending(conn, since_id: int) -> list:
    return conn.execute(
        """
        SELECT trumpstruth_id, created_utc, text, is_retruth, source_handle, truth_url
        FROM posts WHERE trumpstruth_id > ? AND is_deleted = 0
        ORDER BY trumpstruth_id
        """,
        (since_id,),
    ).fetchall()


def _local(created_utc: str) -> str:
    try:
        stamp = datetime.fromisoformat(created_utc)
    except ValueError:
        return created_utc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(LOCAL_TZ).strftime("%a %H:%M")


def _url(row) -> str:
    return row["truth_url"] or f"https://trumpstruth.org/statuses/{row['trumpstruth_id']}"


def format_message(rows: list) -> tuple[str, str]:
    """(title, body) for a batch of new posts."""
    n = len(rows)
    title = "Trump posted" if n == 1 else f"Trump posted — {n} new"

    lines = []
    for row in rows[:MAX_INDIVIDUAL]:
        # The stored timestamp of a ReTruth belongs to the ORIGINAL author, so
        # an alert firing now can carry a date from months ago. Saying which is
        # which costs four words and stops the timestamp reading as lag.
        if row["is_retruth"]:
            stamp = f"[reshared now · original {_local(row['created_utc'])}]"
            tag = f"ReTruth {row['source_handle'] or ''} "
        else:
            stamp, tag = f"[{_local(row['created_utc'])}]", ""
        text = (row["text"] or "").strip() or "(media only)"
        if len(text) > 240:
            text = text[:237] + "…"
        lines.append(f"{stamp} {tag}{text}\n{_url(row)}")

    if n > MAX_INDIVIDUAL:
        lines.append(f"…and {n - MAX_INDIVIDUAL} more in this burst.")
    return title, "\n\n".join(lines)


# --- channels ---------------------------------------------------------------
# Each returns True on success, False on failure, or None when not configured.


def _post(url: str, **kwargs) -> bool:
    try:
        res = requests.post(url, timeout=REQUEST_TIMEOUT_S, **kwargs)
        res.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 - a channel failing must not stop the rest
        log.warning("notify: %s failed: %s", url.split("/")[2] if "//" in url else url, exc)
        return False


def _ntfy(title: str, body: str, payload: dict) -> bool | None:
    topic = os.environ.get("TTF_NTFY_TOPIC", "").strip()
    if not topic:
        return None
    url = topic if topic.startswith("http") else f"https://ntfy.sh/{topic}"
    return _post(
        url,
        data=body.encode("utf-8"),
        headers={"Title": title, "Tags": "loudspeaker", "Click": payload["posts"][0]["url"]},
    )


def _telegram(title: str, body: str, payload: dict) -> bool | None:
    token = os.environ.get("TTF_TELEGRAM_TOKEN", "").strip()
    chat = os.environ.get("TTF_TELEGRAM_CHAT_ID", "").strip()
    if not (token and chat):
        return None
    return _post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": f"{title}\n\n{body}", "disable_web_page_preview": True},
    )


def _discord(title: str, body: str, payload: dict) -> bool | None:
    url = os.environ.get("TTF_DISCORD_WEBHOOK", "").strip()
    if not url:
        return None
    return _post(url, json={"content": f"**{title}**\n{body}"[:1900]})


def _slack(title: str, body: str, payload: dict) -> bool | None:
    url = os.environ.get("TTF_SLACK_WEBHOOK", "").strip()
    if not url:
        return None
    return _post(url, json={"text": f"*{title}*\n{body}"})


def _generic(title: str, body: str, payload: dict) -> bool | None:
    url = os.environ.get("TTF_WEBHOOK_URL", "").strip()
    if not url:
        return None
    return _post(url, json={"title": title, "body": body, **payload})


CHANNELS = (_ntfy, _telegram, _discord, _slack, _generic)


def configured() -> list[str]:
    return [
        name for name, var in (
            ("ntfy", "TTF_NTFY_TOPIC"),
            ("telegram", "TTF_TELEGRAM_TOKEN"),
            ("discord", "TTF_DISCORD_WEBHOOK"),
            ("slack", "TTF_SLACK_WEBHOOK"),
            ("webhook", "TTF_WEBHOOK_URL"),
        )
        if os.environ.get(var, "").strip()
    ]


def notify(dry_run: bool = False) -> int:
    """Send everything stored since the last successful notification."""
    with connect() as conn:
        mark = get_state(conn, STATE_KEY)
        newest = conn.execute("SELECT MAX(trumpstruth_id) m FROM posts").fetchone()["m"]
        if newest is None:
            return 0

        if mark is None:
            # First run: adopt the present rather than replaying the archive.
            set_state(conn, STATE_KEY, str(newest))
            log.info("notify: first run, watermark set at id %s — nothing sent", newest)
            return 0

        rows = _pending(conn, int(mark))
        if not rows:
            return 0

        title, body = format_message(rows)
        payload = {
            "count": len(rows),
            "posts": [
                {
                    "id": int(r["trumpstruth_id"]),
                    "at": r["created_utc"],
                    "text": r["text"],
                    "is_retruth": bool(r["is_retruth"]),
                    "url": _url(r),
                }
                for r in rows
            ],
        }

        if dry_run:
            print(f"{title}\n\n{body}")
            return len(rows)

        results = [fn(title, body, payload) for fn in CHANNELS]
        sent = [r for r in results if r is not None]
        if not sent:
            log.warning("notify: %s new post(s) but no channel configured", len(rows))
            return 0

        if all(sent):
            set_state(conn, STATE_KEY, str(rows[-1]["trumpstruth_id"]))
            log.info("notify: sent %s post(s) to %s", len(rows), ", ".join(configured()))
        else:
            # Watermark stays put: the next run retries. Duplicates on the
            # channels that worked are the acceptable half of that trade.
            log.warning("notify: partial failure, watermark held for retry")
        return len(rows)


DIRECT_STATE_KEY = "notified_direct_max_id"


def notify_direct(posts: list) -> int:
    """Alert on posts read straight from Truth Social, bypassing the mirror.

    Separate watermark, keyed on Truth Social's own status id, and nothing is
    written to the posts table. The archive is keyed on the mirror's id, so
    storing these alongside it could double-count a day — and a double-counted
    day is exactly the silent corruption the rest of this codebase exists to
    prevent. Speed is worth having; it is not worth the series.
    """
    if not posts:
        return 0
    newest = max(p.status_id for p in posts)

    with connect() as conn:
        mark = get_state(conn, DIRECT_STATE_KEY)
        if mark is None:
            set_state(conn, DIRECT_STATE_KEY, str(newest))
            log.info("notify(direct): first run, watermark at %s — nothing sent", newest)
            return 0

        fresh = sorted((p for p in posts if p.status_id > int(mark)), key=lambda p: p.status_id)
        if not fresh:
            return 0

        rows = [
            {
                "trumpstruth_id": p.status_id,
                "created_utc": p.created_utc.isoformat(),
                "text": p.text,
                "is_retruth": 0,
                "source_handle": None,
                "truth_url": p.url,
            }
            for p in fresh
        ]
        title, body = format_message(rows)
        payload = {
            "count": len(rows),
            "source": "truthsocial.com (direct)",
            "posts": [
                {"id": r["trumpstruth_id"], "at": r["created_utc"],
                 "text": r["text"], "is_retruth": False, "url": r["truth_url"]}
                for r in rows
            ],
        }

        results = [fn(title, body, payload) for fn in CHANNELS]
        sent = [r for r in results if r is not None]
        if not sent:
            log.warning("notify(direct): %s new post(s) but no channel configured", len(rows))
            return 0
        if all(sent):
            set_state(conn, DIRECT_STATE_KEY, str(fresh[-1].status_id))
            log.info("notify(direct): sent %s post(s)", len(rows))
        else:
            log.warning("notify(direct): partial failure, watermark held for retry")
        return len(rows)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Send new posts to the configured channels")
    p.add_argument("--dry-run", action="store_true", help="print instead of sending")
    p.add_argument("--reset", action="store_true",
                   help="drop the watermark; the next run adopts the present and sends nothing")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.reset:
        with connect() as conn:
            conn.execute("DELETE FROM ingest_state WHERE key = ?", (STATE_KEY,))
        log.info("notify: watermark cleared")
        return 0

    log.info("notify: channels configured: %s", ", ".join(configured()) or "none")
    notify(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
