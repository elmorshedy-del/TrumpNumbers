"""Reading Truth Social directly, for alerts only.

Every other path in this project goes through trumpstruth.org, which is a mirror
— it crawls Truth Social on its own schedule, and nothing downstream of it can
be fresher than that crawl. This module skips it.

Truth Social is a Mastodon fork, so an account exposes a standard RSS feed at
`/@user.rss`. It refuses datacenter IPs: from Google Cloud every endpoint
returns 403, which is the documented reason the archive uses the mirror at all.
From a home connection it may simply work — that is what `probe()` is for, and
it is the only way to find out, because the answer depends on where you are
rather than on anything in this repo.

**Alerts only, deliberately.** The archive is keyed on the mirror's own post id,
which a direct read does not have. Merging the two sources into one series would
risk storing the same post twice under different keys, and a double-counted day
is precisely the silent corruption the rest of this codebase is built to avoid.
So the direct path keeps its own small state, keyed on Truth Social's status id,
and never touches the modelled series.

    python -m truthforecast.ingest.direct --probe     # can this network reach it?
    python -m truthforecast.ingest.direct --watch     # poll and notify, no mirror
"""

from __future__ import annotations

import email.utils
import logging
import re
from html import unescape
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from ..config import REQUEST_TIMEOUT_S, USER_AGENT

log = logging.getLogger(__name__)

BASE = "https://truthsocial.com"
ACCOUNT = "realDonaldTrump"
FEED_URL = f"{BASE}/@{ACCOUNT}.rss"

ITEM_RE = re.compile(r"<item>(.*?)</item>", re.S)
LINK_RE = re.compile(r"<link>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</link>", re.S)
GUID_RE = re.compile(r"<guid[^>]*>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</guid>", re.S)
PUBDATE_RE = re.compile(r"<pubDate>([^<]+)</pubDate>")
DESC_RE = re.compile(r"<description>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</description>", re.S)
STATUS_ID_RE = re.compile(r"/(?:posts|statuses)?/?(\d{6,})\s*$")
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class DirectPost:
    """One post as Truth Social itself reports it."""

    status_id: int
    created_utc: datetime
    text: str
    url: str


def _strip_html(raw: str) -> str:
    """Mastodon escapes the post's HTML inside <description>, so unescape first.

    Skipping that step yields text full of `&lt;p&gt;`, which parses without
    error and looks wrong only to a human reading the alert.
    """
    text = unescape(raw or "")
    text = re.sub(r"<br\s*/?>|</p>", " ", text)
    return unescape(re.sub(r"\s+", " ", TAG_RE.sub("", text)).strip())


def parse_account_rss(xml: str) -> list[DirectPost]:
    """Parse a Mastodon-style account feed.

    Raises on markup that no longer matches, for the same reason the mirror
    parser does: a parser that silently returns nothing looks exactly like a
    person who has stopped posting.
    """
    if "<rss" not in xml and "<feed" not in xml:
        raise ValueError("response is not an RSS document")

    posts: list[DirectPost] = []
    for body in ITEM_RE.findall(xml):
        link = LINK_RE.search(body) or GUID_RE.search(body)
        pub = PUBDATE_RE.search(body)
        if not link or not pub:
            continue
        url = link.group(1).strip()
        sid = STATUS_ID_RE.search(urlparse(url).path)
        if not sid:
            continue
        created = email.utils.parsedate_to_datetime(pub.group(1).strip())
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        desc = DESC_RE.search(body)
        posts.append(
            DirectPost(
                status_id=int(sid.group(1)),
                created_utc=created.astimezone(timezone.utc),
                text=_strip_html(desc.group(1) if desc else ""),
                url=url,
            )
        )
    return posts


def fetch(session: requests.Session | None = None) -> list[DirectPost]:
    session = session or requests.Session()
    res = session.get(
        FEED_URL, timeout=REQUEST_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )
    res.raise_for_status()
    return parse_account_rss(res.text)


def robots_allows(session: requests.Session | None = None) -> tuple[bool | None, str]:
    """Ask the site whether this is welcome before doing it repeatedly.

    Returns (allowed, explanation). `None` means the question could not be
    answered — which is itself a reason not to hammer the endpoint.
    """
    session = session or requests.Session()
    try:
        res = session.get(
            f"{BASE}/robots.txt", timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"could not fetch robots.txt: {exc}"
    if res.status_code != 200:
        return None, f"robots.txt returned {res.status_code}"

    from urllib.robotparser import RobotFileParser

    rp = RobotFileParser()
    rp.parse(res.text.splitlines())
    allowed = rp.can_fetch(USER_AGENT, FEED_URL)
    return allowed, "robots.txt " + ("permits" if allowed else "disallows") + " this path"


def probe() -> dict:
    """Can this machine read Truth Social directly? Reports rather than assumes."""
    session = requests.Session()
    allowed, why = robots_allows(session)
    out = {"feed": FEED_URL, "robots_allows": allowed, "robots_note": why}

    try:
        res = session.get(
            FEED_URL, timeout=REQUEST_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
        )
        out["status_code"] = res.status_code
        if res.status_code == 200:
            posts = parse_account_rss(res.text)
            out["reachable"] = True
            out["posts_in_feed"] = len(posts)
            if posts:
                newest = max(posts, key=lambda p: p.created_utc)
                age = (datetime.now(timezone.utc) - newest.created_utc).total_seconds() / 60
                out["newest_post_utc"] = newest.created_utc.isoformat()
                out["newest_post_age_minutes"] = round(age, 1)
        else:
            out["reachable"] = False
            out["hint"] = (
                "403 here means the network is being refused, not that the feed is gone. "
                "This is expected from cloud providers and is why the archive uses the mirror. "
                "Try it from a home connection."
            )
    except Exception as exc:  # noqa: BLE001
        out["reachable"] = False
        out["error"] = str(exc)
    return out


def main(argv=None) -> int:
    import argparse
    import json
    import os
    import time

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--probe", action="store_true", help="report whether this network can read it")
    p.add_argument("--watch", action="store_true", help="poll the direct feed and notify")
    p.add_argument("--interval", type=int, default=int(os.environ.get("TTF_POLL_SECONDS", 30)))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.probe or not args.watch:
        print(json.dumps(probe(), indent=2))
        return 0

    allowed, why = robots_allows()
    if allowed is False:
        log.error("refusing to poll: %s", why)
        return 1

    from ..notify import notify_direct

    log.info("watching %s every %ss", FEED_URL, args.interval)
    while True:
        try:
            notify_direct(fetch())
        except KeyboardInterrupt:
            return 0
        except Exception:
            log.exception("direct poll failed; will retry")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
