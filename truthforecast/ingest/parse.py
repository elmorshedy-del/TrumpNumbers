"""Parsers for the two shapes trumpstruth.org serves.

Both produce the same `Post` record. Two rules govern everything here:

1. **Store UTC.** The RSS feed carries an explicit UTC `pubDate` with seconds.
   Listing pages render US Eastern with no timezone label and no seconds, so
   they are localized through `SITE_TZ` and marked as minute-precision.
2. **Fail loudly.** A parser that silently yields nothing when the site's markup
   changes would write a run of zero-post days into the series, and every model
   downstream would faithfully learn from the lie. Structural failures raise.
"""

from __future__ import annotations

import email.utils
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from selectolax.parser import HTMLParser

from ..config import SITE_TZ

# "March 1, 2026, 5:30 PM" as rendered in .status-info__meta
LISTING_TS_FORMAT = "%B %d, %Y, %I:%M %p"
LISTING_TS_RE = re.compile(
    r"^[A-Z][a-z]+ \d{1,2}, \d{4}, \d{1,2}:\d{2} (?:AM|PM)$"
)
STATUS_ID_RE = re.compile(r"/statuses/(\d+)")
TRUTH_ID_RE = re.compile(r"/(\d+)\s*$")
# In RSS the mirror renders a ReTruth as a leading "RT @handle" in the body.
RETRUTH_RE = re.compile(r"^RT\s*@([A-Za-z0-9_]+)")
# On listing pages there is no RT prefix; the block is instead attributed to the
# original author, so a handle other than Trump's is the ReTruth signal.
OWNER_HANDLE = "@realdonaldtrump"


class ParseError(RuntimeError):
    """Raised when the markup no longer matches what the parser expects."""


@dataclass
class Post:
    """One row of the mirror.

    `created_utc` is the *content's* creation time. For an original post that is
    the moment Trump posted — which is what the whole project models. For a
    ReTruth it is when the ORIGINAL author posted, not when Trump reshared it:
    id 40332 is a ReTruth of a June 26 @SebGorka post but was ingested among the
    July 27 ids. Both the RSS feed and the listing pages agree on this, so the
    reshare time is simply not recoverable from this source.

    That is why `is_retruth` exists and why the primary modeled series is
    originals only (see `series.py`). Including ReTruths would sprinkle counts
    onto days on which Trump did nothing.
    """

    trumpstruth_id: int
    created_utc: datetime
    text: str
    truth_id: str | None = None
    truth_url: str | None = None
    is_retruth: bool = False
    source_handle: str | None = None
    is_deleted: bool = False
    has_media: bool = False
    media_kinds: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    # "second" for RSS-sourced rows, "minute" for listing-sourced rows. Daily
    # counts do not care; the Hawkes fit does, so it is recorded rather than
    # quietly lost.
    time_precision: str = "minute"

    def as_row(self) -> dict:
        return {
            "trumpstruth_id": self.trumpstruth_id,
            "truth_id": self.truth_id,
            "truth_url": self.truth_url,
            "created_utc": self.created_utc.astimezone(timezone.utc).isoformat(),
            "text": self.text,
            "is_retruth": int(self.is_retruth),
            "source_handle": self.source_handle,
            "is_deleted": int(self.is_deleted),
            "has_media": int(self.has_media),
            "media_kinds": ",".join(sorted(set(self.media_kinds))),
            "links": "\n".join(self.links),
            "time_precision": self.time_precision,
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_listing_timestamp(label: str) -> datetime:
    """Localize a listing timestamp label to UTC.

    The site prints Eastern wall-clock time with no zone marker. Using
    `America/New_York` rather than a fixed -4h keeps the DST boundary correct;
    a hardcoded offset would shift half the archive by an hour.
    """
    label = _clean(label)
    if not LISTING_TS_RE.match(label):
        raise ParseError(f"unrecognized listing timestamp: {label!r}")
    naive = datetime.strptime(label, LISTING_TS_FORMAT)
    return naive.replace(tzinfo=SITE_TZ).astimezone(timezone.utc)


def parse_rss(xml: str) -> list[Post]:
    """Parse the /feed RSS document (100 newest posts, exact UTC)."""
    items = re.findall(r"<item>(.*?)</item>", xml, re.S)
    if not items:
        if "<rss" not in xml:
            raise ParseError("response is not an RSS document")
        return []

    posts: list[Post] = []
    for body in items:
        link = re.search(r"<link>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</link>", body, re.S)
        sid = STATUS_ID_RE.search(link.group(1)) if link else None
        pub = re.search(r"<pubDate>([^<]+)</pubDate>", body)
        if not sid or not pub:
            raise ParseError("RSS item missing status link or pubDate")

        created = email.utils.parsedate_to_datetime(pub.group(1).strip())
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        title = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", body, re.S)
        desc = re.search(r"<description><!\[CDATA\[(.*?)\]\]></description>", body, re.S)
        html = desc.group(1) if desc else ""
        text = _clean(HTMLParser(html).text()) if html else _clean(title.group(1) if title else "")

        orig_id = re.search(r"<truth:originalId>([^<]+)</truth:originalId>", body)
        orig_url = re.search(r"<truth:originalUrl>([^<]+)</truth:originalUrl>", body)

        node = HTMLParser(html) if html else None
        links = _extract_links(node) if node else []
        rt = RETRUTH_RE.match(text)

        posts.append(
            Post(
                trumpstruth_id=int(sid.group(1)),
                created_utc=created.astimezone(timezone.utc),
                text=text,
                truth_id=orig_id.group(1).strip() if orig_id else None,
                truth_url=orig_url.group(1).strip() if orig_url else None,
                is_retruth=bool(rt),
                source_handle=f"@{rt.group(1)}" if rt else None,
                has_media=bool(re.search(r"<(img|video)\b", html)),
                media_kinds=_media_kinds_from_html(html),
                links=links,
                time_precision="second",
            )
        )
    return posts


def _extract_links(node: HTMLParser) -> list[str]:
    out: list[str] = []
    for a in node.css("a"):
        href = (a.attributes.get("href") or "").strip()
        if href.startswith("http") and "truthsocial.com/@" not in href:
            out.append(href)
    return out


def _media_kinds_from_html(html: str) -> list[str]:
    kinds = []
    if re.search(r"<video\b", html):
        kinds.append("video")
    if re.search(r"<img\b", html):
        kinds.append("image")
    return kinds


def parse_listing(html: str) -> tuple[list[Post], str | None]:
    """Parse a cursor listing page.

    Returns the posts and the *next* cursor (None when the walk is exhausted).
    """
    tree = HTMLParser(html)
    # Only top-level posts. A ReTruth or quote embeds the *original* post as a
    # nested .status block — on a typical page 159 .status nodes correspond to
    # 100 actual posts. Counting the nested ones would both inflate the daily
    # totals and inject the original's (often much older) timestamp into the
    # series. Top-level blocks are exactly those carrying data-status-url.
    nodes = [n for n in tree.css(".status") if n.attributes.get("data-status-url")]
    if not nodes:
        # A genuinely empty page is possible at the end of the archive, but the
        # page still carries the site chrome. No .status AND no pagination form
        # means the markup moved.
        if "per_page" not in html:
            raise ParseError("listing page has no .status blocks and no pagination form")
        return [], None

    posts: list[Post] = []
    seen: set[int] = set()
    for node in nodes:
        post = _parse_status_node(node)
        if post is None or post.trumpstruth_id in seen:
            continue
        seen.add(post.trumpstruth_id)
        posts.append(post)

    if not posts:
        raise ParseError(f"{len(nodes)} .status blocks but none parsed")

    return posts, extract_next_cursor(html)


def _parse_status_node(node) -> Post | None:
    url = (node.attributes.get("data-status-url") or "").strip()
    sid_match = STATUS_ID_RE.search(url)
    if not sid_match:
        return None

    # Document order matters: this post's own header precedes any nested
    # quoted post's header, so the FIRST timestamp-shaped label — and the first
    # @handle — are ours.
    label = None
    handle = None
    for a in node.css(".status-info__meta a"):
        candidate = _clean(a.text())
        if handle is None and candidate.startswith("@"):
            handle = candidate
        if label is None and LISTING_TS_RE.match(candidate):
            label = candidate
        if label and handle:
            break
    if label is None:
        raise ParseError(f"status {sid_match.group(1)} has no parsable timestamp")

    content = node.css_first(".status__content")
    text = _clean(content.text()) if content else ""

    external = node.css_first("a.status__external-link")
    truth_url = external.attributes.get("href") if external else None
    truth_id = None
    if truth_url:
        m = TRUTH_ID_RE.search(truth_url)
        truth_id = m.group(1) if m else None

    is_retruth = bool(RETRUTH_RE.match(text)) or (
        handle is not None and handle.lower() != OWNER_HANDLE
    )
    return Post(
        trumpstruth_id=int(sid_match.group(1)),
        created_utc=parse_listing_timestamp(label),
        text=text,
        truth_id=truth_id,
        truth_url=truth_url,
        is_retruth=is_retruth,
        source_handle=handle,
        is_deleted=node.css_first(".status__deleted-badge") is not None,
        has_media=node.css_first(".status__attachments") is not None,
        media_kinds=_media_kinds_from_node(node),
        links=[
            a.attributes.get("href", "")
            for a in node.css("a.status-attachment__link, .status-card a")
            if (a.attributes.get("href") or "").startswith("http")
        ],
        time_precision="minute",
    )


def _media_kinds_from_node(node) -> list[str]:
    kinds = []
    if node.css_first(".status-attachment--video"):
        kinds.append("video")
    if node.css_first(".status-attachment--image"):
        kinds.append("image")
    if node.css_first(".status-card"):
        kinds.append("card")
    return kinds


CURSOR_RE = re.compile(r"cursor=([A-Za-z0-9+/=_-]{16,})")


def extract_next_cursor(html: str) -> str | None:
    """Pull the last cursor token on the page — the 'next page' pointer.

    The site emits both a previous and a next cursor; the next one is rendered
    last. `client.cursor_for` can synthesize these from scratch, so this is a
    convenience rather than the only way to walk.
    """
    matches = CURSOR_RE.findall(html)
    return matches[-1] if matches else None
