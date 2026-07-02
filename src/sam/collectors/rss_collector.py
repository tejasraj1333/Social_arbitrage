"""RSS recon collector (keyless).

Pulls market-news headlines from the feeds in config/sources.yaml via feedparser.
Feeds are tried in order and entries accumulated until `target` is reached, so a
dead/empty feed (Reuters retired its public RSS) is tolerated as long as the
others backfill the sample.

Record schema: title, summary, url, published_date (ISO-8601 UTC), source.
"""

from __future__ import annotations

import calendar
import re
from datetime import UTC, datetime
from typing import Any

import feedparser

from sam.recon.collector_base import ReconCollector
from sam.recon.sources import load_sources

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SUMMARY_MAX = 600


def _clean_html(text: str) -> str:
    """Strip tags + collapse whitespace from an RSS summary; cap length."""
    stripped = _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()
    return stripped[:_SUMMARY_MAX]


def _parse_published(entry: Any) -> str | None:
    """Return an ISO-8601 UTC timestamp from a feed entry, or a raw string fallback."""
    for key in ("published_parsed", "updated_parsed"):
        struct = entry.get(key)
        if struct:
            # struct_time is UTC; timegm avoids local-tz drift.
            return datetime.fromtimestamp(calendar.timegm(struct), tz=UTC).isoformat()
    published: str | None = entry.get("published") or entry.get("updated")
    return published


class RSSCollector(ReconCollector):
    source_name = "rss"
    required_fields = ["title", "url", "source", "published_date"]
    timestamp_field = "published_date"
    sample_size = 100

    def __init__(
        self,
        feeds: list[dict[str, str]] | None = None,
        target: int = 100,
    ) -> None:
        super().__init__()
        cfg = load_sources().get("rss", {})
        self.feeds: list[dict[str, str]] = feeds if feeds is not None else cfg.get("feeds", [])
        self.target = target

    def fetch(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_urls: set[str] = set()  # dedupe articles syndicated across feeds
        for feed in self.feeds:
            name, url = feed["name"], feed["url"]
            parsed = feedparser.parse(url)
            entries = parsed.get("entries", [])
            if not entries:
                self.log.warning(
                    "rss_feed_empty",
                    feed=name,
                    bozo=str(parsed.get("bozo_exception", "")),
                )
                continue
            kept = 0
            for entry in entries:
                link = entry.get("link")
                if link and link in seen_urls:
                    continue
                if link:
                    seen_urls.add(link)
                records.append(self._to_record(entry, name))
                kept += 1
                if len(records) >= self.target:
                    self.log.info("rss_feed_ok", feed=name, kept=kept, entries=len(entries))
                    return records
            self.log.info("rss_feed_ok", feed=name, kept=kept, entries=len(entries))
        return records

    @staticmethod
    def _to_record(entry: Any, source_name: str) -> dict[str, Any]:
        return {
            "title": entry.get("title"),
            "summary": _clean_html(entry.get("summary", "")),
            "url": entry.get("link"),
            "published_date": _parse_published(entry),
            "source": source_name,
        }
