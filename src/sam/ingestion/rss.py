"""Production RSS ingestion collector.

Fetch is delegated to the recon RSSCollector (proven feed parsing, URL dedup
across feeds, dead-feed tolerance) with the sample cap effectively removed —
ingestion wants every currently-exposed entry, and idempotent persistence
makes re-fetching the overlap a no-op.

Legal constraint (docs/legal_register.md): store headline + short summary +
link only — never full article text.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy.orm import Session

from sam.collectors.rss_collector import RSSCollector as _RSSFetcher
from sam.ingestion.base import DocumentIngestionCollector, parse_utc
from sam.ingestion.hashing import content_hash

_FETCH_ALL = 10_000  # effectively uncapped; feeds expose ~10-30 entries each


class RSSIngestionCollector(DocumentIngestionCollector):
    source_type = "rss"
    source_name = "rss"
    config_ref = "config/sources.yaml:rss"

    def __init__(
        self,
        session: Session,
        source_id: int,
        feeds: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(session, source_id)
        self._fetcher = _RSSFetcher(feeds=feeds, target=_FETCH_ALL)

    def fetch(self) -> Iterable[dict[str, Any]]:
        return self._fetcher.fetch()

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        title = raw.get("title")
        summary = raw.get("summary")
        url = raw.get("url")
        return {
            "source_id": self.source_id,
            "external_id": url,
            "url": url,
            "author": None,
            "title": title,
            "raw_text": summary,  # headline+summary only, per legal register
            "lang": None,
            "content_hash": content_hash(self.source_type, url, title, summary),
            "published_at": parse_utc(raw.get("published_date")),
            "engagement": {"feed": raw.get("source")},
        }
