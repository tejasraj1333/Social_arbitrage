"""Production Hacker News ingestion collector.

Fetch is delegated to the recon HackerNewsCollector (Firebase API, tenacity
retry, dead-item skipping). Engagement (score/comments) is stored as the
first-seen snapshot in the JSON column and is excluded from the content hash —
it changes on every fetch and must not defeat dedup.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from sam.collectors.hn_collector import HackerNewsCollector as _HNFetcher
from sam.ingestion.base import DocumentIngestionCollector
from sam.ingestion.hashing import content_hash


class HackerNewsIngestionCollector(DocumentIngestionCollector):
    source_type = "hackernews"
    source_name = "hackernews"
    config_ref = "config/sources.yaml:hackernews"

    def __init__(self, session: Session, source_id: int, limit: int | None = None) -> None:
        super().__init__(session, source_id)
        self._fetcher = _HNFetcher(limit=limit)

    def fetch(self) -> Iterable[dict[str, Any]]:
        return self._fetcher.fetch()

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        item_id = raw.get("id")
        external_id = str(item_id) if item_id is not None else None
        title = raw.get("title")
        ts = raw.get("timestamp")
        published_at = (
            datetime.fromtimestamp(float(ts), tz=UTC) if isinstance(ts, (int, float)) else None
        )
        return {
            "source_id": self.source_id,
            "external_id": external_id,
            "url": raw.get("url"),
            "author": raw.get("by"),
            "title": title,
            "raw_text": None,  # top-stories listing carries no body text
            "lang": None,
            "content_hash": content_hash(self.source_type, external_id, title, None),
            "published_at": published_at,
            "engagement": {"score": raw.get("score"), "comments": raw.get("comments")},
        }
