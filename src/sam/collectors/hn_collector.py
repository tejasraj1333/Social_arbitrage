"""Hacker News recon collector (keyless).

Uses the official Firebase API: fetch the top-stories id list, then hydrate each
item. httpx for transport, tenacity for transient-failure retry. HN is a useful
early-attention signal (tech/consumer products surface here before mainstream news).

Record schema: title, score, comments (descendants), timestamp (epoch s), url.
Ask/Show-HN text posts legitimately have no url, so url completeness < 1 is expected.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sam.recon.collector_base import ReconCollector
from sam.recon.sources import load_sources

_BASE = "https://hacker-news.firebaseio.com/v0"


class HackerNewsCollector(ReconCollector):
    source_name = "hackernews"
    required_fields = ["title", "score", "comments", "timestamp", "url"]
    timestamp_field = "timestamp"
    sample_size = 100

    def __init__(self, limit: int | None = None, timeout: float = 10.0) -> None:
        super().__init__()
        cfg = load_sources().get("hackernews", {})
        self.limit = limit if limit is not None else cfg.get("limit", 100)
        self.timeout = timeout

    def fetch(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": "sam-recon/0.1"}) as client:
            ids = self._get(client, f"{_BASE}/topstories.json") or []
            for item_id in ids[: self.limit]:
                item = self._get(client, f"{_BASE}/item/{item_id}.json")
                if not item:  # deleted/dead items return null
                    continue
                records.append(self._to_record(item))
        return records

    @staticmethod
    def _to_record(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "title": item.get("title"),
            "score": item.get("score"),
            "comments": item.get("descendants", 0),
            "timestamp": item.get("time"),
            "url": item.get("url"),
            "by": item.get("by"),
            "type": item.get("type"),
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    def _get(self, client: httpx.Client, url: str) -> Any:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()
