"""Collector contract.

Every source implements this 3-stage shape so the orchestrator treats them
uniformly and each is independently testable:

    fetch()     -> raw payloads from the source (network, rate-limited)
    normalize() -> map raw payloads to canonical Document dicts
    persist()   -> idempotent upsert keyed on content_hash (dedup)

run() ties them together and records an ingestion_runs row (added in M1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class Collector(ABC):
    """Abstract base for all data-source collectors."""

    source_type: str

    @abstractmethod
    def fetch(self) -> Iterable[dict[str, Any]]:
        """Pull raw payloads from the source."""

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Map a raw payload to a canonical Document dict."""

    @abstractmethod
    def persist(self, documents: Iterable[dict[str, Any]]) -> int:
        """Idempotently upsert documents; return count newly inserted."""

    def run(self) -> int:
        docs = (self.normalize(raw) for raw in self.fetch())
        return self.persist(docs)
