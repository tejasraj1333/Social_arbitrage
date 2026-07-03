"""Collector contract.

Every source implements this 3-stage shape so the orchestrator treats them
uniformly and each is independently testable:

    fetch()     -> raw payloads from the source (network, rate-limited)
    normalize() -> map raw payloads to canonical row dicts
    persist()   -> idempotent upsert (documents dedup on content_hash;
                   market bars upsert on their natural key)

run() ties the stages together. The production path is the ingestion runner
(sam.ingestion.runner), which additionally writes the bronze lake and records
an ingestion_runs row around these stages.

Collectors are DB-thin: a Session and the resolved source_id are injected, and
all SQL goes through the repository layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from sam.core.logging import get_logger
from sam.storage.repositories import DocumentRepository


def parse_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string to an aware UTC datetime.

    Naive values are pinned to UTC, never reinterpreted in local time
    (point-in-time rule — same discipline as recon's _to_epoch).
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


class Collector(ABC):
    """Abstract base for all data-source collectors."""

    source_type: str
    source_name: str
    config_ref: str | None = None

    @abstractmethod
    def fetch(self) -> Iterable[dict[str, Any]]:
        """Pull raw payloads from the source."""

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Map a raw payload to a canonical row dict."""

    @abstractmethod
    def persist(self, documents: Iterable[dict[str, Any]]) -> int:
        """Idempotently upsert rows; return count newly inserted/updated."""

    def run(self) -> int:
        docs = (self.normalize(raw) for raw in self.fetch())
        return self.persist(docs)


class DocumentIngestionCollector(Collector):
    """Shared base for text sources persisting into the documents table."""

    def __init__(self, session: Session, source_id: int) -> None:
        self.session = session
        self.source_id = source_id
        self.log = get_logger(f"ingestion.{self.source_name}")

    def persist(self, documents: Iterable[dict[str, Any]]) -> int:
        inserted = DocumentRepository(self.session).upsert_many(list(documents))
        self.log.info("documents_persisted", source=self.source_name, inserted=inserted)
        return inserted
