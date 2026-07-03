"""Resolution-pipeline tests (in-memory SQLite, same upsert paths as prod)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.processing.pipeline import ResolutionPipeline
from sam.storage.models import Document, DocumentEntity
from sam.storage.repositories import (
    DocumentRepository,
    EntityRepository,
    SourceRepository,
)

UNIVERSE = [
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "aliases": ["Nvidia"]},
    {"ticker": "AAPL", "name": "Apple Inc.", "aliases": ["Apple"]},
]


def _doc(hash_: str, title: str, raw_text: str | None = None) -> dict[str, object]:
    return {
        "source_id": 1,
        "external_id": f"ext-{hash_[:8]}",
        "url": f"https://example.com/{hash_[:8]}",
        "author": None,
        "title": title,
        "raw_text": raw_text,
        "lang": None,
        "content_hash": hash_,
        "published_at": datetime(2026, 7, 1, tzinfo=UTC),
        "engagement": {},
    }


def _seed_world(session: Session) -> None:
    SourceRepository(session).get_or_create("rss", "rss")
    EntityRepository(session).seed(UNIVERSE)
    DocumentRepository(session).upsert_many(
        [
            _doc("a" * 64, "$NVDA rips higher ahead of earnings"),
            _doc("b" * 64, "New iPhone lineup", "Apple unveiled its new lineup today"),
            _doc("c" * 64, "Fed leaves rates unchanged"),
        ]
    )
    session.commit()


def _pipeline(session: Session) -> ResolutionPipeline:
    return ResolutionPipeline(session_factory=lambda: session)


def test_resolve_links_documents_to_entities(db_session: Session) -> None:
    _seed_world(db_session)
    result = _pipeline(db_session).run()

    assert result.docs_scanned == 3
    assert result.docs_matched == 2
    assert result.links_written == 2

    links = (
        db_session.execute(select(DocumentEntity).order_by(DocumentEntity.document_id))
        .scalars()
        .all()
    )
    assert [(link.method, link.confidence) for link in links] == [
        ("cashtag", 1.0),
        ("alias", 0.8),
    ]

    # Every scanned document is stamped — including the no-match one.
    unresolved = (
        db_session.execute(select(Document).where(Document.resolved_at.is_(None))).scalars().all()
    )
    assert unresolved == []


def test_resolve_is_incremental(db_session: Session) -> None:
    _seed_world(db_session)
    pipeline = _pipeline(db_session)
    pipeline.run()

    second = pipeline.run()
    assert second.docs_scanned == 0
    assert second.links_written == 0


def test_resolve_all_rescans_after_dictionary_change(db_session: Session) -> None:
    _seed_world(db_session)
    pipeline = _pipeline(db_session)
    pipeline.run()

    # The Fed headline gains a matching alias — only --all picks it up.
    curated = [*UNIVERSE, {"ticker": "FED", "name": "Fed Corp", "aliases": ["Fed"]}]
    EntityRepository(db_session).seed(curated, update=True)
    db_session.commit()

    assert pipeline.run().docs_scanned == 0  # incremental run skips stamped docs
    result = pipeline.run(re_resolve=True)
    assert result.docs_scanned == 3
    tickers = set(
        db_session.execute(
            select(DocumentEntity.entity_id).join(
                Document, Document.id == DocumentEntity.document_id
            )
        ).scalars()
    )
    assert len(tickers) == 3  # NVDA + AAPL links refreshed, FED link added


def test_resolve_small_batches_cover_all_documents(db_session: Session) -> None:
    _seed_world(db_session)
    result = _pipeline(db_session).run(batch_size=1)
    assert result.docs_scanned == 3
    assert result.links_written == 2


def test_resolve_with_empty_universe_still_stamps_documents(db_session: Session) -> None:
    SourceRepository(db_session).get_or_create("rss", "rss")
    DocumentRepository(db_session).upsert_many([_doc("d" * 64, "Nvidia beats estimates")])
    db_session.commit()

    result = _pipeline(db_session).run()
    assert result.docs_scanned == 1
    assert result.links_written == 0
    doc = db_session.execute(select(Document)).scalar_one()
    assert doc.resolved_at is not None
