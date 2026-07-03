"""Enrichment-pipeline tests (in-memory SQLite + fake models — no torch)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.core.errors import EnrichmentError
from sam.nlp.pipeline import EnrichmentPipeline, document_text
from sam.storage.models import Document, Embedding, SentimentScore
from sam.storage.repositories import DocumentRepository, SourceRepository
from tests.fakes import FakeEmbeddingModel, FakeSentimentModel


def _doc(hash_: str, title: str | None, raw_text: str | None = None) -> dict[str, object]:
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
    DocumentRepository(session).upsert_many(
        [
            _doc("a" * 64, "Nvidia shares surge on earnings"),
            _doc("b" * 64, "Markets plunge", "Broad selloff hits tech"),
            _doc("c" * 64, None, None),  # no usable text — stamped, not scored
        ]
    )
    session.commit()


def _pipeline(session: Session) -> EnrichmentPipeline:
    return EnrichmentPipeline(
        sentiment=FakeSentimentModel(),
        embedder=FakeEmbeddingModel(),
        session_factory=lambda: session,
    )


def test_document_text_composition() -> None:
    doc = Document(source_id=1, title="Title", raw_text="Body", content_hash="x" * 64)
    assert document_text(doc) == "Title\nBody"
    assert document_text(Document(source_id=1, title="Only", content_hash="y" * 64)) == "Only"
    assert document_text(Document(source_id=1, content_hash="z" * 64)) == ""


def test_enrich_writes_scores_vectors_and_stamps(db_session: Session) -> None:
    _seed_world(db_session)
    result = _pipeline(db_session).run()

    assert result.docs_scanned == 3
    assert result.docs_enriched == 2  # the text-less doc is skipped but stamped
    assert result.sentiments_written == 2
    assert result.embeddings_written == 2

    scores = (
        db_session.execute(select(SentimentScore).order_by(SentimentScore.document_id))
        .scalars()
        .all()
    )
    assert [(s.label, s.model) for s in scores] == [
        ("positive", "fake-sentiment-v1"),
        ("negative", "fake-sentiment-v1"),
    ]
    vectors = db_session.execute(select(Embedding)).scalars().all()
    assert {v.model for v in vectors} == {"fake-embedder-v1"}

    # Every scanned document is stamped — including the text-less one.
    unenriched = (
        db_session.execute(select(Document).where(Document.enriched_at.is_(None))).scalars().all()
    )
    assert unenriched == []


def test_enrich_is_incremental(db_session: Session) -> None:
    _seed_world(db_session)
    pipeline = _pipeline(db_session)
    pipeline.run()

    second = pipeline.run()
    assert second.docs_scanned == 0
    assert second.sentiments_written == 0


def test_enrich_all_re_scores_after_model_change(db_session: Session) -> None:
    _seed_world(db_session)
    _pipeline(db_session).run()

    class UpgradedSentiment(FakeSentimentModel):
        model_id = "fake-sentiment-v2"

    upgraded = EnrichmentPipeline(
        sentiment=UpgradedSentiment(),
        embedder=FakeEmbeddingModel(),
        session_factory=lambda: db_session,
    )
    assert upgraded.run().docs_scanned == 0  # incremental run skips stamped docs
    result = upgraded.run(re_enrich=True)
    assert result.docs_scanned == 3

    models = set(db_session.execute(select(SentimentScore.model)).scalars())
    assert models == {"fake-sentiment-v1", "fake-sentiment-v2"}  # versions coexist
    # Same embedder id: vectors refreshed in place, not duplicated.
    assert len(db_session.execute(select(Embedding)).scalars().all()) == 2


def test_enrich_small_batches_cover_all_documents(db_session: Session) -> None:
    _seed_world(db_session)
    result = _pipeline(db_session).run(batch_size=1)
    assert result.docs_scanned == 3
    assert result.docs_enriched == 2


def test_enrich_rejects_wrong_vector_width(db_session: Session) -> None:
    _seed_world(db_session)

    class NarrowEmbedder(FakeEmbeddingModel):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

    pipeline = EnrichmentPipeline(
        sentiment=FakeSentimentModel(),
        embedder=NarrowEmbedder(),
        session_factory=lambda: db_session,
    )
    with pytest.raises(EnrichmentError, match="schema expects"):
        pipeline.run()
    # Nothing was written or stamped (the batch transaction never committed).
    db_session.rollback()
    assert db_session.execute(select(SentimentScore)).scalars().all() == []
    stamped = (
        db_session.execute(select(Document).where(Document.enriched_at.is_not(None)))
        .scalars()
        .all()
    )
    assert stamped == []
