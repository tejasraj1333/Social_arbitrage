"""Topic-pipeline tests (in-memory SQLite + fake fitter — bertopic never imports)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.nlp.topics import DiscoveredTopic, TopicFit, TopicPipeline
from sam.storage.models import Document, DocumentTopic, Topic
from sam.storage.repositories import (
    DocumentRepository,
    EmbeddingRepository,
    SourceRepository,
)


def _doc(hash_: str, title: str) -> dict[str, object]:
    return {
        "source_id": 1,
        "external_id": f"ext-{hash_[:8]}",
        "url": f"https://example.com/{hash_[:8]}",
        "author": None,
        "title": title,
        "raw_text": None,
        "lang": None,
        "content_hash": hash_,
        "published_at": datetime(2026, 7, 1, tzinfo=UTC),
        "engagement": {},
    }


def _seed_embedded_docs(session: Session, n: int, model: str = "fake-embedder-v1") -> list[int]:
    """n documents, each with a stored embedding under ``model``."""
    SourceRepository(session).get_or_create("rss", "rss")
    DocumentRepository(session).upsert_many([_doc(f"{i:064x}", f"Headline {i}") for i in range(n)])
    ids = list(session.execute(select(Document.id).order_by(Document.id)).scalars())
    EmbeddingRepository(session).upsert_many(
        [
            {"document_id": doc_id, "model": model, "vector": [float(i), 0.5]}
            for i, doc_id in enumerate(ids)
        ]
    )
    session.commit()
    return ids


class _RecordingFitter:
    """Fake fitter: two topics; the last doc is left as an outlier."""

    def __init__(self) -> None:
        self.texts: list[str] | None = None
        self.vectors: list[list[float]] | None = None

    def __call__(self, texts: list[str], vectors: list[list[float]]) -> TopicFit:
        self.texts = texts
        self.vectors = vectors
        return TopicFit(
            topics=[
                DiscoveredTopic(label="0_ai_chips", keywords=("nvidia", "gpu")),
                DiscoveredTopic(label="1_macro", keywords=("fed", "rates")),
            ],
            assignments=[(0, 0, 0.9), (1, 1, 0.7)],
        )


def _pipeline(session: Session, fitter: _RecordingFitter, min_docs: int = 3) -> TopicPipeline:
    return TopicPipeline(
        fitter=fitter,
        session_factory=lambda: session,
        embedding_model="fake-embedder-v1",
        min_docs=min_docs,
    )


def test_topics_persist_versioned_run(db_session: Session) -> None:
    ids = _seed_embedded_docs(db_session, n=3)
    fitter = _RecordingFitter()
    result = _pipeline(db_session, fitter).run()

    assert result.skipped is None
    assert result.version is not None and result.version.startswith("bertopic-")
    assert (result.docs_used, result.topics_found) == (3, 2)
    assert result.outliers == 1  # the unassigned third doc
    assert result.assignments_written == 2

    # The fitter received the stored vectors, not re-encoded ones.
    assert fitter.vectors is not None and fitter.vectors[2][0] == 2.0
    assert fitter.texts is not None and fitter.texts[0] == "Headline 0"

    topics = db_session.execute(select(Topic).order_by(Topic.id)).scalars().all()
    assert [t.label for t in topics] == ["0_ai_chips", "1_macro"]
    assert topics[0].keywords == ["nvidia", "gpu"]
    assert {t.topic_model_version for t in topics} == {result.version}

    assignments = db_session.execute(select(DocumentTopic)).scalars().all()
    assert {(a.document_id, a.probability) for a in assignments} == {(ids[0], 0.9), (ids[1], 0.7)}


def test_topics_skip_below_min_corpus(db_session: Session) -> None:
    _seed_embedded_docs(db_session, n=2)
    fitter = _RecordingFitter()
    result = _pipeline(db_session, fitter, min_docs=10).run()

    assert result.skipped is not None and "needs >= 10" in result.skipped
    assert result.docs_used == 2
    assert fitter.texts is None  # fit never ran
    assert db_session.execute(select(Topic)).scalars().all() == []


def test_topics_reruns_append_new_versions(db_session: Session) -> None:
    _seed_embedded_docs(db_session, n=3)
    pipeline = _pipeline(db_session, _RecordingFitter())
    first = pipeline.run()
    second = pipeline.run()

    versions = {t.topic_model_version for t in db_session.execute(select(Topic)).scalars()}
    assert first.version in versions and second.version in versions
    # Append-only history: both versions' topic rows coexist.
    assert len(db_session.execute(select(Topic)).scalars().all()) == 4


def test_topics_only_sees_configured_embedding_model(db_session: Session) -> None:
    _seed_embedded_docs(db_session, n=3, model="someone-else")
    result = _pipeline(db_session, _RecordingFitter(), min_docs=3).run()
    assert result.skipped is not None  # 0 rows for fake-embedder-v1
    assert result.docs_used == 0
