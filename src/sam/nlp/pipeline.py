"""Batch NLP-enrichment pipeline (Phase 4 / architecture M3).

Scans documents and writes one sentiment score and one embedding per document
(per model id). Incremental by default via the ``documents.enriched_at``
watermark — every scanned document is stamped, even those with no usable
text, so a run only ever touches new documents. ``re_enrich=True``
(``sam enrich --all``) re-scans everything after a model change; rows refresh
via DO UPDATE keyed on (document_id, model).

Commits per batch, so an interrupted run loses at most one batch and never
stamps a document without its rows (same transaction). Model inference is
deterministic (no sampling), so a rebuild writes identical values — the P4
reproducibility gate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from sam.core.db import default_session
from sam.core.errors import EnrichmentError
from sam.core.logging import get_logger
from sam.nlp.models import (
    EmbeddingModel,
    FinBertSentiment,
    SentenceTransformerEmbedder,
    SentimentModel,
)
from sam.storage.models import EMBEDDING_DIM, Document
from sam.storage.repositories import (
    DocumentRepository,
    EmbeddingRepository,
    SentimentRepository,
)

log = get_logger("nlp.pipeline")

_BATCH_SIZE = 500


def document_text(doc: Document) -> str:
    """Compose the text the NLP models see: title + body, newline-joined.

    Same field set the entity resolver matches on, so sentiment/embeddings
    describe exactly the text that produced the entity links.
    """
    return "\n".join(part for part in (doc.title, doc.raw_text) if part)


@dataclass(slots=True)
class EnrichResult:
    """Outcome of one enrichment run."""

    docs_scanned: int = 0
    docs_enriched: int = 0
    sentiments_written: int = 0
    embeddings_written: int = 0


class EnrichmentPipeline:
    """Walks the enrichment watermark, scoring and embedding new documents.

    Models are injectable (tests pass fakes; production wiring uses the
    configured FinBERT + SentenceTransformer). Construction never loads
    weights — a run over an empty backlog stays model-free.
    """

    def __init__(
        self,
        sentiment: SentimentModel | None = None,
        embedder: EmbeddingModel | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        # Resolved lazily so tests can monkeypatch module-level default_session.
        self._session_factory = session_factory or default_session
        self._sentiment = sentiment if sentiment is not None else FinBertSentiment()
        self._embedder = embedder if embedder is not None else SentenceTransformerEmbedder()

    def run(self, *, re_enrich: bool = False, batch_size: int = _BATCH_SIZE) -> EnrichResult:
        session = self._session_factory()
        try:
            return self._run_in_session(session, re_enrich=re_enrich, batch_size=batch_size)
        finally:
            session.close()

    def _run_in_session(
        self, session: Session, *, re_enrich: bool, batch_size: int
    ) -> EnrichResult:
        docs_repo = DocumentRepository(session)
        sentiment_repo = SentimentRepository(session)
        embedding_repo = EmbeddingRepository(session)
        result = EnrichResult()
        last_id = 0

        while True:
            batch = docs_repo.enrichment_batch(
                after_id=last_id, limit=batch_size, include_enriched=re_enrich
            )
            if not batch:
                break
            now = datetime.now(tz=UTC)

            with_text = [(doc, text) for doc in batch if (text := document_text(doc))]
            texts = [text for _, text in with_text]
            scores = self._sentiment.score(texts)
            vectors = self._embedder.embed(texts)
            self._check_shapes(len(texts), scores, vectors)

            result.sentiments_written += sentiment_repo.upsert_many(
                [
                    {
                        "document_id": doc.id,
                        "model": self._sentiment.model_id,
                        "label": score.label,
                        "score": score.score,
                        "scored_at": now,
                    }
                    for (doc, _), score in zip(with_text, scores, strict=True)
                ]
            )
            result.embeddings_written += embedding_repo.upsert_many(
                [
                    {
                        "document_id": doc.id,
                        "model": self._embedder.model_id,
                        "vector": vector,
                        "embedded_at": now,
                    }
                    for (doc, _), vector in zip(with_text, vectors, strict=True)
                ]
            )
            docs_repo.mark_enriched([doc.id for doc in batch], at=now)
            session.commit()  # per-batch: progress survives interruption
            result.docs_scanned += len(batch)
            result.docs_enriched += len(with_text)
            last_id = batch[-1].id

        log.info(
            "enrich_done",
            scanned=result.docs_scanned,
            enriched=result.docs_enriched,
            sentiments=result.sentiments_written,
            embeddings=result.embeddings_written,
            re_enrich=re_enrich,
            sentiment_model=self._sentiment.model_id,
            embedding_model=self._embedder.model_id,
        )
        return result

    def _check_shapes(
        self, expected: int, scores: Sequence[object], vectors: list[list[float]]
    ) -> None:
        """Fail loudly before writing anything schema-incompatible.

        Postgres enforces vector(384) itself, but SQLite (tests) and a
        mispaired count would corrupt silently — this is the one place model
        output meets storage, so validate here.
        """
        if len(scores) != expected or len(vectors) != expected:
            raise EnrichmentError(
                f"model returned {len(scores)} scores / {len(vectors)} vectors for {expected} texts"
            )
        if vectors and len(vectors[0]) != EMBEDDING_DIM:
            raise EnrichmentError(
                f"embedding model {self._embedder.model_id!r} emits "
                f"{len(vectors[0])}-dim vectors; schema expects {EMBEDDING_DIM} "
                "(changing width requires a migration)"
            )
