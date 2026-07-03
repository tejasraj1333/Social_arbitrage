"""Topic discovery over stored embeddings (Phase 4 / architecture M3).

Fits BERTopic on the documents' *persisted* embeddings (no re-encoding — the
vectors in Postgres are the single source of truth) and writes the discovered
topics + assignments under a fresh ``topic_model_version``. Runs are
append-only: past signal values were computed against past topic versions
(point-in-time rule), so old versions are never mutated.

Determinism: UMAP is seeded (fixed ``random_state``) and HDBSCAN is
deterministic given identical input, so re-fitting the same corpus yields the
same clusters. Topic *stability* additionally needs corpus volume — below
``nlp.topics_min_docs`` the run reports itself as skipped instead of fitting
degenerate clusters (same honesty as DQ's "insufficient history").

The actual BERTopic call is isolated behind the ``TopicFitter`` seam so unit
tests exercise persistence with a fake fitter and never import bertopic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from sam.core.config import get_settings
from sam.core.db import default_session
from sam.core.logging import get_logger
from sam.storage.repositories import EmbeddingRepository, TopicRepository

log = get_logger("nlp.topics")

UMAP_RANDOM_STATE = 42  # pinned: deterministic re-fit is a design invariant
OUTLIER_TOPIC = -1  # BERTopic's "no cluster" bucket; deliberately not persisted


@dataclass(frozen=True, slots=True)
class DiscoveredTopic:
    """One cluster from a fit: human-readable label + top keywords."""

    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopicFit:
    """A fit's output: topics and (doc_index, topic_index, probability) triples."""

    topics: list[DiscoveredTopic]
    assignments: list[tuple[int, int, float]]


TopicFitter = Callable[[list[str], list[list[float]]], TopicFit]


@dataclass(slots=True)
class TopicResult:
    """Outcome of one topic run."""

    version: str | None = None
    docs_used: int = 0
    topics_found: int = 0
    outliers: int = 0
    assignments_written: int = 0
    skipped: str | None = None  # reason, when no fit happened


def bertopic_fitter(texts: list[str], vectors: list[list[float]]) -> TopicFit:
    """Production fitter: BERTopic on precomputed embeddings (lazy imports)."""
    import numpy as np
    from bertopic import BERTopic  # heavy import, deferred
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    # Stopword-free keyword representation (labels like "nvidia_ai_chips"
    # instead of "the_to_in_of"); affects only c-TF-IDF, not the clustering.
    vectorizer = CountVectorizer(stop_words="english")
    model = BERTopic(umap_model=umap_model, vectorizer_model=vectorizer, verbose=False)
    embeddings = np.asarray(vectors, dtype=np.float32)
    topic_ids, probs = model.fit_transform(texts, embeddings=embeddings)

    # Map BERTopic's topic ids (excluding the -1 outlier bucket) to 0..n-1.
    info = model.get_topic_info()
    kept = [int(t) for t in info["Topic"] if int(t) != OUTLIER_TOPIC]
    index_of = {topic_id: idx for idx, topic_id in enumerate(kept)}

    topics = [
        DiscoveredTopic(
            label=str(info.loc[info["Topic"] == topic_id, "Name"].iloc[0]),
            keywords=tuple(word for word, _ in (model.get_topic(topic_id) or []))[:10],
        )
        for topic_id in kept
    ]
    assignments = [
        (
            doc_idx,
            index_of[int(topic_id)],
            float(probs[doc_idx]) if probs is not None else 1.0,
        )
        for doc_idx, topic_id in enumerate(topic_ids)
        if int(topic_id) != OUTLIER_TOPIC
    ]
    return TopicFit(topics=topics, assignments=assignments)


class TopicPipeline:
    """Fit topics over every document embedded by the configured model."""

    def __init__(
        self,
        fitter: TopicFitter | None = None,
        session_factory: Callable[[], Session] | None = None,
        embedding_model: str | None = None,
        min_docs: int | None = None,
    ) -> None:
        settings = get_settings().nlp
        # Resolved lazily so tests can monkeypatch module-level default_session.
        self._session_factory = session_factory or default_session
        self._fitter = fitter if fitter is not None else bertopic_fitter
        self._embedding_model = embedding_model or settings.embedding_model
        self._min_docs = min_docs if min_docs is not None else settings.topics_min_docs

    def run(self) -> TopicResult:
        session = self._session_factory()
        try:
            return self._run_in_session(session)
        finally:
            session.close()

    def _run_in_session(self, session: Session) -> TopicResult:
        rows = EmbeddingRepository(session).rows_with_documents(self._embedding_model)
        if len(rows) < self._min_docs:
            reason = (
                f"only {len(rows)} embedded documents for model "
                f"{self._embedding_model!r}; topic fit needs >= {self._min_docs} "
                "(run `sam enrich` first, or grow the corpus)"
            )
            log.warning("topics_skipped", reason=reason)
            return TopicResult(docs_used=len(rows), skipped=reason)

        doc_ids = [row[0] for row in rows]
        texts = ["\n".join(part for part in (title, raw) if part) for _, title, raw, _ in rows]
        vectors = [[float(x) for x in vector] for *_ignored, vector in rows]

        fit = self._fitter(texts, vectors)
        version = datetime.now(tz=UTC).strftime("bertopic-%Y%m%dT%H%M%SZ")

        repo = TopicRepository(session)
        topic_rows = repo.create_topics(
            version,
            [{"label": t.label, "keywords": list(t.keywords)} for t in fit.topics],
        )
        written = repo.assign_documents(
            [
                {
                    "document_id": doc_ids[doc_idx],
                    "topic_id": topic_rows[topic_idx].id,
                    "probability": probability,
                }
                for doc_idx, topic_idx, probability in fit.assignments
            ]
        )
        session.commit()

        result = TopicResult(
            version=version,
            docs_used=len(rows),
            topics_found=len(fit.topics),
            outliers=len(rows) - len(fit.assignments),
            assignments_written=written,
        )
        log.info(
            "topics_done",
            version=version,
            docs=result.docs_used,
            topics=result.topics_found,
            outliers=result.outliers,
            assignments=result.assignments_written,
            embedding_model=self._embedding_model,
        )
        return result
