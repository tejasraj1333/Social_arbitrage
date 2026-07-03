"""Model abstraction for NLP enrichment (Phase 4).

Thin protocols + concrete wrappers around the actual libraries. Design rules:

- **Lazy heavy imports.** torch/transformers/sentence-transformers are only
  imported inside ``_load()``, on first use — importing this module needs
  nothing beyond the standard library, so the core package, CLI startup and
  CI stay independent of the ``nlp`` extra. Tests inject fakes.
- **Model id on every output.** Callers persist ``model_id`` next to each
  score/vector, so outputs from different models never mix silently and a
  rebuild is attributable to an exact model (reproducibility invariant).
- **Batched, deterministic inference.** Inputs are scored in configured
  batches; no sampling anywhere, so re-running a model over the same corpus
  yields identical rows (the P4 "features rebuild deterministically" gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sam.core.config import get_settings
from sam.core.logging import get_logger

log = get_logger("nlp.models")

# FinBERT's label set — mirrored by the CHECK constraint on sentiment_scores.
SENTIMENT_LABELS = ("positive", "negative", "neutral")


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """One scored text: predicted label and the model's confidence in it."""

    label: str  # 'positive' | 'negative' | 'neutral'
    score: float


@runtime_checkable
class SentimentModel(Protocol):
    """Anything that scores texts into (label, confidence) pairs."""

    @property
    def model_id(self) -> str: ...

    def score(self, texts: list[str]) -> list[SentimentResult]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Anything that embeds texts into fixed-width float vectors."""

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FinBertSentiment:
    """Finance-domain sentiment via a transformers text-classification pipeline.

    Defaults to ProsusAI/finbert (config: nlp.sentiment_model). The pipeline
    is loaded on first :meth:`score` call and cached; construction is free so
    CLI wiring can build one unconditionally.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings().nlp
        self._model_id = model_id or settings.sentiment_model
        self._device = device or settings.device
        self._batch_size = batch_size or settings.batch_size
        self._pipeline: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline  # heavy import, deferred

            log.info("sentiment_model_loading", model=self._model_id, device=self._device)
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_id,
                device=self._device,
            )
        return self._pipeline

    def score(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        pipe = self._load()
        # truncation: headlines fit easily, but RSS summaries can exceed the
        # 512-token window — truncate rather than crash mid-batch.
        outputs = pipe(texts, batch_size=self._batch_size, truncation=True)
        results: list[SentimentResult] = []
        for out in outputs:
            label = str(out["label"]).lower()
            if label not in SENTIMENT_LABELS:  # defend the DB CHECK constraint
                raise ValueError(f"unexpected sentiment label from {self._model_id}: {label!r}")
            results.append(SentimentResult(label=label, score=float(out["score"])))
        return results


class SentenceTransformerEmbedder:
    """Semantic embeddings via sentence-transformers.

    Defaults to all-MiniLM-L6-v2 (config: nlp.embedding_model), whose 384-dim
    output matches the embeddings schema (sam.storage.models.EMBEDDING_DIM).
    Vectors are L2-normalized so cosine similarity reduces to dot product.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings().nlp
        self._model_id = model_id or settings.embedding_model
        self._device = device or settings.device
        self._batch_size = batch_size or settings.batch_size
        self._model: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            log.info("embedding_model_loading", model=self._model_id, device=self._device)
            self._model = SentenceTransformer(self._model_id, device=self._device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in vector] for vector in vectors]
