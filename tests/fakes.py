"""Deterministic in-memory NLP model fakes.

Unit tests inject these instead of the real FinBERT/SentenceTransformer so
the suite never needs torch (the ``nlp`` extra) and stays fast. They satisfy
the sam.nlp.models protocols — asserted via isinstance in
tests/test_nlp_models.py (the protocols are runtime_checkable).
"""

from __future__ import annotations

from sam.nlp.models import SentimentResult
from sam.storage.models import EMBEDDING_DIM


class FakeSentimentModel:
    """Keyword-driven sentiment: 'surge' -> positive, 'plunge' -> negative."""

    model_id = "fake-sentiment-v1"

    def score(self, texts: list[str]) -> list[SentimentResult]:
        results: list[SentimentResult] = []
        for text in texts:
            lowered = text.lower()
            if "surge" in lowered:
                results.append(SentimentResult(label="positive", score=0.9))
            elif "plunge" in lowered:
                results.append(SentimentResult(label="negative", score=0.8))
            else:
                results.append(SentimentResult(label="neutral", score=0.6))
        return results


class FakeEmbeddingModel:
    """Deterministic text-dependent vectors at the real schema width."""

    model_id = "fake-embedder-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[(len(text) % 7) / 7.0] * EMBEDDING_DIM for text in texts]
