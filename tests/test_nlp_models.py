"""Model-abstraction tests (Phase 4 M2) — run without torch installed.

The real models are exercised live via the CLI (documented in
docs/nlp_enrichment.md); here we test the wrapper logic with injected fake
pipelines and prove the lazy-import contract that keeps the ``nlp`` extra
optional.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from sam.nlp.models import (
    SENTIMENT_LABELS,
    EmbeddingModel,
    FinBertSentiment,
    SentenceTransformerEmbedder,
    SentimentModel,
    SentimentResult,
)
from sam.storage.models import EMBEDDING_DIM
from tests.fakes import FakeEmbeddingModel, FakeSentimentModel


def test_nlp_package_imports_without_heavy_libs() -> None:
    """Importing sam.nlp must not pull torch/transformers (lazy-load contract)."""
    code = (
        "import sys; import sam.nlp.models, sam.nlp; "
        "heavy = {'torch', 'transformers', 'sentence_transformers'} & set(sys.modules); "
        "assert not heavy, f'heavy libs imported eagerly: {heavy}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_fakes_and_real_wrappers_satisfy_protocols() -> None:
    assert isinstance(FakeSentimentModel(), SentimentModel)
    assert isinstance(FakeEmbeddingModel(), EmbeddingModel)
    assert isinstance(FinBertSentiment("m"), SentimentModel)
    assert isinstance(SentenceTransformerEmbedder("m"), EmbeddingModel)


def test_fake_sentiment_emits_valid_labels() -> None:
    results = FakeSentimentModel().score(["Nvidia shares surge", "Stocks plunge", "Fed holds"])
    assert [r.label for r in results] == ["positive", "negative", "neutral"]
    assert all(r.label in SENTIMENT_LABELS for r in results)


def test_fake_embedder_matches_schema_width() -> None:
    vectors = FakeEmbeddingModel().embed(["a", "bb"])
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


class _FakePipeline:
    """Stands in for the transformers pipeline object."""

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str], **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(texts)
        return self.outputs


def test_finbert_normalizes_labels_and_scores() -> None:
    model = FinBertSentiment("test-model")
    model._pipeline = _FakePipeline(
        [{"label": "Positive", "score": "0.93"}, {"label": "NEUTRAL", "score": 0.55}]
    )
    results = model.score(["up big", "sideways"])
    assert results == [
        SentimentResult(label="positive", score=0.93),
        SentimentResult(label="neutral", score=0.55),
    ]
    assert model.model_id == "test-model"


def test_finbert_rejects_labels_outside_schema() -> None:
    model = FinBertSentiment("test-model")
    model._pipeline = _FakePipeline([{"label": "bullish", "score": 0.9}])
    with pytest.raises(ValueError, match="unexpected sentiment label"):
        model.score(["to the moon"])


def test_finbert_empty_input_never_loads_the_model() -> None:
    model = FinBertSentiment("test-model")
    assert model.score([]) == []
    assert model._pipeline is None  # nothing was loaded


class _FakeSbert:
    """Stands in for the SentenceTransformer object."""

    def encode(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.25, 0.5] for _ in texts]


def test_embedder_converts_output_to_float_lists() -> None:
    model = SentenceTransformerEmbedder("test-embedder")
    model._model = _FakeSbert()
    vectors = model.embed(["a", "b"])
    assert vectors == [[0.25, 0.5], [0.25, 0.5]]
    assert all(isinstance(x, float) for vector in vectors for x in vector)


def test_embedder_empty_input_never_loads_the_model() -> None:
    model = SentenceTransformerEmbedder("test-embedder")
    assert model.embed([]) == []
    assert model._model is None
