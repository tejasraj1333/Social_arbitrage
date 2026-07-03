"""Sentiment-eval tests (Phase 4 M5).

Metric math and label-file integrity run everywhere (model-free). The real
FinBERT gate test is skipped when the ``nlp`` extra is absent (e.g. CI) and
enforced locally + via `sam enrich --evaluate`.
"""

from __future__ import annotations

import importlib.util

import pytest

from sam.nlp.evaluate import (
    F1_GATE,
    LabeledSentiment,
    evaluate_sentiment,
    load_sentiment_labels,
)
from sam.nlp.models import SENTIMENT_LABELS, SentimentResult
from tests.fakes import FakeSentimentModel

_HAS_NLP_EXTRA = importlib.util.find_spec("transformers") is not None


class _ScriptedModel:
    """Returns a fixed prediction sequence (metric-math fixture)."""

    model_id = "scripted"

    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    def score(self, texts: list[str]) -> list[SentimentResult]:
        assert len(texts) == len(self._labels)
        return [SentimentResult(label=label, score=0.9) for label in self._labels]


def _labels(*labels: str) -> list[LabeledSentiment]:
    return [LabeledSentiment(text=f"headline {i}", label=label) for i, label in enumerate(labels)]


def test_perfect_predictions_score_one() -> None:
    labels = _labels("positive", "negative", "neutral")
    report = evaluate_sentiment(
        model=_ScriptedModel(["positive", "negative", "neutral"]), labels=labels
    )
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.passes_gate
    assert report.misclassified == []


def test_macro_f1_hand_computed_confusion() -> None:
    # truth:      pos, pos, neg, neu
    # predicted:  pos, neu, neg, neu
    report = evaluate_sentiment(
        model=_ScriptedModel(["positive", "neutral", "negative", "neutral"]),
        labels=_labels("positive", "positive", "negative", "neutral"),
    )
    # positive: tp=1 fp=0 fn=1 -> P=1, R=.5, F1=2/3
    # negative: tp=1 fp=0 fn=0 -> F1=1
    # neutral:  tp=1 fp=1 fn=0 -> P=.5, R=1, F1=2/3
    assert report.accuracy == 0.75
    assert abs(report.macro_f1 - (2 / 3 + 1.0 + 2 / 3) / 3) < 1e-9
    assert report.misclassified == [("headline 1", "positive", "neutral")]


def test_macro_f1_skips_absent_labels() -> None:
    # Only two labels present anywhere -> macro over those two, not three.
    report = evaluate_sentiment(
        model=_ScriptedModel(["positive", "negative"]),
        labels=_labels("positive", "negative"),
    )
    assert report.macro_f1 == 1.0


def test_labeled_sample_integrity() -> None:
    """The committed eval set stays big and clean enough to gate on."""
    labels = load_sentiment_labels()
    assert len(labels) >= 50
    assert {example.label for example in labels} == set(SENTIMENT_LABELS)  # all classes present
    per_class = {label: sum(1 for e in labels if e.label == label) for label in SENTIMENT_LABELS}
    assert all(count >= 10 for count in per_class.values()), per_class
    assert len({example.text for example in labels}) == len(labels)  # no duplicate texts


def test_fake_model_runs_through_eval_plumbing() -> None:
    labels = [
        LabeledSentiment(text="Shares surge on earnings", label="positive"),
        LabeledSentiment(text="Markets plunge on tariffs", label="negative"),
    ]
    report = evaluate_sentiment(model=FakeSentimentModel(), labels=labels)
    assert report.examples == 2
    assert report.accuracy == 1.0


@pytest.mark.skipif(not _HAS_NLP_EXTRA, reason="needs the nlp extra (transformers/torch)")
def test_finbert_meets_macro_f1_gate() -> None:
    """THE Phase-4 sentiment gate: macro-F1 >= 0.70 on the labeled sample.

    Loads the real FinBERT (cached under ~/.cache/huggingface after the first
    run). If a model or dictionary change drops macro-F1 below the gate, this
    fails the suite — sentiment quality can never silently regress.
    """
    report = evaluate_sentiment()
    assert report.examples >= 50
    assert report.macro_f1 >= F1_GATE, (
        f"macro-F1 {report.macro_f1:.3f} below gate {F1_GATE}; misses: {report.misclassified[:5]}"
    )
