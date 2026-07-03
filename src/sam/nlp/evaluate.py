"""Sentiment evaluation against the hand-labeled sample (the Phase-4 gate).

``data/eval/sentiment_labels.jsonl`` holds real ingested headlines labeled
positive/negative/neutral under the policy in docs/nlp_enrichment.md (label
by the implied market/business direction for the headline's subject; neutral
for purely factual or balanced items). **Macro-F1 across the three labels is
the gate: >= 0.70**, pre-registered before the first measurement — the same
no-p-hacking discipline as the P3 precision gate.

Runs as ``sam enrich --evaluate`` and as a pytest that is skipped when the
``nlp`` extra isn't installed (metric math itself is model-free and always
tested).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sam.core.config import PROJECT_ROOT
from sam.core.logging import get_logger
from sam.nlp.models import SENTIMENT_LABELS, FinBertSentiment, SentimentModel

log = get_logger("nlp.evaluate")

SENTIMENT_LABELS_PATH = PROJECT_ROOT / "data" / "eval" / "sentiment_labels.jsonl"
F1_GATE = 0.70


@dataclass(frozen=True, slots=True)
class LabeledSentiment:
    """One hand-labeled headline: the text and its sentiment label."""

    text: str
    label: str


@dataclass(slots=True)
class SentimentEvalReport:
    """Per-label and macro metrics of the model on the labeled sample."""

    examples: int = 0
    correct: int = 0
    tp: dict[str, int] = field(default_factory=dict)
    fp: dict[str, int] = field(default_factory=dict)
    fn: dict[str, int] = field(default_factory=dict)
    # (text, expected, predicted) for every miss — debugging, not just a number.
    misclassified: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.examples if self.examples else 1.0

    def f1(self, label: str) -> float:
        tp = self.tp.get(label, 0)
        fp = self.fp.get(label, 0)
        fn = self.fn.get(label, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    @property
    def macro_f1(self) -> float:
        """Mean F1 over labels that occur in the sample or the predictions."""
        present = [
            label
            for label in SENTIMENT_LABELS
            if (self.tp.get(label, 0) + self.fp.get(label, 0) + self.fn.get(label, 0)) > 0
        ]
        return sum(self.f1(label) for label in present) / len(present) if present else 0.0

    @property
    def passes_gate(self) -> bool:
        return self.macro_f1 >= F1_GATE


def load_sentiment_labels(path: Path = SENTIMENT_LABELS_PATH) -> list[LabeledSentiment]:
    """Parse the labeled JSONL sample, validating labels against the schema."""
    examples: list[LabeledSentiment] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["label"] not in SENTIMENT_LABELS:
                raise ValueError(f"invalid label in {path.name}: {record['label']!r}")
            examples.append(LabeledSentiment(text=record["text"], label=record["label"]))
    return examples


def evaluate_sentiment(
    model: SentimentModel | None = None,
    labels: list[LabeledSentiment] | None = None,
) -> SentimentEvalReport:
    """Score the model against the labeled sample (loads FinBERT by default)."""
    model = model if model is not None else FinBertSentiment()
    labels = labels if labels is not None else load_sentiment_labels()

    predictions = model.score([example.text for example in labels])
    report = SentimentEvalReport(examples=len(labels))
    for example, prediction in zip(labels, predictions, strict=True):
        if prediction.label == example.label:
            report.correct += 1
            report.tp[example.label] = report.tp.get(example.label, 0) + 1
        else:
            report.fp[prediction.label] = report.fp.get(prediction.label, 0) + 1
            report.fn[example.label] = report.fn.get(example.label, 0) + 1
            report.misclassified.append((example.text, example.label, prediction.label))

    log.info(
        "sentiment_evaluated",
        model=model.model_id,
        examples=report.examples,
        accuracy=round(report.accuracy, 4),
        macro_f1=round(report.macro_f1, 4),
        per_label={label: round(report.f1(label), 4) for label in SENTIMENT_LABELS},
        gate=F1_GATE,
        passes=report.passes_gate,
    )
    return report
