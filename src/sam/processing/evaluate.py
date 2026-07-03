"""Resolver evaluation against the hand-labeled sample (the Phase-3 gate).

``data/eval/entity_labels.jsonl`` holds real ingested headlines labeled with
the universe tickers they actually mention (empty list = mentions none).
Precision over (example, ticker) prediction pairs is the architecture-M2
gate: **>= 0.90**. The gate runs both as `sam resolve --evaluate` and as a
permanent pytest, so resolver precision can never silently regress.

Evaluation is DB-free: the dictionary is built straight from the config
universe, which `sam seed --update` keeps in lockstep with the entities
table — the eval measures the same rules production applies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sam.core.config import PROJECT_ROOT
from sam.core.logging import get_logger
from sam.processing.resolver import EntityMatcher, EntityRef
from sam.recon.sources import load_sources

log = get_logger("processing.evaluate")

EVAL_LABELS_PATH = PROJECT_ROOT / "data" / "eval" / "entity_labels.jsonl"
PRECISION_GATE = 0.90


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """One hand-labeled headline: the text and the tickers it truly mentions."""

    text: str
    tickers: frozenset[str]


@dataclass(slots=True)
class EvalReport:
    """Pairwise precision/recall of the resolver on the labeled sample."""

    examples: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # (text, ticker) evidence for every miss — debugging, not just a number.
    false_positives: list[tuple[str, str]] = field(default_factory=list)
    false_negatives: list[tuple[str, str]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def passes_gate(self) -> bool:
        return self.precision >= PRECISION_GATE


def load_labels(path: Path = EVAL_LABELS_PATH) -> list[LabeledExample]:
    """Parse the labeled JSONL sample."""
    examples: list[LabeledExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(
                LabeledExample(text=record["text"], tickers=frozenset(record["tickers"]))
            )
    return examples


def universe_matcher() -> EntityMatcher:
    """Matcher over the config universe (ids are synthetic — eval never hits a DB)."""
    universe = load_sources().get("universe", [])
    refs = [
        EntityRef(
            entity_id=idx,
            ticker=item["ticker"],
            name=item.get("name", item["ticker"]),
            aliases=tuple(item.get("aliases", [])),
        )
        for idx, item in enumerate(universe, start=1)
    ]
    return EntityMatcher(refs)


def evaluate(
    matcher: EntityMatcher | None = None,
    labels: list[LabeledExample] | None = None,
) -> EvalReport:
    """Score the matcher against the labeled sample, pairwise per ticker."""
    matcher = matcher if matcher is not None else universe_matcher()
    labels = labels if labels is not None else load_labels()

    report = EvalReport(examples=len(labels))
    for example in labels:
        predicted = {match.ticker for match in matcher.match(example.text)}
        for ticker in sorted(predicted - example.tickers):
            report.fp += 1
            report.false_positives.append((example.text, ticker))
        for ticker in sorted(example.tickers - predicted):
            report.fn += 1
            report.false_negatives.append((example.text, ticker))
        report.tp += len(predicted & example.tickers)

    log.info(
        "resolver_evaluated",
        examples=report.examples,
        precision=round(report.precision, 4),
        recall=round(report.recall, 4),
        f1=round(report.f1, 4),
        tp=report.tp,
        fp=report.fp,
        fn=report.fn,
        gate=PRECISION_GATE,
        passes=report.passes_gate,
    )
    return report
