"""Resolver-evaluation tests — including THE Phase-3 precision gate.

`test_resolver_precision_gate` is the architecture-M2 acceptance criterion
(">=90% precision on labeled sample") encoded as a permanent regression test:
any dictionary or rule change that drops precision below 0.90 fails CI.
"""

from __future__ import annotations

from pathlib import Path

from sam.processing.evaluate import (
    EVAL_LABELS_PATH,
    PRECISION_GATE,
    LabeledExample,
    evaluate,
    load_labels,
    universe_matcher,
)
from sam.processing.resolver import EntityMatcher, EntityRef


def test_labeled_sample_loads_and_is_substantial() -> None:
    labels = load_labels()
    assert EVAL_LABELS_PATH.exists()
    assert len(labels) >= 50
    positives = [ex for ex in labels if ex.tickers]
    negatives = [ex for ex in labels if not ex.tickers]
    assert len(positives) >= 10  # enough signal to measure recall
    assert len(negatives) >= 30  # enough noise to measure precision


def test_resolver_precision_gate() -> None:
    """Phase-3 gate: >=90% precision on the hand-labeled real sample."""
    report = evaluate()
    assert report.examples >= 50
    assert report.precision >= PRECISION_GATE, (
        f"resolver precision {report.precision:.3f} fell below the {PRECISION_GATE} gate; "
        f"false positives: {report.false_positives}"
    )
    # Not a formal gate, but a recall collapse means dictionary rot — fail loudly.
    assert report.recall >= 0.80, f"resolver recall collapsed: {report.false_negatives}"


def test_evaluate_metric_math() -> None:
    matcher = EntityMatcher(
        [
            EntityRef(entity_id=1, ticker="NVDA", name="NVIDIA Corporation", aliases=("Nvidia",)),
            EntityRef(entity_id=2, ticker="AAPL", name="Apple Inc.", aliases=("Apple",)),
        ]
    )
    labels = [
        LabeledExample("Nvidia beats estimates", frozenset({"NVDA"})),  # TP
        LabeledExample("Apple pie recipes for July 4th", frozenset()),  # FP (known weakness)
        LabeledExample("Cupertino giant ships new phone", frozenset({"AAPL"})),  # FN
        LabeledExample("Rates unchanged", frozenset()),  # true negative
    ]
    report = evaluate(matcher=matcher, labels=labels)
    assert (report.tp, report.fp, report.fn) == (1, 1, 1)
    assert report.precision == 0.5
    assert report.recall == 0.5
    assert report.false_positives == [("Apple pie recipes for July 4th", "AAPL")]
    assert report.false_negatives == [("Cupertino giant ships new phone", "AAPL")]


def test_evaluate_empty_predictions_yield_perfect_precision() -> None:
    matcher = EntityMatcher([])
    labels = [LabeledExample("Nothing to see", frozenset())]
    report = evaluate(matcher=matcher, labels=labels)
    assert report.precision == 1.0  # vacuous: no predictions, no false positives


def test_universe_matcher_builds_from_config() -> None:
    matcher = universe_matcher()
    (match,) = matcher.match("Nvidia rallies")
    assert match.ticker == "NVDA"


def test_labels_file_round_trips_tmp(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"text": "Nvidia up", "tickers": ["NVDA"]}\n\n{"text": "Fed holds", "tickers": []}\n',
        encoding="utf-8",
    )
    labels = load_labels(path)
    assert labels == [
        LabeledExample("Nvidia up", frozenset({"NVDA"})),
        LabeledExample("Fed holds", frozenset()),
    ]
