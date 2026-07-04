"""Social Arbitrage Index computation (Phase 5 / architecture M4).

Turns resolved + enriched documents into the daily per-entity SAI panel
(``sai_daily``): four sub-signals (mention growth, sentiment momentum, topic
velocity, engagement growth) combined into a composite score and rank.

Dependency rule: this package imports only ``sam.core`` and ``sam.storage``.
It never imports ``sam.ingestion``, ``sam.processing`` or ``sam.nlp`` —
stages communicate exclusively through the database. All math lives in
:mod:`sam.signals.compute` as pure deterministic functions; orchestration and
persistence live in :mod:`sam.signals.pipeline`.

Full methodology (definitions, point-in-time rules, operational caveats):
docs/sai_methodology.md.
"""
