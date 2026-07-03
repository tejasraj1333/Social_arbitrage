"""Processing stage: entity resolution and data quality (Phase 3).

Sits between ingestion (bronze) and nlp/signals in the pipeline:
documents are linked to tradable entities (document_entities) and the
dataset is continuously checked for quality (data_quality_checks).
This package never imports sam.ingestion — the two stages communicate
only through the database.
"""
