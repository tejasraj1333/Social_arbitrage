"""NLP enrichment stage (Phase 4 / architecture M3).

Scores documents with finance-domain sentiment, embeds them for semantic
search (pgvector), and discovers topics. Like ``processing``, this package
never imports ``sam.ingestion`` — pipeline stages communicate only through
the database.

The heavy ML libraries (torch, transformers, sentence-transformers, bertopic)
live in the optional ``nlp`` extra and are imported lazily inside model
wrappers, so importing this package — and everything that only needs its
interfaces — works without them.
"""
