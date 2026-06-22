"""Core cross-cutting concerns: config, logging, db, errors.

This package is a leaf dependency — it must not import from feature
modules (ingestion, nlp, signals, ...). Everything depends on core;
core depends on nothing internal.
"""
