"""Phase-1 source reconnaissance: prove each source can supply usable data.

This is recon, not production ingestion. Collectors here fetch a small sample,
validate its schema, and save it locally (no Postgres). See `collector_base`.
"""
