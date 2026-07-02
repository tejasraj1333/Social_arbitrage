# Source Scorecard

_Generated 2026-07-02 10:04 UTC by `sam recon`. Phase-1 source reconnaissance._

| source | status | record_count | schema_completeness | freshness_hours | estimated_monthly_volume | collection_difficulty | recommended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rss | ok | 100 | 1.0 | 0.11 | 70.0 | low | True |
| yahoo | ok | 1255 | 1.0 | 34.07 | 103.0 | low | True |
| hackernews | ok | 100 | 0.99 | 1.51 | 1135.0 | low | True |
| reddit | needs_credentials | 0 | 0.0 |  |  | medium | False |
| kaggle | needs_credentials | 0 | 0.0 |  |  | medium | False |

## Legend

- **schema_completeness** — mean fraction of required fields present (non-null) across fetched records (1.0 = every required field on every record).
- **freshness_hours** — age of the most recent record at collection time (blank where not time-stamped, e.g. price bars / dataset metadata).
- **estimated_monthly_volume** — records/month extrapolated from the sample's observed time span (order-of-magnitude; top-N feeds compress the span).
- **collection_difficulty** — operational effort/risk to run at scale.
- **recommended** — `True` when status is `ok` and completeness ≥ 0.80.
- **needs_credentials** — collector is built + unit-tested; runs live once credentials are supplied (see `docs/legal_register.md`).

## Collection-difficulty notes

- **rss** — public feeds, no auth; only flakiness is dead/changed feed URLs
- **yahoo** — unofficial API, no auth; risk is undocumented rate limits / breakage
- **hackernews** — official keyless API; 1 request per item is the only friction
- **reddit** — OAuth app required; 100 req/min and listing-depth caps
- **kaggle** — API token required; redistribution constrained by per-dataset license
