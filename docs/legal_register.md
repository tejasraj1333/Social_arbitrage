# Legal & Terms-of-Service Register — Phase 1 Sources

Operational/legal register for every source evaluated in Phase-1 recon. It records
how each source is accessed, what authentication and rate limits apply, the
material Terms-of-Service constraints, and attribution requirements.

> **Disclaimer:** This is engineering due-diligence, not legal advice. Terms change;
> re-verify against the live ToS before any production or commercial use, and route
> commercial/redistribution questions through counsel.

## Summary

| Source | Access method | Auth | Rate limit | Commercial use / redistribution | Attribution |
| --- | --- | --- | --- | --- | --- |
| Reddit | Data API (OAuth2) via PRAW | client_id + secret (script app) | ~100 QPM/client (10-min avg) | Restricted — commercial/ML use needs Reddit agreement | Per Reddit brand/API terms; honor deletions |
| News RSS | RSS/Atom over HTTP via feedparser | None | None formal — be polite, cache | Headline/summary/link only; no full-text reproduction | Link back + name publisher |
| Yahoo Finance | Unofficial endpoints via yfinance | None | Undocumented; throttle to avoid bans | ToS = personal, non-commercial only | Cite Yahoo; migrate off for production |
| Hacker News | Official Firebase REST API | None | None official — stay reasonable | Permissive for non-abusive use | Link to HN item / original |
| Kaggle | Official Kaggle API | API token (kaggle.json) | Reasonable-use | **Per-dataset license governs** | Per dataset license (e.g. CC-BY) |

---

## Reddit

- **API usage method:** Official Reddit Data API over OAuth2, accessed through PRAW
  (read-only mode — public listings, no user context). We pull subreddit `hot`
  listings for `wallstreetbets`, `stocks`, `investing`.
- **Authentication:** Registered "script" app at <https://www.reddit.com/prefs/apps>
  → `client_id`, `client_secret`, and a descriptive `user_agent`. Supplied via
  `SAM_REDDIT__CLIENT_ID`, `SAM_REDDIT__CLIENT_SECRET`, `SAM_REDDIT__USER_AGENT`.
- **Rate limits:** ~100 queries/minute per OAuth client, averaged over a 10-minute
  window (post-2023 API terms). Listing depth is capped (~1000 items per listing).
  PRAW handles backoff via the `X-Ratelimit-*` headers.
- **ToS concerns:** Governed by the Reddit Data API Terms (2023). Commercial use,
  data resale, and using Reddit content to train ML models generally require a
  separate commercial agreement with Reddit. Bulk export/redistribution of raw
  content is prohibited. **Content deletions must be honored** — if a user deletes a
  post/comment, downstream stores must drop it.
- **Attribution / PII:** Follow Reddit brand guidelines when surfacing content.
  Author usernames are public but are PII-adjacent — store only what the signal needs
  and avoid building user-level profiles.
- **Phase-1 status:** Built + unit-tested; not yet run live (no credentials).

## News RSS (CNBC, MarketWatch, WSJ Markets, Nasdaq; Reuters retired)

- **API usage method:** Standard RSS/Atom feeds fetched over HTTPS and parsed with
  feedparser. Feeds + names live in `config/sources.yaml`.
- **Authentication:** None.
- **Rate limits:** No formal published limit. Be a good citizen: cache, use
  conditional GET (`ETag`/`If-Modified-Since`), poll at sane intervals (≥15 min), and
  set a descriptive User-Agent.
- **ToS concerns:** Each publisher's website terms apply. RSS feeds are offered for
  personal aggregation and linking — **store headline + short summary + link only;
  do not reproduce or scrape full article bodies.** Full-text or paywalled content
  (e.g. WSJ articles) must not be harvested. Reuters retired its public RSS feeds, so
  it is listed but expected to be empty/unreliable.
- **Attribution:** Display the source/publisher name and link to the original article.
- **Phase-1 status:** Live; 100 records, schema completeness 1.0.

## Yahoo Finance (yfinance)

- **API usage method:** `yfinance` library, which calls Yahoo's **unofficial** chart/
  quote endpoints. Daily OHLCV + Adjusted Close for the configured universe.
- **Authentication:** None.
- **Rate limits:** Undocumented and enforced opaquely; aggressive polling triggers
  temporary IP throttling/blocks. Keep request volume modest and add backoff.
- **ToS concerns:** Yahoo's Terms restrict data to **personal, non-commercial use**
  and prohibit redistribution; there is no API SLA and the endpoints can change or
  break without notice. Treating an unofficial scraper as production market data is a
  legal **and** reliability risk.
- **Attribution:** Cite Yahoo Finance as the source.
- **Recommendation:** Excellent for research/backtests; **migrate to a licensed
  market-data vendor** (e.g. a paid EOD/intraday feed) before any production or
  commercial deployment.
- **Phase-1 status:** Live; 1255 rows (5 tickers × ~1y daily), schema completeness 1.0.

## Hacker News (Firebase API)

- **API usage method:** Official public HN Firebase REST API
  (`/v0/topstories.json`, `/v0/item/{id}.json`). One request per item to hydrate.
- **Authentication:** None.
- **Rate limits:** None officially documented; the API is intended for reasonable,
  non-abusive use. We retry transient failures with exponential backoff (tenacity).
- **ToS concerns:** Content is user-generated and owned by submitters; the API is
  provided by Y Combinator for general use. No bulk-abuse, no misrepresentation.
- **Attribution:** Link back to the HN discussion (`news.ycombinator.com/item?id=…`)
  and/or the original story URL. Usernames are public.
- **Phase-1 status:** Live; 100 records, schema completeness 1.0.

## Kaggle Datasets

- **API usage method:** Official Kaggle API. Phase-1 uses **metadata-only** queries
  (`dataset_list(search=…)`) — ref, title, size, license, last-updated, popularity.
  **No bulk dataset downloads.**
- **Authentication:** Kaggle API token — `kaggle.json` (`username` + `key`) in
  `~/.kaggle/`, or `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars. The client
  authenticates eagerly at import and exits if unconfigured.
- **Rate limits:** Reasonable-use; abusive automated downloading is disallowed.
- **ToS concerns:** Kaggle's site Terms apply, **and each dataset carries its own
  license** (CC0, CC-BY, CC-BY-SA, GPL, "other", or custom). Redistribution and
  commercial use depend entirely on that per-dataset license — **check `licenseName`
  before ingesting any dataset.** Do not redistribute raw datasets; derive features
  only where the license permits.
- **Attribution:** Per the dataset's license (e.g. CC-BY requires crediting the author).
- **Phase-1 status:** Built + unit-tested; not yet run live (no credentials).

---

## Compliance checklist (carry into Phase 2)

- [ ] Store `ingested_at` separately from `published_at` (point-in-time correctness).
- [ ] Persist the source license/ToS basis alongside ingested records.
- [ ] RSS/news: store headline + summary + link only — never full article text.
- [ ] Reddit: implement a deletion-honoring path; do not train models on content
      without a commercial agreement.
- [ ] Yahoo: plan migration to a licensed market-data vendor before production.
- [ ] Kaggle: gate ingestion on a per-dataset license allowlist (CC0/CC-BY first).
- [ ] Respect `robots.txt`, conditional GET, polite rate limits, descriptive
      User-Agent on every collector.
