# Phase 2: Data Ingestion

**Date:** 2026-03-14

## What was built

- **sources/base.py**: `PaperMetadata` Pydantic model and `PaperSource` protocol
- **sources/openreview.py**: `OpenReviewSource` using `openreview.api.OpenReviewClient` V2 API. Venue discovery, pagination via `get_all_notes`, LLM-based venue filtering (gpt-4o-mini with graceful fallback when no API key)
- **normalize/metadata.py**: `extract_value` for V2 API unwrapping, `clean_author_name` (whitespace, comma inversion, unicode NFC), `normalize_paper_metadata` (V1/V2 auto-detection), `simplify_paper_data` batch helper
- **ingest/pdf_downloader.py**: `AsyncPDFDownloader` with aiohttp, semaphore concurrency control, exponential backoff + rate-limit backoff (extra delay on HTTP 429), relative URL resolution for OpenReview `/pdf/` paths
- **ingest/pdf_validator.py**: pikepdf validation (structure, pages > 0, size > 1KB, no encryption)
- **pipeline.py** updated: `_run_fetch` registers papers in DB; `_run_download` uses async context manager
- **28 new unit tests**

## Decisions

- **`get_all_notes` for pagination**: OpenReview's library handles the 1000-item page limit internally; no manual offset logic needed
- **Inline retry in downloader**: the rate-limit backoff (HTTP 429 → extra delay) needed special logic not easily expressed with a generic retry decorator
- **DB-driven skip logic**: downloader checks stage status before downloading; "already done" is a DB query not a filesystem check, so it works with S3 too

## Outcome

Fetch and download stages fully wired. 101 unit tests passing.
