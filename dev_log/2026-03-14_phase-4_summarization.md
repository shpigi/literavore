# Phase 4: Summary and Tag Generation

**Date:** 2026-03-14

## What was built

- **summarize/llm_client.py**: `LLMClient` with sync/async completion, cost tracking, and mock mode (canned response when no API key — useful for dev and tests)
- **summarize/prompts.py**: `format_summary_prompt` and `format_tag_prompt` helpers building messages lists; prompts request structured JSON output
- **summarize/summarizer.py**: `Summarizer` with async parallel processing via `asyncio.gather` + semaphore, content-hash-based caching (skip if `summaries/{id}.json` exists and hash matches)
- **summarize/tagger.py**: `Tagger` with 8 tag categories (`research_concepts`, `application_domains`, `methodological_approaches`, `key_phrases`, `datasets_benchmarks`, `task_types`, `hardware_platforms`, `theoretical_foundations`), LLM extraction with keyword fallback
- **pipeline.py** updated: `_run_summarize` wired
- **33 new unit tests** with mocked OpenAI

## Decisions

- **Content-hash caching**: summaries are keyed on a SHA-256 of the extracted text + model + prompt version. Re-running with the same text hits the cache; changing the model or prompt invalidates it
- **Mock mode over test fixtures**: having the client self-mock when no key is set makes dev workflows much smoother — no env var juggling
- **8 tag categories**: extended conf_digest's 5 categories with `task_types`, `hardware_platforms`, `theoretical_foundations` for better search faceting

## Outcome

Summarize stage fully wired. 152 unit tests passing.
