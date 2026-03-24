# Phase 9: Batched Pipeline Processing & Extraction Overhaul

**Date:** 2026-03-15

## What was built

### Batched pipeline processing
- **config.py**: Added `PipelineConfig` with `batch_size` (default 0 = all-at-once)
- **pipeline.py**: Added `_run_batched()` method. When `batch_size > 0`, download/extract/summarize process papers in chunks of N. Fetch runs first, embed runs last, both unbatched. Stage methods (`_run_download`, `_run_extract`, `_run_summarize`) now accept optional `papers` parameter for pre-selected batches
- **cli.py**: Added `--batch-size` / `-b` flag to override config at runtime

### PDF extraction: pymupdf4llm → pypdf + pdfplumber
- **Replaced** `pymupdf` + `pymupdf4llm` with `pypdf` + `pdfplumber` in `pyproject.toml`
- **extract/pdf_extractor.py**: Complete rewrite. Fast path uses `pypdf` (pure Python, fast), falls back to `pdfplumber` (more robust for complex layouts) if pypdf returns <500 chars. Section parsing handles both markdown headings and uppercase academic section headers (ABSTRACT, INTRODUCTION, etc.)
- **ingest/pdf_validator.py**: Replaced `fitz` (pymupdf) validation with `pypdf`
- **Dropped figure extraction** — not needed for summarization/keywords use case

### Extraction timeout with hard kill
- **Problem**: pymupdf4llm's C extension (MuPDF) held the GIL indefinitely on complex PDFs. `signal.alarm` couldn't interrupt C code. `ProcessPoolExecutor` couldn't kill individual workers
- **Solution**: Replaced `ProcessPoolExecutor` with manual `multiprocessing.Process` pool. Parent thread polls workers every 0.5s, tracks wall-clock time, and calls `proc.kill()` on timeout. This is an OS-level kill that works regardless of GIL state
- **Config**: `ExtractConfig.timeout_per_paper` (default 90s), `ExtractConfig.max_retries` (default 3)

### Download tuning
- **config/default.yml**: `pdf.max_concurrent` 10→20, `pdf.delay_between_requests` 1.5s→0.5s, `pipeline.batch_size` 200. ~5x faster downloads despite occasional 429 rate limits (backoff handles them)
- **Fixed** `summary.pricing` defaults to match actual gpt-4.1-nano pricing ($0.20/$0.80 per 1M tokens)

## Decisions

- **pypdf over pymupdf4llm**: pymupdf4llm adds markdown conversion and layout analysis overhead that caused hangs on image-heavy robotics papers. For summarization + keyword extraction, plain text from pypdf is sufficient. pdfplumber fallback handles edge cases with complex layouts
- **Manual process pool over ProcessPoolExecutor**: ProcessPoolExecutor doesn't support killing individual workers. Manual Process management allows per-paper timeout enforcement via `proc.kill()`
- **No retry-with-redownload for extraction timeouts**: Extraction hangs are caused by PDF structure, not corruption. Re-downloading the same PDF won't help. Papers that timeout are marked failed immediately
- **batch_size=0 backward-compatible**: All existing behavior preserved unless explicitly configured

## Outcome

12,509 papers across 4 conferences (CoRL, ICML, ICLR, NeurIPS 2025) processing in batched mode. Download throughput ~2.2 papers/sec. Extraction completes without indefinite hangs. 249 tests passing.
