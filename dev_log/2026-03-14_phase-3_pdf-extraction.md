# Phase 3: PDF Text Extraction

**Date:** 2026-03-14

## What was built

- **extract/pdf_extractor.py**: `extract_pdf(bytes) -> dict` using `pymupdf4llm.to_markdown()` via temp file. Section parsing with regex on `#`/`##` headings, abstract extraction (explicit heading or first paragraph fallback), figure caption extraction from markdown image references. `extract_papers_batch` with `ThreadPoolExecutor` (PyMuPDF releases GIL), optional PDF deletion after extraction
- **pipeline.py** updated: `_run_extract` calls `extract_papers_batch`
- **tests/fixtures/sample.pdf**: minimal 3-section fixture PDF created with pymupdf
- **18 unit tests** covering all helpers and batch processing

## Decisions

- **pymupdf4llm over GROBID**: eliminates the Docker sidecar entirely. ~15MB library, <1s per PDF (vs GROBID's 7s), no network dependency, Markdown output is directly LLM-ready
- **Temp file for pymupdf4llm**: the library accepts a file path, not bytes; writing to a NamedTemporaryFile is the cleanest approach
- **ThreadPoolExecutor for CPU-bound work**: PyMuPDF releases the GIL during rendering, so threads actually parallelize
- **Keep-PDFs flag**: in dev mode PDFs are preserved for inspection; in prod they're deleted after extraction to bound disk usage

## Outcome

Extract stage fully wired. `pymupdf4llm` produces clean Markdown from the fixture PDF. 119 unit tests passing.
