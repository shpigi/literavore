# Literavore — Developer Guide

## Overview

Literavore is a conference paper processing pipeline that fetches academic papers from OpenReview, extracts text with pymupdf4llm, generates AI summaries/tags, builds vector indexes, and serves semantic search. It replaces conf_digest, dropping Kedro and GROBID in favor of plain Python with SQLite state tracking.

## Architecture

Six pipeline stages + serve layer:

```
fetch → download → extract → summarize → embed → [serve]
```

Each stage is a function: read DB → process → write storage → update DB.

## Key Patterns

- **Config**: Pydantic v2 models loaded from YAML (`config/default.yml`), env var overrides
- **State**: SQLite database tracks every paper and its processing status per stage
- **Storage**: Abstract protocol (`StorageBackend`) with local filesystem and S3 backends
- **Sources**: Abstract protocol (`PaperSource`) for fetching paper metadata
- **Pipeline**: Plain Python orchestrator (~200 LOC), no framework

## Project Structure

```
src/literavore/
├── cli.py          # Typer CLI
├── config.py       # Pydantic config models
├── db.py           # SQLite state management
├── pipeline.py     # Stage runner
├── storage/        # Storage backends (local, s3)
├── sources/        # Paper sources (openreview)
├── ingest/         # PDF download + validation
├── extract/        # pymupdf4llm text extraction
├── normalize/      # Metadata normalization
├── summarize/      # LLM summaries + tags
├── embed/          # Embeddings + FAISS index
├── serve/          # FastAPI + MCP + Streamlit
└── utils/          # Logging, retry utilities
```

## Commands

```bash
# Install
uv sync

# Run pipeline
literavore run --config config/default.yml
literavore run --stage extract --force  # Re-run single stage
literavore run --dev                    # Dev mode (keeps PDFs)

# Serve
literavore serve                        # FastAPI on :8000
literavore ui                           # Streamlit on :8501
literavore mcp                          # MCP server (stdio)

# Status
literavore status                       # Paper counts per stage

# Test
pytest                                  # All tests
pytest tests/unit/                      # Unit tests only
ruff check src/                         # Lint
```

## Environment Variables

- `OPENAI_API_KEY` — Required for summarization and embedding
- `LITERAVORE_CONFIG` — Config file path (default: `config/default.yml`)
- `LITERAVORE_DEV_MODE` — Keep PDFs after extraction, smaller batches
- `LITERAVORE_DATA_DIR` — Data directory (default: `data`)
- `LITERAVORE_STORAGE_BACKEND` — `local` or `s3`

## Style Guide

- Python 3.11+, type hints on all public functions
- ruff formatting, line-length 100
- Pydantic v2 for all data models
- async for I/O-bound work (downloads, LLM calls)
- ThreadPoolExecutor for CPU-bound work (PDF extraction)

## Agentic Development

This project is built by an Opus orchestrator coordinating parallel Sonnet workers. See `agents.md` for coordination rules. Commit messages follow `phase-N.M: description` format. Dev logs in `dev_log/`.
