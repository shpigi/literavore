# Phase 1: Core Infrastructure

**Date:** 2026-03-14

## What was built

- **config.py**: Pydantic v2 models (`LiteravoreConfig`, `ConferenceConfig`, `PdfConfig`, etc.) with YAML loading and env var overrides (`LITERAVORE_DEV_MODE`, `LITERAVORE_DATA_DIR`, `LITERAVORE_STORAGE_BACKEND`)
- **db.py**: SQLite state management with tables `papers`, `processing_state`, `runs`. Key query: `get_papers_needing_stage(stage, force=False)` — the backbone of idempotent pipeline execution
- **storage/**: `StorageBackend` runtime-checkable protocol, `LocalStorage` implementation, `S3Storage` stub
- **utils/**: `setup_logging`/`get_logger` with optional stage context, `retry` and `async_retry` decorators with exponential backoff + jitter
- **pipeline.py**: `Pipeline` class with stage dispatch, timing, per-stage error handling, config hashing, DB run tracking
- **cli.py**: Typer commands: `run`, `serve`, `ui`, `mcp`, `status`, `reset`
- **73 unit tests** passing

## Decisions

- **SQLite over Postgres**: single-file, survives container restarts without a sidebase, transactional enough for our write patterns
- **Protocol for StorageBackend**: allows LocalStorage and S3Storage to be swapped without changing call sites
- **`check_same_thread=False` + WAL mode**: makes SQLite safe across the ThreadPoolExecutor used in extract

## Outcome

`literavore --help` works. All CLI commands respond. Pipeline runs with stub stages and completes cleanly.
