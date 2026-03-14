# Phase 0: Project Scaffold

**Date:** 2026-03-14
**Phases:** 0

## What was built

- `pyproject.toml` with all dependencies (pymupdf4llm, faiss-cpu, FastAPI, FastMCP, Streamlit, etc.)
- `CLAUDE.md` and `agents.md` — developer guide and agent coordination rules
- `config/default.yml` — default config with CoRL-2025, ICML-2025, ICLR-2025, NeurIPS-2024
- `Dockerfile` (multi-stage, uv-based) and `docker-compose.yml`
- `.env.example`, `.gitignore`, `README.md` stub
- Full directory scaffold with `__init__.py` files for all modules
- `tests/conftest.py` with shared fixtures

## Decisions

- **uv over pip/poetry**: faster installs, better Docker layer caching, lockfile discipline
- **hatchling build backend**: simpler than setuptools for pure Python projects
- **`src/literavore/` layout**: standard src layout avoids accidental import of development tree

## Outcome

`uv sync` succeeds, `import literavore` works. Foundation ready for Phase 1.
