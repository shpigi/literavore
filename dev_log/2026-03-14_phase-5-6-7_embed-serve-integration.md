# Phases 5–7: Embedding, Serving, Integration

**Date:** 2026-03-14

## Phase 5: Embedding and Indexing

- **embed/embedder.py**: `Embedder` with 3 views (`title_abstract`, `paper_card`, `keyword_enriched`), OpenAI `text-embedding-3-large`, L2 normalization via numpy, SHA-256 text-hash cache, mock mode with zero vectors
- **embed/index.py**: `PaperIndex` wrapping FAISS `IndexFlatIP`. `build`, `add`, `search` (with venue filtering), `save`/`load` via `StorageBackend`
- **pipeline.py** updated: `_run_embed` loads summaries, builds views, generates embeddings, saves index
- 48 new unit tests

## Phase 6: Serving

- **serve/models.py**: `SearchRequest`, `SearchResult`, `SearchResponse`, `PaperDetail`, `HealthResponse` (Pydantic v2)
- **serve/api.py**: FastAPI with `POST /search`, `GET /papers/{id}`, `GET /papers`, `GET /health`. Lazy-loaded global state, 503 when index not ready
- **serve/mcp_server.py**: 8 MCP tools via FastMCP — semantic search, author search, paper details, statistics, conference overview, list conferences, recent papers, keyword search
- **serve/streamlit_app.py**: search UI with result cards, plotly scatter plot, paper detail view, graceful API-not-available handling
- **cli.py** updated: `serve` → uvicorn, `ui` → streamlit subprocess, `mcp` → FastMCP stdio
- 37 new tests (unit + integration)

## Phase 7: Integration and Polish

- **tests/integration/test_full_pipeline.py**: 15 end-to-end tests covering state transitions through all 4 stages, idempotency (second run skips done papers, timestamps unchanged), and force rerun. Uses real SQLite/storage/PDF extraction with mocked OpenReview and LLM calls
- **README.md**: full documentation with quick start, commands, config reference, API docs, MCP tools, tech choices
- **Dockerfile**: finalized multi-stage build with src copy in final stage, health check on `/health`

## Final state

- **252 tests passing** (200 unit + 52 integration)
- Full pipeline: fetch → download → extract → summarize → embed → serve
- All stages idempotent with `--force` for re-processing
- Mock mode throughout for development without API keys
- Single Docker container deployment

## Agent collaboration notes

Phases 1–6 used parallel Sonnet workers in isolated git worktrees. Workers consistently wrote correct code but didn't always commit — orchestrator merged by copying files directly. Workers that did commit used their own branch; orchestrator cherry-picked or re-merged as needed. The worktree cleanup policy (cleaned if no commits) caused some file loss that required re-running workers.
