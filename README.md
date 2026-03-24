# Literavore

Conference paper processing pipeline — fetch, extract, summarize, embed, search.

A reimplementation of [conf_digest](../conf_digest) that drops Kedro and GROBID in favor of plain Python with SQLite state tracking and pypdf/pdfplumber for fast in-process PDF extraction.

## Architecture

```
fetch → download → extract → summarize → embed → serve
```

Each stage is a function: read DB → process → write storage → update DB.

```
src/literavore/
├── cli.py          # Typer CLI
├── config.py       # Pydantic v2 config models
├── db.py           # SQLite state management
├── pipeline.py     # Stage orchestrator (~175 LOC)
├── storage/        # Local + S3 backends
├── sources/        # OpenReview fetcher
├── ingest/         # Async PDF downloader + pikepdf validator
├── extract/        # PDF text extraction (pypdf + pdfplumber fallback)
├── normalize/      # Metadata normalization
├── summarize/      # OpenAI summaries + tag extraction
├── embed/          # OpenAI embeddings + FAISS index
└── serve/          # FastAPI + MCP + Streamlit
```

## Quick Start

### Docker

```bash
cp .env.example .env
# edit .env — set OPENAI_API_KEY
docker compose up
# API at http://localhost:8000
# UI  at http://localhost:8501
```

### Local

```bash
uv sync
export OPENAI_API_KEY=sk-...
literavore run --config config/default.yml --dev
literavore serve    # FastAPI on :8000
literavore ui       # Streamlit on :8501
literavore mcp      # MCP server (stdio)
```

## Commands

```bash
# Pipeline
literavore run                          # Run all stages
literavore run --stage extract          # Run one stage
literavore run --from-stage summarize   # Resume from a stage
literavore run --force                  # Re-process everything
literavore run --dev                    # Keep PDFs, smaller batches

# Serve
literavore serve [--host HOST] [--port PORT]
literavore ui    [--port PORT]
literavore mcp

# Inspect
literavore status
literavore reset [--stage STAGE]
```

## Configuration

Edit `config/default.yml` or use environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for summarization and embedding |
| `LITERAVORE_CONFIG` | `config/default.yml` | Config file path |
| `LITERAVORE_DEV_MODE` | `false` | Keep PDFs, smaller batches |
| `LITERAVORE_DATA_DIR` | `data` | Data directory |
| `LITERAVORE_STORAGE_BACKEND` | `local` | `local` or `s3` |

## API

```bash
# Health check
curl localhost:8000/health

# List conferences
curl localhost:8000/conferences

# Semantic search
curl -X POST localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "transformer robotics", "top_k": 5}'

# Paper detail (includes summary, tags, openreview_url, pdf_url)
curl localhost:8000/papers/PAPER_ID

# List papers
curl 'localhost:8000/papers?conference=CoRL-2025'

# UMAP 2D projection of all paper embeddings (cached after first call)
curl localhost:8000/umap
```

## UI

The Streamlit UI (`literavore ui`) provides:

- **Search**: semantic query via text input (Enter or button)
- **UMAP paper map**: all papers projected to 2D; search results highlighted in red with rank labels
- **Result cards**: clickable title (opens OpenReview in new tab), AI summary, tags
- **Paper detail**: full author list, abstract, AI summary, tags, OpenReview + PDF links

## MCP Tools

The MCP server exposes 8 tools to Claude Desktop / Gemini CLI:

- `search_papers_semantic` — vector search across all papers
- `search_papers_by_author` — find papers by author name
- `get_paper_details` — full metadata, AI summary, tags, OpenReview + PDF URLs
- `get_paper_statistics` — counts per conference and stage
- `get_conference_overview` — per-conference breakdown
- `list_conferences` — all conferences in the index
- `get_recent_papers` — latest additions
- `search_by_keywords` — keyword AND-match search

## Development

```bash
uv sync
pytest                   # 252 tests
pytest tests/unit/       # Unit tests only
pytest tests/integration # Integration tests
ruff check src/          # Lint
```

## Technology Choices

| Concern | Choice | Why |
|---|---|---|
| PDF extraction | pypdf + pdfplumber | Pure Python, fast, no C extension hangs |
| Pipeline | Plain Python (~175 LOC) | Kedro was more hassle than help |
| State | SQLite | Single file, SQL-queryable, transactional |
| Storage | Local + S3-compatible | Start local, swap via config |
| LLM | gpt-4o-mini | Proven in conf_digest |
| CLI | Typer | Type-hint-based, auto-generated help |
| API | FastAPI | Same as conf_digest |
| UI | Streamlit | Interactive search + UMAP visualization |
| Agents | MCP (FastMCP) | Claude Desktop / Gemini CLI |
| Dim. reduction | UMAP + numba | 2D paper map from high-dim embeddings |

## Agentic Development

This project was built by a Claude Opus orchestrator coordinating parallel Claude Sonnet workers. Each worker implemented one task in an isolated git worktree; the orchestrator merged, tested, and committed. See `agents.md` for coordination rules and `dev_log/` for timestamped build entries.
