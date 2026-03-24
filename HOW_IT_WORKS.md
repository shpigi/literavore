# How Literavore Works

Literavore is a paper-processing pipeline that turns a conference listing on OpenReview into a searchable, AI-enriched research library. It fetches PDFs, reads them, asks an LLM to summarize and classify each one, embeds the results as semantic vectors, and exposes everything through a search API, an MCP server, and an interactive UI.

This document walks through how each stage works, what the LLMs are asked to do, and why specific packages were chosen.

---

## The Big Picture

Every paper moves through five sequential stages:

```
fetch → download → extract → summarize → embed → serve
```

State is tracked in SQLite: each paper has a status per stage (`pending`, `running`, `done`, `failed`). Stages are idempotent — if a run is interrupted, re-running picks up where it left off. Any stage can be re-forced individually without re-running the whole pipeline. The orchestrator is ~200 lines of plain Python with no external task framework.

---

## Stage 1 — Fetch

The pipeline queries the OpenReview API to collect paper metadata: titles, abstracts, author lists, PDF URLs, and any keywords the authors provided. These records are written to SQLite and serve as the canonical source of truth for downstream stages.

OpenReview is the only source implemented today, but the `PaperSource` protocol is abstract — adding arXiv or Semantic Scholar would mean implementing a single interface.

### LLM-Assisted Venue Filtering

**Model: `gpt-4o-mini` | Temperature: 0**

OpenReview's venue structure is opaque. A single conference is broken into many sub-venues with machine-generated IDs like `CoRL.cc/2025/Conference/Oral_Submissions` or `CoRL.cc/2025/Conference/Workshop`. Filtering to only "Oral" or "Spotlight" papers via string matching would require brittle, conference-specific rules.

Instead, when a `filter_for` list is configured (e.g., `["Oral", "Spotlight"]`), the full list of discovered venue IDs is sent to `gpt-4o-mini` and the model is asked to return only those that match. Temperature is 0 — this is pure classification, the most deterministic possible setting. If the API key is absent, the call fails, or the model returns an empty list, the pipeline falls back to using all venue IDs rather than dropping papers silently.

---

## Stage 2 — Download

PDFs are fetched asynchronously with a configurable concurrency limit, per-request delay, and an exponential-backoff retry strategy with jitter. Rate limiting is intentionally conservative: 10 concurrent requests and a 1.5-second delay between them. Each PDF is validated after download.

In production mode, PDFs are deleted after the extraction stage to keep storage costs down. In dev mode, they are kept.

---

## Stage 3 — Extract

**Packages: `pypdf` (primary), `pdfplumber` (fallback)**

PDF parsing is notoriously messy. The extractor uses a two-tier approach: `pypdf` handles the fast path (pure Python, reliable for most born-digital PDFs), and `pdfplumber` kicks in as a fallback when pypdf extracts too little text (complex layouts, unusual fonts).

The extractor:
1. Extracts text via `pypdf.PdfReader`, falling back to `pdfplumber` if the result is under 500 characters
2. Splits the result on section headers (Markdown headings or uppercase academic headers like ABSTRACT, INTRODUCTION)
3. Extracts the abstract (explicit "Abstract" section, or the first substantial paragraph)

Each paper is extracted in a separate `multiprocessing.Process` with a configurable wall-clock timeout (default 90s). The parent process polls workers and calls `proc.kill()` on any that exceed the timeout — a hard OS-level kill that works even if a C extension is holding the GIL. Four workers run in parallel.

Output is stored as structured JSON per paper: `{full_text, abstract, sections, figures}`.

---

## Stage 4 — Summarize

This is where the LLMs do their work. Two sequential LLM calls are made per paper.

### Call 1: Summary

**Model: `gpt-4o-mini` | Temperature: 0.3 | Max tokens: 500**

The model receives the paper's title, abstract, and up to 3,000 characters of extracted body text. It is asked to return a JSON object with a 3–5 sentence summary and a flat list of 3–8 lowercase, hyphenated tags.

System prompt:
```
You are a scientific paper analyst. Given a paper's title, abstract, and extracted text,
produce a concise summary and extract relevant tags.

Respond with valid JSON only, using this exact schema:
{
  "summary": "<3-5 sentence summary of the paper's contribution and methods>",
  "tags": ["<tag1>", "<tag2>", "..."]
}

Tags should be lowercase, hyphenated phrases (e.g., "deep-learning", "natural-language-processing").
Include 3-8 tags covering the paper's domain, methods, and key contributions.
```

Temperature 0.3 produces focused, factual summaries without the model embellishing beyond what the paper says.

### Call 2: Structured Tag Extraction

**Model: `gpt-4o-mini` | Temperature: 0.1 | Max tokens: 300**

A second call refines the classification into a structured schema with four fields:

```json
{
  "key_phrases": ["..."],
  "domains": ["..."],
  "methods": ["..."],
  "datasets_benchmarks": ["imagenet", "glue", "..."]
}
```

The tagger receives the title, abstract, summary from Call 1, and any author-supplied keywords from OpenReview. Temperature 0.1 makes this call highly consistent — it's classification, not generation.

Separating `datasets_benchmarks` into its own field is deliberate: knowing exactly which benchmarks a paper uses is valuable for filtering (e.g., "show me all papers that evaluate on COCO") and for assessing coverage.

### Concurrency and Cost Tracking

Up to 10 summaries run concurrently via an asyncio semaphore, with a configured RPM cap to stay within API rate limits. The client accumulates token counts across all calls and computes estimated cost at the configured rates ($0.15/1M input, $0.60/1M output for gpt-4o-mini).

Summaries are cached by content hash: if a paper's extracted text hasn't changed since the last run, the stored summary is reused without calling the API.

---

## Stage 5 — Embed

**Package: OpenAI `text-embedding-3-large` | Dimensions: 3072 | Package: `faiss`**

### Why text-embedding-3-large

OpenAI's `text-embedding-3-large` is the highest-quality embedding model available in the API as of this writing. At 3072 dimensions, it captures fine-grained semantic relationships that matter for research paper retrieval — the difference between "policy gradient" and "value-based reinforcement learning" is meaningful and needs to be preserved in the vector space.

The model supports Matryoshka dimensionality reduction, meaning vectors can be truncated to smaller sizes while retaining most of their quality. The 3072-dimension setting is used by default but is configurable.

### Three Embedding Views

Rather than embedding papers once, Literavore generates three vectors per paper from progressively richer text:

| View | Input |
|------|-------|
| `title_abstract` | Title + abstract only |
| `paper_card` | Title + abstract + AI summary + flat tags |
| `keyword_enriched` | Title + abstract + AI summary + all structured tags (domains, methods, datasets) |

This gives the search layer flexibility. A query for a specific method name benefits from `keyword_enriched`. A more conceptual query ("papers about efficient attention mechanisms") may work better against `paper_card`. The default for search is `keyword_enriched`.

All vectors are L2-normalized before storage so that inner-product similarity is equivalent to cosine similarity.

### FAISS Index

**Package: `faiss` (IndexFlatIP)**

FAISS (Facebook AI Similarity Search) is the industry-standard library for fast vector search. `IndexFlatIP` is the simplest index type: it computes exact inner products between the query vector and every stored vector. There is no approximation.

For a few thousand conference papers, brute-force exact search is both fast enough and semantically superior to approximate methods (like HNSW), which trade some accuracy for speed at larger scales. As the dataset grows, swapping in an approximate index would require changing only the `build` method of the `PaperIndex` class.

One FAISS index is built per embedding view and saved to disk as a binary file. A companion metadata JSON file maps index positions back to paper IDs.

At query time:
1. The query string is embedded using the same model
2. The vector is L2-normalized
3. FAISS returns the top-K nearest neighbors by inner product
4. Optional venue filtering is applied (with over-fetching to compensate for filtered-out results)

### UMAP Visualization

**Package: `umap-learn`**

On first request to the `/umap` API endpoint, all paper embeddings (from the `keyword_enriched` view) are projected to 2D using UMAP with `n_neighbors=15`. The result is cached in memory for subsequent requests. The Streamlit UI uses these coordinates to render an interactive scatter plot of the paper landscape, with search results highlighted in red over a grey background.

---

## Stage 6 — Serve

Three serving interfaces are available simultaneously:

### FastAPI REST API

Endpoints:
- `GET /papers` — list all papers, with optional conference filter
- `GET /papers/{id}` — full paper detail including summary and tags
- `POST /search` — semantic search (query string, top_k, view, venue filter)
- `GET /umap` — 2D coordinates for visualization
- `GET /conferences` — list all indexed conferences

### MCP Server

**Package: `mcp` (FastMCP)**

The MCP server exposes Literavore's capabilities as tools that any MCP-compatible LLM client (including Claude) can call. Tools include:

- `search_papers_semantic` — semantic vector search
- `search_by_keywords` — AND-logic keyword search across title, abstract, tags, and summary
- `search_papers_by_author` — author lookup
- `get_paper_details` — full paper metadata
- `get_conference_overview` — summary statistics per conference
- `get_paper_statistics` — global pipeline stats
- `get_recent_papers` — newest additions

All tools return JSON strings. Errors are caught and returned as JSON error objects rather than exceptions, so an LLM calling these tools always gets a parseable response.

### Streamlit UI

A browser-based interface that connects to the FastAPI backend. It shows a UMAP scatter plot of all papers with search results highlighted, and displays result cards with title, authors, venue, relevance score, AI summary, and tags. Each paper links back to its OpenReview page.

---

## Package Summary

| Package | Role | Why |
|---------|------|-----|
| `pypdf` + `pdfplumber` | PDF → text | Pure Python, fast, no C extension hangs; pdfplumber fallback for complex layouts |
| `openai` | LLM + embedding calls | Access to gpt-4o-mini and text-embedding-3-large |
| `faiss` | Vector index + search | Fast exact inner-product search; battle-tested at scale |
| `umap-learn` | 2D projection | Non-linear dimensionality reduction for visualization |
| `fastapi` | REST API | Fast async API framework with automatic OpenAPI docs |
| `mcp` (FastMCP) | MCP server | Native LLM tool integration via Model Context Protocol |
| `streamlit` | Browser UI | Rapid interactive UI without frontend build tooling |
| `pydantic` v2 | Config + data models | Validated, typed configuration with YAML loading |
| `aiohttp` / `asyncio` | Async downloads | Concurrent PDF fetching without threading |

---

## LLM Design Principles

LLMs are used in three distinct places in the pipeline: venue ID classification during fetch, paper summarization, and structured tag extraction. A few decisions shaped how they are used throughout:

**Structured output only.** Both LLM calls return JSON. The prompts specify exact schemas and instruct the model to return JSON only. This eliminates the need for brittle parsing of free-text responses and makes failures obvious (a JSON parse error, not a silent truncation).

**Two temperatures for two tasks.** Summary generation uses temperature 0.3 — the model should be focused and factual, but a little flexibility helps it produce readable prose. Tag extraction uses temperature 0.1 — it's essentially classification, and consistency across papers matters more than any individual paper's tags being creative.

**Minimal context, maximum signal.** The LLM receives at most 3,000 characters of extracted body text, not the full paper. The abstract and summary already carry most of the semantic content. Sending the full text would cost more and dilute the model's attention without meaningfully improving output.

**Cache by content hash.** Summaries are cached using a SHA-256 hash of the extracted text. Re-running the pipeline after a configuration change (e.g., fetching more papers) doesn't re-summarize papers that haven't changed. This keeps incremental runs cheap.

**Cost visibility.** The pipeline accumulates token counts and prints an estimated cost at the end of each summarization run. For a typical conference of 500 papers at 3,000 characters each, the cost is a few dollars.
