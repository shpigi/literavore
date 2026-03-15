"""FastAPI application for the Literavore serve layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from literavore.config import LiteravoreConfig, load_config
from literavore.db import Database
from literavore.embed.embedder import Embedder
from literavore.embed.index import PaperIndex
from literavore.serve.models import (
    HealthResponse,
    PaperDetail,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from literavore.storage.local import LocalStorage
from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.storage.base import StorageBackend

logger = get_logger(__name__, stage="serve")

app = FastAPI(title="Literavore", version="1.0.0")

# ---------------------------------------------------------------------------
# Module-level global state (lazy-initialised at startup)
# ---------------------------------------------------------------------------

_config: LiteravoreConfig | None = None
_db: Database | None = None
_storage: StorageBackend | None = None
_embedder: Embedder | None = None
_index: PaperIndex | None = None


def _get_config() -> LiteravoreConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_db() -> Database:
    global _db
    if _db is None:
        cfg = _get_config()
        data_dir = Path(cfg.storage.data_dir)
        _db = Database(data_dir / "literavore.db")
    return _db


def _get_storage() -> StorageBackend:
    global _storage
    if _storage is None:
        cfg = _get_config()
        data_dir = Path(cfg.storage.data_dir)
        _storage = LocalStorage(data_dir)
    return _storage


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        cfg = _get_config()
        _embedder = Embedder(cfg.embedding)
    return _embedder


def _get_index() -> PaperIndex | None:
    global _index
    if _index is None:
        storage = _get_storage()
        try:
            _index = PaperIndex.load(storage)
        except FileNotFoundError:
            logger.warning("FAISS index not found in storage — semantic search unavailable")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load FAISS index: %s", exc)
            return None
    return _index


def _parse_authors(authors_raw: str | list | None) -> list[str]:
    """Return a list of author name strings from various stored formats."""
    if authors_raw is None:
        return []
    if isinstance(authors_raw, list):
        result = []
        for a in authors_raw:
            if isinstance(a, dict):
                result.append(a.get("name", str(a)))
            else:
                result.append(str(a))
        return result
    try:
        parsed = json.loads(authors_raw)
        return _parse_authors(parsed)
    except (json.JSONDecodeError, TypeError):
        return [authors_raw]


def _parse_keywords(keywords_raw: str | list | None) -> list[str]:
    """Return a list of keyword strings from various stored formats."""
    if keywords_raw is None:
        return []
    if isinstance(keywords_raw, list):
        return [str(k) for k in keywords_raw]
    try:
        parsed = json.loads(keywords_raw)
        if isinstance(parsed, list):
            return [str(k) for k in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health status."""
    try:
        db = _get_db()
        papers = db.get_papers()
        paper_count = len(papers)
    except Exception:  # noqa: BLE001
        paper_count = 0

    index = _get_index()
    return HealthResponse(
        status="ok",
        index_loaded=index is not None,
        paper_count=paper_count,
    )


@app.get("/papers", response_model=list[dict])
def list_papers(conference: str | None = None) -> list[dict]:
    """Return all papers, optionally filtered by conference."""
    db = _get_db()
    papers = db.get_papers(conference=conference)
    result = []
    for p in papers:
        result.append(
            {
                "paper_id": p["id"],
                "title": p.get("title", ""),
                "authors": _parse_authors(p.get("authors")),
                "conference": p.get("conference", ""),
                "abstract": (p.get("abstract") or "")[:300],
            }
        )
    return result


@app.get("/papers/{paper_id}", response_model=PaperDetail)
def get_paper(paper_id: str) -> PaperDetail:
    """Return full details for a single paper."""
    db = _get_db()
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id!r}")

    storage = _get_storage()
    summary_data: dict = {}
    summary_key = f"summaries/{paper_id}.json"
    if storage.exists(summary_key):
        try:
            summary_data = json.loads(storage.get(summary_key).decode())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load summary for %s: %s", paper_id, exc)

    stage_status: dict[str, str] = {}
    for stage in ("fetch", "download", "extract", "summarize", "embed"):
        status = db.get_stage_status(paper_id, stage)
        if status:
            stage_status[stage] = status.get("status", "")

    return PaperDetail(
        paper_id=paper["id"],
        title=paper.get("title", "") or "",
        authors=_parse_authors(paper.get("authors")),
        conference=paper.get("conference", "") or "",
        abstract=paper.get("abstract", "") or "",
        pdf_url=paper.get("pdf_url", "") or "",
        source_url=paper.get("source_url", "") or "",
        keywords=_parse_keywords(paper.get("keywords")),
        published_date=paper.get("published_date"),
        created_at=paper.get("created_at", "") or "",
        stage_status=stage_status,
        summary=summary_data.get("summary", ""),
        tags=summary_data.get("tags", []),
        structured_tags=summary_data.get("structured_tags", {}),
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Semantic search over paper embeddings."""
    index = _get_index()
    if index is None:
        raise HTTPException(
            status_code=503, detail="Search index not loaded. Run embed stage first."
        )

    embedder = _get_embedder()
    query_vector = embedder.embed_texts([request.query])[0]

    # Choose view — fall back to first available if requested view not present
    view = request.view if request.view in index.views else index.views[0]

    paper_venues: dict[str, str] | None = None
    if request.venue_filter:
        db = _get_db()
        all_papers = db.get_papers()
        paper_venues = {p["id"]: (p.get("conference") or "") for p in all_papers}

    hits = index.search(
        query_vector=query_vector,
        view=view,
        top_k=request.top_k,
        venue_filter=request.venue_filter,
        paper_venues=paper_venues,
    )

    db = _get_db()
    results: list[SearchResult] = []
    for hit in hits:
        paper = db.get_paper(hit["paper_id"])
        if paper is None:
            continue
        results.append(
            SearchResult(
                paper_id=hit["paper_id"],
                title=paper.get("title", "") or "",
                authors=_parse_authors(paper.get("authors")),
                conference=paper.get("conference", "") or "",
                abstract=(paper.get("abstract") or "")[:300],
                score=round(hit["score"], 4),
                rank=hit["rank"],
            )
        )

    return SearchResponse(query=request.query, results=results, total=len(results))
