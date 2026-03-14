"""Pydantic models for the Literavore FastAPI serve layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    view: str = "keyword_enriched"
    venue_filter: str | None = None


class SearchResult(BaseModel):
    """A single result returned from /search."""

    paper_id: str
    title: str
    authors: list[str]
    conference: str
    abstract: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: str
    results: list[SearchResult]
    total: int


class PaperDetail(BaseModel):
    """Full paper detail returned from GET /papers/{id}."""

    paper_id: str
    title: str
    authors: list[str]
    conference: str
    abstract: str
    pdf_url: str
    created_at: str
    stage_status: dict[str, str]
    summary: str
    tags: list[str]
    structured_tags: dict


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    index_loaded: bool
    paper_count: int
    version: str = "1.0.0"
