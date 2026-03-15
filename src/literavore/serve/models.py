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
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    openreview_url: str = ""


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
    source_url: str
    keywords: list[str]
    published_date: str | None
    created_at: str
    stage_status: dict[str, str]
    summary: str
    tags: list[str]
    structured_tags: dict
    openreview_url: str = ""


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    index_loaded: bool
    paper_count: int
    version: str = "1.0.0"
