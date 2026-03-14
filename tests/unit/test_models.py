"""Unit tests for serve layer Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from literavore.serve.models import (
    HealthResponse,
    PaperDetail,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


class TestSearchRequest:
    def test_defaults(self) -> None:
        req = SearchRequest(query="attention mechanisms")
        assert req.query == "attention mechanisms"
        assert req.top_k == 10
        assert req.view == "keyword_enriched"
        assert req.venue_filter is None

    def test_custom_values(self) -> None:
        req = SearchRequest(
            query="transformers",
            top_k=5,
            view="title_abstract",
            venue_filter="ICLR 2024",
        )
        assert req.query == "transformers"
        assert req.top_k == 5
        assert req.view == "title_abstract"
        assert req.venue_filter == "ICLR 2024"

    def test_top_k_minimum(self) -> None:
        req = SearchRequest(query="test", top_k=1)
        assert req.top_k == 1

    def test_top_k_maximum(self) -> None:
        req = SearchRequest(query="test", top_k=100)
        assert req.top_k == 100

    def test_top_k_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)

    def test_top_k_above_maximum_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=101)

    def test_query_required(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest()  # type: ignore[call-arg]


class TestSearchResult:
    def test_construction(self) -> None:
        result = SearchResult(
            paper_id="paper-001",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            conference="NeurIPS 2017",
            abstract="We propose a new simple network architecture...",
            score=0.9312,
            rank=1,
        )
        assert result.paper_id == "paper-001"
        assert result.title == "Attention Is All You Need"
        assert result.authors == ["Vaswani", "Shazeer"]
        assert result.conference == "NeurIPS 2017"
        assert result.abstract == "We propose a new simple network architecture..."
        assert result.score == 0.9312
        assert result.rank == 1

    def test_empty_authors(self) -> None:
        result = SearchResult(
            paper_id="p1",
            title="Some Paper",
            authors=[],
            conference="",
            abstract="",
            score=0.5,
            rank=1,
        )
        assert result.authors == []

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(title="missing fields")  # type: ignore[call-arg]


class TestSearchResponse:
    def test_with_results(self) -> None:
        result = SearchResult(
            paper_id="p1",
            title="Test Paper",
            authors=["Author A"],
            conference="ICML 2024",
            abstract="Abstract text",
            score=0.85,
            rank=1,
        )
        response = SearchResponse(query="neural networks", results=[result], total=1)
        assert response.query == "neural networks"
        assert len(response.results) == 1
        assert response.total == 1
        assert response.results[0].paper_id == "p1"

    def test_empty_results(self) -> None:
        response = SearchResponse(query="obscure topic", results=[], total=0)
        assert response.query == "obscure topic"
        assert response.results == []
        assert response.total == 0

    def test_multiple_results(self) -> None:
        results = [
            SearchResult(
                paper_id=f"p{i}",
                title=f"Paper {i}",
                authors=[],
                conference="",
                abstract="",
                score=1.0 - i * 0.1,
                rank=i + 1,
            )
            for i in range(3)
        ]
        response = SearchResponse(query="test", results=results, total=3)
        assert len(response.results) == 3
        assert response.total == 3


class TestPaperDetail:
    def test_defaults(self) -> None:
        detail = PaperDetail(
            paper_id="p1",
            title="My Paper",
            authors=["Alice"],
            conference="ICML",
            abstract="Abstract",
            pdf_url="",
            created_at="",
            stage_status={},
            summary="",
            tags=[],
            structured_tags={},
        )
        assert detail.paper_id == "p1"
        assert detail.summary == ""
        assert detail.tags == []
        assert detail.structured_tags == {}
        assert detail.stage_status == {}

    def test_full_fields(self) -> None:
        detail = PaperDetail(
            paper_id="paper-123",
            title="Deep Learning Survey",
            authors=["LeCun", "Bengio", "Hinton"],
            conference="NeurIPS 2024",
            abstract="A comprehensive survey...",
            pdf_url="https://arxiv.org/pdf/1234.pdf",
            created_at="2024-01-15T10:00:00+00:00",
            stage_status={"fetch": "done", "embed": "done"},
            summary="This paper surveys deep learning methods.",
            tags=["deep learning", "survey"],
            structured_tags={"domains": ["ML"], "methods": ["CNN", "RNN"]},
        )
        assert detail.paper_id == "paper-123"
        assert len(detail.authors) == 3
        assert detail.stage_status["fetch"] == "done"
        assert "deep learning" in detail.tags
        assert detail.structured_tags["domains"] == ["ML"]


class TestHealthResponse:
    def test_defaults(self) -> None:
        health = HealthResponse(status="ok", index_loaded=True, paper_count=42)
        assert health.status == "ok"
        assert health.index_loaded is True
        assert health.paper_count == 42
        assert health.version == "1.0.0"

    def test_index_not_loaded(self) -> None:
        health = HealthResponse(status="ok", index_loaded=False, paper_count=0)
        assert health.index_loaded is False
        assert health.paper_count == 0

    def test_custom_version(self) -> None:
        health = HealthResponse(status="ok", index_loaded=True, paper_count=5, version="2.0.0")
        assert health.version == "2.0.0"

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            HealthResponse(status="ok")  # type: ignore[call-arg]
