"""Integration tests for the Literavore FastAPI serve layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import literavore.serve.api as api_module
from literavore.db import Database
from literavore.serve.api import app
from literavore.storage.local import LocalStorage


@pytest.fixture(autouse=True)
def reset_api_globals() -> None:
    """Reset all module-level globals in api.py before each test."""
    api_module._config = None
    api_module._db = None
    api_module._storage = None
    api_module._embedder = None
    api_module._index = None
    yield
    api_module._config = None
    api_module._db = None
    api_module._storage = None
    api_module._embedder = None
    api_module._index = None


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    """In-memory SQLite database backed by a temp file."""
    return Database(tmp_path / "test.db")


@pytest.fixture
def tmp_storage(tmp_path: Path) -> LocalStorage:
    """Local storage backed by a temp directory."""
    return LocalStorage(tmp_path / "storage")


@pytest.fixture
def client(tmp_db: Database, tmp_storage: LocalStorage) -> TestClient:
    """TestClient with db and storage injected, index left as None."""
    api_module._db = tmp_db
    api_module._storage = tmp_storage
    # _index stays None → search will return 503
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_structure(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data
        assert "index_loaded" in data
        assert "paper_count" in data
        assert "version" in data

    def test_health_status_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_index_not_loaded_when_none(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["index_loaded"] is False

    def test_health_paper_count_empty_db(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["paper_count"] == 0

    def test_health_paper_count_with_papers(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        tmp_db.get_or_create_paper("p1", title="Paper 1", conference="ICML")
        tmp_db.get_or_create_paper("p2", title="Paper 2", conference="ICML")
        api_module._db = tmp_db
        api_module._storage = tmp_storage
        client = TestClient(app, raise_server_exceptions=False)
        data = client.get("/health").json()
        assert data["paper_count"] == 2


class TestPapersListEndpoint:
    def test_empty_db_returns_empty_list(self, client: TestClient) -> None:
        response = client.get("/papers")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_added_papers(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        tmp_db.get_or_create_paper(
            "paper-001",
            title="Attention Is All You Need",
            authors='["Vaswani"]',
            conference="NeurIPS 2017",
            abstract="We propose a new network architecture.",
        )
        api_module._db = tmp_db
        api_module._storage = tmp_storage
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/papers")
        assert response.status_code == 200
        papers = response.json()
        assert len(papers) == 1
        assert papers[0]["paper_id"] == "paper-001"
        assert papers[0]["title"] == "Attention Is All You Need"

    def test_conference_filter(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        tmp_db.get_or_create_paper("p1", title="Paper 1", conference="ICML")
        tmp_db.get_or_create_paper("p2", title="Paper 2", conference="NeurIPS")
        api_module._db = tmp_db
        api_module._storage = tmp_storage
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/papers?conference=ICML")
        assert response.status_code == 200
        papers = response.json()
        assert len(papers) == 1
        assert papers[0]["paper_id"] == "p1"


class TestPaperDetailEndpoint:
    def test_missing_paper_returns_404(self, client: TestClient) -> None:
        response = client.get("/papers/nonexistent-id")
        assert response.status_code == 404

    def test_404_detail_message(self, client: TestClient) -> None:
        response = client.get("/papers/nonexistent-id")
        data = response.json()
        assert "detail" in data

    def test_existing_paper_returns_200(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        tmp_db.get_or_create_paper(
            "paper-42",
            title="Test Paper",
            authors='["Alice", "Bob"]',
            conference="ICLR 2024",
            abstract="An interesting abstract.",
            pdf_url="https://example.com/paper.pdf",
        )
        api_module._db = tmp_db
        api_module._storage = tmp_storage
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/papers/paper-42")
        assert response.status_code == 200

    def test_paper_detail_fields(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        tmp_db.get_or_create_paper(
            "paper-42",
            title="Test Paper",
            authors='["Alice", "Bob"]',
            conference="ICLR 2024",
            abstract="An interesting abstract.",
            pdf_url="https://example.com/paper.pdf",
        )
        api_module._db = tmp_db
        api_module._storage = tmp_storage
        client = TestClient(app, raise_server_exceptions=False)

        data = client.get("/papers/paper-42").json()
        assert data["paper_id"] == "paper-42"
        assert data["title"] == "Test Paper"
        assert data["conference"] == "ICLR 2024"
        assert isinstance(data["authors"], list)
        assert "Alice" in data["authors"]
        assert data["summary"] == ""
        assert data["tags"] == []
        assert data["stage_status"] == {}


class TestSearchEndpoint:
    def test_search_returns_503_when_index_not_loaded(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "neural networks"})
        assert response.status_code == 503

    def test_search_503_detail(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "test"})
        data = response.json()
        assert "detail" in data

    def test_search_with_mock_index(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        """Full search: inject mock index + embedder, paper in DB, POST /search."""
        # Add a paper to the database
        tmp_db.get_or_create_paper(
            "paper-001",
            title="Transformers for NLP",
            authors='["Smith"]',
            conference="ACL 2024",
            abstract="We present a new transformer architecture.",
        )

        # Build a real mock index that returns a hit for paper-001
        mock_index = MagicMock()
        mock_index.views = ["keyword_enriched"]
        mock_index.search.return_value = [
            {
                "paper_id": "paper-001",
                "score": 0.9123,
                "rank": 1,
                "view_type": "keyword_enriched",
            }
        ]

        # Mock embedder that returns a zero vector
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.0] * 8]

        api_module._db = tmp_db
        api_module._storage = tmp_storage
        api_module._index = mock_index
        api_module._embedder = mock_embedder

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/search", json={"query": "transformer NLP"})

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "transformer NLP"
        assert len(data["results"]) == 1
        assert data["total"] == 1
        result = data["results"][0]
        assert result["paper_id"] == "paper-001"
        assert result["title"] == "Transformers for NLP"
        assert result["rank"] == 1
        assert result["score"] == pytest.approx(0.9123)

    def test_search_missing_paper_in_db_skipped(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        """If index returns a hit but the paper is not in DB, it is skipped."""
        mock_index = MagicMock()
        mock_index.views = ["title_abstract"]
        mock_index.search.return_value = [
            {
                "paper_id": "nonexistent-paper",
                "score": 0.75,
                "rank": 1,
                "view_type": "title_abstract",
            }
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.0] * 8]

        api_module._db = tmp_db
        api_module._storage = tmp_storage
        api_module._index = mock_index
        api_module._embedder = mock_embedder

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/search", json={"query": "missing paper"})

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0

    def test_search_top_k_passed_to_index(
        self, tmp_db: Database, tmp_storage: LocalStorage
    ) -> None:
        """top_k from the request should be forwarded to index.search."""
        mock_index = MagicMock()
        mock_index.views = ["keyword_enriched"]
        mock_index.search.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.0] * 8]

        api_module._db = tmp_db
        api_module._storage = tmp_storage
        api_module._index = mock_index
        api_module._embedder = mock_embedder

        client = TestClient(app, raise_server_exceptions=False)
        client.post("/search", json={"query": "test", "top_k": 25})

        mock_index.search.assert_called_once()
        call_kwargs = mock_index.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 25 or call_kwargs.args[2] == 25
