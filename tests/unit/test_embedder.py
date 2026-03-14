"""Tests for literavore.embed.embedder."""

from __future__ import annotations

import pytest

from literavore.config import EmbeddingConfig
from literavore.embed.embedder import (
    ALL_VIEWS,
    VIEW_KEYWORD_ENRICHED,
    VIEW_PAPER_CARD,
    VIEW_TITLE_ABSTRACT,
    Embedder,
    build_view_text,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model="text-embedding-3-large",
        dimensions=16,
        batch_size=8,
        views=list(ALL_VIEWS),
    )


@pytest.fixture
def paper() -> dict:
    return {
        "id": "paper-1",
        "title": "Attention Is All You Need",
        "abstract": "We propose a novel architecture based solely on attention mechanisms.",
        "conference": "NeurIPS",
    }


@pytest.fixture
def summary_data() -> dict:
    return {
        "paper_id": "paper-1",
        "summary": "A transformer architecture without recurrence.",
        "tags": ["transformer", "attention"],
        "structured_tags": {
            "key_phrases": ["self-attention", "multi-head attention"],
            "domains": ["nlp"],
            "methods": ["transformer"],
        },
    }


# ---------------------------------------------------------------------------
# build_view_text
# ---------------------------------------------------------------------------


class TestBuildViewText:
    def test_title_abstract_with_abstract(self, paper: dict):
        text = build_view_text(VIEW_TITLE_ABSTRACT, paper)
        assert paper["title"] in text
        assert paper["abstract"] in text

    def test_title_abstract_without_abstract(self, paper: dict):
        p = {**paper, "abstract": ""}
        text = build_view_text(VIEW_TITLE_ABSTRACT, p)
        assert paper["title"] in text
        # No double-blank-line separator when abstract is empty
        assert "\n\n" not in text

    def test_title_abstract_no_summary_no_keywords(self, paper: dict):
        text = build_view_text(VIEW_TITLE_ABSTRACT, paper)
        assert "Summary:" not in text
        assert "Keywords:" not in text

    def test_paper_card_includes_summary(self, paper: dict, summary_data: dict):
        text = build_view_text(VIEW_PAPER_CARD, paper, summary_data)
        assert "Summary:" in text
        assert summary_data["summary"] in text

    def test_paper_card_includes_tags(self, paper: dict, summary_data: dict):
        text = build_view_text(VIEW_PAPER_CARD, paper, summary_data)
        assert "Tags:" in text
        assert "transformer" in text

    def test_paper_card_without_summary(self, paper: dict):
        text = build_view_text(VIEW_PAPER_CARD, paper)
        assert "Summary:" not in text
        assert paper["title"] in text
        assert paper["abstract"] in text

    def test_keyword_enriched_includes_keywords(self, paper: dict, summary_data: dict):
        text = build_view_text(VIEW_KEYWORD_ENRICHED, paper, summary_data)
        assert "Keywords:" in text
        assert "self-attention" in text

    def test_keyword_enriched_deduplicates(self, paper: dict, summary_data: dict):
        # "transformer" appears in both tags and methods — should appear once in keywords
        text = build_view_text(VIEW_KEYWORD_ENRICHED, paper, summary_data)
        kw_line = [line for line in text.splitlines() if line.startswith("Keywords:")][0]
        assert kw_line.count("transformer") == 1

    def test_keyword_enriched_without_summary_data(self, paper: dict):
        text = build_view_text(VIEW_KEYWORD_ENRICHED, paper)
        assert "Keywords:" not in text
        assert paper["title"] in text

    def test_unknown_view_raises(self, paper: dict):
        with pytest.raises(ValueError, match="Unknown view"):
            build_view_text("invalid_view", paper)

    def test_all_views_return_non_empty(self, paper: dict, summary_data: dict):
        for view in ALL_VIEWS:
            text = build_view_text(view, paper, summary_data)
            assert len(text) > 0

    def test_none_abstract_handled(self):
        p = {"id": "x", "title": "Title", "abstract": None}
        text = build_view_text(VIEW_TITLE_ABSTRACT, p)
        assert "Title" in text

    def test_empty_summary_data(self, paper: dict):
        text = build_view_text(VIEW_PAPER_CARD, paper, {})
        assert paper["title"] in text


# ---------------------------------------------------------------------------
# Embedder — mock mode (no OPENAI_API_KEY)
# ---------------------------------------------------------------------------


class TestEmbedderMockMode:
    def test_mock_mode_enabled_when_no_api_key(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        assert embedder._mock_mode is True

    def test_mock_returns_zero_vectors(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        vectors = embedder.embed_texts(["hello world", "another text"])
        assert len(vectors) == 2
        for vec in vectors:
            assert len(vec) == config.dimensions
            assert all(v == 0.0 for v in vec)

    def test_mock_correct_dimension(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = EmbeddingConfig(dimensions=64)
        embedder = Embedder(cfg)
        vectors = embedder.embed_texts(["test"])
        assert len(vectors[0]) == 64

    def test_embed_texts_empty_list(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        result = embedder.embed_texts([])
        assert result == []


# ---------------------------------------------------------------------------
# embed_papers
# ---------------------------------------------------------------------------


class TestEmbedPapers:
    def test_returns_correct_structure(
        self, config: EmbeddingConfig, paper: dict, summary_data: dict, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        records = embedder.embed_papers([paper], {paper["id"]: summary_data})

        # 3 views × 1 paper = 3 records
        assert len(records) == len(ALL_VIEWS)
        for rec in records:
            assert "paper_id" in rec
            assert "view_type" in rec
            assert "vector" in rec
            assert "dimensions" in rec
            assert rec["paper_id"] == paper["id"]
            assert rec["view_type"] in ALL_VIEWS
            assert rec["dimensions"] == config.dimensions

    def test_embed_papers_multiple_papers(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        papers = [
            {"id": "p1", "title": "Paper One", "abstract": "Abstract one."},
            {"id": "p2", "title": "Paper Two", "abstract": "Abstract two."},
        ]
        records = embedder.embed_papers(papers, {})
        assert len(records) == 2 * len(ALL_VIEWS)
        paper_ids = {r["paper_id"] for r in records}
        assert paper_ids == {"p1", "p2"}

    def test_embed_papers_custom_views(self, config: EmbeddingConfig, paper: dict, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        records = embedder.embed_papers([paper], {}, views=[VIEW_TITLE_ABSTRACT])
        assert len(records) == 1
        assert records[0]["view_type"] == VIEW_TITLE_ABSTRACT

    def test_embed_papers_empty_list(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        records = embedder.embed_papers([], {})
        assert records == []

    def test_embed_papers_vector_length(
        self, config: EmbeddingConfig, paper: dict, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)
        records = embedder.embed_papers([paper], {})
        for rec in records:
            assert len(rec["vector"]) == config.dimensions


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestEmbedTextsCache:
    def test_cache_hit_on_second_call(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)

        call_count = 0
        original_embed_batch = embedder._embed_batch

        def counting_embed_batch(texts):
            nonlocal call_count
            call_count += len(texts)
            return original_embed_batch(texts)

        embedder._embed_batch = counting_embed_batch  # type: ignore[method-assign]

        text = "unique text for cache test"
        embedder.embed_texts([text])
        first_count = call_count

        # Second call — should hit cache, _embed_batch not called again
        embedder.embed_texts([text])
        assert call_count == first_count  # no additional calls

    def test_different_texts_both_embedded(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)

        call_count = 0
        original_embed_batch = embedder._embed_batch

        def counting_embed_batch(texts):
            nonlocal call_count
            call_count += len(texts)
            return original_embed_batch(texts)

        embedder._embed_batch = counting_embed_batch  # type: ignore[method-assign]

        embedder.embed_texts(["text A", "text B"])
        assert call_count == 2

    def test_partial_cache_hit(self, config: EmbeddingConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = Embedder(config)

        embedder.embed_texts(["cached text"])

        call_count = 0
        original_embed_batch = embedder._embed_batch

        def counting_embed_batch(texts):
            nonlocal call_count
            call_count += len(texts)
            return original_embed_batch(texts)

        embedder._embed_batch = counting_embed_batch  # type: ignore[method-assign]

        # One cached + one new
        embedder.embed_texts(["cached text", "new text"])
        assert call_count == 1  # only the new text triggered _embed_batch
