"""Tests for literavore.embed.index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from literavore.embed.index import PaperIndex
from literavore.storage.local import LocalStorage

# Dimension used across tests (small for speed)
DIM = 32
VIEWS = ["title_abstract", "paper_card"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vector(dim: int = DIM) -> list[float]:
    rng = np.random.default_rng(42)
    vec = rng.random(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_records(paper_ids: list[str], views: list[str] = VIEWS) -> list[dict]:
    """Create fake embedding records for the given paper IDs and views."""
    rng = np.random.default_rng(0)
    records = []
    for paper_id in paper_ids:
        for view in views:
            vec = rng.random(DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            records.append(
                {
                    "paper_id": paper_id,
                    "view_type": view,
                    "vector": vec.tolist(),
                    "dimensions": DIM,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_index() -> PaperIndex:
    return PaperIndex(dimensions=DIM, views=VIEWS)


@pytest.fixture
def populated_index() -> PaperIndex:
    idx = PaperIndex(dimensions=DIM, views=VIEWS)
    records = _make_records(["p1", "p2", "p3"])
    idx.build(records)
    return idx


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "data")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class TestPaperIndexBuild:
    def test_build_increases_size(self, empty_index: PaperIndex):
        records = _make_records(["p1", "p2"])
        empty_index.build(records)
        # 2 papers × 2 views = 4 vectors total
        assert empty_index.size == 4

    def test_build_correct_size_per_view(self, empty_index: PaperIndex):
        records = _make_records(["p1", "p2", "p3"])
        empty_index.build(records)
        assert empty_index.size == 6

    def test_empty_build(self, empty_index: PaperIndex):
        empty_index.build([])
        assert empty_index.size == 0

    def test_views_property(self, empty_index: PaperIndex):
        assert empty_index.views == VIEWS

    def test_dimensions_property(self, empty_index: PaperIndex):
        assert empty_index.dimensions == DIM


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------


class TestPaperIndexAdd:
    def test_add_increments_size(self, populated_index: PaperIndex):
        before = populated_index.size
        populated_index.add(_make_records(["p4"]))
        assert populated_index.size == before + len(VIEWS)

    def test_add_to_empty(self, empty_index: PaperIndex):
        empty_index.add(_make_records(["p1"]))
        assert empty_index.size == len(VIEWS)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestPaperIndexSearch:
    def test_search_returns_list(self, populated_index: PaperIndex):
        q = _random_vector()
        results = populated_index.search(q, view="title_abstract", top_k=2)
        assert isinstance(results, list)

    def test_search_result_structure(self, populated_index: PaperIndex):
        q = _random_vector()
        results = populated_index.search(q, view="title_abstract", top_k=2)
        assert len(results) > 0
        for rec in results:
            assert "paper_id" in rec
            assert "score" in rec
            assert "rank" in rec
            assert "view_type" in rec
            assert rec["view_type"] == "title_abstract"

    def test_search_top_k_limit(self, populated_index: PaperIndex):
        q = _random_vector()
        results = populated_index.search(q, view="title_abstract", top_k=2)
        assert len(results) <= 2

    def test_search_rank_sequence(self, populated_index: PaperIndex):
        q = _random_vector()
        results = populated_index.search(q, view="title_abstract", top_k=3)
        for i, rec in enumerate(results, start=1):
            assert rec["rank"] == i

    def test_search_scores_descending(self, populated_index: PaperIndex):
        q = _random_vector()
        results = populated_index.search(q, view="title_abstract", top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_unknown_view_raises(self, populated_index: PaperIndex):
        q = _random_vector()
        with pytest.raises(ValueError, match="Unknown view"):
            populated_index.search(q, view="nonexistent_view", top_k=5)

    def test_search_on_empty_index_returns_empty(self, empty_index: PaperIndex):
        q = _random_vector()
        results = empty_index.search(q, view="title_abstract", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Venue filter
# ---------------------------------------------------------------------------


class TestVenueFilter:
    def test_venue_filter_restricts_results(self):
        idx = PaperIndex(dimensions=DIM, views=["title_abstract"])
        records = _make_records(["p1", "p2", "p3"], views=["title_abstract"])
        idx.build(records)

        paper_venues = {"p1": "NeurIPS", "p2": "ICML", "p3": "NeurIPS"}
        q = _random_vector()
        results = idx.search(
            q,
            view="title_abstract",
            top_k=10,
            venue_filter="NeurIPS",
            paper_venues=paper_venues,
        )
        assert all(r["paper_id"] in {"p1", "p3"} for r in results)
        assert not any(r["paper_id"] == "p2" for r in results)

    def test_venue_filter_no_matches_returns_empty(self):
        idx = PaperIndex(dimensions=DIM, views=["title_abstract"])
        records = _make_records(["p1", "p2"], views=["title_abstract"])
        idx.build(records)

        paper_venues = {"p1": "NeurIPS", "p2": "NeurIPS"}
        q = _random_vector()
        results = idx.search(
            q,
            view="title_abstract",
            top_k=10,
            venue_filter="ICLR",
            paper_venues=paper_venues,
        )
        assert results == []

    def test_no_venue_filter_returns_all(self):
        idx = PaperIndex(dimensions=DIM, views=["title_abstract"])
        records = _make_records(["p1", "p2", "p3"], views=["title_abstract"])
        idx.build(records)

        q = _random_vector()
        results = idx.search(q, view="title_abstract", top_k=10)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_storage_keys(
        self, populated_index: PaperIndex, storage: LocalStorage
    ):
        populated_index.save(storage)
        assert storage.exists("index/metadata.json")
        for view in VIEWS:
            assert storage.exists(f"index/{view}.faiss")

    def test_load_restores_size(
        self, populated_index: PaperIndex, storage: LocalStorage
    ):
        original_size = populated_index.size
        populated_index.save(storage)
        loaded = PaperIndex.load(storage)
        assert loaded.size == original_size

    def test_load_restores_dimensions(
        self, populated_index: PaperIndex, storage: LocalStorage
    ):
        populated_index.save(storage)
        loaded = PaperIndex.load(storage)
        assert loaded.dimensions == DIM

    def test_load_restores_views(
        self, populated_index: PaperIndex, storage: LocalStorage
    ):
        populated_index.save(storage)
        loaded = PaperIndex.load(storage)
        assert set(loaded.views) == set(VIEWS)

    def test_load_produces_consistent_search(
        self, populated_index: PaperIndex, storage: LocalStorage
    ):
        q = _random_vector()
        original_results = populated_index.search(q, view="title_abstract", top_k=3)

        populated_index.save(storage)
        loaded = PaperIndex.load(storage)
        loaded_results = loaded.search(q, view="title_abstract", top_k=3)

        orig_ids = [r["paper_id"] for r in original_results]
        loaded_ids = [r["paper_id"] for r in loaded_results]
        assert orig_ids == loaded_ids

    def test_load_missing_metadata_raises(self, storage: LocalStorage):
        with pytest.raises(FileNotFoundError):
            PaperIndex.load(storage)
