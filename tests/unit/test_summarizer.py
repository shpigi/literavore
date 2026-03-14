"""Tests for literavore.summarize.summarizer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from literavore.config import SummaryConfig
from literavore.db import Database
from literavore.storage.local import LocalStorage
from literavore.summarize.summarizer import Summarizer, _content_hash


@pytest.fixture
def config() -> SummaryConfig:
    return SummaryConfig(
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.3,
        max_concurrent=2,
        batch_size=5,
        enable_tag_extraction=False,
        cache_enabled=True,
    )


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "data")


@pytest.fixture
def summarizer(config: SummaryConfig, db: Database, storage: LocalStorage) -> Summarizer:
    return Summarizer(config, db, storage)


_MOCK_LLM_RESPONSE = json.dumps(
    {"summary": "This paper presents a novel approach.", "tags": ["deep-learning", "nlp"]}
)

_MOCK_TAG_RESPONSE = json.dumps(
    {"key_phrases": ["neural network"], "domains": ["ml"], "methods": ["transformer"]}
)


class TestSummarizerCreation:
    def test_creates_summarizer(
        self, config: SummaryConfig, db: Database, storage: LocalStorage
    ):
        s = Summarizer(config, db, storage)
        assert s._config is config
        assert s._db is db
        assert s._storage is storage

    def test_has_llm_client(self, summarizer: Summarizer):
        assert summarizer._llm is not None

    def test_has_tagger(self, summarizer: Summarizer):
        assert summarizer._tagger is not None


class TestSummarizePapers:
    def _make_extract_json(self, paper_id: str, storage: LocalStorage, text: str = "Full text."):
        data = {
            "paper_id": paper_id,
            "full_text": text,
            "abstract": "Test abstract.",
            "sections": [],
            "figures": [],
        }
        storage.put(f"extract/{paper_id}.json", json.dumps(data).encode())

    def test_summarize_papers_success(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
        monkeypatch,
    ):
        paper = {"id": "p1", "title": "Test Paper", "abstract": "An abstract."}
        db.get_or_create_paper("p1", title="Test Paper")
        self._make_extract_json("p1", storage)

        with patch(
            "literavore.summarize.summarizer.LLMClient.achat_complete",
            new_callable=AsyncMock,
            return_value=_MOCK_LLM_RESPONSE,
        ):
            summarizer = Summarizer(config, db, storage)
            results = asyncio.run(summarizer.summarize_papers([paper]))

        assert len(results) == 1
        assert results[0]["paper_id"] == "p1"
        assert results[0]["summary"] == "This paper presents a novel approach."
        assert "deep-learning" in results[0]["tags"]

    def test_summarize_papers_writes_to_storage(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
    ):
        paper = {"id": "p2", "title": "Paper Two", "abstract": "Another abstract."}
        db.get_or_create_paper("p2", title="Paper Two")
        self._make_extract_json("p2", storage)

        with patch(
            "literavore.summarize.summarizer.LLMClient.achat_complete",
            new_callable=AsyncMock,
            return_value=_MOCK_LLM_RESPONSE,
        ):
            summarizer = Summarizer(config, db, storage)
            asyncio.run(summarizer.summarize_papers([paper]))

        assert storage.exists("summaries/p2.json")
        stored = json.loads(storage.get("summaries/p2.json").decode())
        assert stored["paper_id"] == "p2"

    def test_summarize_papers_updates_db_status(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
    ):
        paper = {"id": "p3", "title": "Paper Three", "abstract": "Abstract three."}
        db.get_or_create_paper("p3", title="Paper Three")
        self._make_extract_json("p3", storage)

        with patch(
            "literavore.summarize.summarizer.LLMClient.achat_complete",
            new_callable=AsyncMock,
            return_value=_MOCK_LLM_RESPONSE,
        ):
            summarizer = Summarizer(config, db, storage)
            asyncio.run(summarizer.summarize_papers([paper]))

        status = db.get_stage_status("p3", "summarize")
        assert status["status"] == "done"

    def test_summarize_papers_empty_list(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
    ):
        summarizer = Summarizer(config, db, storage)
        results = asyncio.run(summarizer.summarize_papers([]))
        assert results == []


class TestSkipMissingExtract:
    def test_skips_when_extract_json_missing(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
    ):
        paper = {"id": "missing", "title": "Missing Paper", "abstract": ""}
        db.get_or_create_paper("missing", title="Missing Paper")
        # Do NOT store extract JSON

        summarizer = Summarizer(config, db, storage)
        results = asyncio.run(summarizer.summarize_papers([paper]))

        assert results == []
        status = db.get_stage_status("missing", "summarize")
        assert status["status"] == "failed"

    def test_skips_on_invalid_extract_json(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
    ):
        paper = {"id": "bad", "title": "Bad JSON Paper", "abstract": ""}
        db.get_or_create_paper("bad", title="Bad JSON Paper")
        storage.put("extract/bad.json", b"not valid json{{{")

        summarizer = Summarizer(config, db, storage)
        results = asyncio.run(summarizer.summarize_papers([paper]))

        assert results == []
        status = db.get_stage_status("bad", "summarize")
        assert status["status"] == "failed"


class TestCaching:
    def _make_extract_json(self, paper_id: str, storage: LocalStorage, text: str = "Full text."):
        data = {
            "paper_id": paper_id,
            "full_text": text,
            "abstract": "Test abstract.",
            "sections": [],
            "figures": [],
        }
        storage.put(f"extract/{paper_id}.json", json.dumps(data).encode())

    def test_does_not_re_summarize_if_cache_matches(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
    ):
        paper = {"id": "cached", "title": "Cached Paper", "abstract": "Abstract."}
        db.get_or_create_paper("cached", title="Cached Paper")
        full_text = "Full text."
        self._make_extract_json("cached", storage, text=full_text)

        # Pre-store a summary with the matching content hash
        existing_summary = {
            "paper_id": "cached",
            "summary": "Cached summary.",
            "tags": ["cached-tag"],
            "content_hash": _content_hash(full_text),
        }
        storage.put("summaries/cached.json", json.dumps(existing_summary).encode())

        call_count = 0

        async def mock_achat(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return _MOCK_LLM_RESPONSE

        with patch(
            "literavore.summarize.summarizer.LLMClient.achat_complete",
            new_callable=AsyncMock,
            side_effect=mock_achat,
        ):
            summarizer = Summarizer(config, db, storage)
            results = asyncio.run(summarizer.summarize_papers([paper]))

        # LLM should NOT have been called
        assert call_count == 0
        assert len(results) == 1
        assert results[0]["summary"] == "Cached summary."

    def test_re_summarizes_if_content_hash_differs(
        self,
        config: SummaryConfig,
        db: Database,
        storage: LocalStorage,
    ):
        paper = {"id": "changed", "title": "Changed Paper", "abstract": "Abstract."}
        db.get_or_create_paper("changed", title="Changed Paper")
        full_text = "Updated full text."
        self._make_extract_json("changed", storage, text=full_text)

        # Pre-store a summary with a DIFFERENT content hash
        existing_summary = {
            "paper_id": "changed",
            "summary": "Old summary.",
            "tags": [],
            "content_hash": "stale_hash_000000",
        }
        storage.put("summaries/changed.json", json.dumps(existing_summary).encode())

        with patch(
            "literavore.summarize.summarizer.LLMClient.achat_complete",
            new_callable=AsyncMock,
            return_value=_MOCK_LLM_RESPONSE,
        ):
            summarizer = Summarizer(config, db, storage)
            results = asyncio.run(summarizer.summarize_papers([paper]))

        assert len(results) == 1
        assert results[0]["summary"] == "This paper presents a novel approach."
