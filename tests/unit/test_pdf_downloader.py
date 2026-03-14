"""Tests for literavore.ingest.pdf_downloader."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from literavore.config import PdfConfig
from literavore.ingest.pdf_downloader import OPENREVIEW_BASE, AsyncPDFDownloader


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> PdfConfig:
    """Return a PdfConfig with fast settings for tests."""
    defaults = {
        "max_concurrent": 2,
        "delay_between_requests": 0,
        "max_retries": 0,
        "timeout": 5,
        "chunk_size": 1024,
        "base_backoff": 2.0,
        "rate_limit_backoff": 0.1,
        "max_backoff": 5.0,
        "backoff_jitter": False,
    }
    defaults.update(kwargs)
    return PdfConfig(**defaults)


def _make_db(stage_status=None) -> MagicMock:
    db = MagicMock()
    db.get_stage_status.return_value = stage_status
    db.update_stage_status.return_value = None
    return db


def _make_storage() -> MagicMock:
    storage = MagicMock()
    storage.put.return_value = None
    return storage


# ---------------------------------------------------------------------------
# AsyncPDFDownloader creation
# ---------------------------------------------------------------------------


class TestAsyncPDFDownloaderCreation:
    def test_can_create_with_defaults(self, tmp_path: Path):
        config = _make_config()
        db = _make_db()
        storage = _make_storage()
        downloader = AsyncPDFDownloader(config, db, storage)
        assert downloader._config is config
        assert downloader._db is db
        assert downloader._storage is storage

    def test_semaphore_max_concurrent(self, tmp_path: Path):
        config = _make_config(max_concurrent=5)
        downloader = AsyncPDFDownloader(config, _make_db(), _make_storage())
        # asyncio.Semaphore stores the value in _value attribute
        assert downloader._semaphore._value == 5

    def test_session_initially_none(self):
        downloader = AsyncPDFDownloader(_make_config(), _make_db(), _make_storage())
        assert downloader._session is None


# ---------------------------------------------------------------------------
# Context manager (aenter / aexit)
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    def test_session_set_on_enter(self):
        async def _run():
            config = _make_config()
            downloader = AsyncPDFDownloader(config, _make_db(), _make_storage())
            async with downloader as d:
                assert d._session is not None
            assert downloader._session is None

        asyncio.get_event_loop().run_until_complete(_run())

    def test_session_cleared_on_exit(self):
        async def _run():
            config = _make_config()
            downloader = AsyncPDFDownloader(config, _make_db(), _make_storage())
            async with downloader:
                pass
            assert downloader._session is None

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# URL resolution (relative /pdf/ URLs)
# ---------------------------------------------------------------------------


class TestUrlResolution:
    """Test that relative OpenReview PDF paths are resolved to absolute URLs."""

    def test_openreview_base_constant(self):
        assert OPENREVIEW_BASE == "https://openreview.net"

    def test_relative_url_resolved_in_download_one(self):
        """_download_one should prepend OPENREVIEW_BASE to /pdf/ paths."""

        async def _run():
            config = _make_config()
            db = _make_db()
            storage = _make_storage()
            downloader = AsyncPDFDownloader(config, db, storage)

            # Patch _fetch_url to capture the URL it was called with
            called_urls: list[str] = []

            async def fake_fetch(url: str) -> bytes:
                called_urls.append(url)
                return b"%PDF-fake-content"

            downloader._fetch_url = fake_fetch  # type: ignore[method-assign]

            async with downloader:
                paper = {"id": "test-paper", "pdf_url": "/pdf/abc123"}
                result = await downloader._download_one(paper)

            assert len(called_urls) == 1
            assert called_urls[0] == "https://openreview.net/pdf/abc123"
            assert result["success"] is True

        asyncio.get_event_loop().run_until_complete(_run())

    def test_absolute_url_not_modified(self):
        """Absolute URLs should be used as-is."""

        async def _run():
            config = _make_config()
            db = _make_db()
            storage = _make_storage()
            downloader = AsyncPDFDownloader(config, db, storage)

            called_urls: list[str] = []

            async def fake_fetch(url: str) -> bytes:
                called_urls.append(url)
                return b"%PDF-1.4-fake"

            downloader._fetch_url = fake_fetch  # type: ignore[method-assign]

            async with downloader:
                paper = {"id": "p2", "pdf_url": "https://arxiv.org/pdf/1234.5678"}
                await downloader._download_one(paper)

            assert called_urls[0] == "https://arxiv.org/pdf/1234.5678"

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# download_papers skips papers without pdf_url
# ---------------------------------------------------------------------------


class TestDownloadPapersFiltering:
    def test_no_papers_with_pdf_url_returns_empty(self):
        async def _run():
            downloader = AsyncPDFDownloader(_make_config(), _make_db(), _make_storage())
            async with downloader:
                results = await downloader.download_papers([
                    {"id": "p1"},
                    {"id": "p2", "pdf_url": ""},
                    {"id": "p3", "pdf_url": None},
                ])
            return results

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert results == []

    def test_skips_paper_without_pdf_url_key(self):
        async def _run():
            downloader = AsyncPDFDownloader(_make_config(), _make_db(), _make_storage())
            async with downloader:
                results = await downloader.download_papers([{"id": "no-url"}])
            return results

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert results == []

    def test_skips_already_done_papers(self):
        """Papers with stage status 'done' in the DB are not re-downloaded."""

        async def _run():
            done_status = {"status": "done"}
            db = _make_db(stage_status=done_status)
            storage = _make_storage()
            downloader = AsyncPDFDownloader(_make_config(), db, storage)

            fetched_urls: list[str] = []

            async def fake_fetch(url: str) -> bytes:
                fetched_urls.append(url)
                return b"%PDF-done"

            downloader._fetch_url = fake_fetch  # type: ignore[method-assign]

            async with downloader:
                results = await downloader.download_papers([
                    {"id": "already-done", "pdf_url": "https://example.com/paper.pdf"},
                ])
            return results, fetched_urls

        results, fetched_urls = asyncio.get_event_loop().run_until_complete(_run())
        # Result list is empty because the paper was skipped
        assert results == []
        assert fetched_urls == []

    def test_downloads_paper_with_pdf_url(self):
        """A paper with a valid pdf_url should be downloaded."""

        async def _run():
            db = _make_db(stage_status=None)  # not done yet
            storage = _make_storage()
            downloader = AsyncPDFDownloader(_make_config(), db, storage)

            async def fake_fetch(url: str) -> bytes:
                return b"%PDF-1.4 fake content here"

            downloader._fetch_url = fake_fetch  # type: ignore[method-assign]

            async with downloader:
                results = await downloader.download_papers([
                    {"id": "new-paper", "pdf_url": "https://example.com/new.pdf"},
                ])
            return results

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["paper_id"] == "new-paper"
        assert results[0]["file_path"] == "pdfs/new-paper.pdf"


# ---------------------------------------------------------------------------
# _fetch_url raises RuntimeError when session is None
# ---------------------------------------------------------------------------


class TestFetchUrlWithoutSession:
    def test_raises_runtime_error_without_context_manager(self):
        async def _run():
            downloader = AsyncPDFDownloader(_make_config(), _make_db(), _make_storage())
            # Do NOT use async with — session stays None
            with pytest.raises(RuntimeError, match="async context manager"):
                await downloader._fetch_url("https://example.com/paper.pdf")

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Storage key format
# ---------------------------------------------------------------------------


class TestStorageKey:
    def test_storage_key_uses_paper_id(self):
        async def _run():
            db = _make_db()
            storage = _make_storage()
            downloader = AsyncPDFDownloader(_make_config(), db, storage)

            async def fake_fetch(url: str) -> bytes:
                return b"%PDF"

            downloader._fetch_url = fake_fetch  # type: ignore[method-assign]

            async with downloader:
                result = await downloader._download_one(
                    {"id": "my-paper-id", "pdf_url": "https://example.com/x.pdf"}
                )
            return result, storage

        result, storage = asyncio.get_event_loop().run_until_complete(_run())
        storage.put.assert_called_once_with("pdfs/my-paper-id.pdf", b"%PDF")
        assert result["file_path"] == "pdfs/my-paper-id.pdf"
