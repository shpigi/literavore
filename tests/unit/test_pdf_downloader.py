"""Tests for literavore.ingest.pdf_downloader."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from literavore.config import PdfConfig
from literavore.db import Database
from literavore.ingest.pdf_downloader import OPENREVIEW_BASE, AsyncPDFDownloader
from literavore.storage.local import LocalStorage


def _make_downloader(tmp_path: Path) -> AsyncPDFDownloader:
    config = PdfConfig(max_concurrent=2, max_retries=0, timeout=5, backoff_jitter=False)
    db = Database(tmp_path / "test.db")
    storage = LocalStorage(tmp_path / "data")
    return AsyncPDFDownloader(config, db, storage)


class TestCreation:
    def test_creates_downloader(self, tmp_path: Path):
        dl = _make_downloader(tmp_path)
        assert dl._session is None

    def test_semaphore_value(self, tmp_path: Path):
        dl = _make_downloader(tmp_path)
        assert dl._semaphore._value == 2


class TestUrlResolution:
    def test_openreview_base_constant(self):
        assert OPENREVIEW_BASE == "https://openreview.net"


class TestDownloadPapersFiltering:
    def test_no_pdf_url_returns_empty(self, tmp_path: Path):
        dl = _make_downloader(tmp_path)
        papers = [{"id": "p1", "title": "No PDF"}]

        async def run():
            async with dl:
                return await dl.download_papers(papers)

        results = asyncio.run(run())
        assert len(results) == 0

    def test_missing_pdf_url_key(self, tmp_path: Path):
        dl = _make_downloader(tmp_path)
        papers = [{"id": "p1"}]

        async def run():
            async with dl:
                return await dl.download_papers(papers)

        results = asyncio.run(run())
        assert len(results) == 0
