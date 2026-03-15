"""Async PDF downloader for Literavore."""

from __future__ import annotations

import asyncio
import random
from types import TracebackType

import aiohttp

from literavore.config import PdfConfig
from literavore.db import Database
from literavore.ingest.pdf_validator import validate_pdf
from literavore.storage.base import StorageBackend
from literavore.utils import get_logger

logger = get_logger(__name__, stage="download")

OPENREVIEW_BASE = "https://openreview.net"


class AsyncPDFDownloader:
    """Async PDF downloader with concurrency control, retries, and rate-limit handling."""

    def __init__(self, config: PdfConfig, db: Database, storage: StorageBackend) -> None:
        self._config = config
        self._db = db
        self._storage = storage
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> AsyncPDFDownloader:
        timeout = aiohttp.ClientTimeout(total=self._config.timeout)
        headers = {"User-Agent": self._config.user_agent}
        self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def download_papers(self, papers: list[dict]) -> list[dict]:
        """Download PDFs for all papers that have a pdf_url.

        Papers already marked 'done' in the DB are skipped. The pipeline is
        responsible for filtering by force-flag before calling this method.

        Returns a list of result dicts with keys:
            paper_id, success, error, file_path, size_bytes
        """
        eligible = [p for p in papers if p.get("pdf_url")]
        if not eligible:
            logger.info("No papers with pdf_url — nothing to download")
            return []

        # Skip papers already done
        to_download: list[dict] = []
        for paper in eligible:
            paper_id = paper["id"]
            stage_status = self._db.get_stage_status(paper_id, "download")
            if stage_status and stage_status.get("status") == "done":
                logger.debug("Skipping %s — already downloaded", paper_id)
            else:
                to_download.append(paper)

        already_done = len(eligible) - len(to_download)
        logger.info("Downloading %d PDFs (%d already done)", len(to_download), already_done)

        tasks = [self._download_one(paper) for paper in to_download]
        results: list[dict] = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def _download_one(self, paper: dict) -> dict:
        """Download a single PDF with semaphore, delay, and exponential-backoff retries."""
        paper_id: str = paper["id"]
        raw_url: str = paper["pdf_url"]

        # Normalise relative OpenReview PDF URLs
        if raw_url.startswith("/pdf/"):
            url = f"{OPENREVIEW_BASE}{raw_url}"
        else:
            url = raw_url

        storage_key = f"pdfs/{paper_id}.pdf"

        async with self._semaphore:
            # Polite delay before each request
            if self._config.delay_between_requests > 0:
                await asyncio.sleep(self._config.delay_between_requests)

            self._db.update_stage_status(paper_id, "download", "running")

            last_error: str = ""
            for attempt in range(self._config.max_retries + 1):
                if attempt > 0:
                    backoff = min(
                        self._config.base_backoff ** attempt,
                        self._config.max_backoff,
                    )
                    if self._config.backoff_jitter:
                        backoff += random.uniform(0, backoff * 0.25)
                    logger.debug(
                        "Retry %d/%d for %s — sleeping %.1fs",
                        attempt,
                        self._config.max_retries,
                        paper_id,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

                try:
                    data = await self._fetch_url(url)
                    if self._config.validate_pdfs:
                        valid, reason = validate_pdf(data)
                        if not valid:
                            last_error = f"Validation failed: {reason}"
                            logger.warning("PDF validation failed for %s: %s", paper_id, reason)
                            continue  # retry
                    self._storage.put(storage_key, data)
                    self._db.update_stage_status(paper_id, "download", "done")
                    logger.info("Downloaded %s (%d bytes)", paper_id, len(data))
                    return {
                        "paper_id": paper_id,
                        "success": True,
                        "error": None,
                        "file_path": storage_key,
                        "size_bytes": len(data),
                    }

                except aiohttp.ClientResponseError as exc:
                    last_error = f"HTTP {exc.status}: {exc.message}"
                    if exc.status == 429:
                        logger.warning(
                            "Rate-limited on %s (attempt %d) — extra delay %.1fs",
                            paper_id,
                            attempt,
                            self._config.rate_limit_backoff,
                        )
                        await asyncio.sleep(self._config.rate_limit_backoff)
                    else:
                        logger.warning("HTTP error for %s: %s", paper_id, last_error)

                except (aiohttp.ClientError, TimeoutError) as exc:
                    last_error = str(exc)
                    logger.warning("Download error for %s: %s", paper_id, last_error)

            self._db.update_stage_status(paper_id, "download", "failed", error=last_error)
            total_attempts = self._config.max_retries + 1
            logger.error(
                "Failed to download %s after %d attempts: %s",
                paper_id,
                total_attempts,
                last_error,
            )
            return {
                "paper_id": paper_id,
                "success": False,
                "error": last_error,
                "file_path": None,
                "size_bytes": 0,
            }

    async def _fetch_url(self, url: str) -> bytes:
        """Perform an HTTP GET and return the full response body as bytes."""
        if self._session is None:
            raise RuntimeError("AsyncPDFDownloader must be used as an async context manager")

        async with self._session.get(url, raise_for_status=True) as response:
            chunks: list[bytes] = []
            async for chunk in response.content.iter_chunked(self._config.chunk_size):
                chunks.append(chunk)
            return b"".join(chunks)
