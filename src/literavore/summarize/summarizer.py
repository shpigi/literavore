"""LLM-based paper summarization."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

from literavore.config import SummaryConfig
from literavore.summarize.llm_client import LLMClient
from literavore.summarize.prompts import (
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER_TEMPLATE,
)
from literavore.summarize.tagger import Tagger
from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.db import Database
    from literavore.storage.base import StorageBackend

logger = get_logger(__name__, stage="summarize")


def _content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class Summarizer:
    """Generates summaries and tags for extracted papers."""

    def __init__(
        self,
        config: SummaryConfig,
        db: Database,
        storage: StorageBackend,
    ) -> None:
        self._config = config
        self._db = db
        self._storage = storage
        self._llm = LLMClient(config)
        self._tagger = Tagger(config, self._llm)

    async def _summarize_single(self, paper: dict) -> dict | None:
        """Summarize one paper.  Returns result dict or None on failure."""
        paper_id: str = paper["id"]
        title: str = paper.get("title", "")
        abstract: str = paper.get("abstract", "") or ""

        raw_keywords = paper.get("keywords") or []
        if isinstance(raw_keywords, str):
            try:
                raw_keywords = json.loads(raw_keywords)
            except json.JSONDecodeError:
                raw_keywords = []
        paper_keywords: list[str] = [str(k) for k in raw_keywords] if raw_keywords else []

        extract_key = f"extract/{paper_id}.json"

        # Load extracted text from storage
        try:
            extract_bytes = self._storage.get(extract_key)
            extract_data = json.loads(extract_bytes.decode())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s — extract JSON missing or invalid: %s", paper_id, exc)
            self._db.update_stage_status(paper_id, "summarize", "failed", error=str(exc))
            return None

        full_text: str = extract_data.get("full_text", "")
        text_excerpt = full_text[:self._config.max_text_excerpt_chars]

        # Cache check: skip if summary already exists for same content
        summary_key = f"summaries/{paper_id}.json"
        new_hash = _content_hash(full_text)
        if self._config.cache_enabled and self._storage.exists(summary_key):
            try:
                existing_bytes = self._storage.get(summary_key)
                existing = json.loads(existing_bytes.decode())
                if existing.get("content_hash") == new_hash:
                    logger.debug("Cache hit for %s — skipping re-summarization", paper_id)
                    self._db.update_stage_status(paper_id, "summarize", "done")
                    return existing
            except Exception:  # noqa: BLE001
                pass  # If cache read fails, proceed with re-summarization

        self._db.update_stage_status(paper_id, "summarize", "running")

        try:
            messages = [
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {
                    "role": "user",
                    "content": SUMMARIZE_USER_TEMPLATE.format(
                        title=title,
                        abstract=abstract,
                        text_excerpt=text_excerpt,
                    ),
                },
            ]
            raw = await self._llm.achat_complete(messages)

            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            parsed = json.loads(cleaned)
            summary_text: str = parsed.get("summary", "")
            tags: list = parsed.get("tags", [])

            # Optionally enrich with structured tags
            structured_tags = await self._tagger.extract_tags(
                title, abstract, summary_text, keywords=paper_keywords or None
            )

            result = {
                "paper_id": paper_id,
                "title": title,
                "summary": summary_text,
                "tags": tags,
                "structured_tags": structured_tags,
                "content_hash": new_hash,
            }

            self._storage.put(summary_key, json.dumps(result).encode())
            self._db.update_stage_status(paper_id, "summarize", "done")
            logger.info("Summarized paper %s", paper_id)
            return result

        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to summarize paper %s: %s", paper_id, error_msg)
            self._db.update_stage_status(paper_id, "summarize", "failed", error=error_msg)
            return None

    async def summarize_papers(self, papers: list[dict]) -> list[dict]:
        """Summarize a list of papers concurrently.

        Args:
            papers: List of paper dicts (must include 'id').

        Returns:
            List of result dicts for successfully summarized papers.
        """
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        async def _with_sem(paper: dict) -> dict | None:
            async with semaphore:
                return await self._summarize_single(paper)

        tasks = [_with_sem(paper) for paper in papers]
        raw_results = await asyncio.gather(*tasks, return_exceptions=False)
        results = [r for r in raw_results if r is not None]
        logger.info("Summarization complete: %d/%d papers succeeded", len(results), len(papers))
        return results
