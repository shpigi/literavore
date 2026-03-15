"""Tag extraction for summarized papers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from literavore.config import SummaryConfig
from literavore.summarize.prompts import TAG_SYSTEM, TAG_USER_TEMPLATE
from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.summarize.llm_client import LLMClient

logger = get_logger(__name__, stage="summarize")


class Tagger:
    """Extracts structured tags from paper metadata using an LLM."""

    def __init__(self, config: SummaryConfig, llm_client: LLMClient) -> None:
        self._config = config
        self._llm = llm_client

    def extract_tags_from_keywords(self, keywords: list[str]) -> dict:
        """Build a tag dict directly from a keywords list (no LLM call).

        Args:
            keywords: List of keyword strings from paper metadata.

        Returns:
            dict with key_phrases, domains, methods keys.
        """
        return {
            "key_phrases": [kw.lower() for kw in keywords],
            "domains": [],
            "methods": [],
            "datasets_benchmarks": [],
        }

    async def extract_tags(
        self,
        title: str,
        abstract: str,
        summary: str,
        keywords: list[str] | None = None,
    ) -> dict:
        """Extract structured tags using the LLM.

        Falls back to an empty tag structure when enable_tag_extraction is False
        or when the LLM response cannot be parsed as JSON.

        Args:
            title: Paper title.
            abstract: Paper abstract.
            summary: Previously generated summary string.

        Returns:
            dict with key_phrases, domains, methods keys.
        """
        empty: dict = {"key_phrases": [], "domains": [], "methods": [], "datasets_benchmarks": []}

        if not self._config.enable_tag_extraction:
            logger.debug("Tag extraction disabled — returning empty tags")
            return empty

        keywords_section = ""
        if keywords:
            keywords_section = "\nAuthor-supplied keywords: " + ", ".join(keywords)

        messages = [
            {"role": "system", "content": TAG_SYSTEM},
            {
                "role": "user",
                "content": TAG_USER_TEMPLATE.format(
                    title=title,
                    abstract=abstract,
                    summary=summary,
                    keywords_section=keywords_section,
                ),
            },
        ]

        try:
            raw = await self._llm.achat_complete(
                messages,
                max_tokens=self._config.max_tag_tokens,
                temperature=0.1,
                model=self._config.tag_model,
            )
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            parsed = json.loads(cleaned)
            return {
                "key_phrases": parsed.get("key_phrases", []),
                "domains": parsed.get("domains", []),
                "methods": parsed.get("methods", []),
                "datasets_benchmarks": parsed.get("datasets_benchmarks", []),
            }
        except (json.JSONDecodeError, KeyError, Exception) as exc:  # noqa: BLE001
            logger.warning("Tag extraction failed (%s) — returning empty tags", exc)
            return empty
