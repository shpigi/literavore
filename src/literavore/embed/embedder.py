"""Text embedding via OpenAI or mock zero-vectors."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import numpy as np

from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.config import EmbeddingConfig

logger = get_logger(__name__, stage="embed")

# View names supported
VIEW_TITLE_ABSTRACT = "title_abstract"
VIEW_PAPER_CARD = "paper_card"
VIEW_KEYWORD_ENRICHED = "keyword_enriched"
ALL_VIEWS = [VIEW_TITLE_ABSTRACT, VIEW_PAPER_CARD, VIEW_KEYWORD_ENRICHED]


def build_view_text(view: str, paper: dict, summary_data: dict | None = None) -> str:
    """Build the text string for a given embedding view.

    Args:
        view: One of "title_abstract", "paper_card", "keyword_enriched".
        paper: Paper dict (must include at minimum "title").
        summary_data: Optional summary JSON dict with "summary", "tags", "structured_tags".

    Returns:
        Text string ready for embedding.
    """
    title: str = paper.get("title", "") or ""
    abstract: str = paper.get("abstract", "") or ""

    summary: str = ""
    tags: list[str] = []
    keywords: list[str] = []

    if summary_data is not None:
        summary = summary_data.get("summary", "") or ""
        tags = summary_data.get("tags", []) or []
        structured = summary_data.get("structured_tags", {}) or {}
        keywords = (
            structured.get("key_phrases", [])
            + structured.get("domains", [])
            + structured.get("methods", [])
        )

    if view == VIEW_TITLE_ABSTRACT:
        parts = [title]
        if abstract:
            parts.append(abstract)
        return "\n\n".join(parts)

    if view == VIEW_PAPER_CARD:
        parts = [title]
        if abstract:
            parts.append(abstract)
        if summary:
            parts.append(f"Summary: {summary}")
        if tags:
            parts.append("Tags: " + ", ".join(tags))
        return "\n\n".join(parts)

    if view == VIEW_KEYWORD_ENRICHED:
        parts = [title]
        if abstract:
            parts.append(abstract)
        if summary:
            parts.append(f"Summary: {summary}")
        all_kw = list(dict.fromkeys(tags + keywords))  # deduplicate preserving order
        if all_kw:
            parts.append("Keywords: " + ", ".join(all_kw))
        return "\n\n".join(parts)

    raise ValueError(f"Unknown view: {view!r}. Expected one of {ALL_VIEWS}")


def _text_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Embedder:
    """Generates multi-view embeddings for papers.

    In mock mode (no OPENAI_API_KEY) returns zero vectors of the configured
    dimension.  In live mode calls OpenAI's embedding endpoint.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._cache: dict[str, list[float]] = {}
        self._total_tokens: int = 0

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._mock_mode = not api_key

        if not self._mock_mode:
            try:
                from openai import OpenAI  # noqa: PLC0415

                self._client = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not available — falling back to mock mode")
                self._mock_mode = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, using an in-memory cache.

        Args:
            texts: Input strings.

        Returns:
            List of float vectors, one per input text.
        """
        results: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = _text_cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append([])  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_vectors = self._embed_batch(uncached_texts)
            for i, vec in zip(uncached_indices, new_vectors, strict=True):
                key = _text_cache_key(texts[i])
                self._cache[key] = vec
                results[i] = vec

        return results

    def embed_papers(
        self,
        papers: list[dict],
        summaries: dict[str, dict],
        views: list[str] | None = None,
    ) -> list[dict]:
        """Generate multi-view embeddings for each paper.

        Args:
            papers: List of paper dicts.
            summaries: Mapping from paper_id to summary data dict.
            views: View names to embed; defaults to config.views.

        Returns:
            List of embedding records, each a dict with keys:
            paper_id, view_type, vector, dimensions.
        """
        if views is None:
            views = list(self._config.views)

        # Build all (paper, view) texts in one pass, then embed in batch
        records: list[tuple[str, str, str]] = []  # (paper_id, view, text)
        for paper in papers:
            paper_id: str = paper["id"]
            summary_data = summaries.get(paper_id)
            for view in views:
                text = build_view_text(view, paper, summary_data)
                records.append((paper_id, view, text))

        texts = [r[2] for r in records]
        vectors = self.embed_texts(texts)

        results: list[dict] = []
        for (paper_id, view, _text), vector in zip(records, vectors, strict=True):
            results.append(
                {
                    "paper_id": paper_id,
                    "view_type": view,
                    "vector": vector,
                    "dimensions": len(vector),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (no cache layer)."""
        if self._mock_mode:
            return [[0.0] * self._config.dimensions for _ in texts]

        # Live mode: call OpenAI in sub-batches
        all_vectors: list[list[float]] = []
        batch_size = self._config.batch_size
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            response = self._client.embeddings.create(
                model=self._config.model,
                input=chunk,
                dimensions=self._config.dimensions,
            )
            if response.usage:
                self._total_tokens += response.usage.total_tokens
            for item in sorted(response.data, key=lambda x: x.index):
                vec = item.embedding
                # Ensure correct dimensionality (some models ignore the param)
                if len(vec) != self._config.dimensions:
                    arr = np.array(vec, dtype=np.float32)
                    arr = arr[: self._config.dimensions]
                    vec = arr.tolist()
                all_vectors.append(vec)

        return all_vectors
