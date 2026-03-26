"""FAISS-backed vector index for multi-view paper search."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import faiss
import numpy as np

from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.storage.base import StorageBackend

logger = get_logger(__name__, stage="embed")

# Storage keys
INDEX_METADATA_KEY = "index/metadata.json"
INDEX_FAISS_KEY_TEMPLATE = "index/{view}.faiss"


class PaperIndex:
    """Multi-view FAISS index for semantic paper search.

    One FAISS flat-IP index is maintained per embedding view.  The
    metadata list tracks (paper_id, view_type) so results can be
    returned with full context.
    """

    def __init__(self, dimensions: int, views: list[str]) -> None:
        self._dimensions = dimensions
        self._views = list(views)

        # Per-view FAISS IndexFlatIP (inner-product, works like cosine for
        # unit-normalised vectors)
        self._indexes: dict[str, faiss.IndexFlatIP] = {
            view: faiss.IndexFlatIP(dimensions) for view in views
        }

        # Parallel metadata: metadata[view][i] → paper_id for FAISS row i
        self._metadata: dict[str, list[str]] = {view: [] for view in views}

    # ------------------------------------------------------------------
    # Build / add
    # ------------------------------------------------------------------

    def build(self, embedding_records: list[dict]) -> None:
        """Populate the index from a list of embedding records.

        Each record must have: paper_id, view_type, vector.

        Args:
            embedding_records: Output of Embedder.embed_papers.
        """
        # Group by view
        by_view: dict[str, list[dict]] = {view: [] for view in self._views}
        for rec in embedding_records:
            view = rec["view_type"]
            if view in by_view:
                by_view[view].append(rec)

        for view, recs in by_view.items():
            if not recs:
                continue
            vecs = np.array([r["vector"] for r in recs], dtype=np.float32)
            # L2-normalise for cosine similarity via inner product
            faiss.normalize_L2(vecs)
            self._indexes[view].add(vecs)
            self._metadata[view].extend(r["paper_id"] for r in recs)

        logger.info(
            "Built index: %d views, %d records",
            len(self._views),
            len(embedding_records),
        )

    def add(self, embedding_records: list[dict]) -> None:
        """Add new embedding records to an existing index.

        Args:
            embedding_records: List of embedding records (same format as build).
        """
        self.build(embedding_records)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        view: str,
        top_k: int = 10,
        venue_filter: list[str] | None = None,
        paper_venues: dict[str, str] | None = None,
    ) -> list[dict]:
        """Search for nearest neighbours in *view*.

        Args:
            query_vector: Query embedding (same dimension as the index).
            view: Which view's index to search.
            top_k: Maximum number of results.
            venue_filter: If set, only return results whose venue is in this list.
            paper_venues: Optional mapping from paper_id to venue string.

        Returns:
            List of dicts with keys: paper_id, score, rank, view_type.
        """
        if view not in self._indexes:
            raise ValueError(f"Unknown view {view!r}. Available: {self._views}")

        index = self._indexes[view]
        meta = self._metadata[view]

        if index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)

        # Fetch more results when filtering so we can satisfy top_k after filtering
        fetch_k = min(index.ntotal, top_k * 10 if venue_filter else top_k)
        scores, ids = index.search(q, fetch_k)

        venue_set = set(venue_filter) if venue_filter else None
        results: list[dict] = []
        rank = 1
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0:
                continue
            paper_id = meta[idx]
            if venue_set and paper_venues:
                if paper_venues.get(paper_id) not in venue_set:
                    continue
            results.append(
                {
                    "paper_id": paper_id,
                    "score": float(score),
                    "rank": rank,
                    "view_type": view,
                }
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, storage: StorageBackend) -> None:
        """Serialise the index to *storage*.

        Writes:
        - ``index/<view>.faiss`` for each view
        - ``index/metadata.json`` with the metadata lists
        """
        # Save each FAISS index
        for view, faiss_index in self._indexes.items():
            buf = faiss.serialize_index(faiss_index)
            storage.put(INDEX_FAISS_KEY_TEMPLATE.format(view=view), buf.tobytes())

        # Save metadata
        meta_payload = {
            "dimensions": self._dimensions,
            "views": self._views,
            "metadata": self._metadata,
        }
        storage.put(INDEX_METADATA_KEY, json.dumps(meta_payload).encode())
        logger.info("Saved index to storage (%d views)", len(self._views))

    @classmethod
    def load(cls, storage: StorageBackend) -> PaperIndex:
        """Load a previously saved index from *storage*.

        Args:
            storage: Storage backend that contains the index files.

        Returns:
            Populated PaperIndex instance.

        Raises:
            FileNotFoundError: if any required file is missing.
        """
        meta_bytes = storage.get(INDEX_METADATA_KEY)
        meta_payload = json.loads(meta_bytes.decode())

        dimensions: int = meta_payload["dimensions"]
        views: list[str] = meta_payload["views"]
        metadata: dict[str, list[str]] = meta_payload["metadata"]

        instance = cls(dimensions=dimensions, views=views)
        instance._metadata = metadata

        for view in views:
            faiss_key = INDEX_FAISS_KEY_TEMPLATE.format(view=view)
            faiss_bytes = storage.get(faiss_key)
            arr = np.frombuffer(faiss_bytes, dtype=np.uint8)
            instance._indexes[view] = faiss.deserialize_index(arr)

        logger.info("Loaded index from storage (%d views)", len(views))
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of vectors across all views."""
        return sum(idx.ntotal for idx in self._indexes.values())

    @property
    def views(self) -> list[str]:
        return list(self._views)

    @property
    def dimensions(self) -> int:
        return self._dimensions
