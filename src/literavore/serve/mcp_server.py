"""MCP server for Literavore — semantic paper search via FastMCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from literavore.config import LiteravoreConfig, load_config
from literavore.db import Database
from literavore.embed.embedder import Embedder
from literavore.embed.index import PaperIndex
from literavore.storage.local import LocalStorage
from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.storage.base import StorageBackend

logger = get_logger(__name__, stage="serve")

mcp = FastMCP("literavore")

# ---------------------------------------------------------------------------
# Lazy-initialised singletons
# ---------------------------------------------------------------------------

_config: LiteravoreConfig | None = None
_db: Database | None = None
_storage: StorageBackend | None = None
_embedder: Embedder | None = None
_index: PaperIndex | None = None


def _get_config() -> LiteravoreConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_db() -> Database:
    global _db
    if _db is None:
        cfg = _get_config()
        data_dir = Path(cfg.storage.data_dir)
        _db = Database(data_dir / "literavore.db")
    return _db


def _get_storage() -> StorageBackend:
    global _storage
    if _storage is None:
        cfg = _get_config()
        data_dir = Path(cfg.storage.data_dir)
        _storage = LocalStorage(data_dir)
    return _storage


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        cfg = _get_config()
        _embedder = Embedder(cfg.embedding)
    return _embedder


def _get_index() -> PaperIndex | None:
    global _index
    if _index is None:
        storage = _get_storage()
        try:
            _index = PaperIndex.load(storage)
        except FileNotFoundError:
            logger.warning("FAISS index not found in storage — semantic search unavailable")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load FAISS index: %s", exc)
            return None
    return _index


def _parse_authors(authors_raw: str | list | None) -> list[str]:
    """Return a list of author name strings from various stored formats."""
    if authors_raw is None:
        return []
    if isinstance(authors_raw, list):
        result = []
        for a in authors_raw:
            if isinstance(a, dict):
                result.append(a.get("name", str(a)))
            else:
                result.append(str(a))
        return result
    # JSON string stored by db.py
    try:
        parsed = json.loads(authors_raw)
        return _parse_authors(parsed)
    except (json.JSONDecodeError, TypeError):
        return [authors_raw]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_papers_semantic(
    query: str,
    top_k: int = 10,
    venue_filter: str | None = None,
) -> str:
    """Semantic search over paper embeddings.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.
        venue_filter: If provided, restrict results to papers from this conference/venue.

    Returns:
        JSON string with a list of matching papers ranked by relevance.
    """
    try:
        index = _get_index()
        if index is None:
            return json.dumps({"error": "Search index not available. Run the embed stage first."})

        embedder = _get_embedder()
        query_vector = embedder.embed_texts([query])[0]

        # Build venue→paper_id mapping when a filter is requested
        paper_venues: dict[str, str] | None = None
        if venue_filter:
            db = _get_db()
            all_papers = db.get_papers()
            paper_venues = {p["id"]: (p.get("conference") or "") for p in all_papers}

        # Search the best available view
        view = "keyword_enriched" if "keyword_enriched" in index.views else index.views[0]
        hits = index.search(
            query_vector=query_vector,
            view=view,
            top_k=top_k,
            venue_filter=venue_filter,
            paper_venues=paper_venues,
        )

        # Enrich hits with paper metadata from DB
        db = _get_db()
        results = []
        for hit in hits:
            paper = db.get_paper(hit["paper_id"])
            if paper is None:
                continue
            results.append(
                {
                    "rank": hit["rank"],
                    "score": round(hit["score"], 4),
                    "paper_id": hit["paper_id"],
                    "title": paper.get("title", ""),
                    "authors": _parse_authors(paper.get("authors")),
                    "conference": paper.get("conference", ""),
                    "abstract": (paper.get("abstract") or "")[:300],
                }
            )

        return json.dumps({"query": query, "results": results}, indent=2)

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Semantic search failed: {exc}"})


@mcp.tool()
def search_papers_by_author(author_name: str, top_k: int = 10) -> str:
    """Find papers by author name (case-insensitive partial match).

    Args:
        author_name: Name or partial name to search for.
        top_k: Maximum number of results to return.

    Returns:
        JSON string with a list of matching papers.
    """
    try:
        db = _get_db()
        all_papers = db.get_papers()
        needle = author_name.lower().strip()
        matches = []
        for paper in all_papers:
            authors = _parse_authors(paper.get("authors"))
            if any(needle in a.lower() for a in authors):
                matches.append(
                    {
                        "paper_id": paper["id"],
                        "title": paper.get("title", ""),
                        "authors": authors,
                        "conference": paper.get("conference", ""),
                        "abstract": (paper.get("abstract") or "")[:300],
                    }
                )
                if len(matches) >= top_k:
                    break

        return json.dumps(
            {"author_query": author_name, "count": len(matches), "results": matches}, indent=2
        )

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Author search failed: {exc}"})


@mcp.tool()
def get_paper_details(paper_id: str) -> str:
    """Return full details and summary for a single paper.

    Args:
        paper_id: The unique paper identifier stored in the database.

    Returns:
        JSON string with paper metadata, abstract, and AI summary (if available).
    """
    try:
        db = _get_db()
        paper = db.get_paper(paper_id)
        if paper is None:
            return json.dumps({"error": f"Paper not found: {paper_id!r}"})

        storage = _get_storage()
        summary_data: dict = {}
        summary_key = f"summaries/{paper_id}.json"
        if storage.exists(summary_key):
            try:
                summary_data = json.loads(storage.get(summary_key).decode())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load summary for %s: %s", paper_id, exc)

        stage_status = {}
        for stage in ("fetch", "download", "extract", "summarize", "embed"):
            status = db.get_stage_status(paper_id, stage)
            if status:
                stage_status[stage] = status.get("status")

        return json.dumps(
            {
                "paper_id": paper["id"],
                "title": paper.get("title", ""),
                "authors": _parse_authors(paper.get("authors")),
                "conference": paper.get("conference", ""),
                "abstract": paper.get("abstract", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "openreview_url": f"https://openreview.net/forum?id={paper['id']}",
                "created_at": paper.get("created_at", ""),
                "stage_status": stage_status,
                "summary": summary_data.get("summary", ""),
                "tags": summary_data.get("tags", []),
                "structured_tags": summary_data.get("structured_tags", {}),
            },
            indent=2,
        )

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to retrieve paper details: {exc}"})


@mcp.tool()
def get_paper_statistics() -> str:
    """Return counts per conference and per-stage processing statistics.

    Returns:
        JSON string with total paper counts broken down by conference and pipeline stage.
    """
    try:
        db = _get_db()
        all_papers = db.get_papers()

        conference_counts: dict[str, int] = {}
        for paper in all_papers:
            conf = paper.get("conference") or "unknown"
            conference_counts[conf] = conference_counts.get(conf, 0) + 1

        stage_stats = db.get_run_stats()

        return json.dumps(
            {
                "total_papers": len(all_papers),
                "papers_per_conference": conference_counts,
                "stage_stats": stage_stats,
            },
            indent=2,
        )

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to retrieve statistics: {exc}"})


@mcp.tool()
def get_conference_overview(conference: str | None = None) -> str:
    """Return an overview of papers for one or all conferences.

    Args:
        conference: Conference name to filter by.  If omitted, all conferences are shown.

    Returns:
        JSON string with per-conference paper counts and basic stage completion stats.
    """
    try:
        db = _get_db()

        if conference:
            papers = db.get_papers(conference=conference)
            conferences = [conference]
        else:
            papers = db.get_papers()
            conferences = sorted({p.get("conference") or "unknown" for p in papers})

        overview: dict[str, dict] = {}
        for conf in conferences:
            conf_papers = [p for p in papers if (p.get("conference") or "unknown") == conf]
            # Count papers that completed the embed stage (fully processed)
            fully_processed = 0
            summarised = 0
            for p in conf_papers:
                embed_status = db.get_stage_status(p["id"], "embed")
                if embed_status and embed_status.get("status") == "done":
                    fully_processed += 1
                sum_status = db.get_stage_status(p["id"], "summarize")
                if sum_status and sum_status.get("status") == "done":
                    summarised += 1
            overview[conf] = {
                "total": len(conf_papers),
                "summarised": summarised,
                "embedded": fully_processed,
            }

        return json.dumps({"conference_overview": overview}, indent=2)

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to retrieve conference overview: {exc}"})


@mcp.tool()
def list_conferences() -> str:
    """List all conferences present in the database.

    Returns:
        JSON string with a sorted list of conference names and paper counts.
    """
    try:
        db = _get_db()
        all_papers = db.get_papers()

        counts: dict[str, int] = {}
        for paper in all_papers:
            conf = paper.get("conference") or "unknown"
            counts[conf] = counts.get(conf, 0) + 1

        conferences = [
            {"name": name, "paper_count": count}
            for name, count in sorted(counts.items())
        ]
        return json.dumps({"conferences": conferences, "total": len(conferences)}, indent=2)

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to list conferences: {exc}"})


@mcp.tool()
def get_recent_papers(limit: int = 20) -> str:
    """Return the most recently added papers.

    Args:
        limit: Maximum number of papers to return (default 20).

    Returns:
        JSON string with a list of recently added papers ordered by created_at descending.
    """
    try:
        db = _get_db()
        all_papers = db.get_papers()

        # Sort by created_at descending (ISO strings sort lexicographically)
        sorted_papers = sorted(
            all_papers, key=lambda p: p.get("created_at") or "", reverse=True
        )

        results = []
        for paper in sorted_papers[:limit]:
            results.append(
                {
                    "paper_id": paper["id"],
                    "title": paper.get("title", ""),
                    "authors": _parse_authors(paper.get("authors")),
                    "conference": paper.get("conference", ""),
                    "created_at": paper.get("created_at", ""),
                    "abstract": (paper.get("abstract") or "")[:300],
                }
            )

        return json.dumps({"count": len(results), "papers": results}, indent=2)

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to retrieve recent papers: {exc}"})


@mcp.tool()
def search_by_keywords(keywords: list[str], top_k: int = 10) -> str:
    """Find papers matching a list of keywords against title, abstract, and tags.

    All keywords must appear (AND logic) in the combined text of a paper.

    Args:
        keywords: List of keyword strings to search for.
        top_k: Maximum number of results to return.

    Returns:
        JSON string with a list of matching papers and which keywords matched.
    """
    try:
        db = _get_db()
        storage = _get_storage()
        all_papers = db.get_papers()

        lower_keywords = [kw.lower().strip() for kw in keywords if kw.strip()]
        if not lower_keywords:
            return json.dumps({"error": "No keywords provided."})

        matches = []
        for paper in all_papers:
            title = (paper.get("title") or "").lower()
            abstract = (paper.get("abstract") or "").lower()

            # Try to load summary/tag text
            tag_text = ""
            summary_key = f"summaries/{paper['id']}.json"
            if storage.exists(summary_key):
                try:
                    summary_data = json.loads(storage.get(summary_key).decode())
                    tags: list[str] = summary_data.get("tags", []) or []
                    tag_text = " ".join(t.lower() for t in tags)
                    tag_text += " " + (summary_data.get("summary") or "").lower()
                except Exception:  # noqa: BLE001
                    pass

            combined = f"{title} {abstract} {tag_text}"
            matched_keywords = [kw for kw in lower_keywords if kw in combined]

            if len(matched_keywords) == len(lower_keywords):
                matches.append(
                    {
                        "paper_id": paper["id"],
                        "title": paper.get("title", ""),
                        "authors": _parse_authors(paper.get("authors")),
                        "conference": paper.get("conference", ""),
                        "abstract": (paper.get("abstract") or "")[:300],
                        "matched_keywords": matched_keywords,
                    }
                )
                if len(matches) >= top_k:
                    break

        return json.dumps(
            {"keywords": keywords, "count": len(matches), "results": matches}, indent=2
        )

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Keyword search failed: {exc}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")
