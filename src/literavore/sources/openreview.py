"""OpenReview paper source implementation."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import parse_qs, urlparse

from literavore.config import ConferenceConfig, FetchConfig
from literavore.sources.base import PaperMetadata
from literavore.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import openreview  # type: ignore[import]

    _OPENREVIEW_AVAILABLE = True
except ImportError:
    _OPENREVIEW_AVAILABLE = False
    logger.warning("openreview package not available; OpenReviewSource will not function")


def _parse_group_id(url: str) -> str:
    """Extract the OpenReview group ID from a URL.

    For example: https://openreview.net/group?id=NeurIPS.cc/2024/Conference
    returns: NeurIPS.cc/2024/Conference
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    ids = qs.get("id", [])
    if not ids:
        raise ValueError(f"Could not extract group id from URL: {url!r}")
    return ids[0]


def _extract_value(field: Any) -> Any:
    """Extract the value from an OpenReview V2 API field.

    V2 API wraps field values in a dict like {"value": <actual_value>}.
    This helper unwraps that pattern, or returns the field as-is when it
    is already a plain value.
    """
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


class OpenReviewSource:
    """Fetch paper metadata from OpenReview using the V2 API."""

    def __init__(self, fetch_config: FetchConfig | None = None) -> None:
        if not _OPENREVIEW_AVAILABLE:
            raise ImportError(
                "The 'openreview' package is required to use OpenReviewSource. "
                "Install it with: pip install openreview-py"
            )
        self._fetch_config = fetch_config or FetchConfig()
        self._client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, config: ConferenceConfig) -> list[PaperMetadata]:
        """Fetch all papers for the conference described by *config*."""
        group_id = _parse_group_id(config.openreview_url)
        logger.info("Fetching papers for group %s", group_id)

        venue_ids = self._discover_venue_ids(group_id)
        logger.info("Discovered %d venue(s): %s", len(venue_ids), venue_ids)

        if config.filter_for:
            venue_ids = self._filter_venues(venue_ids, config.filter_for)
            logger.info("After filtering: %d venue(s): %s", len(venue_ids), venue_ids)

        papers: list[PaperMetadata] = []
        for venue_id in venue_ids:
            notes = self._fetch_notes(venue_id)
            logger.info("Fetched %d notes from venue %s", len(notes), venue_id)
            for note in notes:
                paper = self._note_to_metadata(note)
                if paper is not None:
                    papers.append(paper)
                if config.max_papers and len(papers) >= config.max_papers:
                    break
            if config.max_papers and len(papers) >= config.max_papers:
                break

        if config.max_papers:
            papers = papers[: config.max_papers]

        logger.info("Total papers fetched: %d", len(papers))
        return papers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_venue_ids(self, group_id: str) -> list[str]:
        """Return a list of venue IDs to query.

        Tries ``{group_id}/Conference`` first; falls back to ``group_id`` itself.
        """
        candidate = f"{group_id}/Conference"
        try:
            notes = self._client.get_all_notes(content={"venueid": candidate})
            if notes:
                return [candidate]
        except Exception:
            pass

        return [group_id]

    def _fetch_notes(self, venue_id: str) -> list[Any]:
        """Retrieve all notes for a given venue ID."""
        try:
            return self._client.get_all_notes(content={"venueid": venue_id})
        except Exception as exc:
            logger.error("Failed to fetch notes for venue %s: %s", venue_id, exc)
            return []

    def _note_to_metadata(self, note: Any) -> PaperMetadata | None:
        """Convert a single OpenReview note to a PaperMetadata object."""
        try:
            content = note.content or {}

            title = _extract_value(content.get("title", "")) or ""
            abstract = _extract_value(content.get("abstract", "")) or ""

            raw_authors = _extract_value(content.get("authors", [])) or []
            authors = [str(a) for a in raw_authors] if isinstance(raw_authors, list) else []

            raw_keywords = _extract_value(content.get("keywords", [])) or []
            keywords = [str(k) for k in raw_keywords] if isinstance(raw_keywords, list) else []

            venue = _extract_value(content.get("venue", "")) or ""

            pdf_path = _extract_value(content.get("pdf", "")) or ""
            if pdf_path and not pdf_path.startswith("http"):
                pdf_url = f"https://openreview.net{pdf_path}"
            else:
                pdf_url = pdf_path

            published_date: str | None = None
            pdate = getattr(note, "pdate", None)
            if pdate is not None:
                try:
                    from datetime import UTC, datetime  # noqa: PLC0415

                    published_date = datetime.fromtimestamp(pdate / 1000, tz=UTC).date().isoformat()
                except Exception:  # noqa: BLE001
                    pass

            return PaperMetadata(
                id=note.id,
                title=title,
                authors=authors,
                abstract=abstract,
                keywords=keywords,
                venue=venue,
                pdf_url=pdf_url,
                source_url=f"https://openreview.net/forum?id={note.id}",
                published_date=published_date,
                raw_data=dict(content),
            )
        except Exception as exc:
            logger.warning("Skipping note %s due to error: %s", getattr(note, "id", "?"), exc)
            return None

    def _filter_venues(self, venue_ids: list[str], filter_for: list[str]) -> list[str]:
        """Use an LLM to filter venue IDs to those matching *filter_for* criteria.

        Falls back to returning all venue IDs if the OpenAI key is absent or the
        call fails.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set; skipping venue filtering and using all venues"
            )
            return venue_ids

        try:
            import openai  # type: ignore[import]

            client = openai.OpenAI(api_key=api_key)
            prompt = (
                "You are helping to filter OpenReview venue IDs.\n"
                f"Filter criteria: {filter_for}\n"
                f"Venue IDs to filter: {venue_ids}\n\n"
                "Return a JSON object with a single key 'venues' whose value is a list of "
                "venue IDs from the input that match the filter criteria. "
                "If none match, return all venue IDs."
            )
            response = client.chat.completions.create(
                model=self._fetch_config.venue_filter_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content or "{}")
            filtered: list[str] = result.get("venues", venue_ids)
            if not filtered:
                logger.warning("LLM returned empty venue list; using all venues")
                return venue_ids
            return filtered
        except Exception as exc:
            logger.warning("Venue filtering via LLM failed (%s); using all venues", exc)
            return venue_ids
