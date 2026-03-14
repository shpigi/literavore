"""Metadata normalization utilities for OpenReview paper data."""

import unicodedata
from typing import Any

from literavore.utils import get_logger

logger = get_logger(__name__)

OPENREVIEW_BASE_URL = "https://openreview.net"


def extract_value(field: Any) -> Any:
    """Return the value from an OpenReview V2 nested field dict, or the field as-is.

    OpenReview V2 API wraps many fields as {"value": <actual_value>}. This helper
    unwraps that structure, passing through anything that is not a dict with a
    single "value" key unchanged.
    """
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def clean_author_name(name: str) -> str:
    """Normalize a single author name string.

    Steps applied:
    1. Strip leading/trailing whitespace.
    2. Normalize unicode to NFC form.
    3. Convert "Last, First" comma-separated format to "First Last".
    """
    name = name.strip()
    name = unicodedata.normalize("NFC", name)
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            name = f"{parts[1]} {parts[0]}"
    return name


def normalize_paper_metadata(raw: dict) -> dict:
    """Normalize a raw OpenReview paper dict into a clean, flat representation.

    Handles both V1 (content nested under "content" key) and V2 (direct fields
    with {"value": ...} wrappers) API structures.

    Returns a dict with keys:
        id, title, authors, abstract, keywords, venue, pdf_url,
        created_date, modified_date, forum, number
    """
    paper_id: str = raw.get("id", "")

    # Determine API version: V1 stores most fields under raw["content"]
    content = raw.get("content", {})
    is_v1 = bool(content) and not isinstance(next(iter(content.values()), None), dict)

    def _get(field_name: str, default: Any = None) -> Any:
        """Extract a field, checking content dict for V1 and top-level for V2."""
        if is_v1:
            return content.get(field_name, raw.get(field_name, default))
        # V2: try top-level first, then content (some fields stay in content)
        value = raw.get(field_name, content.get(field_name, default))
        return extract_value(value)

    def _get_content(field_name: str, default: Any = None) -> Any:
        """Extract a field from the content sub-dict, unwrapping V2 value wrappers."""
        value = content.get(field_name, default)
        return extract_value(value)

    # --- Title ---
    title_raw = _get_content("title") or _get("title", "")
    title = str(extract_value(title_raw)).strip() if title_raw is not None else ""

    # --- Authors ---
    authors_raw = _get_content("authors") or _get("authors", [])
    authors_raw = extract_value(authors_raw) or []
    if isinstance(authors_raw, str):
        authors_raw = [authors_raw]
    authors = [clean_author_name(str(a)) for a in authors_raw if a]

    # --- Abstract ---
    abstract_raw = _get_content("abstract") or _get("abstract", "")
    abstract = str(extract_value(abstract_raw)).strip() if abstract_raw else ""

    # --- Keywords ---
    keywords_raw = _get_content("keywords") or _get("keywords", [])
    keywords_raw = extract_value(keywords_raw) or []
    if isinstance(keywords_raw, str):
        keywords: list[str] = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    else:
        keywords = [str(k).strip() for k in keywords_raw if k]

    # --- Venue ---
    venue_raw = _get_content("venue") or _get("venue", "")
    venue = str(extract_value(venue_raw)).strip() if venue_raw else ""

    # --- PDF URL ---
    pdf_raw = _get_content("pdf") or _get("pdf", "")
    pdf_str = str(extract_value(pdf_raw)).strip() if pdf_raw else ""
    if pdf_str.startswith("/pdf/"):
        pdf_url = f"{OPENREVIEW_BASE_URL}{pdf_str}"
    elif pdf_str.startswith("http"):
        pdf_url = pdf_str
    else:
        pdf_url = pdf_str  # empty string or unexpected format — pass through

    # --- Dates ---
    created_date: int | None = raw.get("cdate") or raw.get("created", None)
    modified_date: int | None = raw.get("mdate") or raw.get("modified", None)

    # --- Forum / Number ---
    forum: str = raw.get("forum", "")
    number: int | None = raw.get("number")

    result = {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "keywords": keywords,
        "venue": venue,
        "pdf_url": pdf_url,
        "created_date": created_date,
        "modified_date": modified_date,
        "forum": forum,
        "number": number,
    }

    logger.debug("Normalized paper %s: title=%r authors=%d", paper_id, title, len(authors))
    return result


def simplify_paper_data(papers: list[dict], api_version: str = "v2") -> list[dict]:
    """Batch-normalize a list of raw OpenReview paper dicts.

    Args:
        papers: List of raw paper dicts as returned by the OpenReview API.
        api_version: Hint for logging; actual detection is done per-paper by
            ``normalize_paper_metadata``.

    Returns:
        List of normalized paper dicts, skipping any papers that raise an
        exception during normalization (with a warning logged).
    """
    results: list[dict] = []
    for raw in papers:
        try:
            results.append(normalize_paper_metadata(raw))
        except Exception:
            paper_id = raw.get("id", "<unknown>")
            logger.warning("Failed to normalize paper %s", paper_id, exc_info=True)
    logger.info(
        "simplify_paper_data: normalized %d/%d papers (api_version=%s)",
        len(results),
        len(papers),
        api_version,
    )
    return results
