"""PDF text extraction using pymupdf4llm."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pymupdf4llm

logging.getLogger("fitz").setLevel(logging.ERROR)

from literavore.config import ExtractConfig
from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.db import Database
    from literavore.storage.base import StorageBackend

logger = get_logger(__name__, stage="extract")

# Regex to match markdown image references: ![alt text](path)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")

# Regex to split on markdown headings (# or ## at line start)
_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)


def extract_pdf(pdf_data: bytes) -> dict:
    """Extract structured text from PDF bytes using pymupdf4llm.

    Args:
        pdf_data: Raw PDF file bytes.

    Returns:
        dict with keys: full_text, abstract, sections, figures.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_data)
        tmp.flush()
        full_text: str = pymupdf4llm.to_markdown(tmp.name)

    sections = _parse_sections(full_text)
    abstract = _extract_abstract(full_text, sections)
    figures = _extract_figures(full_text)

    return {
        "full_text": full_text,
        "abstract": abstract,
        "sections": sections,
        "figures": figures,
    }


def _parse_sections(text: str) -> list[dict]:
    """Split markdown text into heading/text pairs."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        # No headings found — treat entire text as a single unnamed section
        return [{"heading": "", "text": text.strip()}]

    sections: list[dict] = []

    # Text before first heading (preamble)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({"heading": "", "text": preamble})

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[content_start:content_end].strip()
        sections.append({"heading": heading, "text": section_text})

    return sections


def _extract_abstract(text: str, sections: list[dict]) -> str:
    """Return abstract text: explicit Abstract section, or first paragraph."""
    for section in sections:
        if section["heading"].lower() == "abstract":
            return section["text"]

    # Fall back to first non-empty paragraph in the full text
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        # Skip lines that look like headings
        if not para.startswith("#"):
            return para

    return ""


def _extract_figures(text: str) -> list[dict]:
    """Extract figure captions from markdown image references."""
    figures: list[dict] = []
    for match in _IMAGE_RE.finditer(text):
        caption = match.group(1).strip()
        figures.append({"caption": caption})
    return figures


def _extract_worker(
    args: tuple[str, str, str],
) -> tuple[str, dict | None, str | None]:
    """Worker function for process pool: extract a single PDF from disk.

    Reads the PDF file directly rather than receiving bytes, so no large buffers
    are serialised across the process boundary.

    Args:
        args: (paper_id, pdf_path, title)

    Returns:
        (paper_id, result_dict, error_message) — result_dict is None on failure.
    """
    paper_id, pdf_path, title = args
    try:
        with open(pdf_path, "rb") as fh:
            pdf_data = fh.read()
        result = extract_pdf(pdf_data)
        result["paper_id"] = paper_id
        result["title"] = title
        return paper_id, result, None
    except Exception as exc:  # noqa: BLE001
        return paper_id, None, f"{type(exc).__name__}: {exc}"


def extract_papers_batch(
    papers: list[dict],
    config: ExtractConfig,
    db: Database,
    storage: StorageBackend,
    keep_pdfs: bool = False,
) -> list[dict]:
    """Extract text from a batch of papers in parallel using multiple processes.

    PDF extraction via pymupdf4llm is CPU-bound and holds the Python GIL, so
    ProcessPoolExecutor is used to achieve real parallelism.  Storage I/O and
    DB updates remain in the calling thread.

    Args:
        papers: List of paper dicts (must include 'id' and 'title').
        config: Extraction configuration (batch_size, max_workers).
        db: Database instance for stage status updates.
        storage: Storage backend for reading PDFs and writing JSON.
        keep_pdfs: When False, delete the source PDF after successful extraction.

    Returns:
        List of result dicts for successfully processed papers.
    """
    results: list[dict] = []
    batch_size = config.batch_size
    max_workers = config.max_workers

    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start : batch_start + batch_size]
        logger.info(
            "Processing extraction batch %d-%d of %d",
            batch_start + 1,
            batch_start + len(batch),
            len(papers),
        )

        # Resolve PDF paths and mark papers as running before entering the process pool.
        # Workers read files directly — no bytes are serialised across the process boundary.
        work_items: list[tuple[str, str, str]] = []
        for paper in batch:
            paper_id: str = paper["id"]
            pdf_key = f"pdfs/{paper_id}.pdf"
            pdf_path = storage.get_local_path(pdf_key)
            if pdf_path is None or not pdf_path.exists():
                error_msg = f"PDF not found in local storage: {pdf_key}"
                logger.error("Failed to locate PDF for %s: %s", paper_id, error_msg)
                db.update_stage_status(paper_id, "extract", "failed", error=error_msg)
                continue
            db.update_stage_status(paper_id, "extract", "running")
            work_items.append((paper_id, str(pdf_path), paper.get("title", "")))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(_extract_worker, item): item[0] for item in work_items
            }
            for future in as_completed(future_to_id):
                paper_id = future_to_id[future]
                pdf_key = f"pdfs/{paper_id}.pdf"
                try:
                    pid, result, error_msg = future.result()
                except Exception as exc:  # noqa: BLE001
                    result, error_msg = None, f"{type(exc).__name__}: {exc}"

                if result is not None:
                    extract_key = f"extract/{paper_id}.json"
                    storage.put(extract_key, json.dumps(result).encode())
                    if not keep_pdfs:
                        try:
                            storage.delete(pdf_key)
                        except FileNotFoundError:
                            logger.warning("PDF already absent: %s", pdf_key)
                    db.update_stage_status(paper_id, "extract", "done")
                    logger.info("Extracted paper %s", paper_id)
                    results.append(result)
                else:
                    logger.error("Failed to extract paper %s: %s", paper_id, error_msg)
                    db.update_stage_status(paper_id, "extract", "failed", error=error_msg)

    logger.info(
        "Extraction complete: %d/%d papers succeeded",
        len(results),
        len(papers),
    )
    return results
