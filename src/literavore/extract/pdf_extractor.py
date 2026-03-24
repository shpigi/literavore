"""PDF text extraction using pypdf with pdfplumber fallback."""

from __future__ import annotations

import json
import logging
import multiprocessing
import re
import tempfile
from typing import TYPE_CHECKING

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

from literavore.config import ExtractConfig
from literavore.utils import get_logger

if TYPE_CHECKING:
    from literavore.db import Database
    from literavore.storage.base import StorageBackend

logger = get_logger(__name__, stage="extract")

# Regex to split on markdown-style headings (# or ## at line start)
_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)

# Minimum text length to accept from the fast path (pypdf)
_MIN_TEXT_LENGTH = 500


def extract_pdf(pdf_data: bytes) -> dict:
    """Extract text from PDF bytes using pypdf, falling back to pdfplumber.

    Args:
        pdf_data: Raw PDF file bytes.

    Returns:
        dict with keys: full_text, abstract, sections, figures.
    """
    text = _extract_with_pypdf(pdf_data)
    if len(text.strip()) < _MIN_TEXT_LENGTH:
        text = _extract_with_pdfplumber(pdf_data)

    sections = _parse_sections(text)
    abstract = _extract_abstract(text, sections)

    return {
        "full_text": text,
        "abstract": abstract,
        "sections": sections,
        "figures": [],
    }


def _extract_with_pypdf(pdf_data: bytes) -> str:
    """Fast text extraction using pypdf."""
    import io  # noqa: PLC0415

    import pypdf  # noqa: PLC0415

    reader = pypdf.PdfReader(io.BytesIO(pdf_data))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # pypdf sometimes produces lone surrogates from exotic PDF font encodings;
            # strip them so downstream UTF-8 encoding (JSON, API calls) doesn't crash.
            page_text = page_text.encode("utf-8", errors="ignore").decode("utf-8")
            pages.append(page_text)
    return "\n".join(pages)


def _extract_with_pdfplumber(pdf_data: bytes) -> str:
    """Fallback text extraction using pdfplumber (more robust for complex layouts)."""
    import io  # noqa: PLC0415

    import pdfplumber  # noqa: PLC0415

    pages = []
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def _parse_sections(text: str) -> list[dict]:
    """Split text into heading/text pairs based on common academic paper structure."""
    # Try markdown-style headings first
    matches = list(_HEADING_RE.finditer(text))
    if matches:
        return _split_by_matches(text, matches)

    # Try uppercase section headers (common in extracted PDF text)
    uppercase_re = re.compile(
        r"^((?:Abstract|Introduction|Related Work|Method(?:s|ology)?|"
        r"Experiment(?:s|al)?(?:\s+(?:Setup|Results))?|Results|"
        r"Discussion|Conclusion(?:s)?|References|Appendix|"
        r"Background|Approach|Evaluation|Analysis|"
        r"Acknowledgement(?:s)?|Supplementary)(?:\s+Material)?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(uppercase_re.finditer(text))
    if matches:
        return _split_by_section_headers(text, matches)

    # No headings found — treat entire text as a single unnamed section
    return [{"heading": "", "text": text.strip()}]


def _split_by_matches(text: str, matches: list[re.Match]) -> list[dict]:
    """Split text by regex matches into heading/text pairs."""
    sections: list[dict] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({"heading": "", "text": preamble})

    for i, match in enumerate(matches):
        heading = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else match.group(1).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[content_start:content_end].strip()
        sections.append({"heading": heading, "text": section_text})

    return sections


def _split_by_section_headers(text: str, matches: list[re.Match]) -> list[dict]:
    """Split text by section header matches."""
    sections: list[dict] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({"heading": "", "text": preamble})

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
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
        # Skip lines that look like headings or titles
        if not para.startswith("#") and len(para) > 100:
            return para

    return ""


def _extract_worker_target(
    paper_id: str, pdf_path: str, title: str, result_queue: multiprocessing.Queue,
) -> None:
    """Worker target for extraction. Runs in a child process that can be killed."""
    try:
        with open(pdf_path, "rb") as fh:
            pdf_data = fh.read()
        result = extract_pdf(pdf_data)
        result["paper_id"] = paper_id
        result["title"] = title
        result_queue.put((paper_id, result, None))
    except Exception as exc:  # noqa: BLE001
        result_queue.put((paper_id, None, f"{type(exc).__name__}: {exc}"))


def extract_papers_batch(
    papers: list[dict],
    config: ExtractConfig,
    db: Database,
    storage: StorageBackend,
    keep_pdfs: bool = False,
) -> list[dict]:
    """Extract text from a batch of papers using parallel processes with hard timeouts.

    Each paper is extracted in a separate process that is killed if it exceeds
    the configured timeout.

    Papers that time out are marked as failed (re-downloading won't help since
    the PDF structure itself causes the hang).

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
    timeout = config.timeout_per_paper

    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start : batch_start + batch_size]
        logger.info(
            "Processing extraction batch %d-%d of %d",
            batch_start + 1,
            batch_start + len(batch),
            len(papers),
        )

        _process_extract_batch(
            batch, max_workers, timeout,
            db, storage, keep_pdfs, results,
        )

    logger.info(
        "Extraction complete: %d/%d papers succeeded",
        len(results),
        len(papers),
    )
    return results


def _process_extract_batch(
    batch: list[dict],
    max_workers: int,
    timeout: int,
    db: Database,
    storage: StorageBackend,
    keep_pdfs: bool,
    results: list[dict],
) -> None:
    """Process a single extraction batch with per-paper timeouts."""
    import time  # noqa: PLC0415

    # Resolve PDF paths and build work items
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

    if not work_items:
        return

    # Process papers with bounded concurrency using individual processes.
    # Each process can be killed independently on timeout.
    active: dict[str, tuple[multiprocessing.Process, multiprocessing.Queue, float]] = {}
    pending = list(work_items)
    # Collected results keyed by paper_id, drained eagerly to avoid pipe-buffer deadlock.
    # Child processes put ~100-200KB pickled results into the queue; Linux pipe buffers
    # are only 64KB, so the child's feeder thread blocks until the parent reads.
    # If the parent only reads after is_alive()==False, neither side ever unblocks.
    collected: dict[str, tuple] = {}

    while pending or active:
        # Fill worker slots
        while pending and len(active) < max_workers:
            paper_id, pdf_path, title = pending.pop(0)
            result_queue: multiprocessing.Queue = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_extract_worker_target,
                args=(paper_id, pdf_path, title, result_queue),
            )
            proc.start()
            active[paper_id] = (proc, result_queue, time.monotonic())

        # Drain all queues every iteration so child feeder threads can flush and exit.
        for paper_id, (_, rq, _st) in list(active.items()):
            if paper_id not in collected:
                try:
                    collected[paper_id] = rq.get_nowait()
                except Exception:  # noqa: BLE001
                    pass

        # Check for completed or timed-out workers
        now = time.monotonic()
        completed: list[str] = []
        for paper_id, (proc, rq, start_time) in list(active.items()):
            if not proc.is_alive():
                item = collected.pop(paper_id, None)
                if item is not None:
                    _, result, error_msg = item
                else:
                    result, error_msg = None, f"Worker exited with code {proc.exitcode}"
                proc.join(timeout=1)
                _handle_result(
                    paper_id, result, error_msg, db, storage, keep_pdfs, results,
                )
                completed.append(paper_id)
            elif timeout > 0 and (now - start_time) > timeout:
                # Kill the hung process
                elapsed = now - start_time
                logger.warning(
                    "Killing hung extraction worker for %s (%.0fs)", paper_id, elapsed,
                )
                proc.kill()
                proc.join(timeout=5)
                collected.pop(paper_id, None)
                _handle_result(
                    paper_id, None, "Extraction timed out", db, storage, keep_pdfs, results,
                )
                completed.append(paper_id)

        for pid in completed:
            del active[pid]

        if not completed and active:
            time.sleep(0.5)


def _handle_result(
    paper_id: str,
    result: dict | None,
    error_msg: str | None,
    db: Database,
    storage: StorageBackend,
    keep_pdfs: bool,
    results: list[dict],
) -> None:
    """Handle extraction result for a single paper."""
    pdf_key = f"pdfs/{paper_id}.pdf"
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
