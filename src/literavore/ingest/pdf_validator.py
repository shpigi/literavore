"""PDF validation utilities for literavore ingest stage."""

import io
from pathlib import Path

import pikepdf

from literavore.utils.logging import get_logger

logger = get_logger(__name__)


def validate_pdf(data: bytes) -> tuple[bool, str | None]:
    """Validate PDF bytes for structural integrity.

    Checks performed:
    - Minimum size (> 1024 bytes)
    - PDF magic bytes header (%PDF)
    - Valid pikepdf-parseable structure
    - At least one page
    - Not encrypted

    Args:
        data: Raw PDF bytes.

    Returns:
        (True, None) if all checks pass, (False, reason) on failure.
    """
    if len(data) <= 1024:
        msg = f"PDF too small: {len(data)} bytes (minimum 1024)"
        logger.debug(msg)
        return False, msg

    if not data.startswith(b"%PDF"):
        msg = "Missing PDF magic bytes (%PDF header not found)"
        logger.debug(msg)
        return False, msg

    try:
        with pikepdf.open(io.BytesIO(data)) as pdf:
            num_pages = len(pdf.pages)
    except pikepdf.PasswordError:
        msg = "PDF is encrypted (password required)"
        logger.debug(msg)
        return False, msg
    except Exception as exc:
        msg = f"PDF structure invalid: {exc}"
        logger.debug(msg)
        return False, msg

    if num_pages == 0:
        msg = "PDF has no pages"
        logger.debug(msg)
        return False, msg

    return True, None


def validate_pdf_file(path: Path) -> tuple[bool, str | None]:
    """Validate a PDF file on disk.

    Reads the file contents and delegates to validate_pdf().

    Args:
        path: Path to the PDF file.

    Returns:
        (True, None) if valid, (False, reason) on failure.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        msg = f"Could not read file {path}: {exc}"
        logger.debug(msg)
        return False, msg

    return validate_pdf(data)
