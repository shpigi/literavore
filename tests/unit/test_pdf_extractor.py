"""Tests for literavore.extract.pdf_extractor."""

import json
from pathlib import Path

import pytest

from literavore.extract.pdf_extractor import (
    _extract_abstract,
    _parse_sections,
    extract_papers_batch,
    extract_pdf,
)


FIXTURE_PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"


class TestExtractPdf:
    def test_extracts_from_fixture(self):
        pdf_data = FIXTURE_PDF.read_bytes()
        result = extract_pdf(pdf_data)
        assert "full_text" in result
        assert "sections" in result
        assert "abstract" in result
        assert "figures" in result
        assert isinstance(result["full_text"], str)
        assert len(result["full_text"]) > 0

    def test_returns_sections_list(self):
        pdf_data = FIXTURE_PDF.read_bytes()
        result = extract_pdf(pdf_data)
        assert isinstance(result["sections"], list)
        for section in result["sections"]:
            assert "heading" in section
            assert "text" in section

    def test_figures_is_list(self):
        pdf_data = FIXTURE_PDF.read_bytes()
        result = extract_pdf(pdf_data)
        assert isinstance(result["figures"], list)


class TestParseSections:
    def test_no_headings(self):
        text = "This is plain text with no headings."
        sections = _parse_sections(text)
        assert len(sections) == 1
        assert sections[0]["heading"] == ""
        assert "plain text" in sections[0]["text"]

    def test_single_heading(self):
        text = "# Introduction\nSome content here."
        sections = _parse_sections(text)
        assert any(s["heading"] == "Introduction" for s in sections)

    def test_multiple_headings(self):
        text = "# Abstract\nThe abstract.\n\n## Methods\nThe methods."
        sections = _parse_sections(text)
        headings = [s["heading"] for s in sections]
        assert "Abstract" in headings
        assert "Methods" in headings

    def test_preamble_before_first_heading(self):
        text = "Preamble text.\n\n# Section One\nContent."
        sections = _parse_sections(text)
        assert sections[0]["heading"] == ""
        assert "Preamble" in sections[0]["text"]

    def test_empty_text(self):
        sections = _parse_sections("")
        assert isinstance(sections, list)


class TestExtractAbstract:
    def test_finds_abstract_section(self):
        sections = [
            {"heading": "Abstract", "text": "This is the abstract."},
            {"heading": "Introduction", "text": "Intro text."},
        ]
        result = _extract_abstract("", sections)
        assert result == "This is the abstract."

    def test_case_insensitive(self):
        sections = [{"heading": "ABSTRACT", "text": "The abstract content."}]
        result = _extract_abstract("", sections)
        assert result == "The abstract content."

    def test_fallback_to_first_paragraph(self):
        long_para = "This is the first paragraph. " * 10  # > 100 chars
        text = f"{long_para}\n\nThis is the second."
        sections = [{"heading": "Introduction", "text": "Intro."}]
        result = _extract_abstract(text, sections)
        assert result == long_para.strip()

    def test_skips_heading_lines_in_fallback(self):
        long_para = "Actual first paragraph content. " * 10  # > 100 chars
        text = f"# Title\n\n{long_para}"
        sections = []
        result = _extract_abstract(text, sections)
        assert result == long_para.strip()


class TestExtractPapersBatch:
    def test_skips_missing_pdf(self, tmp_path: Path):
        from literavore.config import ExtractConfig
        from literavore.db import Database
        from literavore.storage.local import LocalStorage

        db = Database(tmp_path / "test.db")
        storage = LocalStorage(tmp_path / "data")
        config = ExtractConfig(batch_size=5, max_workers=1)

        db.get_or_create_paper("p1", title="Test")
        papers = [{"id": "p1", "title": "Test"}]
        results = extract_papers_batch(papers, config, db, storage)
        assert results == []
        status = db.get_stage_status("p1", "extract")
        assert status["status"] == "failed"

    def test_extracts_real_pdf(self, tmp_path: Path):
        from literavore.config import ExtractConfig
        from literavore.db import Database
        from literavore.storage.local import LocalStorage

        db = Database(tmp_path / "test.db")
        storage = LocalStorage(tmp_path / "data")
        config = ExtractConfig(batch_size=5, max_workers=1)

        pdf_data = FIXTURE_PDF.read_bytes()
        db.get_or_create_paper("p1", title="Sample Paper")
        storage.put("pdfs/p1.pdf", pdf_data)

        papers = [{"id": "p1", "title": "Sample Paper"}]
        results = extract_papers_batch(papers, config, db, storage, keep_pdfs=True)
        assert len(results) == 1
        assert results[0]["paper_id"] == "p1"
        # JSON stored in storage
        assert storage.exists("extract/p1.json")
        status = db.get_stage_status("p1", "extract")
        assert status["status"] == "done"

    def test_deletes_pdf_when_keep_pdfs_false(self, tmp_path: Path):
        from literavore.config import ExtractConfig
        from literavore.db import Database
        from literavore.storage.local import LocalStorage

        db = Database(tmp_path / "test.db")
        storage = LocalStorage(tmp_path / "data")
        config = ExtractConfig(batch_size=5, max_workers=1)

        pdf_data = FIXTURE_PDF.read_bytes()
        db.get_or_create_paper("p1", title="Sample Paper")
        storage.put("pdfs/p1.pdf", pdf_data)

        extract_papers_batch(
            [{"id": "p1", "title": "Sample Paper"}], config, db, storage, keep_pdfs=False
        )
        assert not storage.exists("pdfs/p1.pdf")
