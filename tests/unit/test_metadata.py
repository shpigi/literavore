"""Tests for literavore.normalize.metadata."""

from literavore.normalize.metadata import (
    clean_author_name,
    extract_value,
    normalize_paper_metadata,
    simplify_paper_data,
)


class TestExtractValue:
    def test_plain_value(self):
        assert extract_value("hello") == "hello"

    def test_dict_with_value_key(self):
        assert extract_value({"value": "world"}) == "world"

    def test_nested_dict_without_value(self):
        d = {"foo": "bar"}
        assert extract_value(d) == d

    def test_none(self):
        assert extract_value(None) is None

    def test_list(self):
        assert extract_value([1, 2]) == [1, 2]


class TestCleanAuthorName:
    def test_strips_whitespace(self):
        assert clean_author_name("  Alice  ") == "Alice"

    def test_last_first_format(self):
        assert clean_author_name("Smith, John") == "John Smith"

    def test_already_normal(self):
        assert clean_author_name("John Smith") == "John Smith"

    def test_unicode_normalization(self):
        # NFC normalization
        result = clean_author_name("Müller")
        assert result == "Müller"


class TestNormalizePaperMetadata:
    def test_v2_structure(self):
        raw = {
            "id": "paper1",
            "content": {
                "title": {"value": "Deep Learning"},
                "authors": {"value": ["Alice", "Bob"]},
                "abstract": {"value": "An abstract."},
                "keywords": {"value": ["ml", "dl"]},
                "venue": {"value": "NeurIPS 2024"},
                "pdf": {"value": "/pdf/abc123"},
            },
            "forum": "forum1",
            "number": 42,
        }
        result = normalize_paper_metadata(raw)
        assert result["id"] == "paper1"
        assert result["title"] == "Deep Learning"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["abstract"] == "An abstract."
        assert result["keywords"] == ["ml", "dl"]
        assert result["pdf_url"] == "https://openreview.net/pdf/abc123"

    def test_v1_structure(self):
        raw = {
            "id": "paper2",
            "content": {
                "title": "Plain Title",
                "authors": ["Carol"],
                "abstract": "Plain abstract.",
                "keywords": ["test"],
                "venue": "ICML",
                "pdf": "https://example.com/paper.pdf",
            },
        }
        result = normalize_paper_metadata(raw)
        assert result["title"] == "Plain Title"
        assert result["authors"] == ["Carol"]
        assert result["pdf_url"] == "https://example.com/paper.pdf"

    def test_relative_pdf_url(self):
        raw = {"id": "p3", "content": {"pdf": {"value": "/pdf/xyz"}}}
        result = normalize_paper_metadata(raw)
        assert result["pdf_url"] == "https://openreview.net/pdf/xyz"

    def test_missing_fields(self):
        raw = {"id": "p4"}
        result = normalize_paper_metadata(raw)
        assert result["id"] == "p4"
        assert result["title"] == ""
        assert result["authors"] == []


class TestSimplifyPaperData:
    def test_empty_list(self):
        assert simplify_paper_data([]) == []

    def test_batch_processing(self):
        papers = [
            {"id": "p1", "content": {"title": "T1"}},
            {"id": "p2", "content": {"title": "T2"}},
        ]
        result = simplify_paper_data(papers)
        assert len(result) == 2
        assert result[0]["id"] == "p1"
        assert result[1]["id"] == "p2"
