"""Tests for literavore.normalize.metadata."""

from __future__ import annotations

import pytest

from literavore.normalize.metadata import (
    clean_author_name,
    extract_value,
    normalize_paper_metadata,
    simplify_paper_data,
)


class TestExtractValue:
    def test_plain_string(self):
        assert extract_value("hello") == "hello"

    def test_plain_int(self):
        assert extract_value(42) == 42

    def test_plain_list(self):
        assert extract_value(["a", "b"]) == ["a", "b"]

    def test_dict_with_value_key(self):
        assert extract_value({"value": "unwrapped"}) == "unwrapped"

    def test_dict_with_value_key_int(self):
        assert extract_value({"value": 99}) == 99

    def test_dict_with_value_key_list(self):
        assert extract_value({"value": ["x", "y"]}) == ["x", "y"]

    def test_dict_without_value_key_returned_as_is(self):
        d = {"other": "field"}
        assert extract_value(d) is d

    def test_nested_dict_does_not_double_unwrap(self):
        # Only one level is unwrapped
        inner = {"value": "deep"}
        result = extract_value({"value": inner})
        assert result == inner

    def test_none_returned_as_is(self):
        assert extract_value(None) is None

    def test_empty_dict_returned_as_is(self):
        assert extract_value({}) == {}


class TestCleanAuthorName:
    def test_strips_whitespace(self):
        assert clean_author_name("  Alice Bob  ") == "Alice Bob"

    def test_no_change_for_plain_name(self):
        assert clean_author_name("Alice Bob") == "Alice Bob"

    def test_last_first_format(self):
        assert clean_author_name("Smith, John") == "John Smith"

    def test_last_first_with_extra_spaces(self):
        assert clean_author_name("  Smith ,  John  ") == "John Smith"

    def test_unicode_nfc_normalization(self):
        # Compose NFC: e + combining acute -> é
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        assert clean_author_name(decomposed) == composed

    def test_unicode_name_no_comma(self):
        assert clean_author_name("José García") == "José García"

    def test_comma_only_one_part_no_swap(self):
        # If splitting on comma yields an empty part, do not swap
        result = clean_author_name("Smith,")
        # parts[1] would be empty, so no swap
        assert result == "Smith,"

    def test_comma_with_empty_first_no_swap(self):
        result = clean_author_name(",Jones")
        assert result == ",Jones"


class TestNormalizePaperMetadata:
    # ------------------------------------------------------------------
    # V1 structure: content is a flat dict of plain values
    # ------------------------------------------------------------------

    def _v1_raw(self, **overrides):
        base = {
            "id": "v1-paper-1",
            "content": {
                "title": "A V1 Paper",
                "authors": ["Smith, John", "Doe, Jane"],
                "abstract": "This is the abstract.",
                "keywords": ["deep learning", "NLP"],
                "venue": "NeurIPS 2020",
                "pdf": "/pdf/v1abc.pdf",
            },
            "cdate": 1609459200,
            "mdate": 1612137600,
            "forum": "v1forum",
            "number": 7,
        }
        base.update(overrides)
        return base

    def test_v1_id(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["id"] == "v1-paper-1"

    def test_v1_title(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["title"] == "A V1 Paper"

    def test_v1_authors_inverted(self):
        result = normalize_paper_metadata(self._v1_raw())
        # "Last, First" format is converted to "First Last"
        assert result["authors"] == ["John Smith", "Jane Doe"]

    def test_v1_abstract(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["abstract"] == "This is the abstract."

    def test_v1_keywords(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["keywords"] == ["deep learning", "NLP"]

    def test_v1_venue(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["venue"] == "NeurIPS 2020"

    def test_v1_relative_pdf_url_resolved(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["pdf_url"] == "https://openreview.net/pdf/v1abc.pdf"

    def test_v1_dates(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["created_date"] == 1609459200
        assert result["modified_date"] == 1612137600

    def test_v1_forum_and_number(self):
        result = normalize_paper_metadata(self._v1_raw())
        assert result["forum"] == "v1forum"
        assert result["number"] == 7

    # ------------------------------------------------------------------
    # V2 structure: content fields are {"value": ...} dicts
    # ------------------------------------------------------------------

    def _v2_raw(self, **overrides):
        base = {
            "id": "v2-paper-1",
            "content": {
                "title": {"value": "A V2 Paper"},
                "authors": {"value": ["Alice", "Bob"]},
                "abstract": {"value": "V2 abstract text."},
                "keywords": {"value": ["machine learning", "CV"]},
                "venue": {"value": "ICML 2023"},
                "pdf": {"value": "/pdf/v2xyz.pdf"},
            },
            "cdate": 1700000000,
            "mdate": 1700100000,
            "forum": "v2forum",
            "number": 42,
        }
        base.update(overrides)
        return base

    def test_v2_title_unwrapped(self):
        result = normalize_paper_metadata(self._v2_raw())
        assert result["title"] == "A V2 Paper"

    def test_v2_authors_unwrapped(self):
        result = normalize_paper_metadata(self._v2_raw())
        assert result["authors"] == ["Alice", "Bob"]

    def test_v2_abstract_unwrapped(self):
        result = normalize_paper_metadata(self._v2_raw())
        assert result["abstract"] == "V2 abstract text."

    def test_v2_keywords_unwrapped(self):
        result = normalize_paper_metadata(self._v2_raw())
        assert result["keywords"] == ["machine learning", "CV"]

    def test_v2_venue_unwrapped(self):
        result = normalize_paper_metadata(self._v2_raw())
        assert result["venue"] == "ICML 2023"

    def test_v2_relative_pdf_url_resolved(self):
        result = normalize_paper_metadata(self._v2_raw())
        assert result["pdf_url"] == "https://openreview.net/pdf/v2xyz.pdf"

    def test_absolute_pdf_url_kept_as_is(self):
        raw = self._v2_raw()
        raw["content"]["pdf"] = {"value": "https://example.com/paper.pdf"}
        result = normalize_paper_metadata(raw)
        assert result["pdf_url"] == "https://example.com/paper.pdf"

    def test_empty_pdf_url(self):
        raw = self._v2_raw()
        raw["content"]["pdf"] = {"value": ""}
        result = normalize_paper_metadata(raw)
        assert result["pdf_url"] == ""

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_missing_fields_use_defaults(self):
        result = normalize_paper_metadata({"id": "empty"})
        assert result["title"] == ""
        assert result["authors"] == []
        assert result["abstract"] == ""
        assert result["keywords"] == []
        assert result["venue"] == ""
        assert result["pdf_url"] == ""

    def test_result_has_all_expected_keys(self):
        result = normalize_paper_metadata({"id": "check-keys"})
        expected_keys = {
            "id",
            "title",
            "authors",
            "abstract",
            "keywords",
            "venue",
            "pdf_url",
            "created_date",
            "modified_date",
            "forum",
            "number",
        }
        assert set(result.keys()) == expected_keys


class TestSimplifyPaperData:
    def test_empty_list_returns_empty(self):
        result = simplify_paper_data([])
        assert result == []

    def test_batch_normalizes_all_papers(self):
        papers = [
            {"id": "p1", "content": {"title": "Paper 1", "authors": ["Alice"]}},
            {"id": "p2", "content": {"title": "Paper 2", "authors": ["Bob"]}},
        ]
        result = simplify_paper_data(papers)
        assert len(result) == 2
        assert result[0]["id"] == "p1"
        assert result[1]["id"] == "p2"

    def test_skips_bad_papers_without_raising(self, monkeypatch):
        """A paper that raises during normalization should be skipped."""
        import literavore.normalize.metadata as mod

        call_count = 0

        def flaky_normalize(raw):
            nonlocal call_count
            call_count += 1
            if raw.get("id") == "bad":
                raise ValueError("intentional failure")
            return {"id": raw["id"], "title": "", "authors": [], "abstract": "",
                    "keywords": [], "venue": "", "pdf_url": "", "created_date": None,
                    "modified_date": None, "forum": "", "number": None}

        monkeypatch.setattr(mod, "normalize_paper_metadata", flaky_normalize)
        papers = [{"id": "good"}, {"id": "bad"}, {"id": "also-good"}]
        result = simplify_paper_data(papers)
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "good" in ids
        assert "also-good" in ids
        assert "bad" not in ids

    def test_returns_list_of_dicts(self):
        papers = [{"id": "x", "content": {"title": "X"}}]
        result = simplify_paper_data(papers)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_api_version_hint_accepted(self):
        papers = [{"id": "p1"}]
        result = simplify_paper_data(papers, api_version="v1")
        assert len(result) == 1
