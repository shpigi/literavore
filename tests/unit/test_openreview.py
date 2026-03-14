"""Tests for literavore.sources.openreview and base."""

import pytest

from literavore.sources.base import PaperMetadata, PaperSource


class TestPaperMetadata:
    def test_required_fields(self):
        pm = PaperMetadata(id="p1", title="Test", authors=["A"], abstract="Abs")
        assert pm.id == "p1"
        assert pm.title == "Test"

    def test_defaults(self):
        pm = PaperMetadata(id="p1", title="T", authors=[], abstract="A")
        assert pm.keywords == []
        assert pm.venue == ""
        assert pm.pdf_url == ""
        assert pm.raw_data == {}

    def test_full_fields(self):
        pm = PaperMetadata(
            id="p1",
            title="Title",
            authors=["A", "B"],
            abstract="Abstract",
            keywords=["ml"],
            venue="NeurIPS",
            pdf_url="https://example.com/p.pdf",
            source_url="https://openreview.net/forum?id=p1",
            raw_data={"key": "val"},
        )
        assert pm.venue == "NeurIPS"
        assert pm.raw_data["key"] == "val"


class TestParseGroupId:
    def test_standard_url(self):
        from literavore.sources.openreview import _parse_group_id

        result = _parse_group_id(
            "https://openreview.net/group?id=NeurIPS.cc/2024/Conference"
        )
        assert result == "NeurIPS.cc/2024/Conference"

    def test_url_with_extra_params(self):
        from literavore.sources.openreview import _parse_group_id

        result = _parse_group_id(
            "https://openreview.net/group?id=ICML.cc/2025/Conference&foo=bar"
        )
        assert result == "ICML.cc/2025/Conference"

    def test_missing_id_raises(self):
        from literavore.sources.openreview import _parse_group_id

        with pytest.raises(ValueError, match="Could not extract"):
            _parse_group_id("https://openreview.net/group")


class TestExtractValue:
    def test_plain(self):
        from literavore.sources.openreview import _extract_value

        assert _extract_value("hello") == "hello"

    def test_wrapped(self):
        from literavore.sources.openreview import _extract_value

        assert _extract_value({"value": "world"}) == "world"
