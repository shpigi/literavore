"""Tests for literavore.sources.openreview."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from literavore.sources.base import PaperMetadata
from literavore.sources.openreview import _extract_value, _parse_group_id


# ---------------------------------------------------------------------------
# _parse_group_id
# ---------------------------------------------------------------------------


class TestParseGroupId:
    def test_standard_neurips_url(self):
        url = "https://openreview.net/group?id=NeurIPS.cc/2024/Conference"
        assert _parse_group_id(url) == "NeurIPS.cc/2024/Conference"

    def test_iclr_url(self):
        url = "https://openreview.net/group?id=ICLR.cc/2025/Conference"
        assert _parse_group_id(url) == "ICLR.cc/2025/Conference"

    def test_icml_url(self):
        url = "https://openreview.net/group?id=ICML.cc/2024/Conference"
        assert _parse_group_id(url) == "ICML.cc/2024/Conference"

    def test_url_with_extra_query_params(self):
        url = "https://openreview.net/group?id=NeurIPS.cc/2024/Conference&mode=edit"
        assert _parse_group_id(url) == "NeurIPS.cc/2024/Conference"

    def test_url_without_id_raises(self):
        with pytest.raises(ValueError, match="Could not extract group id"):
            _parse_group_id("https://openreview.net/group?other=value")

    def test_url_with_no_query_raises(self):
        with pytest.raises(ValueError, match="Could not extract group id"):
            _parse_group_id("https://openreview.net/group")

    def test_plain_id_string_raises(self):
        with pytest.raises(ValueError, match="Could not extract group id"):
            _parse_group_id("NeurIPS.cc/2024/Conference")


# ---------------------------------------------------------------------------
# _extract_value
# ---------------------------------------------------------------------------


class TestExtractValueOpenreview:
    def test_plain_string_unchanged(self):
        assert _extract_value("hello") == "hello"

    def test_plain_int_unchanged(self):
        assert _extract_value(7) == 7

    def test_none_unchanged(self):
        assert _extract_value(None) is None

    def test_list_unchanged(self):
        lst = ["a", "b"]
        assert _extract_value(lst) is lst

    def test_dict_with_value_key_unwrapped(self):
        assert _extract_value({"value": "actual"}) == "actual"

    def test_dict_with_value_key_none(self):
        assert _extract_value({"value": None}) is None

    def test_dict_without_value_key_unchanged(self):
        d = {"other": "field"}
        assert _extract_value(d) is d

    def test_empty_dict_unchanged(self):
        assert _extract_value({}) == {}


# ---------------------------------------------------------------------------
# PaperMetadata model validation
# ---------------------------------------------------------------------------


class TestPaperMetadataModel:
    def test_minimal_valid(self):
        pm = PaperMetadata(id="abc", title="Title", authors=["Alice"], abstract="Abstract.")
        assert pm.id == "abc"
        assert pm.title == "Title"
        assert pm.authors == ["Alice"]
        assert pm.abstract == "Abstract."

    def test_defaults(self):
        pm = PaperMetadata(id="x", title="T", authors=[], abstract="A")
        assert pm.keywords == []
        assert pm.venue == ""
        assert pm.pdf_url == ""
        assert pm.source_url == ""
        assert pm.raw_data == {}

    def test_full_fields(self):
        pm = PaperMetadata(
            id="full",
            title="Full Paper",
            authors=["Alice", "Bob"],
            abstract="Some abstract.",
            keywords=["ml", "nlp"],
            venue="NeurIPS 2024",
            pdf_url="https://openreview.net/pdf/abc.pdf",
            source_url="https://openreview.net/forum?id=abc",
            raw_data={"extra": "data"},
        )
        assert pm.keywords == ["ml", "nlp"]
        assert pm.venue == "NeurIPS 2024"
        assert pm.pdf_url == "https://openreview.net/pdf/abc.pdf"
        assert pm.raw_data == {"extra": "data"}

    def test_multiple_authors(self):
        authors = ["Alice", "Bob", "Carol"]
        pm = PaperMetadata(id="m", title="T", authors=authors, abstract="A")
        assert pm.authors == authors

    def test_model_is_pydantic(self):
        from pydantic import BaseModel

        assert issubclass(PaperMetadata, BaseModel)


# ---------------------------------------------------------------------------
# OpenReviewSource instantiation (mocking openreview package)
# ---------------------------------------------------------------------------


class TestOpenReviewSourceInstantiation:
    def test_raises_import_error_when_package_unavailable(self):
        """When openreview package is not importable, instantiation raises ImportError."""
        import literavore.sources.openreview as mod

        with patch.object(mod, "_OPENREVIEW_AVAILABLE", False):
            from literavore.sources.openreview import OpenReviewSource

            with pytest.raises(ImportError, match="openreview"):
                OpenReviewSource()

    def test_instantiates_when_package_available(self):
        """When openreview is available, OpenReviewSource should create a client."""
        # Build a minimal fake openreview module
        fake_or = types.ModuleType("openreview")
        fake_api = types.ModuleType("openreview.api")
        mock_client_cls = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_cls.return_value = mock_client_instance
        fake_api.OpenReviewClient = mock_client_cls
        fake_or.api = fake_api

        import literavore.sources.openreview as mod

        with (
            patch.object(mod, "_OPENREVIEW_AVAILABLE", True),
            patch.object(mod, "openreview", fake_or, create=True),
        ):
            from literavore.sources.openreview import OpenReviewSource

            src = OpenReviewSource.__new__(OpenReviewSource)
            # Manually call __init__ with the patched module in place
            OpenReviewSource.__init__(src)
            mock_client_cls.assert_called_once_with(baseurl="https://api2.openreview.net")

    def test_parse_group_id_used_in_fetch(self):
        """fetch() calls _parse_group_id with the config's openreview_url."""
        import literavore.sources.openreview as mod
        from literavore.config import ConferenceConfig

        fake_or = types.ModuleType("openreview")
        fake_api = types.ModuleType("openreview.api")
        mock_client = MagicMock()
        mock_client.get_all_notes.return_value = []
        fake_api.OpenReviewClient = MagicMock(return_value=mock_client)
        fake_or.api = fake_api

        with (
            patch.object(mod, "_OPENREVIEW_AVAILABLE", True),
            patch.object(mod, "openreview", fake_or, create=True),
        ):
            from literavore.sources.openreview import OpenReviewSource

            src = OpenReviewSource.__new__(OpenReviewSource)
            src._client = mock_client

            conf = ConferenceConfig(
                name="NeurIPS",
                year=2024,
                openreview_url="https://openreview.net/group?id=NeurIPS.cc/2024/Conference",
            )
            papers = src.fetch(conf)
            assert papers == []
            # Verify the client was asked about the group
            mock_client.get_all_notes.assert_called()
