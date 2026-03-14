"""Tests for literavore.summarize.tagger."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from literavore.config import SummaryConfig
from literavore.summarize.llm_client import LLMClient
from literavore.summarize.tagger import Tagger


@pytest.fixture
def config() -> SummaryConfig:
    return SummaryConfig(
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.3,
        enable_tag_extraction=True,
        max_tag_tokens=100,
    )


@pytest.fixture
def config_no_tags() -> SummaryConfig:
    return SummaryConfig(
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.3,
        enable_tag_extraction=False,
        max_tag_tokens=100,
    )


@pytest.fixture
def llm_client(config: SummaryConfig, monkeypatch) -> LLMClient:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return LLMClient(config)


@pytest.fixture
def tagger(config: SummaryConfig, llm_client: LLMClient) -> Tagger:
    return Tagger(config, llm_client)


_MOCK_TAG_JSON = json.dumps(
    {
        "key_phrases": ["neural network", "attention mechanism"],
        "domains": ["machine learning"],
        "methods": ["transformer"],
    }
)


class TestTaggerCreation:
    def test_creates_tagger(self, config: SummaryConfig, llm_client: LLMClient):
        t = Tagger(config, llm_client)
        assert t._config is config
        assert t._llm is llm_client


class TestExtractTagsFromKeywords:
    def test_puts_keywords_in_key_phrases(self, tagger: Tagger):
        result = tagger.extract_tags_from_keywords(["Deep Learning", "NLP"])
        assert result["key_phrases"] == ["deep learning", "nlp"]

    def test_empty_keywords(self, tagger: Tagger):
        result = tagger.extract_tags_from_keywords([])
        assert result["key_phrases"] == []

    def test_domains_and_methods_are_empty(self, tagger: Tagger):
        result = tagger.extract_tags_from_keywords(["some keyword"])
        assert result["domains"] == []
        assert result["methods"] == []

    def test_lowercases_keywords(self, tagger: Tagger):
        result = tagger.extract_tags_from_keywords(["UPPER", "MiXeD"])
        assert result["key_phrases"] == ["upper", "mixed"]


class TestExtractTags:
    def test_extract_tags_with_mock_llm(
        self, config: SummaryConfig, llm_client: LLMClient
    ):
        tagger = Tagger(config, llm_client)
        with patch.object(
            llm_client,
            "achat_complete",
            new_callable=AsyncMock,
            return_value=_MOCK_TAG_JSON,
        ):
            result = asyncio.run(
                tagger.extract_tags("Test Title", "Test abstract.", "Test summary.")
            )

        assert result["key_phrases"] == ["neural network", "attention mechanism"]
        assert result["domains"] == ["machine learning"]
        assert result["methods"] == ["transformer"]

    def test_extract_tags_returns_correct_structure(
        self, config: SummaryConfig, llm_client: LLMClient
    ):
        tagger = Tagger(config, llm_client)
        with patch.object(
            llm_client,
            "achat_complete",
            new_callable=AsyncMock,
            return_value=_MOCK_TAG_JSON,
        ):
            result = asyncio.run(
                tagger.extract_tags("Title", "Abstract.", "Summary.")
            )

        assert "key_phrases" in result
        assert "domains" in result
        assert "methods" in result


class TestFallbackWhenTagExtractionDisabled:
    def test_returns_empty_tags_when_disabled(
        self, config_no_tags: SummaryConfig, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = LLMClient(config_no_tags)
        tagger = Tagger(config_no_tags, llm)

        result = asyncio.run(tagger.extract_tags("Title", "Abstract.", "Summary."))

        assert result == {"key_phrases": [], "domains": [], "methods": []}

    def test_no_llm_call_when_disabled(self, config_no_tags: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = LLMClient(config_no_tags)
        tagger = Tagger(config_no_tags, llm)

        call_count = 0

        async def mock_achat(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return _MOCK_TAG_JSON

        with patch.object(llm, "achat_complete", new_callable=AsyncMock, side_effect=mock_achat):
            asyncio.run(tagger.extract_tags("Title", "Abstract.", "Summary."))

        assert call_count == 0


class TestJsonParseErrorFallback:
    def test_returns_empty_on_invalid_json(
        self, config: SummaryConfig, llm_client: LLMClient
    ):
        tagger = Tagger(config, llm_client)
        with patch.object(
            llm_client,
            "achat_complete",
            new_callable=AsyncMock,
            return_value="this is not json {{{",
        ):
            result = asyncio.run(
                tagger.extract_tags("Title", "Abstract.", "Summary.")
            )

        assert result == {"key_phrases": [], "domains": [], "methods": []}

    def test_returns_empty_on_missing_keys(
        self, config: SummaryConfig, llm_client: LLMClient
    ):
        tagger = Tagger(config, llm_client)
        # Valid JSON but missing expected keys — should still return defaults
        with patch.object(
            llm_client,
            "achat_complete",
            new_callable=AsyncMock,
            return_value=json.dumps({"unexpected_key": "value"}),
        ):
            result = asyncio.run(
                tagger.extract_tags("Title", "Abstract.", "Summary.")
            )

        assert result["key_phrases"] == []
        assert result["domains"] == []
        assert result["methods"] == []

    def test_returns_empty_on_exception(
        self, config: SummaryConfig, llm_client: LLMClient
    ):
        tagger = Tagger(config, llm_client)
        with patch.object(
            llm_client,
            "achat_complete",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network error"),
        ):
            result = asyncio.run(
                tagger.extract_tags("Title", "Abstract.", "Summary.")
            )

        assert result == {"key_phrases": [], "domains": [], "methods": []}
