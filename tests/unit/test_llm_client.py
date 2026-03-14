"""Tests for literavore.summarize.llm_client."""

from __future__ import annotations

import pytest

from literavore.config import SummaryConfig
from literavore.summarize.llm_client import _MOCK_RESPONSE, LLMClient


@pytest.fixture
def config() -> SummaryConfig:
    return SummaryConfig(
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.3,
        pricing={"input_per_1m_tokens": 0.15, "output_per_1m_tokens": 0.60},
    )


class TestMockMode:
    def test_mock_mode_true_when_no_api_key(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        assert client.mock_mode is True

    def test_mock_mode_false_when_api_key_set(self, config: SummaryConfig, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        client = LLMClient(config)
        assert client.mock_mode is False

    def test_chat_complete_returns_mock_response(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        result = client.chat_complete([{"role": "user", "content": "hello"}])
        assert result == _MOCK_RESPONSE

    def test_achat_complete_returns_mock_response(self, config: SummaryConfig, monkeypatch):
        import asyncio

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        result = asyncio.run(client.achat_complete([{"role": "user", "content": "hello"}]))
        assert result == _MOCK_RESPONSE


class TestCostTracking:
    def test_track_usage_accumulates_tokens(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        client.track_usage(100, 50)
        client.track_usage(200, 75)
        assert client._total_prompt_tokens == 300
        assert client._total_completion_tokens == 125

    def test_initial_token_counts_are_zero(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        assert client._total_prompt_tokens == 0
        assert client._total_completion_tokens == 0

    def test_get_cost_summary_structure(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        summary = client.get_cost_summary()
        assert "total_prompt_tokens" in summary
        assert "total_completion_tokens" in summary
        assert "total_cost_usd" in summary

    def test_get_cost_summary_values_after_tracking(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        client.track_usage(1_000_000, 0)
        summary = client.get_cost_summary()
        assert summary["total_prompt_tokens"] == 1_000_000
        assert summary["total_completion_tokens"] == 0
        # 1M input tokens at $0.15/1M = $0.15
        assert abs(summary["total_cost_usd"] - 0.15) < 1e-9

    def test_get_cost_summary_output_tokens(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        client.track_usage(0, 1_000_000)
        summary = client.get_cost_summary()
        # 1M output tokens at $0.60/1M = $0.60
        assert abs(summary["total_cost_usd"] - 0.60) < 1e-9

    def test_get_cost_summary_zero_when_no_usage(self, config: SummaryConfig, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(config)
        summary = client.get_cost_summary()
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_prompt_tokens"] == 0
        assert summary["total_completion_tokens"] == 0
