"""OpenAI client wrapper for summarization."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai

from literavore.config import SummaryConfig
from literavore.utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_MOCK_RESPONSE = '{"summary": "Mock summary for testing.", "tags": []}'


class LLMClient:
    """Wrapper around the OpenAI API with cost tracking and mock mode support."""

    def __init__(self, config: SummaryConfig) -> None:
        self._config = config
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self._client = openai.OpenAI(api_key=api_key)
            self._async_client = openai.AsyncOpenAI(api_key=api_key)
        else:
            self._client = None  # type: ignore[assignment]
            self._async_client = None  # type: ignore[assignment]

    @property
    def mock_mode(self) -> bool:
        """True when OPENAI_API_KEY is not set."""
        return not bool(os.environ.get("OPENAI_API_KEY"))

    def chat_complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call chat completions synchronously and return the message content string.

        Falls back to a canned mock response when mock_mode is True.
        """
        if self.mock_mode:
            logger.debug("mock_mode active — returning canned response")
            return _MOCK_RESPONSE

        resolved_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        resolved_temperature = temperature if temperature is not None else self._config.temperature

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
        )
        usage = response.usage
        if usage is not None:
            self.track_usage(usage.prompt_tokens, usage.completion_tokens)

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned an empty message content")
        return content

    async def achat_complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call chat completions asynchronously and return the message content string.

        Falls back to a canned mock response when mock_mode is True.
        """
        if self.mock_mode:
            logger.debug("mock_mode active — returning canned response")
            return _MOCK_RESPONSE

        resolved_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        resolved_temperature = temperature if temperature is not None else self._config.temperature

        response = await self._async_client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
        )
        usage = response.usage
        if usage is not None:
            self.track_usage(usage.prompt_tokens, usage.completion_tokens)

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned an empty message content")
        return content

    def track_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Accumulate token counts and compute running cost."""
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

    def get_cost_summary(self) -> dict:
        """Return accumulated token counts and estimated USD cost."""
        prompt_cost = (
            self._total_prompt_tokens / 1_000_000 * self._config.pricing.input_per_1m_tokens
        )
        completion_cost = (
            self._total_completion_tokens / 1_000_000 * self._config.pricing.output_per_1m_tokens
        )
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_cost_usd": prompt_cost + completion_cost,
        }
