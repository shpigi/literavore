"""Base abstractions for paper sources."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from literavore.config import ConferenceConfig


class PaperMetadata(BaseModel):
    """Metadata for a single academic paper."""

    id: str
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str] = Field(default_factory=list)
    venue: str = ""
    pdf_url: str = ""
    source_url: str = ""
    raw_data: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class PaperSource(Protocol):
    """Protocol for paper source implementations."""

    def fetch(self, config: ConferenceConfig) -> list[PaperMetadata]:
        """Fetch paper metadata for the given conference configuration."""
        ...
