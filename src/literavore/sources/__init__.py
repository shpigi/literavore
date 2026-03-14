"""Paper source abstractions and implementations."""

from literavore.sources.base import PaperMetadata, PaperSource
from literavore.sources.openreview import OpenReviewSource

__all__ = ["PaperMetadata", "PaperSource", "OpenReviewSource"]
