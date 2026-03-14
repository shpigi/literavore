"""Prompt templates for LLM-based paper summarization and tagging."""

from __future__ import annotations

SUMMARIZE_SYSTEM = """\
You are a scientific paper analyst. Given a paper's title, abstract, and extracted text,
produce a concise summary and extract relevant tags.

Respond with valid JSON only, using this exact schema:
{
  "summary": "<2-4 sentence summary of the paper's contribution and methods>",
  "tags": ["<tag1>", "<tag2>", "..."]
}

Tags should be lowercase, hyphenated phrases (e.g., "deep-learning", "natural-language-processing").
Include 3-8 tags covering the paper's domain, methods, and key contributions.
"""

SUMMARIZE_USER_TEMPLATE = """\
Title: {title}

Abstract: {abstract}

Extracted text (truncated):
{text_excerpt}
"""

TAG_SYSTEM = """\
You are a scientific paper classifier. Given a paper's title, abstract, and summary,
extract relevant topic tags.

Respond with valid JSON only, using this exact schema:
{
  "key_phrases": ["<phrase1>", "<phrase2>", "..."],
  "domains": ["<domain1>", "..."],
  "methods": ["<method1>", "..."]
}

All values should be lowercase strings.
"""

TAG_USER_TEMPLATE = """\
Title: {title}

Abstract: {abstract}

Summary: {summary}
"""

# Maximum characters of extracted text to include in summary prompt
MAX_TEXT_EXCERPT_CHARS = 3000
