"""Shared test fixtures for literavore tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_config_dict() -> dict:
    """Provide a minimal config dictionary for testing."""
    return {
        "conferences": [
            {
                "name": "Test-2025",
                "year": 2025,
                "max_papers": 10,
                "openreview_url": "https://openreview.net/group?id=Test/2025",
                "filter_for": [],
            }
        ],
        "pdf": {
            "max_concurrent": 2,
            "delay_between_requests": 0.1,
            "max_retries": 1,
            "timeout": 5,
            "chunk_size": 8192,
            "user_agent": "Test/1.0",
            "validate_pdfs": True,
            "keep_pdfs": True,
            "base_backoff": 1.0,
            "rate_limit_backoff": 1.0,
            "max_backoff": 5.0,
            "backoff_jitter": False,
        },
        "extract": {
            "batch_size": 5,
            "max_workers": 2,
        },
        "summary": {
            "model": "gpt-4o-mini",
            "max_tokens": 100,
            "temperature": 0.3,
            "max_concurrent": 2,
            "batch_size": 5,
            "rate_limit_rpm": 10,
            "enable_tag_extraction": True,
            "max_tag_tokens": 100,
            "cache_enabled": True,
            "pricing": {
                "input_per_1m_tokens": 0.15,
                "output_per_1m_tokens": 0.60,
            },
        },
        "embedding": {
            "model": "text-embedding-3-large",
            "dimensions": 3072,
            "batch_size": 10,
            "views": ["title_abstract"],
        },
        "serve": {
            "api_host": "127.0.0.1",
            "api_port": 8000,
            "streamlit_port": 8501,
            "default_top_k": 5,
        },
        "storage": {
            "backend": "local",
            "data_dir": "",  # Will be replaced by tmp_data_dir fixture
        },
        "processing": {
            "min_abstract_length": 50,
        },
    }
