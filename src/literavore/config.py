"""Pydantic v2 configuration models for Literavore."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ConferenceConfig(BaseModel):
    name: str
    year: int
    max_papers: int = 2000
    openreview_url: str = ""
    filter_for: list[str] = Field(default_factory=list)


class PdfConfig(BaseModel):
    max_concurrent: int = 10
    requests_per_second: float = 5.0  # global token-bucket rate limit across all workers; 0 = unlimited
    delay_between_requests: float = 1.5  # fallback per-worker delay when requests_per_second=0
    max_retries: int = 3
    timeout: int = 30
    chunk_size: int = 8192
    user_agent: str = "Literavore/1.0 (PDF Fetcher)"
    validate_pdfs: bool = True
    keep_pdfs: bool = False
    base_backoff: float = 2.0
    rate_limit_backoff: float = 10.0
    max_backoff: float = 60.0
    backoff_jitter: bool = True


class ExtractConfig(BaseModel):
    batch_size: int = 50
    max_workers: int = 4
    timeout_per_paper: int = 90  # seconds; 0 = no timeout
    max_retries: int = 3


class FetchConfig(BaseModel):
    venue_filter_model: str = "gpt-4o-mini"


class SummaryPricingConfig(BaseModel):
    input_per_1m_tokens: float = 0.20
    output_per_1m_tokens: float = 0.80


class SummaryConfig(BaseModel):
    model: str = "gpt-4o-mini"
    tag_model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.3
    max_concurrent: int = 10
    batch_size: int = 20
    rate_limit_rpm: int = 100
    enable_tag_extraction: bool = True
    max_tag_tokens: int = 300
    cache_enabled: bool = True
    max_text_excerpt_chars: int = 3000
    pricing: SummaryPricingConfig = Field(default_factory=SummaryPricingConfig)


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    batch_size: int = 128
    views: list[str] = Field(
        default_factory=lambda: ["title_abstract", "paper_card", "keyword_enriched"]
    )


class ServeConfig(BaseModel):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    default_top_k: int = 10


class StorageConfig(BaseModel):
    backend: Literal["local", "s3"] = "local"
    data_dir: str = "data"


class ProcessingConfig(BaseModel):
    min_abstract_length: int = 50


class PipelineConfig(BaseModel):
    batch_size: int = 0  # 0 = all-at-once (current behavior)


class LiteravoreConfig(BaseModel):
    conferences: list[ConferenceConfig] = Field(default_factory=list)
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    pdf: PdfConfig = Field(default_factory=PdfConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_config(path: Path | None = None) -> LiteravoreConfig:
    """Load configuration from a YAML file with environment variable overrides.

    Resolution order:
    1. Defaults baked into the Pydantic models.
    2. YAML file at *path* (or ``LITERAVORE_CONFIG`` env var, or ``config/default.yml``).
    3. Environment variable overrides applied after loading.
    """
    # Resolve config file path
    if path is None:
        env_path = os.environ.get("LITERAVORE_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            path = Path("config/default.yml")

    raw: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    config = LiteravoreConfig.model_validate(raw)

    # Environment variable overrides
    dev_mode = os.environ.get("LITERAVORE_DEV_MODE", "").strip().lower()
    if dev_mode in {"1", "true", "yes"}:
        config.pdf.keep_pdfs = True

    data_dir = os.environ.get("LITERAVORE_DATA_DIR", "").strip()
    if data_dir:
        config.storage.data_dir = data_dir

    storage_backend = os.environ.get("LITERAVORE_STORAGE_BACKEND", "").strip().lower()
    if storage_backend in {"local", "s3"}:
        config.storage.backend = storage_backend  # type: ignore[assignment]

    return config
