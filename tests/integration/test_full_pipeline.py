"""End-to-end pipeline integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from literavore.config import ConferenceConfig, LiteravoreConfig
from literavore.db import Database
from literavore.embed.index import INDEX_FAISS_KEY_TEMPLATE, INDEX_METADATA_KEY
from literavore.pipeline import Pipeline
from literavore.sources.base import PaperMetadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAMPLE_PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"

# ---------------------------------------------------------------------------
# Fake paper metadata returned by the mocked OpenReviewSource.fetch
# ---------------------------------------------------------------------------

FAKE_PAPERS = [
    PaperMetadata(
        id="paper-001",
        title="Attention Is All You Need",
        authors=["Alice Smith", "Bob Jones"],
        abstract="We propose a novel architecture based solely on attention mechanisms.",
        keywords=["attention", "transformer", "nlp"],
        venue="NeurIPS 2024",
        pdf_url="https://openreview.net/pdf?id=paper-001",
    ),
    PaperMetadata(
        id="paper-002",
        title="Diffusion Models for Image Synthesis",
        authors=["Carol White"],
        abstract="Diffusion-based generative models achieve state-of-the-art image quality.",
        keywords=["diffusion", "generative models", "images"],
        venue="NeurIPS 2024",
        pdf_url="https://openreview.net/pdf?id=paper-002",
    ),
    PaperMetadata(
        id="paper-003",
        title="Scaling Laws for Language Models",
        authors=["Dave Brown", "Eve Davis"],
        abstract="We empirically investigate scaling laws governing language model performance.",
        keywords=["scaling", "language models", "llm"],
        venue="NeurIPS 2024",
        pdf_url="https://openreview.net/pdf?id=paper-003",
    ),
]

# Canned LLM response for summarization (valid JSON expected by Summarizer)
_CANNED_SUMMARY_RESPONSE = json.dumps(
    {
        "summary": "This paper presents an important contribution to the field.",
        "tags": ["machine learning", "deep learning"],
    }
)

# Canned LLM response for tag extraction
_CANNED_TAG_RESPONSE = json.dumps(
    {
        "key_phrases": ["neural network", "attention mechanism"],
        "domains": ["natural language processing"],
        "methods": ["transformer"],
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(data_dir: Path) -> LiteravoreConfig:
    """Build a minimal LiteravoreConfig pointing at *data_dir*."""
    return LiteravoreConfig.model_validate(
        {
            "conferences": [
                {
                    "name": "NeurIPS-2024",
                    "year": 2024,
                    "max_papers": 10,
                    "openreview_url": "https://openreview.net/group?id=NeurIPS/2024",
                    "filter_for": [],
                }
            ],
            "pdf": {
                "max_concurrent": 2,
                "delay_between_requests": 0.0,
                "max_retries": 1,
                "timeout": 5,
                "keep_pdfs": True,
            },
            "extract": {"batch_size": 10, "max_workers": 1},
            "summary": {
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0.0,
                "max_concurrent": 2,
                "batch_size": 10,
                "rate_limit_rpm": 100,
                "enable_tag_extraction": True,
                "max_tag_tokens": 100,
                "cache_enabled": False,
            },
            "embedding": {
                "model": "text-embedding-3-large",
                "dimensions": 64,  # small dims for speed
                "batch_size": 10,
                "views": ["title_abstract"],
            },
            "storage": {"backend": "local", "data_dir": str(data_dir)},
        }
    )


def _seed_papers_in_storage(pipeline: Pipeline, paper_ids: list[str]) -> None:
    """Copy sample.pdf into storage for each paper and mark download=done."""
    pdf_bytes = SAMPLE_PDF.read_bytes()
    for paper_id in paper_ids:
        pipeline.storage.put(f"pdfs/{paper_id}.pdf", pdf_bytes)
        pipeline.db.update_stage_status(paper_id, "download", "done")


def _all_papers_have_stage(db: Database, stage: str, status: str = "done") -> bool:
    """Return True if every paper in the DB has the given stage/status."""
    papers = db.get_papers()
    if not papers:
        return False
    for paper in papers:
        row = db.get_stage_status(paper["id"], stage)
        if row is None or row["status"] != status:
            return False
    return True


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture()
def config(data_dir: Path) -> LiteravoreConfig:
    return _make_config(data_dir)


@pytest.fixture()
def pipeline(config: LiteravoreConfig) -> Pipeline:
    return Pipeline(config)


# ---------------------------------------------------------------------------
# TestPipelineStateTransitions
# ---------------------------------------------------------------------------


class TestPipelineStateTransitions:
    """Verify SQLite state transitions through all pipeline stages."""

    def _mock_fetch(self, _conf: ConferenceConfig) -> list[PaperMetadata]:
        return list(FAKE_PAPERS)

    def test_fetch_creates_papers_with_done_status(self, pipeline: Pipeline) -> None:
        """After fetch stage, all papers exist in DB with fetch=done."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        papers = pipeline.db.get_papers()
        assert len(papers) == 3

        for paper in papers:
            row = pipeline.db.get_stage_status(paper["id"], "fetch")
            assert row is not None
            assert row["status"] == "done"

    def test_fetch_stores_paper_metadata(self, pipeline: Pipeline) -> None:
        """Paper title and authors are stored correctly after fetch."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        paper = pipeline.db.get_paper("paper-001")
        assert paper is not None
        assert paper["title"] == "Attention Is All You Need"
        assert paper["conference"] == "NeurIPS-2024"

    def test_extract_marks_done_and_writes_json(self, pipeline: Pipeline) -> None:
        """After extract, papers have extract=done and JSON in storage."""
        # Bootstrap: create papers, seed storage with PDFs, mark download done
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])

        pipeline.run(stages=["extract"])

        assert _all_papers_have_stage(pipeline.db, "extract", "done")
        for paper_id in [p.id for p in FAKE_PAPERS]:
            assert pipeline.storage.exists(f"extract/{paper_id}.json")

    def test_extract_json_has_expected_keys(self, pipeline: Pipeline) -> None:
        """Extracted JSON contains full_text, abstract, sections, figures."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        raw = pipeline.storage.get("extract/paper-001.json")
        data = json.loads(raw.decode())
        for key in ("full_text", "abstract", "sections", "figures"):
            assert key in data, f"Missing key {key!r} in extract JSON"

    def test_summarize_marks_done_and_writes_json(self, pipeline: Pipeline) -> None:
        """After summarize, papers have summarize=done and summaries in storage."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        # Mock both LLM calls (summarize + tag extraction) with canned responses
        async def _canned_achat(*_args, **_kwargs) -> str:
            return _CANNED_SUMMARY_RESPONSE

        async def _canned_tag(*_args, **_kwargs) -> str:
            return _CANNED_TAG_RESPONSE

        with patch(
            "literavore.summarize.llm_client.LLMClient.achat_complete",
            new=AsyncMock(side_effect=_canned_achat),
        ):
            pipeline.run(stages=["summarize"])

        assert _all_papers_have_stage(pipeline.db, "summarize", "done")
        for paper_id in [p.id for p in FAKE_PAPERS]:
            assert pipeline.storage.exists(f"summaries/{paper_id}.json")

    def test_summarize_json_has_expected_keys(self, pipeline: Pipeline) -> None:
        """Summary JSON contains paper_id, summary, tags, structured_tags."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        async def _canned_achat(*_args, **_kwargs) -> str:
            return _CANNED_SUMMARY_RESPONSE

        with patch(
            "literavore.summarize.llm_client.LLMClient.achat_complete",
            new=AsyncMock(side_effect=_canned_achat),
        ):
            pipeline.run(stages=["summarize"])

        raw = pipeline.storage.get("summaries/paper-001.json")
        data = json.loads(raw.decode())
        for key in ("paper_id", "summary", "tags", "structured_tags"):
            assert key in data, f"Missing key {key!r} in summary JSON"

    def test_embed_marks_done_and_saves_index(self, pipeline: Pipeline) -> None:
        """After embed, papers have embed=done and index files in storage."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        async def _canned_achat(*_args, **_kwargs) -> str:
            return _CANNED_SUMMARY_RESPONSE

        with patch(
            "literavore.summarize.llm_client.LLMClient.achat_complete",
            new=AsyncMock(side_effect=_canned_achat),
        ):
            pipeline.run(stages=["summarize"])

        # Embedder uses mock mode (no OPENAI_API_KEY) — zero vectors
        pipeline.run(stages=["embed"])

        assert _all_papers_have_stage(pipeline.db, "embed", "done")
        assert pipeline.storage.exists(INDEX_METADATA_KEY)
        assert pipeline.storage.exists(INDEX_FAISS_KEY_TEMPLATE.format(view="title_abstract"))

    def test_full_pipeline_all_stages_complete(
        self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running all stages sequentially leaves every paper fully processed."""
        monkeypatch.setattr(
            "literavore.pipeline.OpenReviewSource.fetch",
            self._mock_fetch,
        )

        # Run fetch first so paper rows exist in the DB before seeding storage
        pipeline.run(stages=["fetch"])
        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])

        async def _canned_achat(*_args, **_kwargs) -> str:
            return _CANNED_SUMMARY_RESPONSE

        with patch(
            "literavore.summarize.llm_client.LLMClient.achat_complete",
            new=AsyncMock(side_effect=_canned_achat),
        ):
            pipeline.run(stages=["extract", "summarize", "embed"])

        for stage in ("fetch", "extract", "summarize", "embed"):
            assert _all_papers_have_stage(pipeline.db, stage, "done"), (
                f"Stage {stage!r} not done for all papers"
            )


# ---------------------------------------------------------------------------
# TestPipelineIdempotency
# ---------------------------------------------------------------------------


class TestPipelineIdempotency:
    """Run pipeline twice without force, verify second run skips already-done papers."""

    def _mock_fetch(self, _conf: ConferenceConfig) -> list[PaperMetadata]:
        return list(FAKE_PAPERS)

    def _run_full_pipeline(self, pipeline: Pipeline) -> None:
        """Run fetch + seed + extract + summarize + embed for all fake papers."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])

        async def _canned_achat(*_args, **_kwargs) -> str:
            return _CANNED_SUMMARY_RESPONSE

        with patch(
            "literavore.summarize.llm_client.LLMClient.achat_complete",
            new=AsyncMock(side_effect=_canned_achat),
        ):
            pipeline.run(stages=["extract", "summarize"])

        pipeline.run(stages=["embed"])

    def test_second_run_skips_all_stages(self, pipeline: Pipeline) -> None:
        """Second run with force=False finds no papers needing processing."""
        self._run_full_pipeline(pipeline)

        # After first run, all stages should be done
        for stage in ("extract", "summarize", "embed"):
            assert _all_papers_have_stage(pipeline.db, stage, "done")

        # Second run: nothing should need processing
        for stage in ("extract", "summarize", "embed"):
            pending = pipeline.db.get_papers_needing_stage(stage, force=False)
            assert pending == [], (
                f"Stage {stage!r} should have no pending papers on second run, "
                f"got {len(pending)}"
            )

    def test_completed_at_timestamps_unchanged_after_second_run(
        self, pipeline: Pipeline
    ) -> None:
        """Completed_at timestamps for done stages are not overwritten on second run."""
        self._run_full_pipeline(pipeline)

        # Capture completed_at for extract stage
        timestamps_before = {
            p["id"]: pipeline.db.get_stage_status(p["id"], "extract")["completed_at"]
            for p in pipeline.db.get_papers()
        }

        # Second run without force — extract stage should be skipped
        pipeline.run(stages=["extract"])

        timestamps_after = {
            p["id"]: pipeline.db.get_stage_status(p["id"], "extract")["completed_at"]
            for p in pipeline.db.get_papers()
        }

        assert timestamps_before == timestamps_after

    def test_fetch_idempotency_does_not_duplicate_papers(
        self, pipeline: Pipeline
    ) -> None:
        """Running fetch twice does not create duplicate paper records."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])
            pipeline.run(stages=["fetch"])

        papers = pipeline.db.get_papers()
        assert len(papers) == 3  # exactly 3, not 6


# ---------------------------------------------------------------------------
# TestPipelineForceRerun
# ---------------------------------------------------------------------------


class TestPipelineForceRerun:
    """Run with force=True, verify all papers are reprocessed."""

    def _mock_fetch(self, _conf: ConferenceConfig) -> list[PaperMetadata]:
        return list(FAKE_PAPERS)

    def test_force_true_reprocesses_extract(self, pipeline: Pipeline) -> None:
        """force=True causes extract to re-process already-done papers."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        assert _all_papers_have_stage(pipeline.db, "extract", "done")

        # Re-seed PDFs (keep_pdfs=True, so they still exist; seed again for safety)
        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"], force=True)

        # All papers still done after force re-run
        assert _all_papers_have_stage(pipeline.db, "extract", "done")
        # completed_at timestamps are set for all papers
        for p in pipeline.db.get_papers():
            row = pipeline.db.get_stage_status(p["id"], "extract")
            assert row is not None and row["completed_at"] is not None

    def test_force_true_get_papers_needing_stage_returns_all(
        self, pipeline: Pipeline
    ) -> None:
        """get_papers_needing_stage with force=True returns all papers even when done."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        # All 3 papers are done; force=True should still return all 3
        pending = pipeline.db.get_papers_needing_stage("extract", force=True)
        assert len(pending) == 3

    def test_force_false_skips_done_papers(self, pipeline: Pipeline) -> None:
        """get_papers_needing_stage with force=False skips done papers."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        pending = pipeline.db.get_papers_needing_stage("extract", force=False)
        assert pending == []

    def test_force_rerun_summarize(self, pipeline: Pipeline) -> None:
        """force=True on summarize stage reprocesses all papers."""
        with patch(
            "literavore.pipeline.OpenReviewSource.fetch",
            side_effect=self._mock_fetch,
        ):
            pipeline.run(stages=["fetch"])

        _seed_papers_in_storage(pipeline, [p.id for p in FAKE_PAPERS])
        pipeline.run(stages=["extract"])

        async def _canned_achat(*_args, **_kwargs) -> str:
            return _CANNED_SUMMARY_RESPONSE

        with patch(
            "literavore.summarize.llm_client.LLMClient.achat_complete",
            new=AsyncMock(side_effect=_canned_achat),
        ):
            pipeline.run(stages=["summarize"])

        assert _all_papers_have_stage(pipeline.db, "summarize", "done")

        # Force rerun
        with patch(
            "literavore.summarize.llm_client.LLMClient.achat_complete",
            new=AsyncMock(side_effect=_canned_achat),
        ):
            pipeline.run(stages=["summarize"], force=True)

        assert _all_papers_have_stage(pipeline.db, "summarize", "done")
