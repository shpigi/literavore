"""Tests for literavore.pipeline."""

from pathlib import Path

import pytest

from literavore.config import LiteravoreConfig
from literavore.pipeline import STAGES, Pipeline


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    config = LiteravoreConfig(storage={"backend": "local", "data_dir": str(tmp_path / "data")})
    return Pipeline(config)


class TestPipelineCreation:
    def test_creates_pipeline(self, pipeline: Pipeline):
        assert pipeline.config is not None
        assert pipeline.db is not None
        assert pipeline.storage is not None

    def test_db_in_data_dir(self, tmp_path: Path):
        config = LiteravoreConfig(
            storage={"backend": "local", "data_dir": str(tmp_path / "data")}
        )
        Pipeline(config)
        assert (tmp_path / "data" / "literavore.db").exists()


class TestStageResolution:
    def test_all_stages_by_default(self, pipeline: Pipeline):
        result = pipeline._resolve_stages(None, None)
        assert result == STAGES

    def test_explicit_stages(self, pipeline: Pipeline):
        result = pipeline._resolve_stages(["fetch", "extract"], None)
        assert result == ["fetch", "extract"]

    def test_from_stage(self, pipeline: Pipeline):
        result = pipeline._resolve_stages(None, "extract")
        assert result == ["extract", "summarize", "embed"]

    def test_from_first_stage(self, pipeline: Pipeline):
        result = pipeline._resolve_stages(None, "fetch")
        assert result == STAGES

    def test_from_last_stage(self, pipeline: Pipeline):
        result = pipeline._resolve_stages(None, "embed")
        assert result == ["embed"]

    def test_both_raises(self, pipeline: Pipeline):
        with pytest.raises(ValueError, match="at most one"):
            pipeline._resolve_stages(["fetch"], "extract")

    def test_unknown_stage_raises(self, pipeline: Pipeline):
        with pytest.raises(ValueError, match="Unknown stage"):
            pipeline._resolve_stages(["bogus"], None)

    def test_unknown_from_stage_raises(self, pipeline: Pipeline):
        with pytest.raises(ValueError, match="Unknown stage"):
            pipeline._resolve_stages(None, "bogus")


class TestStagesConstant:
    def test_stage_order(self):
        assert STAGES == ["fetch", "download", "extract", "summarize", "embed"]

    def test_stages_are_strings(self):
        assert all(isinstance(s, str) for s in STAGES)


class TestRun:
    def test_run_all_stages(self, pipeline: Pipeline):
        pipeline.run()

    def test_run_single_stage(self, pipeline: Pipeline):
        pipeline.run(stages=["fetch"])

    def test_run_from_stage(self, pipeline: Pipeline):
        pipeline.run(from_stage="summarize")

    def test_run_records_in_db(self, pipeline: Pipeline):
        pipeline.run(stages=["fetch"])
        stats = pipeline.db.get_run_stats()
        assert isinstance(stats, dict)

    def test_run_with_force(self, pipeline: Pipeline):
        pipeline.run(stages=["fetch"], force=True)
