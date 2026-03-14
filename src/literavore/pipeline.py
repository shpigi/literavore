"""Stage runner / orchestrator for the Literavore pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from literavore.config import LiteravoreConfig
from literavore.db import Database
from literavore.storage import LocalStorage, StorageBackend
from literavore.utils import get_logger, setup_logging

STAGES = ["fetch", "download", "extract", "summarize", "embed"]


class Pipeline:
    """Orchestrates the five Literavore processing stages."""

    def __init__(self, config: LiteravoreConfig) -> None:
        setup_logging()
        self.logger = get_logger(__name__)
        self.config = config

        data_dir = Path(config.storage.data_dir)
        db_path = data_dir / "literavore.db"
        self.db = Database(db_path)
        self.storage: StorageBackend = LocalStorage(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        stages: list[str] | None = None,
        from_stage: str | None = None,
        force: bool = False,
    ) -> None:
        """Run the pipeline.

        Args:
            stages: Explicit list of stages to run.  Mutually exclusive with
                *from_stage*.
            from_stage: Run this stage and every stage that follows it.
            force: Re-process papers that have already completed a stage.
        """
        stages_to_run = self._resolve_stages(stages, from_stage)

        config_hash = self._hash_config()
        run_id = self.db.start_run(config_hash, stages_to_run)
        self.logger.info("Pipeline run %d started (stages: %s)", run_id, stages_to_run)

        for stage in stages_to_run:
            t0 = time.time()
            self.logger.info("Stage [%s] starting", stage)
            try:
                self._run_stage(stage, force=force)
                elapsed = time.time() - t0
                self.logger.info("Stage [%s] completed in %.2fs", stage, elapsed)
            except Exception as exc:  # noqa: BLE001
                elapsed = time.time() - t0
                self.logger.error(
                    "Stage [%s] failed after %.2fs: %s", stage, elapsed, exc, exc_info=True
                )

        self.db.complete_run(run_id)
        self.logger.info("Pipeline run %d finished", run_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_stages(
        self,
        stages: list[str] | None,
        from_stage: str | None,
    ) -> list[str]:
        if stages is not None and from_stage is not None:
            raise ValueError("Provide at most one of 'stages' or 'from_stage', not both.")

        if stages is not None:
            for s in stages:
                if s not in STAGES:
                    raise ValueError(f"Unknown stage {s!r}. Valid stages: {STAGES}")
            return list(stages)

        if from_stage is not None:
            if from_stage not in STAGES:
                raise ValueError(f"Unknown stage {from_stage!r}. Valid stages: {STAGES}")
            return STAGES[STAGES.index(from_stage) :]

        return list(STAGES)

    def _hash_config(self) -> str:
        config_json = self.config.model_dump_json()
        return hashlib.sha256(config_json.encode()).hexdigest()

    def _run_stage(self, stage: str, force: bool = False) -> None:
        """Dispatch *stage* to the appropriate handler."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Valid stages: {STAGES}")

        dispatch: dict[str, object] = {
            "fetch": self._run_fetch,
            "download": self._run_download,
            "extract": self._run_extract,
            "summarize": self._run_summarize,
            "embed": self._run_embed,
        }
        handler = dispatch[stage]
        handler(force)  # type: ignore[operator]

    # ------------------------------------------------------------------
    # Stage stubs (filled in by later phases)
    # ------------------------------------------------------------------

    def _run_fetch(self, force: bool = False) -> None:
        """Stub: fetch paper metadata from conference sources."""
        self.logger.info("Stage fetch not yet implemented")

    def _run_download(self, force: bool = False) -> None:
        """Stub: download PDFs for fetched papers."""
        self.logger.info("Stage download not yet implemented")

    def _run_extract(self, force: bool = False) -> None:
        """Stub: extract text from downloaded PDFs."""
        self.logger.info("Stage extract not yet implemented")

    def _run_summarize(self, force: bool = False) -> None:
        """Stub: generate LLM summaries and tags."""
        self.logger.info("Stage summarize not yet implemented")

    def _run_embed(self, force: bool = False) -> None:
        """Stub: generate embeddings and build vector index."""
        self.logger.info("Stage embed not yet implemented")
