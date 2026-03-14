"""Stage runner / orchestrator for the Literavore pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

from literavore.config import LiteravoreConfig
from literavore.db import Database
from literavore.embed.embedder import Embedder
from literavore.embed.index import PaperIndex
from literavore.extract.pdf_extractor import extract_papers_batch
from literavore.ingest.pdf_downloader import AsyncPDFDownloader
from literavore.sources.openreview import OpenReviewSource
from literavore.storage import LocalStorage, StorageBackend
from literavore.summarize.summarizer import Summarizer
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
        """Fetch paper metadata from OpenReview for all configured conferences."""
        source = OpenReviewSource()
        for conference_config in self.config.conferences:
            papers = source.fetch(conference_config)
            for paper in papers:
                self.db.get_or_create_paper(
                    paper.id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    pdf_url=paper.pdf_url,
                    conference=conference_config.name,
                    source="openreview",
                )
                self.db.update_stage_status(paper.id, "fetch", "done")
            self.logger.info(
                "Fetched %d papers for conference %s", len(papers), conference_config.name
            )

    def _run_download(self, force: bool = False) -> None:
        """Download PDFs for all papers that have not yet been downloaded."""
        papers = self.db.get_papers_needing_stage("download", force=force)
        if not papers:
            self.logger.info("No papers needing download — skipping")
            return

        self.logger.info("Downloading PDFs for %d papers", len(papers))

        async def _run() -> list[dict]:
            async with AsyncPDFDownloader(self.config.pdf, self.db, self.storage) as downloader:
                return await downloader.download_papers(papers)

        results = asyncio.run(_run())
        succeeded = sum(1 for r in results if r.get("success"))
        failed = len(results) - succeeded
        self.logger.info(
            "Download complete: %d succeeded, %d failed (of %d attempted)",
            succeeded,
            failed,
            len(results),
        )

    def _run_extract(self, force: bool = False) -> None:
        """Extract text from downloaded PDFs using pymupdf4llm."""
        papers = self.db.get_papers_needing_stage("extract", force=force)
        if not papers:
            self.logger.info("No papers needing extraction — skipping")
            return
        self.logger.info("Extracting text from %d papers", len(papers))
        keep_pdfs = self.config.pdf.keep_pdfs
        results = extract_papers_batch(
            papers, self.config.extract, self.db, self.storage, keep_pdfs
        )
        self.logger.info(
            "Extraction complete: %d/%d succeeded", len(results), len(papers)
        )

    def _run_summarize(self, force: bool = False) -> None:
        """Generate LLM summaries and tags for extracted papers."""
        papers = self.db.get_papers_needing_stage("summarize", force=force)
        if not papers:
            self.logger.info("No papers needing summarization — skipping")
            return
        self.logger.info("Summarizing %d papers", len(papers))
        summarizer = Summarizer(self.config.summary, self.db, self.storage)
        results = asyncio.run(summarizer.summarize_papers(papers))
        self.logger.info("Summarization complete: %d/%d succeeded", len(results), len(papers))

    def _run_embed(self, force: bool = False) -> None:
        """Generate multi-view embeddings and build FAISS vector index."""
        import json  # noqa: PLC0415

        papers = self.db.get_papers_needing_stage("embed", force=force)
        if not papers:
            self.logger.info("No papers needing embedding — skipping")
            return

        self.logger.info("Embedding %d papers", len(papers))

        # Load summaries from storage for each paper
        summaries: dict[str, dict] = {}
        for paper in papers:
            paper_id: str = paper["id"]
            summary_key = f"summaries/{paper_id}.json"
            if self.storage.exists(summary_key):
                try:
                    raw = self.storage.get(summary_key)
                    summaries[paper_id] = json.loads(raw.decode())
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "Could not load summary for %s: %s", paper_id, exc
                    )

        # Generate embeddings
        embedder = Embedder(self.config.embedding)
        embedding_records = embedder.embed_papers(papers, summaries)

        # Build and save the index
        paper_index = PaperIndex(
            dimensions=self.config.embedding.dimensions,
            views=list(self.config.embedding.views),
        )
        paper_index.build(embedding_records)
        paper_index.save(self.storage)

        # Update DB stage status for each paper
        for paper in papers:
            self.db.update_stage_status(paper["id"], "embed", "done")

        self.logger.info(
            "Embed complete: %d papers, %d records, index size %d",
            len(papers),
            len(embedding_records),
            paper_index.size,
        )
