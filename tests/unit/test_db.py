"""Tests for literavore.db."""

from pathlib import Path

from literavore.db import Database


class TestDatabaseCreation:
    def test_creates_db_file(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        Database(db_path)
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        db_path = tmp_path / "deep" / "nested" / "test.db"
        Database(db_path)
        assert db_path.exists()

    def test_idempotent_creation(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        Database(db_path)
        Database(db_path)


class TestGetOrCreatePaper:
    def test_create_paper(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        paper = db.get_or_create_paper("p1", title="Test Paper", conference="Test-2025")
        assert paper["id"] == "p1"
        assert paper["title"] == "Test Paper"
        assert paper["conference"] == "Test-2025"

    def test_idempotent(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        p1 = db.get_or_create_paper("p1", title="Test")
        p2 = db.get_or_create_paper("p1", title="Different")
        assert p1["title"] == p2["title"]

    def test_authors_list_serialized(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1", authors=["Alice", "Bob"])
        paper = db.get_paper("p1")
        assert paper["authors"] == '["Alice", "Bob"]'

    def test_timestamps_set(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        paper = db.get_or_create_paper("p1", title="Test")
        assert paper["created_at"] is not None
        assert paper["updated_at"] is not None


class TestUpdatePaper:
    def test_update_fields(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1", title="Old")
        db.update_paper("p1", title="New")
        paper = db.get_paper("p1")
        assert paper["title"] == "New"

    def test_noop_with_no_kwargs(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1", title="Test")
        db.update_paper("p1")


class TestStageStatus:
    def test_update_and_get(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "running")
        status = db.get_stage_status("p1", "fetch")
        assert status["status"] == "running"
        assert status["started_at"] is not None

    def test_done_sets_completed_at(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "done")
        status = db.get_stage_status("p1", "fetch")
        assert status["completed_at"] is not None

    def test_failed_with_error(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "failed", error="timeout")
        status = db.get_stage_status("p1", "fetch")
        assert status["status"] == "failed"
        assert status["error"] == "timeout"

    def test_upsert_preserves_started_at(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "running")
        started = db.get_stage_status("p1", "fetch")["started_at"]
        db.update_stage_status("p1", "fetch", "done")
        status = db.get_stage_status("p1", "fetch")
        assert status["started_at"] == started


class TestGetPapersNeedingStage:
    def test_all_need_stage(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.get_or_create_paper("p2")
        papers = db.get_papers_needing_stage("fetch")
        assert len(papers) == 2

    def test_done_excluded(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.get_or_create_paper("p2")
        db.update_stage_status("p1", "fetch", "done")
        papers = db.get_papers_needing_stage("fetch")
        assert len(papers) == 1
        assert papers[0]["id"] == "p2"

    def test_failed_still_needs_stage(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "failed")
        papers = db.get_papers_needing_stage("fetch")
        assert len(papers) == 1

    def test_force_returns_all(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "done")
        papers = db.get_papers_needing_stage("fetch", force=True)
        assert len(papers) == 1


class TestGetFailedPapers:
    def test_returns_failed(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "failed", error="oops")
        failed = db.get_failed_papers("fetch")
        assert len(failed) == 1

    def test_empty_on_no_failures(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.update_stage_status("p1", "fetch", "done")
        assert db.get_failed_papers("fetch") == []


class TestRunLifecycle:
    def test_start_and_complete(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        run_id = db.start_run("abc123", ["fetch", "download"])
        assert isinstance(run_id, int)
        db.complete_run(run_id)


class TestRunStats:
    def test_empty_stats(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        assert db.get_run_stats() == {}

    def test_counts_by_stage_status(self, tmp_path: Path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_paper("p1")
        db.get_or_create_paper("p2")
        db.update_stage_status("p1", "fetch", "done")
        db.update_stage_status("p2", "fetch", "failed")
        stats = db.get_run_stats()
        assert stats["fetch"]["done"] == 1
        assert stats["fetch"]["failed"] == 1
