import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


CREATE_PAPERS = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    source TEXT,
    conference TEXT,
    title TEXT,
    authors TEXT,
    abstract TEXT,
    pdf_url TEXT,
    status TEXT,
    created_at TEXT,
    updated_at TEXT
)
"""

CREATE_PROCESSING_STATE = """
CREATE TABLE IF NOT EXISTS processing_state (
    paper_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    file_hash TEXT,
    started_at TEXT,
    completed_at TEXT,
    PRIMARY KEY (paper_id, stage),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
)
"""

CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    completed_at TEXT,
    config_hash TEXT,
    stages_run TEXT
)
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.execute(CREATE_PAPERS)
            self._conn.execute(CREATE_PROCESSING_STATE)
            self._conn.execute(CREATE_RUNS)

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return dict(row)

    def get_or_create_paper(self, paper_id: str, **kwargs) -> dict:
        now = _now()
        authors = kwargs.get("authors")
        if isinstance(authors, (list, dict)):
            kwargs["authors"] = json.dumps(authors)

        existing = self.get_paper(paper_id)
        if existing is not None:
            return existing

        fields = {"id": paper_id, "created_at": now, "updated_at": now, **kwargs}
        columns = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        with self._conn:
            self._conn.execute(
                f"INSERT OR IGNORE INTO papers ({columns}) VALUES ({placeholders})",
                list(fields.values()),
            )
        return self.get_paper(paper_id)  # type: ignore[return-value]

    def update_paper(self, paper_id: str, **kwargs) -> None:
        if not kwargs:
            return
        authors = kwargs.get("authors")
        if isinstance(authors, (list, dict)):
            kwargs["authors"] = json.dumps(authors)
        kwargs["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [paper_id]
        with self._conn:
            self._conn.execute(
                f"UPDATE papers SET {set_clause} WHERE id = ?",
                values,
            )

    def get_paper(self, paper_id: str) -> dict | None:
        cursor = self._conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
        row = cursor.fetchone()
        return self._row_to_dict(row)

    def get_papers(self, conference: str | None = None) -> list[dict]:
        if conference is not None:
            cursor = self._conn.execute("SELECT * FROM papers WHERE conference = ?", (conference,))
        else:
            cursor = self._conn.execute("SELECT * FROM papers")
        return [self._row_to_dict(row) for row in cursor.fetchall()]  # type: ignore[misc]

    def update_stage_status(
        self,
        paper_id: str,
        stage: str,
        status: str,
        error: str | None = None,
        file_hash: str | None = None,
    ) -> None:
        now = _now()
        started_at: str | None = None
        completed_at: str | None = None
        if status == "running":
            started_at = now
        elif status in ("done", "failed"):
            completed_at = now

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO processing_state
                    (paper_id, stage, status, error, file_hash, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (paper_id, stage) DO UPDATE SET
                    status = excluded.status,
                    error = excluded.error,
                    file_hash = COALESCE(excluded.file_hash, file_hash),
                    started_at = CASE
                        WHEN excluded.started_at IS NOT NULL THEN excluded.started_at
                        ELSE started_at
                    END,
                    completed_at = CASE
                        WHEN excluded.completed_at IS NOT NULL THEN excluded.completed_at
                        ELSE completed_at
                    END
                """,
                (paper_id, stage, status, error, file_hash, started_at, completed_at),
            )

    def get_papers_needing_stage(self, stage: str, force: bool = False) -> list[dict]:
        if force:
            return self.get_papers()
        cursor = self._conn.execute(
            """
            SELECT p.* FROM papers p
            WHERE p.id NOT IN (
                SELECT ps.paper_id FROM processing_state ps
                WHERE ps.stage = ? AND ps.status = 'done'
            )
            """,
            (stage,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]  # type: ignore[misc]

    def get_failed_papers(self, stage: str) -> list[dict]:
        cursor = self._conn.execute(
            """
            SELECT p.* FROM papers p
            INNER JOIN processing_state ps ON p.id = ps.paper_id
            WHERE ps.stage = ? AND ps.status = 'failed'
            """,
            (stage,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]  # type: ignore[misc]

    def get_stage_status(self, paper_id: str, stage: str) -> dict | None:
        cursor = self._conn.execute(
            "SELECT * FROM processing_state WHERE paper_id = ? AND stage = ?",
            (paper_id, stage),
        )
        row = cursor.fetchone()
        return self._row_to_dict(row)

    def start_run(self, config_hash: str, stages: list[str]) -> int:
        now = _now()
        stages_json = json.dumps(stages)
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO runs (started_at, config_hash, stages_run) VALUES (?, ?, ?)",
                (now, config_hash, stages_json),
            )
        return cursor.lastrowid  # type: ignore[return-value]

    def complete_run(self, run_id: int) -> None:
        now = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET completed_at = ? WHERE id = ?",
                (now, run_id),
            )

    def get_run_stats(self) -> dict:
        cursor = self._conn.execute(
            """
            SELECT stage, status, COUNT(*) as count
            FROM processing_state
            GROUP BY stage, status
            """
        )
        stats: dict = {}
        for row in cursor.fetchall():
            stage = row["stage"]
            status = row["status"]
            count = row["count"]
            if stage not in stats:
                stats[stage] = {}
            stats[stage][status] = count
        return stats
