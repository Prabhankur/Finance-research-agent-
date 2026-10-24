"""
database/report_jobs.py  –  Persistence for the report pipeline's jobs.

Was: an in-memory Python dict keyed by job_id (see api/server.py's old
_JOBS). That had two real problems the user hit:

  1. Every server restart nuked the job list, including successfully-
     completed reports still sitting on disk in output/. The .docx file
     was still there, but the API had no memory of it, so the sidebar
     couldn't offer it and the user had no way to find it.

  2. Reports had no association with the thread that requested them, so a
     multi-thread user could not answer "which report came from which
     conversation" — the whole point of having threads.

Same table backs both SQLite and PostgreSQL — see database/connection.py
and database/schema.py for how the DATABASE_URL env var swaps the backend
with no code change here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from database.connection import execute, init_db, query, query_one

# Schema lives here rather than in database/schema.py because it's not part
# of the financial-facts knowledge layer — it's operational state for the
# HTTP layer. Kept separate so a schema migration to that knowledge layer
# doesn't drag job records along with it.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS report_jobs (
    job_id          TEXT PRIMARY KEY,
    thread_id       TEXT,
    query           TEXT NOT NULL,
    report_type     TEXT,
    status          TEXT NOT NULL,    -- queued | running | completed | failed
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    duration_s      REAL,
    report_path     TEXT,
    error           TEXT,
    UNIQUE(job_id)
)
"""
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_report_jobs_thread ON report_jobs(thread_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_report_jobs_status ON report_jobs(status)",
]


def _init() -> None:
    init_db()  # ensures the DB file/schema/connection are usable
    try:
        from database.connection import get_connection
        with get_connection() as conn:
            cursor = conn.cursor()
            from database.connection import _translate  # local translator, see connection.py
            cursor.execute(_translate(_SCHEMA))
            for stmt in _INDEXES:
                cursor.execute(_translate(stmt))
    except Exception as exc:
        # Same failure posture as the caching layer: a broken DB must not
        # break the whole agent. Report submission still runs against the
        # in-memory fallback in api/server.py — persistence just becomes
        # per-process again, degrading to the pre-existing behaviour.
        print(f"[ReportJobs] Schema init failed ({exc}) — running without persisted jobs.")


_init()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create(job_id: str, query: str, report_type: Optional[str], thread_id: Optional[str]) -> None:
    try:
        execute(
            """
            INSERT INTO report_jobs (job_id, thread_id, query, report_type, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, thread_id, query, report_type, "queued", _now()),
        )
    except Exception as exc:
        print(f"[ReportJobs] create failed ({exc})")


def update(job_id: str, **fields: Any) -> None:
    """Partial update on any known column — used to move a job through the
    status lifecycle without a bespoke method per transition."""
    if not fields:
        return
    columns = list(fields.keys())
    set_clause = ", ".join(f"{c} = ?" for c in columns)
    values = tuple(fields[c] for c in columns) + (job_id,)
    try:
        execute(f"UPDATE report_jobs SET {set_clause} WHERE job_id = ?", values)
    except Exception as exc:
        print(f"[ReportJobs] update({job_id}) failed ({exc})")


def get(job_id: str) -> Optional[dict]:
    try:
        return query_one("SELECT * FROM report_jobs WHERE job_id = ?", (job_id,))
    except Exception:
        return None


def list_all(limit: int = 50) -> list[dict]:
    try:
        return query(
            "SELECT * FROM report_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    except Exception:
        return []


def list_for_thread(thread_id: str, limit: int = 20) -> list[dict]:
    """Reports the sidebar shows next to their conversation — the whole
    reason thread_id got added to this table."""
    try:
        return query(
            """SELECT * FROM report_jobs WHERE thread_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (thread_id, limit),
        )
    except Exception:
        return []
