"""
database/connection.py  –  One connection helper for SQLite and PostgreSQL.

Backend is chosen by DATABASE_URL:
    unset / sqlite:///...     -> SQLite file (default, zero setup)
    postgresql://user@host/db -> PostgreSQL via psycopg2

Why support both rather than committing to Postgres now: the schema is
identical either way, and requiring a running Postgres instance to launch a
dev server is friction that buys nothing until there's a reason to scale.
The switch is one env var with no code change, so the Postgres migration is
a deployment decision rather than a rewrite.

The one real portability wrinkle is parameter style — SQLite uses `?`,
psycopg2 uses `%s`. Rather than making every call site care, queries are
written with `?` throughout and translated on the way out for Postgres.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from database.schema import SCHEMA_STATEMENTS

_DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "output" / "financial_knowledge.db"
_LOCK = threading.Lock()
_INITIALISED = False


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def is_postgres() -> bool:
    return _database_url().startswith(("postgresql://", "postgres://"))


def _translate(sql: str) -> str:
    """SQLite `?` placeholders -> psycopg2 `%s`, and AUTOINCREMENT -> SERIAL."""
    if not is_postgres():
        return sql
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    return sql.replace("?", "%s")


@contextmanager
def get_connection() -> Iterator[Any]:
    """Yields a DB connection, committing on success and rolling back on error."""
    if is_postgres():
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(_database_url(), cursor_factory=psycopg2.extras.DictCursor)
    else:
        url = _database_url()
        path = url.replace("sqlite:///", "") if url.startswith("sqlite:///") else str(_DEFAULT_SQLITE_PATH)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        # WAL lets the report pipeline's background thread write while the
        # chat agent reads, instead of the two blocking each other — this
        # process runs both concurrently by design.
        conn.execute("PRAGMA journal_mode=WAL")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(force: bool = False) -> None:
    """Creates all tables. Idempotent — every statement is IF NOT EXISTS."""
    global _INITIALISED
    with _LOCK:
        if _INITIALISED and not force:
            return
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                for statement in SCHEMA_STATEMENTS:
                    cursor.execute(_translate(statement))
            _INITIALISED = True
            backend = "PostgreSQL" if is_postgres() else f"SQLite ({_DEFAULT_SQLITE_PATH.name})"
            print(f"[DB] Knowledge layer ready ({backend}).")
        except Exception as exc:
            # A database problem must never prevent the agent from starting.
            # Every read path treats a cache miss and a cache error
            # identically (fall through to the live API), so a broken DB
            # degrades this to "no caching" rather than "no agent".
            print(f"[DB] WARNING: init failed ({exc}). Running without the knowledge layer.")


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Runs a SELECT and returns plain dicts."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(_translate(sql), params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    """
    Runs an INSERT/UPDATE/DELETE and returns the affected row's id where
    available.

    REAL BUG FIXED HERE, found via live-Postgres testing rather than
    guessed at: psycopg2 cursors have a `.lastrowid` ATTRIBUTE (so the old
    `getattr(cursor, "lastrowid", 0)` never fell through to the default),
    but Postgres has no session-level "last insert id" concept the way
    SQLite/MySQL do — it always reads back None/0 regardless of whether
    the insert succeeded. The row was being inserted correctly; only the
    REPORTED id was wrong, silently, which then got used as a foreign key
    elsewhere (e.g. a sources.id passed into income_statements.source_id)
    and caused a ForeignKeyViolation two calls later — a confusing failure
    surfacing far from its actual cause.

    Fix: on Postgres, INSERT statements get `RETURNING id` appended (unless
    already present) and the id is read from the query result directly,
    which IS reliable. Falls back to a plain retry without RETURNING if
    the target table has no `id` column at all (report_jobs is keyed by
    job_id, not id) — that fallback intentionally returns 0, matching the
    old best-effort behaviour for tables whose callers never use the
    returned id anyway.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        if is_postgres() and sql.strip().upper().startswith("INSERT") and "RETURNING" not in sql.upper():
            attempt_sql = sql.rstrip().rstrip(";") + " RETURNING id"
            try:
                cursor.execute(_translate(attempt_sql), params)
                row = cursor.fetchone()
                return row[0] if row else 0
            except Exception:
                # Table has no `id` column (report_jobs) — Postgres aborts
                # the whole transaction on a failed statement, so it must be
                # rolled back before this connection can run anything else.
                conn.rollback()
                cursor = conn.cursor()
                cursor.execute(_translate(sql), params)
                return 0

        cursor.execute(_translate(sql), params)
        return getattr(cursor, "lastrowid", 0) or 0
