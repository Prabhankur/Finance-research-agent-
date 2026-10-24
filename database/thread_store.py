"""
database/thread_store.py  –  Lightweight metadata for the thread list.

Deliberately NOT the same thing as the LangGraph checkpointer (graph/chat_
graph.py's SqliteSaver). The checkpointer stores the full conversation STATE
(messages, intent, tool traces) keyed by thread_id — it's what LangGraph
needs to resume a graph. This store answers a much smaller question:
"what threads exist, what are they called, when were they last touched" —
exactly what a sidebar needs to render a list, and nothing the checkpointer
is designed to answer efficiently (it has no "list all distinct thread_ids"
primitive; SqliteSaver's schema is optimized for point lookups by thread_id,
not enumeration with metadata).

Two tiny tables, one SQLite file, no ORM — this is genuinely simple enough
that SQLAlchemy would be more ceremony than the problem needs. When the
Postgres knowledge-layer schema lands, this table (or its equivalent) is a
natural candidate to fold in as `conversations`, per the reviewed schema
proposal — same shape, just a different backend.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).resolve().parent.parent / "output" / "chat_threads.db"
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
"""


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init() -> None:
    with _LOCK, _connect() as conn:
        conn.executescript(_SCHEMA)


_init()


def _title_from_text(text: str, max_len: int = 60) -> str:
    """First user message, trimmed — the same convention ChatGPT/Claude use."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def touch(thread_id: str, first_message: Optional[str] = None) -> None:
    """
    Upserts a conversation's `updated_at` (and increments message_count).
    Called once per turn from the API layer. `first_message` sets the title
    ONLY on first insert — later calls never overwrite an existing title, so
    a conversation stays labelled by what it started as, matching how every
    mainstream chat UI behaves.
    """
    now = datetime.now(timezone.utc).isoformat()
    title = _title_from_text(first_message) if first_message else thread_id

    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversations (thread_id, title, created_at, updated_at, message_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(thread_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                message_count = message_count + 1
            """,
            (thread_id, title, now, now),
        )


def list_threads(limit: int = 50) -> list[dict]:
    """Most recently active conversations first."""
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at, message_count
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_thread(thread_id: str) -> bool:
    """Removes the metadata row. Does NOT touch the checkpointer's own state
    for that thread — a deliberate scope limit, see module docstring. A
    thread with no metadata row simply won't render in the sidebar; its
    checkpoint data, if any, is harmless orphaned storage until a future
    retention job cleans it up (not built yet — noted as a known gap)."""
    with _LOCK, _connect() as conn:
        cursor = conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        return cursor.rowcount > 0
