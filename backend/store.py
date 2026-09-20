"""PostgreSQL + pgvector store.

Every document lands in one transaction — the documents row, its doc_text row
and all its chunk rows commit together — so a failed upload never leaves half
a document behind.

Store calls normally open their own connection and commit on exit. Tests wrap a
test in store.test_transaction(), which pins one ambient connection so every
store call runs in a savepoint and the whole test can be rolled back.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("recite.store")

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_DATABASE_URL = "postgresql://localhost:5432/recite"

_current = threading.local()


class _Rollback(Exception):
    """Raised internally to force a rollback of the ambient test transaction."""


def _connect(register: bool = True) -> psycopg.Connection:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    conn = psycopg.connect(url)
    if register:                                   # vector type must exist first
        from pgvector.psycopg import register_vector
        register_vector(conn)
    return conn


@contextmanager
def _ambient(register: bool = True):
    """Yield the connection for a store call: the ambient test connection (in a
    savepoint) when one is active, otherwise a fresh, self-committing one."""
    conn = getattr(_current, "conn", None)
    if conn is not None:
        with conn.transaction():                   # savepoint inside the test transaction
            yield conn
    else:
        with _connect(register) as conn:           # own connection: commit on exit
            yield conn


@contextmanager
def test_transaction():
    """Run store calls in one transaction, rolled back on exit (for tests)."""
    conn = _connect()
    prev = getattr(_current, "conn", None)
    _current.conn = conn
    try:
        with conn.transaction():
            yield conn
            raise _Rollback
    except _Rollback:
        pass
    finally:
        _current.conn = prev
        conn.close()


def connect(register: bool = True) -> psycopg.Connection:
    """Public connection factory for retrieval; registers the vector type."""
    return _connect(register)


def init_db() -> None:
    """Create tables and indexes; safe to run more than once."""
    with _connect(register=False) as conn:         # schema creates the extension
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    log.info("schema applied")


def has(doc_id: str) -> bool:
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,))
        return cur.fetchone() is not None


def get_chunks(doc_id: str) -> list:
    """Return a document's chunks in order; deferred import avoids a cycle."""
    from backend.ingest import Chunk

    with _ambient() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, d.name, c.page, c.section, c.text
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.doc_id = %s ORDER BY c.chunk_index
            """,
            (doc_id,),
        )
        rows = cur.fetchall()
    return [Chunk(id=r[0], doc_id=r[1], doc_name=r[2], page=r[3], section=r[4], text=r[5]) for r in rows]


def add(doc_id: str, name: str, full_text: str, chunks: list, vectors: np.ndarray) -> bool:
    """Insert a document atomically; returns False if it already existed
    (a concurrent upload of identical content won the insert)."""
    rows = [
        (c.id, c.doc_id, i, c.page, c.section, c.text, np.asarray(v, dtype=np.float32))
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (doc_id, name),
        )
        if cur.rowcount == 0:
            return False
        cur.execute(
            "INSERT INTO doc_text (doc_id, text) VALUES (%s, %s)",
            (doc_id, full_text),
        )
        cur.executemany(
            """
            INSERT INTO chunks (id, doc_id, chunk_index, page, section, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    log.info("stored doc %s: %d chunks", doc_id, len(chunks))
    return True


def quota_take(model: str, day: date) -> int:
    """Atomically increment a model's daily call counter; returns the new count."""
    with _ambient(register=False) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO quota_daily (model, day, calls) VALUES (%s, %s, 1)
            ON CONFLICT (model, day) DO UPDATE SET calls = quota_daily.calls + 1
            RETURNING calls
            """,
            (model, day),
        )
        return cur.fetchone()[0]


def save_pending(tool: str, args: dict) -> str:
    """Insert a pending action and return its id."""
    pid = uuid4().hex
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pending (id, tool, args) VALUES (%s, %s, %s)",
            (pid, tool, Jsonb(args)),
        )
    return pid


def confirm_pending(pid: str) -> tuple[str, dict] | None:
    """Atomically mark a pending action executed if it is still pending and
    younger than 10 minutes; returns (tool, args), or None if already handled."""
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pending SET status = 'executed'
            WHERE id = %(id)s AND status = 'pending'
              AND created_at > now() - interval '10 minutes'
            RETURNING tool, args
            """,
            {"id": pid},
        )
        row = cur.fetchone()
    return None if row is None else (row[0], row[1])


def cancel_pending(pid: str) -> bool:
    """Mark a pending action cancelled; True only if it was still pending."""
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE pending SET status = 'cancelled' WHERE id = %(id)s AND status = 'pending'",
            {"id": pid},
        )
        return cur.rowcount > 0


def mark_pending(pid: str, status: str) -> None:
    """Set a pending action's status (e.g. 'failed' or 'cancelled')."""
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute("UPDATE pending SET status = %s WHERE id = %s", (status, pid))


def list_documents() -> list[dict]:
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.name, d.created_at, count(c.id)
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
            GROUP BY d.id, d.name, d.created_at
            ORDER BY d.created_at DESC
            """,
        )
        rows = cur.fetchall()
    return [{"id": r[0], "name": r[1], "created_at": r[2].isoformat(), "chunks": r[3]}
            for r in rows]


def delete_document(doc_id: str) -> bool:
    """Delete a document; ON DELETE CASCADE removes its text and chunks."""
    with _ambient() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        return cur.rowcount > 0
