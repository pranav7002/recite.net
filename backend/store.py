"""PostgreSQL + pgvector store.

Every document lands in one transaction — the documents row, its doc_text row
and all its chunk rows commit together — so a failed upload never leaves half
a document behind.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("recite.store")

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_DATABASE_URL = "postgresql://localhost:5432/recite"


def _connect(register: bool = True) -> psycopg.Connection:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    conn = psycopg.connect(url)
    if register:                                   # vector type must exist first
        from pgvector.psycopg import register_vector
        register_vector(conn)
    return conn


def init_db() -> None:
    """Create tables and indexes; safe to run more than once."""
    with _connect(register=False) as conn:         # schema creates the extension
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    log.info("schema applied")


def has(doc_id: str) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,))
        return cur.fetchone() is not None


def get_chunks(doc_id: str) -> list:
    """Return a document's chunks in order; deferred import avoids a cycle."""
    from backend.ingest import Chunk

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, d.name, c.page, c.section, c.text
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.doc_id = %s ORDER BY c.chunk_index.id
            """,
            (doc_id,),
        )
        rows = cur.fetchall()
    return [Chunk(id=r[0], doc_id=r[1], doc_name=r[2], page=r[3], section=r[4], text=r[5]) for r in rows]


def add(doc_id: str, name: str, full_text: str, chunks: list, vectors: np.ndarray) -> None:
    """Insert a document atomically: documents row + doc_text row + chunks."""
    rows = [
        (c.id, c.doc_id, i, c.page, c.section, c.text, np.asarray(v, dtype=np.float32))
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    with _connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (id, name) VALUES (%s, %s)",
            (doc_id, name),
        )
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
