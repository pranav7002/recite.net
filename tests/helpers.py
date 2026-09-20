"""Small test helpers: a fake retriever, passages, outbox reading, uploads."""
from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi import UploadFile

from backend.retrieval.base import Passage


class FakeRetriever:
    def __init__(self, passages):
        self._passages = list(passages)
        self.queries: list[str] = []

    def search(self, query, k=5):
        self.queries.append(query)
        return list(self._passages)


def passage(text, doc_name="notes.pdf", page=1, doc_id="d", pid="p1"):
    return Passage(id=pid, doc_id=doc_id, doc_name=doc_name, page=page,
                   section=None, text=text, score=0.9)


def read_outbox(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)
