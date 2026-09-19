"""The interface both retrieval arms implement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Passage:
    id: str
    doc_id: str
    doc_name: str
    page: int
    section: str | None
    text: str
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Passage]: ...
