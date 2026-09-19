"""The embedding arm: embed the query at 768 dimensions, then let Postgres
do the search.

<=> is pgvector's cosine distance, so 1 - distance is cosine similarity — the
score the router's threshold will use. No index on purpose: at this scale an
exact scan over every chunk is what measures retrieval quality.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend import llm, store
from backend.retrieval.base import Passage

_SEARCH_SQL = """
SELECT c.id, c.doc_id, d.name, c.page, c.section, c.text,
       1 - (c.embedding <=> %(q)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.doc_id
ORDER BY c.embedding <=> %(q)s
LIMIT %(k)s
"""


class EmbeddingRetriever:
    def __init__(self, embed: Callable[..., list[list[float]]] = llm.embed) -> None:
        self._embed = embed

    def search(self, query: str, k: int = 5) -> list[Passage]:
        vector = self._embed([query], dimensions=llm.EMBED_DIMENSIONS, task_type="RETRIEVAL_QUERY")[0]
        with store.connect() as conn, conn.cursor() as cur:
            cur.execute(_SEARCH_SQL, {"q": np.asarray(vector, dtype=np.float32), "k": k})
            rows = cur.fetchall()
        return [
            Passage(id=r[0], doc_id=r[1], doc_name=r[2], page=r[3], section=r[4], text=r[5], score=r[6])
            for r in rows
        ]
