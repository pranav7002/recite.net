"""The router: try embeddings first, escalate to the RLM arm only when fixed
rules say the cheap result looks weak.

Three triggers, from the architecture guide (section 10):

1. The top embedding score is below SCORE_THRESHOLD — nothing matched well.
2. The hits are spread across four or more sections — the question is probably
   about structure, not one fact.
3. (applied later, in the grounding retry) an answer built from embedding hits
   fails the grounding check, so retry once with the RLM arm.

Rules, not a model, because the signals already exist and a routing call would
cost the latency the router exists to save. SCORE_THRESHOLD is tuned on the
``tune`` split only.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from backend import trace
from backend.retrieval.base import Passage, Retriever
from backend.retrieval.embeddings import EmbeddingRetriever
from backend.retrieval.rlm_arm import RLMRetriever

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
SCATTER_SECTIONS = 4


class Router:
    """A Retriever that escalates from embeddings to RLM on weak results."""

    def __init__(self, embeddings: Retriever | None = None, rlm: Retriever | None = None,
                 score_threshold: float = SCORE_THRESHOLD) -> None:
        self.embeddings = embeddings or EmbeddingRetriever()
        self.rlm = rlm or RLMRetriever()
        self.score_threshold = score_threshold

    def search(self, query: str, k: int = 5) -> list[Passage]:
        hits = self.embeddings.search(query, k=k)
        if hits and hits[0].score < self.score_threshold:
            trace.record(op="router", escalated=True, reason="weak best match",
                         score=round(hits[0].score, 4))
            return self.rlm.search(query, k=k)
        if len({h.section for h in hits if h.section}) >= SCATTER_SECTIONS:
            trace.record(op="router", escalated=True, reason="scattered sections",
                         sections=len({h.section for h in hits if h.section}))
            return self.rlm.search(query, k=k)
        return hits
