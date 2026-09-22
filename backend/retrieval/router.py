"""The router: try embeddings first, escalate to the RLM arm only when a fixed
rule says the cheap result looks weak, and fall back to embeddings when the RLM
arm comes back empty (so escalation never makes an answer worse).

Two triggers:

1. The top embedding score is below SCORE_THRESHOLD — nothing matched well.
2. The hits are spread across SCATTER_SECTIONS or more sections — the question
   is probably about structure, not one fact.

Rules, not a model, because the signals already exist and a routing call would
cost the latency the router exists to save. Both SCORE_THRESHOLD and
SCATTER_SECTIONS must be tuned on the ``tune`` split: every turn logs the top
score and the section count, and the escalation reason, so the distribution is
measurable before the thresholds are trusted. Trigger 3 from the architecture
guide (escalate after a grounding-check failure) is not wired into the answer
loop yet.
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
SCATTER_SECTIONS = int(os.getenv("SCATTER_SECTIONS", "4"))


class Router:
    """A Retriever that escalates from embeddings to RLM on weak results."""

    def __init__(self, embeddings: Retriever | None = None, rlm: Retriever | None = None,
                 score_threshold: float = SCORE_THRESHOLD) -> None:
        self.embeddings = embeddings or EmbeddingRetriever()
        self.rlm = rlm or RLMRetriever()
        self.score_threshold = score_threshold

    def search(self, query: str, k: int = 5) -> list[Passage]:
        hits = self.embeddings.search(query, k=k)
        top = hits[0].score if hits else None
        sections = len({h.section for h in hits if h.section})
        trace.record(op="router", top_score=round(top, 4) if top is not None else None,
                     sections=sections)

        def escalate(reason: str) -> list[Passage]:
            trace.record(op="router", escalated=True, reason=reason)
            rlm_hits = self.rlm.search(query, k=k)
            return rlm_hits or hits   # empty RLM must not make the answer worse

        if hits and top < self.score_threshold:
            return escalate("weak best match")
        if sections >= SCATTER_SECTIONS:
            return escalate("scattered sections")
        return hits
