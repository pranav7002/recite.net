"""The router: try embeddings first, escalate to the RLM arm only when a fixed
rule says the cheap result looks weak, and fall back to embeddings when the RLM
arm comes back empty (so escalation never makes an answer worse).

Three triggers:

1. The top embedding score is below SCORE_THRESHOLD — nothing matched well.
2. The hits are spread across SCATTER_SECTIONS or more sections — the question
   is probably about structure, not one fact.
3. The grounding check fails on the embeddings arm's passages — that failure
   is itself evidence they were insufficient. Wired into ``backend.agent``'s
   redraft step (``_escalate_after_grounding_failure``), via ``escalate()``
   below, rather than into ``search()``: by the time a grounding check has
   run, the answer loop already has a drafted answer and knows *which*
   passages failed to support it, neither of which ``search()`` sees.

Rules, not a model, because the signals already exist and a routing call would
cost the latency the router exists to save. Both SCORE_THRESHOLD and
SCATTER_SECTIONS must be tuned on the ``tune`` split: every turn logs the top
score and the section count, and the escalation reason, so the distribution is
measurable before the thresholds are trusted.
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
                 score_threshold: float = SCORE_THRESHOLD,
                 scatter_sections: int = SCATTER_SECTIONS) -> None:
        self.embeddings = embeddings or EmbeddingRetriever()
        self.rlm = rlm or RLMRetriever()
        self.score_threshold = score_threshold
        self.scatter_sections = scatter_sections

    def search(self, query: str, k: int = 5) -> list[Passage]:
        hits = self.embeddings.search(query, k=k)
        top = hits[0].score if hits else None
        sections = len({h.section for h in hits if h.section})
        trace.record(op="router", top_score=round(top, 4) if top is not None else None,
                     sections=sections)

        if hits and top < self.score_threshold:
            return self.escalate(query, k, "weak best match") or hits
        if sections >= self.scatter_sections:
            return self.escalate(query, k, "scattered sections") or hits
        return hits

    def escalate(self, query: str, k: int = 5, reason: str = "grounding check failed") -> list[Passage]:
        """Force an RLM search regardless of the score/section triggers. Used
        internally by search()'s two triggers, and directly by the answer
        loop for trigger 3 (a grounding-check failure). The caller decides the
        empty-result fallback: search()'s triggers fall back to the embeddings
        hits already in hand; the answer loop keeps its existing passages."""
        trace.record(op="router", escalated=True, reason=reason)
        return self.rlm.search(query, k=k)
