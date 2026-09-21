"""The recursive-language-model retrieval arm.

RLMs (Zhang, Kraska, Khattab 2025) put the whole document in a sandboxed REPL
as a variable and let the model write code to navigate it, calling itself
recursively on the slices it selects. This arm adapts the open-source ``rlms``
library to the same Gemini endpoint, so uploaded documents work immediately and
citations still map back to pages.

Two design choices matter:

- **No tools.** Neither the RLM nor its sub-calls get any tool, so the mutating
  ``email_summary`` cannot be reached from code running over untrusted document
  text — this breaks the "lethal trifecta" by construction.
- **Every model call routes through ``llm.answer_llm``.** The ``rlms`` client is
  replaced with one that delegates to the rate-limited client, so the RLM shares
  the free-tier daily budget and the existing trace capture still splits model
  time from limiter waits (the two latency columns the router reports).

The ``local`` environment blocks ``eval``/``exec``/``compile`` but still exposes
``__import__`` and ``open``, so it is not a hard sandbox against a determined
escape; a production build would use the Docker environment.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from backend import llm, store, trace
from backend.citations import CITE
from backend.retrieval.base import Passage

BACKEND_NAME = "recite-gemini"
MAX_DEPTH = 2
MAX_ITERATIONS = 15

# "(name, p. 14, 17)" — multi-page aware, unlike the single-page CITE regex.
_CITE_MULTI = re.compile(r"\(([^,()]+),\s*p\.\s*([\d,\s]+)\)")


def _doc_key(name: str) -> str:
    return Path(name).stem.lower()


def _make_client(model_name: str):
    """A duck-typed BaseLM whose calls go through llm.answer_llm."""
    from rlm.core.types import ModelUsageSummary, UsageSummary

    class RateLimitedGemini:
        def __init__(self) -> None:
            self.model_name = model_name

        def completion(self, prompt, model=None) -> str:
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            resp = llm.answer_llm.chat(messages)
            return resp.choices[0].message.content or ""

        async def acompletion(self, prompt, model=None) -> str:
            return await asyncio.to_thread(self.completion, prompt, model)

        def get_usage_summary(self):
            return UsageSummary(model_usage_summaries={model_name: self.get_last_usage()})

        def get_last_usage(self):
            return ModelUsageSummary(total_calls=1, total_input_tokens=0, total_output_tokens=0)

    return RateLimitedGemini()


def _install_backend() -> None:
    """Route the custom backend name to the rate-limited client, once."""
    import rlm.core.rlm as rlm_core

    if getattr(rlm_core, "_recite_backend_installed", False):
        return
    original = rlm_core.get_client

    def get_client(backend, backend_kwargs):
        if backend == BACKEND_NAME:
            return _make_client(backend_kwargs.get("model_name"))
        return original(backend, backend_kwargs)

    rlm_core.get_client = get_client
    rlm_core._recite_backend_installed = True


def rlm_kwargs() -> dict:
    """The RLM configuration. custom_tools is deliberately absent — neither the
    RLM nor its sub-calls get any tool, so email_summary cannot be reached from
    code running over untrusted document text."""
    return {
        "backend": BACKEND_NAME,
        "backend_kwargs": {"model_name": llm.answer_llm.model},
        "environment": "local",
        "max_depth": MAX_DEPTH,
        "max_iterations": MAX_ITERATIONS,
    }


def make_rlm():
    """Build the RLM configured for this project."""
    from rlm import RLM

    _install_backend()
    return RLM(**rlm_kwargs())


def _build_context(docs: list[dict]) -> str:
    return "\n\n".join(f"[[document: {d['name']}]]\n{d['text']}" for d in docs)


def _root_prompt(query: str) -> str:
    return (
        f"{query}\n\n"
        "Answer using only the document context. Cite the pages you relied on "
        "as (name, p. N), where name is the exact document name shown in a "
        "[[document: ...]] marker."
    )


def _cited(answer: str, docs: list[dict]) -> list[tuple[str, int]]:
    """Map the RLM's citations to (document name, page) pairs that exist."""
    names = {_doc_key(d["name"]): d["name"] for d in docs}
    out: list[tuple[str, int]] = []
    for name, pages in _CITE_MULTI.findall(answer):
        if _doc_key(name) not in names:
            continue
        for page in pages.split(","):
            page = page.strip()
            if page.isdigit():
                out.append((names[_doc_key(name)], int(page)))
    # Single-page citations the multi-page regex also matches, but fall back to
    # the canonical CITE regex for a bare "(name, p. N)" with no trailing pages.
    if not out:
        for name, page in CITE.findall(answer):
            if _doc_key(name) in names:
                out.append((names[_doc_key(name)], int(page)))
    return out


def _answer_to_passages(answer: str, docs: list[dict]) -> list[Passage]:
    passages: list[Passage] = []
    seen: set[tuple[str, int]] = set()
    for name, page in _cited(answer, docs):
        if (name, page) in seen:
            continue
        seen.add((name, page))
        for chunk in store.get_chunks_by_page(name, page):
            passages.append(Passage(id=chunk.id, doc_id=chunk.doc_id, doc_name=chunk.doc_name,
                                    page=chunk.page, section=chunk.section, text=chunk.text, score=0.0))
    return passages


class RLMRetriever:
    """A Retriever backed by the RLM arm. rlm is injectable for tests."""

    def __init__(self, rlm=None) -> None:
        self._rlm = rlm

    def search(self, query: str, k: int = 5) -> list[Passage]:
        docs = store.get_doc_texts()
        if not docs:
            return []
        rlm = self._rlm or make_rlm()
        started = time.monotonic()
        completion = rlm.completion(_build_context(docs), root_prompt=_root_prompt(query))
        trace.record(op="rlm", wall_s=round(time.monotonic() - started, 3))
        return _answer_to_passages(completion.response, docs)[:k]
