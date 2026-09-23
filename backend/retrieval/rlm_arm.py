"""The recursive-language-model retrieval arm.

RLMs (Zhang, Kraska, Khattab 2025) put the whole document in a REPL as a
variable and let the model write code to navigate it, calling itself recursively
on the slices it selects. This arm adapts the open-source ``rlms`` library to
the same Gemini endpoint, so uploaded documents work immediately and citations
still map back to pages.

Two design choices matter:

- **No tools.** Neither the RLM nor its sub-calls get any tool, so the mutating
  ``email_summary`` cannot be reached from code running over untrusted document
  text.
- **Every model call routes through ``llm.answer_llm``.** The ``rlms`` client is
  replaced with one that delegates to the rate-limited client, so the RLM shares
  the free-tier daily budget and the existing trace capture still splits model
  time from limiter waits.

The REPL is a **partial sandbox**: ``open``, ``__import__`` and the common
escape primitives (``getattr``, ``type``, ``object``, ``super``, …) are stripped
from the builtins and the document is injected directly rather than read back
with ``open``. This stops the naive ``open('.env').read()`` and ``import os``,
but Python-level escapes such as ``().__class__.__subclasses__()`` still reach
the host, so it is not a hard sandbox. The Docker environment would be one, but
it needs a writable mount and a host proxy, so ``--network none`` is not
available through the library as shipped.
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
SANDBOX_ENV_NAME = "recite-sandbox"
MAX_DEPTH = 2
MAX_ITERATIONS = 15

# Builtins removed from the REPL namespace: the file and import primitives, plus
# the common introspection primitives used to reach them (object.__subclasses__).
_STRIP_BUILTINS = frozenset({
    "open", "__import__", "getattr", "setattr", "delattr", "vars", "dir",
    "type", "object", "super", "property", "staticmethod", "classmethod",
})

# "(name, p. 14, 17)" — multi-page aware, unlike the single-page CITE regex.
_CITE_MULTI = re.compile(r"\(([^,()]+),\s*p\.\s*([\d,\s]+)\)")

# Some free-tier models (gemma-4-26b-a4b-it) wrap their real output in a
# <thought>...</thought> block before the actual response. The rlms library's
# code-block parser looks for a ```repl fence anywhere in the text, so a
# leaked thought block ahead of it does not by itself break parsing — but it
# wastes tokens and adds latency on every call, so it is stripped before the
# text leaves this client. No-op for models that don't leak one.
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>\s*", re.DOTALL)


def _strip_thought(text: str) -> str:
    return _THOUGHT_RE.sub("", text).strip()


def _doc_key(name: str) -> str:
    return Path(name).stem.lower()


def _make_client(model_name: str):
    """A duck-typed BaseLM whose calls go through llm.rlm_llm — a model chosen
    for this arm specifically, not the answer model. gemini-3.1-flash-lite
    (500/day) plans forever and never finalises a grounded answer through the
    rlms code-execution loop; gemma-4-26b-a4b-it (14,400/day) does, in a live
    test, once _strip_thought() removes its <thought> leakage. See
    EXPERIMENTS.md for the run that established this."""
    from rlm.core.types import ModelUsageSummary, UsageSummary

    class RateLimitedGemini:
        def __init__(self) -> None:
            self.model_name = model_name

        def completion(self, prompt, model=None) -> str:
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            resp = llm.rlm_llm.chat(messages)
            return _strip_thought(resp.choices[0].message.content or "")

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


def _sandboxed_repl_class():
    """A LocalREPL with dangerous builtins stripped and the context injected
    directly (no ``open``), so code over untrusted text cannot read ``.env`` or
    import modules. This is a partial sandbox, not a hard one."""
    from rlm.environments.base_env import extract_tool_value
    from rlm.environments.local_repl import _SAFE_BUILTINS, LocalREPL, _AnswerDict

    restricted = {k: v for k, v in _SAFE_BUILTINS.items() if k not in _STRIP_BUILTINS}

    class SandboxedREPL(LocalREPL):
        def setup(self):
            self.globals = {"__builtins__": restricted.copy(), "__name__": "__main__"}
            self.locals = {}
            self._pending_llm_calls = []
            self._last_final_answer = None
            self.globals["SHOW_VARS"] = self._show_vars
            self.globals["llm_query"] = self._llm_query
            self.globals["llm_query_batched"] = self._llm_query_batched
            self.globals["rlm_query"] = self._rlm_query
            self.globals["rlm_query_batched"] = self._rlm_query_batched
            self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)
            for name, entry in (self.custom_tools or {}).items():
                value = extract_tool_value(entry)
                if callable(value):
                    self.globals[name] = value
                else:
                    self.locals[name] = value

        def add_context(self, context_payload, context_index=None):
            if context_index is None:
                context_index = self._context_count
            var_name = f"context_{context_index}"
            self.locals[var_name] = context_payload
            if context_index == 0:
                self.locals["context"] = context_payload
            self._context_count = max(self._context_count, context_index + 1)
            return context_index

    return SandboxedREPL


def make_sandboxed_repl(**kwargs):
    """Build the sandboxed REPL directly, for tests and the attack case."""
    return _sandboxed_repl_class()(**kwargs)


def _install_environment() -> None:
    """Route the custom environment name to the sandboxed REPL, once."""
    import rlm.core.rlm as rlm_core

    if getattr(rlm_core, "_recite_env_installed", False):
        return
    repl_class = _sandboxed_repl_class()
    original = rlm_core.get_environment

    def get_environment(environment, environment_kwargs):
        if environment == SANDBOX_ENV_NAME:
            return repl_class(**environment_kwargs)
        return original(environment, environment_kwargs)

    rlm_core.get_environment = get_environment
    rlm_core._recite_env_installed = True


def rlm_kwargs() -> dict:
    """The RLM configuration. custom_tools is deliberately absent — neither the
    RLM nor its sub-calls get any tool, so email_summary cannot be reached from
    code running over untrusted document text."""
    return {
        "backend": BACKEND_NAME,
        "backend_kwargs": {"model_name": llm.rlm_llm.model},
        "environment": SANDBOX_ENV_NAME,
        "max_depth": MAX_DEPTH,
        "max_iterations": MAX_ITERATIONS,
    }


def make_rlm():
    """Build the RLM configured for this project."""
    from rlm import RLM

    _install_backend()
    _install_environment()
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
        # Every model call the loop makes (root and recursive) goes through
        # llm.rlm_llm.chat(), which records wait_s/backoff_s/call_s per call —
        # capture() collects those so a 400-600s search can be attributed to
        # limiter waits, 429-retry backoff, or genuine model time instead of
        # guessed at. asyncio.to_thread (used by acompletion for recursive
        # sub-calls) copies the current context into its worker thread, so
        # this still sees calls made off the main thread.
        with trace.capture() as calls:
            completion = rlm.completion(_build_context(docs), root_prompt=_root_prompt(query))
        t = trace.timings(calls)
        trace.record(op="rlm", wall_s=round(time.monotonic() - started, 3),
                     wait_s=t["wait_s"], backoff_s=t["backoff_s"], model_s=t["model_s"],
                     calls=t["calls"])
        return _answer_to_passages(completion.response, docs)[:k]
