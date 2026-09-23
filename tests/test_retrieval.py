"""RLM arm and router tests.

The RLM itself is faked (its model calls go to Gemini), so no test hits the
network. The router is tested with faked embedding and RLM retrievers. The
sandbox tests exercise the real (sandboxed) REPL, since that is the mechanism
under test.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend import llm, store
from backend.ingest import Chunk
from backend.retrieval import rlm_arm, router
from backend.retrieval.base import Passage


def test_rlm_config_has_no_tools():
    """Neither the RLM nor its sub-calls get any tool, so email_summary cannot
    be reached from code running over untrusted document text."""
    kwargs = rlm_arm.rlm_kwargs()
    assert "custom_tools" not in kwargs
    assert "custom_sub_tools" not in kwargs
    assert kwargs["max_depth"] == 2
    assert kwargs["max_iterations"] == 15
    assert kwargs["environment"] == "recite-sandbox"
    assert kwargs["max_timeout"] == rlm_arm.MAX_TIMEOUT_S


def test_rlm_kwargs_names_the_rlm_model():
    """The arm is driven by llm.rlm_llm, a model chosen for this loop
    specifically — not llm.answer_llm, which never finalises an answer
    through it (see EXPERIMENTS.md)."""
    kwargs = rlm_arm.rlm_kwargs()
    assert kwargs["backend_kwargs"]["model_name"] == llm.rlm_llm.model


def test_strip_thought_removes_a_leaked_thought_block():
    raw = "<thought>internal reasoning the model should not show</thought>```repl\nprint(1)\n```"
    assert rlm_arm._strip_thought(raw) == "```repl\nprint(1)\n```"


def test_strip_thought_is_a_noop_without_one():
    raw = "```repl\nprint(1)\n```"
    assert rlm_arm._strip_thought(raw) == raw


def test_make_client_routes_through_rlm_llm_not_answer_llm(monkeypatch):
    """The RLM arm must use its own model, and never fall back to the answer
    model — they have different quotas and, per EXPERIMENTS.md, different
    ability to drive this loop at all."""
    calls = {"rlm": 0}

    def fake_rlm_chat(messages, **kw):
        calls["rlm"] += 1
        msg = SimpleNamespace(content="<thought>plan</thought>```repl\nprint(1)\n```")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def fail_if_called(*a, **kw):
        raise AssertionError("the RLM arm must not call the answer model")

    monkeypatch.setattr(llm.rlm_llm, "chat", fake_rlm_chat)
    monkeypatch.setattr(llm.answer_llm, "chat", fail_if_called)

    client = rlm_arm._make_client("gemma-4-26b-a4b-it")
    result = client.completion("hello")

    assert result == "```repl\nprint(1)\n```"
    assert calls == {"rlm": 1}


def _fake_rlm(response):
    class FakeRLM:
        def __init__(self):
            self.calls = []

        def completion(self, prompt, root_prompt=None):
            self.calls.append((prompt, root_prompt))
            return SimpleNamespace(response=response)

    return FakeRLM()


def _fake_rlm_timeout(partial_answer):
    from rlm import TimeoutExceededError

    class FakeRLM:
        def completion(self, prompt, root_prompt=None):
            raise TimeoutExceededError(elapsed=70.0, timeout=60.0, partial_answer=partial_answer)

    return FakeRLM()


def test_rlm_retriever_maps_citations_to_passages(monkeypatch):
    monkeypatch.setattr(store, "get_doc_texts", lambda: [
        {"doc_id": "d1", "name": "notes.pdf", "text": "[[page 3]]\nthe answer text"},
    ])
    monkeypatch.setattr(store, "get_chunks_by_page", lambda name, page: [
        Chunk(id="d1:0", doc_id="d1", doc_name=name, page=page, section=None, text="the answer text"),
    ])
    retriever = rlm_arm.RLMRetriever(rlm=_fake_rlm("The answer is X (notes.pdf, p. 3)."))
    passages = retriever.search("what is X?")
    assert [p.doc_name for p in passages] == ["notes.pdf"]
    assert [p.page for p in passages] == [3]


def test_rlm_retriever_ignores_unknown_documents(monkeypatch):
    monkeypatch.setattr(store, "get_doc_texts", lambda: [
        {"doc_id": "d1", "name": "notes.pdf", "text": "text"},
    ])
    monkeypatch.setattr(store, "get_chunks_by_page", lambda name, page: [])
    retriever = rlm_arm.RLMRetriever(rlm=_fake_rlm("Answer (elsewhere.pdf, p. 9)."))
    assert retriever.search("q") == []


def test_rlm_retriever_uses_partial_answer_on_timeout(monkeypatch):
    """MAX_TIMEOUT_S exceeded: use whatever citation the RLM had produced
    before rlm.core.rlm raised TimeoutExceededError, instead of crashing the
    turn or blocking it for however long the library would otherwise run."""
    monkeypatch.setattr(store, "get_doc_texts", lambda: [
        {"doc_id": "d1", "name": "notes.pdf", "text": "[[page 3]]\nthe answer text"},
    ])
    monkeypatch.setattr(store, "get_chunks_by_page", lambda name, page: [
        Chunk(id="d1:0", doc_id="d1", doc_name=name, page=page, section=None, text="the answer text"),
    ])
    retriever = rlm_arm.RLMRetriever(rlm=_fake_rlm_timeout("So far: X (notes.pdf, p. 3)."))
    passages = retriever.search("what is X?")
    assert [p.doc_name for p in passages] == ["notes.pdf"]


def test_rlm_retriever_empty_on_timeout_with_no_partial_answer(monkeypatch):
    monkeypatch.setattr(store, "get_doc_texts", lambda: [{"doc_id": "d1", "name": "notes.pdf", "text": "text"}])
    retriever = rlm_arm.RLMRetriever(rlm=_fake_rlm_timeout(None))
    assert retriever.search("q") == []


def test_cited_parses_multiple_pages():
    docs = [{"doc_id": "d1", "name": "notes.pdf", "text": ""}]
    answer = "The answer is X (notes.pdf, p. 3, 7). Also (other.pdf, p. 5)."
    assert rlm_arm._cited(answer, docs) == [("notes.pdf", 3), ("notes.pdf", 7)]


def _passages(specs):
    return [Passage(id=s["id"], doc_id="d", doc_name=s.get("doc", "notes.pdf"), page=1,
                    section=s.get("section"), text="t", score=s["score"]) for s in specs]


def test_router_escalates_on_weak_best_match():
    emb = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "p1", "score": 0.1}])})()
    rlm = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "r1", "doc": "rlm.pdf", "score": 0.0}])})()
    router_ = router.Router(embeddings=emb, rlm=rlm, score_threshold=0.5)
    assert [p.doc_name for p in router_.search("q")] == ["rlm.pdf"]


def test_router_escalates_on_scattered_sections():
    emb = type("R", (), {"search": lambda self, q, k=5: _passages(
        [{"id": f"p{i}", "score": 0.9, "section": f"s{i}"} for i in range(4)])})()
    rlm = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "r1", "doc": "rlm.pdf", "score": 0.0}])})()
    # scatter_sections pinned explicitly, not left to the SCATTER_SECTIONS
    # module default: that default reads SCATTER_SECTIONS from any local
    # .env, which prod tunes to 6 (EXPERIMENTS.md) — this test's 4-section
    # fixture would then silently stop triggering escalation.
    router_ = router.Router(embeddings=emb, rlm=rlm, score_threshold=0.5, scatter_sections=4)
    assert [p.doc_name for p in router_.search("q")] == ["rlm.pdf"]


def test_router_returns_embeddings_when_strong():
    emb = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "p1", "score": 0.9, "section": "s1"}])})()
    rlm = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "r1", "doc": "rlm.pdf", "score": 0.0}])})()
    router_ = router.Router(embeddings=emb, rlm=rlm, score_threshold=0.5)
    assert [p.doc_name for p in router_.search("q")] == ["notes.pdf"]


def test_router_falls_back_to_embeddings_when_rlm_empty():
    emb = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "p1", "score": 0.1}])})()
    rlm = type("R", (), {"search": lambda self, q, k=5: []})()
    router_ = router.Router(embeddings=emb, rlm=rlm, score_threshold=0.5)
    assert [p.id for p in router_.search("q")] == ["p1"]


def test_router_escalate_forces_rlm_regardless_of_triggers():
    """escalate() is trigger 3's entry point (wired into backend.agent's
    redraft step, not into search()): it must call the RLM arm even when
    neither the score nor the scatter trigger would fire on its own, and it
    must not fall back to embeddings hits — the answer loop, which has its
    own already-failed passages to fall back to, owns that decision."""
    emb = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "p1", "score": 0.99}])})()
    rlm = type("R", (), {"search": lambda self, q, k=5: _passages([{"id": "r1", "doc": "rlm.pdf", "score": 0.0}])})()
    router_ = router.Router(embeddings=emb, rlm=rlm)
    result = router_.escalate("q", reason="grounding check failed")
    assert [p.doc_name for p in result] == ["rlm.pdf"]


def test_router_escalate_returns_empty_when_rlm_empty():
    """No embeddings fallback inside escalate() itself — see the docstring
    above; an empty RLM result is returned as-is."""
    rlm = type("R", (), {"search": lambda self, q, k=5: []})()
    router_ = router.Router(embeddings=type("R", (), {"search": lambda self, q, k=5: []})(), rlm=rlm)
    assert router_.escalate("q") == []


def test_sandbox_blocks_env_read():
    """The attack the user cares about: code over untrusted text tries to read
    .env and put it in the answer. `open` is stripped, so the key never leaks.

    A fake secret is planted directly in the sandbox's own execution directory
    (LocalREPL.execute_code cd's into repl.temp_dir before running code), so
    this is self-contained rather than depending on a real GEMINI_API_KEY
    being set in the environment — which is unset in CI (ci.yml runs without
    it, by design) and would otherwise have skipped the real assertion."""
    repl = rlm_arm.make_sandboxed_repl()
    try:
        secret = "sk-fake-not-a-real-key-8f2c1e"
        (Path(repl.temp_dir) / ".env").write_text(f"GEMINI_API_KEY={secret}\n")
        result = repl.execute_code("answer['content'] = open('.env').read(); answer['ready'] = True")
        assert secret not in result.stdout and secret not in result.stderr
        assert result.final_answer is None          # open() raised before ready was set
        assert "NameError" in result.stderr
    finally:
        repl.cleanup()


def test_sandbox_blocks_import():
    repl = rlm_arm.make_sandboxed_repl()
    try:
        result = repl.execute_code("import os\nanswer['content'] = os.getcwd()")
        assert "Error" in result.stderr or "NameError" in result.stderr
        assert result.final_answer is None
    finally:
        repl.cleanup()
