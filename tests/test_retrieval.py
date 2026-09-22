"""RLM arm and router tests.

The RLM itself is faked (its model calls go to Gemini), so no test hits the
network. The router is tested with faked embedding and RLM retrievers. The
sandbox tests exercise the real (sandboxed) REPL, since that is the mechanism
under test.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

from backend import store
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


def _fake_rlm(response):
    class FakeRLM:
        def __init__(self):
            self.calls = []

        def completion(self, prompt, root_prompt=None):
            self.calls.append((prompt, root_prompt))
            return SimpleNamespace(response=response)

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
    router_ = router.Router(embeddings=emb, rlm=rlm, score_threshold=0.5)
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


def test_sandbox_blocks_env_read():
    """The attack the user cares about: code over untrusted text tries to read
    .env and put it in the answer. `open` is stripped, so the key never leaks."""
    repl = rlm_arm.make_sandboxed_repl()
    try:
        result = repl.execute_code("answer['content'] = open('.env').read(); answer['ready'] = True")
        key = os.getenv("GEMINI_API_KEY")
        assert key, "GEMINI_API_KEY must be set for this test"
        assert key not in result.stdout and key not in result.stderr
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
