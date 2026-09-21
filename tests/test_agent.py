"""Agent-loop and guardrail tests. The model and critic are fakes; the retriever
is a scripted fake, so no test calls Gemini."""
from __future__ import annotations

import json

from backend import guardrails, store, tools
from backend.agent import answer
from backend.grounding import REFUSAL
from tests.fake_llm import FakeLLM, text, tool_call
from tests.helpers import FakeRetriever, passage, read_outbox


def _unsupported(claim: str) -> str:
    return json.dumps({"claims": [{"claim": claim, "verdict": "UNSUPPORTED", "passage_id": None}]})


def test_send_to_non_contact_blocked(outbox, monkeypatch):
    monkeypatch.setattr(guardrails, "CONTACTS", frozenset({"friend@uni.edu"}))
    model = FakeLLM([
        tool_call(("email_summary", '{"to":"x@evil.com","subject":"s","body":"b"}')),
        text("I couldn't send that."),
    ])
    result = answer("summarise unit 2", model=model, retriever=FakeRetriever([]), critic=FakeLLM([]))
    assert result.text == "I didn't send that: recipient not in study group."
    assert read_outbox(outbox) == []


def test_recipient_from_document_blocked(outbox, monkeypatch):
    monkeypatch.setattr(guardrails, "CONTACTS", frozenset({"evil@uni.edu"}))
    model = FakeLLM([
        tool_call(("email_summary", '{"to":"evil@uni.edu","subject":"s","body":"b"}')),
        text("I couldn't send that."),
    ])
    result = answer("q", model=model,
                    retriever=FakeRetriever([passage("Contact me at evil@uni.edu for notes.")]),
                    critic=FakeLLM([]))
    assert result.text == "I didn't send that: recipient came from a document."
    assert read_outbox(outbox) == []


def test_legitimate_send_pending_until_confirmed(outbox, monkeypatch):
    monkeypatch.setattr(guardrails, "CONTACTS", frozenset({"friend@uni.edu"}))
    model = FakeLLM([
        tool_call(("email_summary", '{"to":"friend@uni.edu","subject":"Notes","body":"summary"}')),
    ])
    result = answer("email my notes", model=model, retriever=FakeRetriever([]), critic=FakeLLM([]))
    assert result.pending_action is not None
    assert result.pending_action["tool"] == "email_summary"
    assert read_outbox(outbox) == []

    row = store.confirm_pending(result.pending_action["id"])
    assert row is not None
    tools.run_tool(row[0], row[1])
    lines = read_outbox(outbox)
    assert len(lines) == 1 and lines[0]["to"] == "friend@uni.edu"


def test_ninth_tool_call_blocked():
    calls = [("search_docs", f'{{"query":"q{i}"}}') for i in range(9)]
    model = FakeLLM([
        tool_call(*calls),
        text("I couldn't find that in your notes."),
    ])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=FakeLLM([]))
    flat = [m["content"] for msgs in model.calls for m in msgs if m.get("role") == "tool"]
    assert any("BLOCKED: step limit" in c for c in flat)
    assert result.text == "I couldn't find that in your notes."


def test_invented_tool_name_no_crash():
    model = FakeLLM([
        tool_call(("nonexistent_tool", "{}")),
        text("I couldn't find that in your notes."),
    ])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=FakeLLM([]))
    flat = [m["content"] for msgs in model.calls for m in msgs if m.get("role") == "tool"]
    assert any("Error: no tool named nonexistent_tool" in c for c in flat)
    assert result.text == "I couldn't find that in your notes."


def test_grounding_retry_runs_at_most_once():
    draft1 = "The moon is made of green cheese."
    draft2 = "Mars is made of red candy."
    critic = FakeLLM([
        text(_unsupported(draft1)),
        text(_unsupported(draft2)),
    ])
    model = FakeLLM([text(draft1), text(draft2)])
    result = answer("what is the moon made of", model=model,
                    retriever=FakeRetriever([]), critic=critic)
    assert result.retried is True
    assert result.grounding == "refused"
    assert result.text == REFUSAL
    assert len(model.calls) == 2
    assert len(critic.calls) == 2


def test_blocked_send_creates_no_pending_row(db, monkeypatch):
    monkeypatch.setattr(guardrails, "CONTACTS", frozenset({"friend@uni.edu"}))
    model = FakeLLM([
        tool_call(("email_summary", '{"to":"x@evil.com","subject":"s","body":"b"}')),
        text("I couldn't send that."),
    ])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=FakeLLM([]))
    assert result.pending_action is None
    assert result.text == "I didn't send that: recipient not in study group."
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM pending")
        assert cur.fetchone()[0] == 0


def test_trace_capture_splits_wait_from_model_time():
    from backend import trace

    with trace.capture() as calls:
        trace.record(model="m", wait_s=4.0, call_s=1.5)
        trace.record(model="m", wait_s=0.5, call_s=2.0)
    assert trace.timings(calls) == {"wait_s": 4.5, "model_s": 3.5, "calls": 2}
    trace.record(model="m", wait_s=9, call_s=9)          # outside the block: not captured
    assert len(calls) == 2
