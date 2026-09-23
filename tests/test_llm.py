"""LLM.chat()'s retry/backoff accounting.

Before this, a call that failed on a 429 a few times before succeeding looked
identical in the trace to one that succeeded on the first try: only the
successful attempt's call_s was ever recorded, and the sleeps between retries
vanished. That gap is what hid the RLM arm's real 400-600s latency cause
behind a guess (see EXPERIMENTS.md, "RLM arm and router").
"""
from __future__ import annotations

import httpx
import pytest

from backend import llm, trace


def _rate_limit_error() -> llm.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
    return llm.RateLimitError("rate limited", response=response, body=None)


class _FakeChatClient:
    """A stand-in for the OpenAI client: fails with a 429 `fail_times` times,
    then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.chat = self  # so `self.chat.completions.create(...)` works
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _rate_limit_error()
        return "ok"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


def test_chat_records_backoff_across_retries(monkeypatch):
    fake = _FakeChatClient(fail_times=2)
    monkeypatch.setattr(llm, "_chat_client", lambda: fake)

    test_llm = llm.LLM("test-model", rpm=1000, rpd=1000)
    with trace.capture() as recorded:
        resp = test_llm.chat([{"role": "user", "content": "hi"}])

    assert resp == "ok"
    assert fake.calls == 3
    assert len(recorded) == 1                 # one record per chat() call, not per attempt
    assert recorded[0]["attempts"] == 3
    assert recorded[0]["backoff_s"] > 0        # the two retry sleeps are no longer invisible


def test_chat_records_backoff_on_exhaustion(monkeypatch):
    fake = _FakeChatClient(fail_times=999)     # never succeeds
    monkeypatch.setattr(llm, "_chat_client", lambda: fake)

    test_llm = llm.LLM("test-model", rpm=1000, rpd=1000)
    with trace.capture() as recorded, pytest.raises(llm.QuotaExhausted):
        test_llm.chat([{"role": "user", "content": "hi"}])

    assert fake.calls == 5
    assert len(recorded) == 1
    assert recorded[0]["attempts"] == 5
    assert recorded[0]["exhausted"] is True
    assert recorded[0]["backoff_s"] > 0        # exhaustion still reports where the time went
