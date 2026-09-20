"""Grounding-check tests. The critic is faked, so these exercise the check and
retry logic without any model call."""
from __future__ import annotations

import json

from backend.agent import answer
from backend.grounding import REFUSAL
from tests.fake_llm import FakeLLM, text
from tests.helpers import FakeRetriever


def _verdict(claim: str, verdict: str) -> str:
    return json.dumps({"claims": [{"claim": claim, "verdict": verdict, "passage_id": None}]})


def test_supported_answer_passes():
    draft = "A latch is level-triggered."
    model = FakeLLM([text(draft)])
    critic = FakeLLM([text(_verdict(draft, "SUPPORTED"))])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=critic)
    assert result.text == draft
    assert result.retried is False and result.grounding == "supported"
    assert len(model.calls) == 1 and len(critic.calls) == 1


def test_retry_fixes_answer():
    draft1 = "The moon is made of green cheese."
    draft2 = "A latch is level-triggered."
    model = FakeLLM([text(draft1), text(draft2)])
    critic = FakeLLM([
        text(_verdict(draft1, "UNSUPPORTED")),
        text(_verdict(draft2, "SUPPORTED")),
    ])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=critic)
    assert result.text == draft2
    assert result.retried is True and result.grounding == "supported"
    assert len(model.calls) == 2 and len(critic.calls) == 2


def test_short_answer_not_refused():
    model = FakeLLM([text("Yes.")])
    critic = FakeLLM([])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=critic)
    assert result.text == "Yes."
    assert result.retried is False and result.grounding == "supported"
    assert len(critic.calls) == 0


def test_honest_refusal_not_retried():
    model = FakeLLM([text(REFUSAL)])
    critic = FakeLLM([])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=critic)
    assert result.text == REFUSAL
    assert result.retried is False
    assert len(model.calls) == 1 and len(critic.calls) == 0
