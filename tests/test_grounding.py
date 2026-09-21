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


# ---- regression tests for the claim-count mismatch that caused false refusals ----

def test_citation_page_number_does_not_split_a_claim():
    from backend.grounding import split_sentences

    draft = ('Leakage makes performance look better than it is (lecture4.pdf, p. 16). '
             'Overfitting is a large train/test gap ("DS-2 (1).pdf", p. 22), so leakage hides it.')
    claims = split_sentences(draft)
    assert len(claims) == 2
    assert claims[0].endswith("p. 16).") and "p. 22)" in claims[1]


def test_verdicts_matched_by_id_not_position():
    from backend.grounding import check

    claims_text = "The mean is 60 here. The median is 50 here."
    reply = json.dumps({"claims": [{"id": 2, "verdict": "UNSUPPORTED"}, {"id": 1, "verdict": "SUPPORTED"}]})
    result = check(claims_text, [], FakeLLM([text(reply)]))
    assert [c.verdict for c in result.claims] == ["SUPPORTED", "UNSUPPORTED"]
    assert result.failing == ["The median is 50 here."]
    assert result.parse_failed is False


def test_missing_claim_id_fails_closed_after_retry():
    from backend.grounding import check

    reply = json.dumps({"claims": [{"id": 1, "verdict": "SUPPORTED"}]})    # claim 2 missing
    critic = FakeLLM([text(reply), text(reply)])
    result = check("The mean is 60 here. The median is 50 here.", [], critic)
    assert result.parse_failed is True and not result.all_supported
    assert len(critic.calls) == 2                                          # asked once more first


def test_fenced_json_with_ids_parses():
    from backend.grounding import check

    reply = '```json\n{"claims": [{"id": 1, "verdict": "SUPPORTED", "passage_id": "a:1"}]}\n```'
    result = check("The mean is 60 here.", [], FakeLLM([text(reply)]))
    assert result.all_supported and result.parse_failed is False


# ---- claim splitting: citations and short claims ----

def test_citation_after_full_stop_stays_with_its_sentence():
    from backend.grounding import split_sentences

    claims = split_sentences("The mean is 50 here. (DAI-101_Lecture_3_Data_Types.pdf, p. 16)")
    assert claims == ["The mean is 50 here. (DAI-101_Lecture_3_Data_Types.pdf, p. 16)"]


def test_short_factual_claim_is_checked_not_dropped():
    from backend.grounding import split_sentences

    assert split_sentences("The median is 60.") == ["The median is 60."]      # 17 chars
    assert split_sentences("The mean is 50. The median is 60.") == ["The mean is 50.", "The median is 60."]


def test_short_wrong_fact_reaches_the_critic():
    critic = FakeLLM([text(_verdict("The median is 99.", "UNSUPPORTED")), text(_verdict("The median is 60.", "SUPPORTED"))])
    model = FakeLLM([text("The median is 99."), text("The median is 60.")])
    result = answer("q", model=model, retriever=FakeRetriever([]), critic=critic)
    assert len(critic.calls) == 2                       # it was checked, and the retry was used
    assert result.text == "The median is 60." and result.retried is True


def test_filler_is_merged_into_a_neighbour_not_checked_alone():
    from backend.grounding import split_sentences

    assert split_sentences("Yes. The median is 60 here.") == ["Yes. The median is 60 here."]
    assert split_sentences("The median is 60 here. Here's why.") == ["The median is 60 here. Here's why."]
    assert split_sentences("Yes.") == []                                       # nothing to check
