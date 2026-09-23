"""Quiz mode tests. The question model and judge are faked, so no test calls Gemini."""
from __future__ import annotations

import random

import pytest

from backend import grounding, quiz, store
from backend.ingest import Chunk
from tests.fake_llm import FakeLLM, text
from tests.helpers import passage


def _chunk(cid, text="The mean is fifty."):
    return Chunk(id=cid, doc_id="d", doc_name="notes.pdf", page=1, section=None, text=text)


def test_pick_passage_weights_missed():
    rng = random.Random(0)
    chunks = [_chunk("c0"), _chunk("c1")]
    picks = [quiz.pick_passage(chunks, {"c1"}, rng=rng).id for _ in range(1000)]
    assert picks.count("c1") > picks.count("c0")   # 3x weight on the missed chunk


def test_leaks_answer_detects_leak():
    assert quiz.leaks_answer("Is the z-score the number of standard deviations from the mean?",
                             "number of standard deviations from the mean")
    assert not quiz.leaks_answer("What is the average?", "fifty")


def test_leaks_answer_allows_questions_that_reuse_passage_words():
    # The old passage-overlap check flagged this good question as a leak.
    assert not quiz.leaks_answer("How is the z-score of a value computed?",
                                 "subtract the mean and divide by the standard deviation")


def test_quizzable_skips_title_slides():
    long_text = "The variance is the average squared deviation from the mean. " * 5
    chunks = [_chunk("title", "Lecture 4"), _chunk("body", long_text), _chunk("end", "Thank you")]
    assert [c.id for c in quiz.quizzable(chunks)] == ["body"]


def test_quizzable_falls_back_when_nothing_qualifies():
    chunks = [_chunk("a", "short"), _chunk("b", "also short")]
    assert quiz.quizzable(chunks) == chunks


def test_leaky_question_is_regenerated(monkeypatch):
    monkeypatch.setattr(store, "get_chunks", lambda unit: [_chunk("c0")])
    monkeypatch.setattr(store, "quiz_missed_chunk_ids", lambda: set())
    llm = FakeLLM([
        text('{"question": "Is the mean fifty?", "answer": "the mean is fifty"}'),
        text('{"question": "What is the average of the marks?", "answer": "fifty"}'),
    ])
    item = quiz.next_question("d", llm=llm)
    assert item.question == "What is the average of the marks?"
    assert len(llm.calls) == 2


def test_next_question_returns_item(monkeypatch):
    monkeypatch.setattr(store, "get_chunks", lambda unit: [_chunk("c0")])
    monkeypatch.setattr(store, "quiz_missed_chunk_ids", lambda: set())
    item = quiz.next_question("d", llm=FakeLLM([text("What is the average?")]))
    assert item.chunk_id == "c0"
    assert item.question == "What is the average?"
    assert item.passage.doc_name == "notes.pdf"


def test_grade_incorrect_records_miss(monkeypatch):
    missed = []
    monkeypatch.setattr(store, "record_quiz_miss", lambda cid: missed.append(cid))
    monkeypatch.setattr(store, "clear_quiz_miss", lambda cid: None)
    monkeypatch.setattr(grounding, "judge",
                        lambda answer, passages, llm_small, question=None:
                        grounding.Grade(grade="incorrect", missing="wrong"))
    item = quiz.QuizItem(chunk_id="c0", passage=passage("t"), question="q?")
    result = quiz.grade(item, "wrong", llm_small=object())
    assert result.grade == "incorrect"
    assert missed == ["c0"]


def test_grade_correct_clears_miss(monkeypatch):
    missed, cleared = [], []
    monkeypatch.setattr(store, "record_quiz_miss", lambda cid: missed.append(cid))
    monkeypatch.setattr(store, "clear_quiz_miss", lambda cid: cleared.append(cid))
    monkeypatch.setattr(grounding, "judge",
                        lambda answer, passages, llm_small, question=None:
                        grounding.Grade(grade="correct"))
    item = quiz.QuizItem(chunk_id="c0", passage=passage("t"), question="q?")
    assert quiz.grade(item, "right", llm_small=object()).grade == "correct"
    assert missed == []
    assert cleared == ["c0"]


def test_miss_is_recorded_then_cleared_in_db(db):
    # Real store round trip inside the rolled-back test transaction.
    import numpy as np

    from backend.ingest import Chunk as C
    chunk = C(id="docq:0", doc_id="docq", doc_name="q.pdf", page=1, section=None,
              text="The mean is fifty.")
    store.add(doc_id="docq", name="q.pdf", full_text="[[page 1]]\nThe mean is fifty.",
              chunks=[chunk], vectors=np.full((1, 768), 0.01, dtype=np.float32))
    store.record_quiz_miss("docq:0")
    assert "docq:0" in store.quiz_missed_chunk_ids()
    store.clear_quiz_miss("docq:0")
    assert "docq:0" not in store.quiz_missed_chunk_ids()


def test_item_from_unknown_raises(monkeypatch):
    monkeypatch.setattr(store, "get_chunk", lambda cid: None)
    with pytest.raises(quiz.UnknownUnit):
        quiz.item_from("nope")


def test_grounding_judge_normalizes_grade():
    llm = FakeLLM([text('{"grade": "partially correct", "missing": "forgot the formula"}')])
    result = grounding.judge("the mean is 50", [passage("the mean is 50")], llm)
    assert result.grade == "partial"
    assert result.missing == "forgot the formula"


def test_grounding_judge_retries_on_bad_json():
    llm = FakeLLM([text("not json"), text('{"grade": "correct", "missing": ""}')])
    result = grounding.judge("the mean is 50", [passage("the mean is 50")], llm)
    assert result.grade == "correct"
