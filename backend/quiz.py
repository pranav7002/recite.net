"""Quiz mode: a fixed workflow, no agent loop.

Pick a passage (in code, weighting missed ones higher), ask a model to write a
question whose answer is stated in it, grade the spoken answer against the same
passage with the grounding checker, and record a miss so the next pick favours
the passages the student is weak on.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from backend import grounding, store
from backend.llm import answer_llm, check_llm
from backend.retrieval.base import Passage

QUESTION_PROMPT = """\
Write one short-answer study question whose answer is stated in the passage. The \
question must be answerable from the passage alone and must not contain or \
restate the answer. Return one JSON object: {"question": "...", "answer": "..."} \
where answer is the short answer taken from the passage."""

REGENERATE_PROMPT = """\
The previous question gave its own answer away. Write a different short-answer \
study question whose answer is stated in the passage, and make sure none of the \
answer's key words appear in the question. Return one JSON object: \
{"question": "...", "answer": "..."}"""

# Title slides, agenda slides and "Thank you" pages have nothing to quiz on.
MIN_QUIZ_CHARS = 200

MISS_WEIGHT = 3.0

_WORD = re.compile(r"[a-z0-9]{4,}")
_STOP = {
    "this", "that", "these", "those", "with", "about", "which", "what", "when",
    "where", "from", "have", "they", "them", "their", "there", "would", "could",
    "should", "because", "after", "before", "between", "answer", "question",
    "passage", "following", "above", "below", "stated", "according", "explain",
    "describe", "give", "name", "list", "why", "how", "does",
}


class UnknownUnit(Exception):
    pass


@dataclass
class QuizItem:
    chunk_id: str
    passage: Passage
    question: str
    expected: str = ""          # the model's short answer, used only for the leak check


def pick_passage(chunks: list, missed_ids: set[str], rng: random.Random = random) -> object:
    """Pick a chunk in code, weighting missed chunks higher (MISS_WEIGHT x)."""
    weights = [MISS_WEIGHT if c.id in missed_ids else 1.0 for c in chunks]
    return rng.choices(chunks, weights=weights, k=1)[0]


def _to_passage(chunk) -> Passage:
    return Passage(id=chunk.id, doc_id=chunk.doc_id, doc_name=chunk.doc_name,
                   page=chunk.page, section=chunk.section, text=chunk.text, score=0.0)


def _significant_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def leaks_answer(question: str, answer: str) -> bool:
    """True when the question gives its own answer away: most of the answer's
    significant words already appear in the question. Comparing against the
    model's stated answer (not the whole passage) keeps ordinary questions,
    which naturally reuse the passage's vocabulary, from being flagged."""
    a = _significant_words(answer)
    if not a:
        return False
    return len(a & _significant_words(question)) / len(a) >= 0.6


def _ask_question(chunk, llm, system_prompt: str) -> tuple[str, str]:
    """Returns (question, expected answer). A reply that is not JSON is taken
    as a bare question with no stated answer, so the leak check is skipped."""
    passage = f'<untrusted_content doc="{chunk.doc_name}" page="{chunk.page}">\n{chunk.text}\n</untrusted_content>'
    resp = llm.chat([{"role": "system", "content": system_prompt},
                     {"role": "user", "content": f"Passage:\n{passage}"}], temperature=0.4)
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(grounding._extract_json(raw))
        question = str(data.get("question", "")).strip()
        return (question, str(data.get("answer", "")).strip()) if question else (raw, "")
    except (ValueError, AttributeError):
        return raw, ""


def generate_question(chunk, llm=answer_llm) -> tuple[str, str]:
    question, expected = _ask_question(chunk, llm, QUESTION_PROMPT)
    if leaks_answer(question, expected):
        question, expected = _ask_question(chunk, llm, REGENERATE_PROMPT)
    return question, expected


def quizzable(chunks: list) -> list:
    """Chunks with enough text to ask about; all chunks if none qualify."""
    good = [c for c in chunks if len(c.text.strip()) >= MIN_QUIZ_CHARS]
    return good or chunks


def next_question(unit: str, llm=answer_llm, rng: random.Random = random) -> QuizItem:
    """Pick a passage from a unit (document) and ask a question about it."""
    chunks = store.get_chunks(unit)
    if not chunks:
        raise UnknownUnit(f"no document with id {unit!r}")
    chunk = pick_passage(quizzable(chunks), store.quiz_missed_chunk_ids(), rng=rng)
    question, expected = generate_question(chunk, llm=llm)
    return QuizItem(chunk_id=chunk.id, passage=_to_passage(chunk),
                    question=question, expected=expected)


def item_from(chunk_id: str, question: str = "") -> QuizItem:
    """Reconstruct a QuizItem from a chunk id (the answer endpoint receives only
    the id and the spoken answer, not the passage)."""
    chunk = store.get_chunk(chunk_id)
    if chunk is None:
        raise UnknownUnit(f"no passage with id {chunk_id!r}")
    return QuizItem(chunk_id=chunk.id, passage=_to_passage(chunk), question=question)


def grade(item: QuizItem, spoken_answer: str, llm_small=check_llm) -> grounding.Grade:
    """Grade the answer against the passage. A wrong or partial answer records a
    miss (so pick_passage favours it); a correct one clears it."""
    result = grounding.judge(spoken_answer, [item.passage], llm_small, question=item.question)
    if result.grade == "correct":
        store.clear_quiz_miss(item.chunk_id)
    else:
        store.record_quiz_miss(item.chunk_id)
    return result
