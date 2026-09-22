"""The grounding check: a smaller, independent model verifies each claim of a
draft answer against the retrieved passages before it is spoken.

The critic is a different model than the drafter — a Flash-Lite against a Flash
— so it does not share the drafter's blind spots. It is asked the narrow,
verifiable question "is this claim stated in these passages?", not "is this a
good answer?".
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from backend.retrieval.base import Passage

log = logging.getLogger("recite.grounding")

CHECK_PROMPT = """\
You check whether numbered claims are supported by the passages. For each claim \
answer SUPPORTED, UNSUPPORTED or PARTIAL using only the passages. SUPPORTED means the \
passages state it or it follows directly from them. Judge the claim, not any \
citation such as "(notes.pdf, p. 3)" that appears in it. Name the passage id you \
relied on.

Return one JSON object with a "claims" array containing exactly one entry per \
numbered claim, using the claim's number as "id":
{"claims": [{"id": 1, "verdict": "SUPPORTED", "passage_id": "abc:2"}, \
{"id": 2, "verdict": "UNSUPPORTED", "passage_id": null}]}

Example. Passage abc:2 says "The mean of 40, 50, 60 is 50." Claims: 1. The mean of \
40, 50, 60 is 50. 2. The median is 60. Correct output: {"claims": [{"id": 1, \
"verdict": "SUPPORTED", "passage_id": "abc:2"}, {"id": 2, "verdict": "UNSUPPORTED", \
"passage_id": null}]}"""

# A claim is a sentence ending in . ? or !, or a paragraph. A boundary needs a
# capital letter, quote or bracket-free start next, so "p. 16)" does not split;
# "(" is deliberately not a boundary, so a citation written after the full stop
# stays with its sentence. Fragments of fewer than MIN_CLAIM_WORDS words ("Yes.",
# "Here's why.") are filler: they are merged into a neighbouring claim, never
# checked alone and never silently dropped with a real claim.
_SENTENCE = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'\[$\\])|\n{2,}")
_CITATION_ONLY = re.compile(r"^[(\[][^()\[\]]*[)\]]\.?$")
MIN_CLAIM_WORDS = 3

REFUSAL = "I couldn't find that in your notes."


class ClaimVerdict(BaseModel):
    claim: str = ""
    verdict: str                      # SUPPORTED | UNSUPPORTED | PARTIAL
    passage_id: str | None = None
    id: int | None = None             # the claim's number in the request


class CheckResult(BaseModel):
    claims: list[ClaimVerdict]
    raw: str = ""                     # the check model's raw output, for diagnosis
    parse_failed: bool = False

    @property
    def all_supported(self) -> bool:
        return all(c.verdict.strip().upper() == "SUPPORTED" for c in self.claims)

    @property
    def failing(self) -> list[str]:
        return [c.claim for c in self.claims if c.verdict.strip().upper() != "SUPPORTED"]


@dataclass
class GroundingOutcome:
    text: str
    retried: bool
    supported: bool
    checks: list[dict] = field(default_factory=list)    # one entry per check call, for the eval traces


def split_sentences(draft: str) -> list[str]:
    """Split a draft into claims. A fragment that is only a citation is folded
    into the sentence before it; filler fragments are merged into a neighbour,
    so a short factual sentence ("The median is 60.") is still checked."""
    parts = [p.strip() for p in _SENTENCE.split(draft.strip()) if p.strip()]
    sentences: list[str] = []
    for part in parts:
        if sentences and _CITATION_ONLY.match(part):
            sentences[-1] += " " + part
        else:
            sentences.append(part)

    claims: list[str] = []
    leading = ""                                  # filler waiting for the next claim
    for sentence in sentences:
        if len(sentence.split()) < MIN_CLAIM_WORDS:
            if claims:
                claims[-1] += " " + sentence
            else:
                leading = f"{leading} {sentence}".strip()
            continue
        claims.append(f"{leading} {sentence}".strip() if leading else sentence)
        leading = ""
    return claims


def check_messages(claims: list[str], passages: list[Passage]) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    return [
        {"role": "system", "content": CHECK_PROMPT},
        {"role": "user", "content": f"Claims:\n{numbered}\n\nPassages:\n{_format_passages(passages)}"},
    ]


def check(draft: str, passages: list[Passage], llm_small) -> CheckResult:
    """Check every claim of `draft` against `passages`. On a JSON parse failure
    the critic is asked once more; only then is the answer treated as unsupported."""
    claims = split_sentences(draft)
    if not claims:
        return CheckResult(claims=[])
    messages = check_messages(claims, passages)
    raw = ""
    for attempt in range(2):
        raw = _ask(llm_small, messages)
        result = _parse(raw, claims)
        if result is not None:
            result.raw = raw
            return result
        log.warning("grounding check returned unparseable JSON (attempt %d)", attempt + 1)
    log.warning("grounding check failed to parse twice; treating answer as unsupported")
    unsupported = _unsupported(claims)
    unsupported.raw, unsupported.parse_failed = raw, True
    return unsupported


def check_and_retry(draft: str, state, messages: list[dict], llm, llm_small,
                    step: Callable[[], str | None]) -> GroundingOutcome:
    """Check `draft`; on failure retry once, then refuse.

    `step` runs the model again from the current messages (it may call
    search_docs) and returns the redraft, or None when no answer is produced.
    """
    if draft.strip() == REFUSAL:
        return GroundingOutcome(text=REFUSAL, retried=False, supported=True)

    checks: list[dict] = []
    result = check(draft, state.passages, llm_small)
    checks.append(_trace(draft, result))
    if result.all_supported:
        return GroundingOutcome(text=draft, retried=False, supported=True, checks=checks)

    state.retried = True
    log.info("grounding check failed; retrying once (failing: %r)", result.failing)
    messages.append({"role": "user", "content":
                     f"These claims were not supported by the passages: {result.failing}. "
                     "Search again or remove them."})

    redraft = step()
    if redraft is None:
        return GroundingOutcome(text=refuse(result.failing), retried=True, supported=False, checks=checks)

    result2 = check(redraft, state.passages, llm_small)
    checks.append(_trace(redraft, result2))
    fixed = result2.all_supported
    log.info("grounding retry %s", "fixed the answer" if fixed else "did not fix the answer")
    if fixed:
        return GroundingOutcome(text=redraft, retried=True, supported=True, checks=checks)
    return GroundingOutcome(text=refuse(result2.failing), retried=True, supported=False, checks=checks)


JUDGE_PROMPT = """\
You grade a student's spoken answer against a passage. Decide whether the answer \
is correct, partially correct, or incorrect with respect to the passage, and say \
in one sentence what is missing or wrong. Return one JSON object:
{"grade": "correct" | "partial" | "incorrect", "missing": "..."}"""


class Grade(BaseModel):
    grade: str = "incorrect"       # correct | partial | incorrect
    missing: str = ""


def judge(student_answer: str, passages: list[Passage], llm_small,
          question: str | None = None) -> Grade:
    """Grade a student's answer against a passage — quiz mode reuses the checker.

    On a JSON parse failure the judge is asked once more; only then is the
    answer graded incorrect (fail closed)."""
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": (
            f"Question: {question or ''}\n\n"
            f"Passage:\n{_format_passages(passages)}\n\n"
            f"Student answer: {student_answer}"
        )},
    ]
    for _ in range(2):
        grade = _parse_grade(_ask(llm_small, messages))
        if grade is not None:
            return grade
    log.warning("quiz grade returned unparseable JSON twice; grading incorrect")
    return Grade(grade="incorrect", missing="could not parse the grade")


def _parse_grade(text: str) -> Grade | None:
    try:
        grade = Grade.model_validate_json(_extract_json(text))
    except ValidationError:
        return None
    grade.grade = _canonical_grade(grade.grade)
    return grade


def _canonical_grade(grade: str) -> str:
    g = (grade or "").strip().lower()
    if g in {"correct", "right"}:
        return "correct"
    if g in {"partial", "partially correct", "partially_correct", "partly"}:
        return "partial"
    return "incorrect"


def _trace(draft: str, result: CheckResult) -> dict:
    """What one check call decided, kept in the eval records so a refused
    answer can be traced to the claim (or parse failure) that caused it."""
    return {"draft": draft, "parse_failed": result.parse_failed, "raw": result.raw,
            "verdicts": [{"claim": c.claim, "verdict": c.verdict, "passage_id": c.passage_id}
                         for c in result.claims]}


def refuse(failing: list[str]) -> str:
    log.info("refused; unsupported claims: %r", failing)
    return REFUSAL


def _ask(llm_small, messages: list[dict]) -> str:
    resp = llm_small.chat(messages, response_format={"type": "json_object"}, temperature=0)
    return resp.choices[0].message.content or ""


def _extract_json(text: str) -> str:
    """Small models often wrap JSON in a code fence, add prose around it, or
    return the bare claims array; pull out the JSON so those still parse."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        return '{"claims": ' + text[start:end + 1] + "}"
    return text


def _parse(text: str, claims: list[str]) -> CheckResult | None:
    """Match verdicts to claims by the numbered id the model echoes back, so a
    reordered or re-split answer still lines up; without ids, by position."""
    try:
        result = CheckResult.model_validate_json(_extract_json(text))
    except ValidationError:
        return None
    parsed = result.claims
    if parsed and all(p.id is not None for p in parsed):
        by_id = {p.id: p for p in parsed}
        if set(by_id) != set(range(1, len(claims) + 1)):
            return None                       # a claim is missing or invented
        parsed = [by_id[i] for i in range(1, len(claims) + 1)]
    elif len(parsed) != len(claims):
        return None
    for original, verdict in zip(claims, parsed, strict=True):
        verdict.claim = original              # keep our wording for the retry message
    return CheckResult(claims=parsed)


def _unsupported(claims: list[str]) -> CheckResult:
    return CheckResult(claims=[ClaimVerdict(claim=c, verdict="UNSUPPORTED") for c in claims])


def _format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "(no matching passages found)"
    return "\n\n".join(
        f'<untrusted_content id="{p.id}" doc="{p.doc_name}" page="{p.page}">\n{p.text}\n</untrusted_content>'
        for p in passages
    )
