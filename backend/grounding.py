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
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from backend.retrieval.base import Passage

log = logging.getLogger("recite.grounding")

CHECK_PROMPT = """\
For each numbered claim, answer SUPPORTED, UNSUPPORTED or PARTIAL using only the \
passages. Name the passage id you relied on. Return a single JSON object with a \
"claims" array, one entry per claim, in order: {"claim": "...", "verdict": "...", \
"passage_id": "..."}."""

# A claim is a sentence ending in . ? or !, with a length floor so fragments
# like "Yes." or "Here's why." are not checked as standalone claims.
_SENTENCE = re.compile(r"(?<=[.?!])\s+")
MIN_CLAIM_CHARS = 20

REFUSAL = "I couldn't find that in your notes."


class ClaimVerdict(BaseModel):
    claim: str
    verdict: str                      # SUPPORTED | UNSUPPORTED | PARTIAL
    passage_id: str | None = None


class CheckResult(BaseModel):
    claims: list[ClaimVerdict]

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


def split_sentences(draft: str) -> list[str]:
    """Split a draft into claims: sentences ending in . ? or !, above a length floor."""
    return [p.strip() for p in _SENTENCE.split(draft.strip()) if len(p.strip()) >= MIN_CLAIM_CHARS]


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
    for attempt in range(2):
        result = _parse(_ask(llm_small, messages), claims)
        if result is not None:
            return result
        log.warning("grounding check returned unparseable JSON (attempt %d)", attempt + 1)
    log.warning("grounding check failed to parse twice; treating answer as unsupported")
    return _unsupported(claims)


def check_and_retry(draft: str, state, messages: list[dict], llm, llm_small,
                    step: Callable[[], str | None]) -> GroundingOutcome:
    """Check `draft`; on failure retry once, then refuse.

    `step` runs the model again from the current messages (it may call
    search_docs) and returns the redraft, or None when no answer is produced.
    """
    if draft.strip() == REFUSAL:
        return GroundingOutcome(text=REFUSAL, retried=False, supported=True)

    result = check(draft, state.passages, llm_small)
    if result.all_supported:
        return GroundingOutcome(text=draft, retried=False, supported=True)

    state.retried = True
    log.info("grounding check failed; retrying once (failing: %r)", result.failing)
    messages.append({"role": "user", "content":
                     f"These claims were not supported by the passages: {result.failing}. "
                     "Search again or remove them."})

    redraft = step()
    if redraft is None:
        return GroundingOutcome(text=refuse(result.failing), retried=True, supported=False)

    result2 = check(redraft, state.passages, llm_small)
    fixed = result2.all_supported
    log.info("grounding retry %s", "fixed the answer" if fixed else "did not fix the answer")
    if fixed:
        return GroundingOutcome(text=redraft, retried=True, supported=True)
    return GroundingOutcome(text=refuse(result2.failing), retried=True, supported=False)


def refuse(failing: list[str]) -> str:
    log.info("refused; unsupported claims: %r", failing)
    return REFUSAL


def _ask(llm_small, messages: list[dict]) -> str:
    resp = llm_small.chat(messages, response_format={"type": "json_object"}, temperature=0)
    return resp.choices[0].message.content or ""


def _parse(text: str, claims: list[str]) -> CheckResult | None:
    try:
        result = CheckResult.model_validate_json(text)
    except ValidationError:
        return None
    if len(result.claims) != len(claims):
        return None
    for original, parsed in zip(claims, result.claims, strict=True):
        parsed.claim = original             # keep our wording for the retry message
    return result


def _unsupported(claims: list[str]) -> CheckResult:
    return CheckResult(claims=[ClaimVerdict(claim=c, verdict="UNSUPPORTED") for c in claims])


def _format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "(no matching passages found)"
    return "\n\n".join(
        f'<untrusted_content id="{p.id}" doc="{p.doc_name}" page="{p.page}">\n{p.text}\n</untrusted_content>'
        for p in passages
    )
