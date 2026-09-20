"""The answer loop.

Retrieve first, build the messages, call the model, run tool calls through the
guardrails, and ground the answer before it is returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from backend import grounding, guardrails
from backend.llm import answer_llm, check_llm
from backend.prompts import SYSTEM_PROMPT
from backend.retrieval.base import Passage, Retriever
from backend.retrieval.embeddings import EmbeddingRetriever
from backend.tools import TOOLS, format_passages

K = 5
MAX_ROUNDS = 3
HISTORY_TURNS = 3
FALLBACK_ANSWER = "I couldn't find a complete answer in your notes. Try asking more specifically."

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


@dataclass
class TurnState:
    """Mutable per-turn state shared by the loop, guardrails and grounding."""
    passages: list[Passage] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    read_untrusted: bool = True      # documents are always untrusted in this app
    pending_action: dict | None = None
    blocked_reason: str | None = None
    retried: bool = False


@dataclass
class TurnResult:
    text: str
    passages: list[Passage] = field(default_factory=list)
    steps: int = 0
    pending_action: dict | None = None
    retried: bool = False
    grounding: str = "supported"     # "supported" or "refused"


def answer(question: str, model=answer_llm, retriever: Retriever | None = None,
           k: int = K, max_rounds: int = MAX_ROUNDS,
           history: list[dict] | None = None, critic=check_llm) -> TurnResult:
    """Answer a question from retrieved passages; the model may call tools.

    history holds earlier turns as {"question", "answer"} dicts. Only their text
    is replayed, never old passages, so document text from past turns never
    re-enters the prompt.
    """
    retriever = retriever or EmbeddingRetriever()
    passages = list(retriever.search(question, k=k))
    messages = _build_messages(question, passages, history)
    state = TurnState(passages=passages)

    draft = _run_until_answer(messages, model, retriever, state, max_rounds)
    if state.pending_action is not None:
        # A mutating tool paused the turn for confirmation; nothing executed.
        return TurnResult(text=draft or FALLBACK_ANSWER, passages=state.passages,
                          steps=state.steps, pending_action=state.pending_action)
    if state.blocked_reason:
        # The send was refused; write the reply in code so the model cannot
        # describe (or misreport) a blocked action.
        return TurnResult(text=f"I didn't send that: {state.blocked_reason}.",
                          passages=state.passages, steps=state.steps)
    if draft is None:
        return TurnResult(text=FALLBACK_ANSWER, passages=state.passages,
                          steps=state.steps, pending_action=state.pending_action)

    outcome = grounding.check_and_retry(
        draft, state, messages, model, critic,
        step=lambda: _redraft(messages, model, retriever, state, max_rounds),
    )
    return TurnResult(text=outcome.text, passages=state.passages, steps=state.steps,
                      pending_action=state.pending_action,
                      retried=outcome.retried,
                      grounding="supported" if outcome.supported else "refused")


def _build_messages(question: str, passages: list[Passage],
                    history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-HISTORY_TURNS:]:
        messages += [{"role": "user", "content": turn["question"]},
                     {"role": "assistant", "content": turn["answer"]}]
    messages.append({"role": "user", "content": _user_turn(question, passages)})
    return messages


def _run_until_answer(messages: list[dict], model, retriever, state: TurnState,
                      max_rounds: int) -> str | None:
    """Run model steps until it answers with plain text (returned) or the round
    cap is hit (None). Tool calls run through the guardrails."""
    for _ in range(max_rounds):
        state.steps += 1
        resp = model.chat(messages, tools=[t.openai_spec() for t in TOOLS])
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}}
                                        for c in msg.tool_calls]})
        for call in msg.tool_calls:
            tool = TOOLS_BY_NAME.get(call.function.name)
            if tool is None:
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"Error: no tool named {call.function.name}"})
                continue
            try:
                args = tool.validate(call.function.arguments)
            except ValidationError as e:
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"Error: invalid arguments for {tool.name}: {e}"})
                continue
            verdict = guardrails.check(tool, args.model_dump(), state)
            if verdict.action == "block":
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"BLOCKED: {verdict.reason}"})
                if tool.mutating:
                    state.blocked_reason = verdict.reason
            elif verdict.action == "needs_confirmation":
                if state.pending_action is None:
                    state.pending_action = {"tool": tool.name, "args": args.model_dump()}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"Pending confirmation: {verdict.summary} (recorded, not executed)"})
                state.tool_calls += 1
                return verdict.summary
            else:
                content = tool.run(args, {"retriever": retriever, "collect": state.passages.extend})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            state.tool_calls += 1
    return None


def _redraft(messages: list[dict], model, retriever, state: TurnState,
             max_rounds: int) -> str | None:
    """Run the loop again after a failed grounding check; None if it pauses for
    confirmation instead of producing a draft."""
    text = _run_until_answer(messages, model, retriever, state, max_rounds)
    return None if state.pending_action is not None else text


def _user_turn(question: str, passages: list[Passage]) -> str:
    return f"Question: {question}\n\nPassages:\n{format_passages(passages)}"
