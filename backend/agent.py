"""The answer loop from the architecture guide.

Retrieve first, build the messages, call the model, run tool calls, and return
the answer. No guardrails or grounding check yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from backend.llm import answer_llm
from backend.prompts import SYSTEM_PROMPT
from backend.retrieval.base import Passage, Retriever
from backend.retrieval.embeddings import EmbeddingRetriever
from backend.tools import ALL_TOOLS, TOOLS, format_passages

K = 5
MAX_ROUNDS = 3
HISTORY_TURNS = 3
FALLBACK_ANSWER = "I couldn't find a complete answer in your notes. Try asking more specifically."

TOOLS_BY_NAME = {t.name: t for t in TOOLS}
ALL_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


@dataclass
class TurnResult:
    text: str
    passages: list[Passage] = field(default_factory=list)  # initial + every search_docs result
    steps: int = 0                                         # model calls made
    pending_action: dict | None = None                     # mutating tool awaiting confirmation


def answer(question: str, model=answer_llm, retriever: Retriever | None = None,
           k: int = K, max_rounds: int = MAX_ROUNDS,
           history: list[dict] | None = None) -> TurnResult:
    """Answer a question from retrieved passages; the model may call tools.

    history holds earlier turns as {"question", "answer"} dicts. Only their text
    is replayed, never old passages, so document text from past turns never
    re-enters the prompt.
    """
    retriever = retriever or EmbeddingRetriever()
    passages = list(retriever.search(question, k=k))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-HISTORY_TURNS:]:
        messages += [{"role": "user", "content": turn["question"]},
                     {"role": "assistant", "content": turn["answer"]}]
    messages.append({"role": "user", "content": _user_turn(question, passages)})
    steps = 0
    pending_action: dict | None = None
    for _ in range(max_rounds):
        steps += 1
        resp = model.chat(messages, tools=[t.openai_spec() for t in TOOLS])
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return TurnResult(text=msg.content or "", passages=passages, steps=steps,
                              pending_action=pending_action)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}}
                                        for c in msg.tool_calls]})
        for call in msg.tool_calls:
            tool = TOOLS_BY_NAME.get(call.function.name)
            if tool is None:
                withheld = ALL_TOOLS_BY_NAME.get(call.function.name)
                if withheld is not None:       # mutating tool, withheld until guardrails exist
                    # TODO(day2): guardrails before pending — run guardrails.check() here;
                    # a blocked call must never reach the user as a confirmation.
                    try:
                        args = withheld.validate(call.function.arguments)
                    except ValidationError as e:
                        messages.append({"role": "tool", "tool_call_id": call.id,
                                         "content": f"Error: invalid arguments for {withheld.name}: {e}"})
                        continue
                    if pending_action is None:
                        pending_action = {"tool": withheld.name, "args": args.model_dump()}
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": f"Pending confirmation: {withheld.name} was recorded but not executed."})
                    continue
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"Error: no tool named {call.function.name}"})
                continue
            try:
                args = tool.validate(call.function.arguments)
            except ValidationError as e:
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"Error: invalid arguments for {tool.name}: {e}"})
                continue
            content = tool.run(args, {"retriever": retriever, "collect": passages.extend})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return TurnResult(text=FALLBACK_ANSWER, passages=passages, steps=steps,
                      pending_action=pending_action)


def _user_turn(question: str, passages: list[Passage]) -> str:
    return f"Question: {question}\n\nPassages:\n{format_passages(passages)}"
