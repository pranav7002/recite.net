"""The answer loop from the architecture guide.

Retrieve first, build the messages, call the model, run any search_docs tool
calls, and return the answer. No guardrails or grounding check yet.
"""
from __future__ import annotations

import json

from backend.llm import answer_llm
from backend.prompts import SYSTEM_PROMPT
from backend.retrieval.base import Passage, Retriever
from backend.retrieval.embeddings import EmbeddingRetriever

K = 5
MAX_ROUNDS = 3

SEARCH_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the student's course material for passages relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in the course material."},
            },
            "required": ["query"],
        },
    },
}


def answer(question: str, model=answer_llm, retriever: Retriever | None = None,
           k: int = K, max_rounds: int = MAX_ROUNDS) -> str:
    """Answer a question from retrieved passages; the model may call search_docs."""
    retriever = retriever or EmbeddingRetriever()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_turn(question, retriever.search(question, k=k))},
    ]
    msg = None
    for _ in range(max_rounds):
        resp = model.chat(messages, tools=[SEARCH_DOCS_TOOL])
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
        for call in msg.tool_calls:
            if call.function.name != "search_docs":
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"Error: no tool named {call.function.name}"})
                continue
            query = _tool_query(call.function.arguments)
            results = retriever.search(query, k=k) if query else []
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": _passages_block(results)})
    return "I couldn't find a complete answer in your notes. Try asking more specifically."


def _tool_query(arguments: str) -> str:
    try:
        return str(json.loads(arguments).get("query", ""))
    except (json.JSONDecodeError, AttributeError):
        return ""


def _passages_block(passages: list[Passage]) -> str:
    if not passages:
        return "(no matching passages found)"
    return "\n\n".join(
        f'<untrusted_content doc="{p.doc_name}" page="{p.page}">\n{p.text}\n</untrusted_content>'
        for p in passages
    )


def _user_turn(question: str, passages: list[Passage]) -> str:
    return f"Question: {question}\n\nPassages:\n{_passages_block(passages)}"
