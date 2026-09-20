"""Tools the model may call, with Pydantic-validated arguments.

search_docs runs a retrieval search. email_summary only appends a JSON line to
data/outbox.json — it never sends anything; the agent loop's guardrails gate it.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from backend.retrieval.base import Passage

OUTBOX_PATH = Path(__file__).resolve().parent.parent / "data" / "outbox.json"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    args_model: type[BaseModel]
    run: Callable[[BaseModel, dict], str]
    mutating: bool = False           # changes something outside the app
    returns_untrusted: bool = False  # output contains document text

    def openai_spec(self) -> dict:
        """The OpenAI tools-format spec sent to the model."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}

    def validate(self, arguments: str) -> BaseModel:
        return self.args_model.model_validate_json(arguments)


class SearchDocsArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500,
                       description="What to search for in the course material.")


class EmailSummaryArgs(BaseModel):
    to: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=200,
                    description="The recipient's email address.")
    subject: str = Field(min_length=1, max_length=200,
                         description="The email subject line.")
    body: str = Field(min_length=1, max_length=5000,
                      description="The email body text.")


def _run_search(args: SearchDocsArgs, ctx: dict) -> str:
    results = ctx["retriever"].search(args.query)
    ctx["collect"](results)
    return format_passages(results)


def _run_email(args: EmailSummaryArgs, ctx: dict) -> str:
    entry = {"ts": datetime.now(timezone.utc).isoformat(),
             "to": args.to,
             "subject": args.subject,
             "body": args.body}
    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTBOX_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return f"Queued: summary recorded for {args.to} (outbox only — nothing sent)."


def format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "(no matching passages found)"
    return "\n\n".join(
        f'<untrusted_content doc="{p.doc_name}" page="{p.page}">\n{p.text}\n</untrusted_content>'
        for p in passages
    )


search_docs = Tool(
    name="search_docs",
    description="Search the student's course material for passages relevant to a query.",
    parameters=SearchDocsArgs.model_json_schema(),
    args_model=SearchDocsArgs,
    run=_run_search,
    returns_untrusted=True,
)

email_summary = Tool(
    name="email_summary",
    description="Queue an email with the answer summary. Nothing is sent; the request is recorded for confirmation.",
    parameters=EmailSummaryArgs.model_json_schema(),
    args_model=EmailSummaryArgs,
    run=_run_email,
    mutating=True,
)

# Both tools are offered; the mutating email_summary is gated by the guardrails
# in the agent loop before anything is recorded or sent.
TOOLS = [search_docs, email_summary]
