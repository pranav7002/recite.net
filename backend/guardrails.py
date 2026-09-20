"""The four guardrails between the model's tool request and the real action.
The model proposes; code disposes. Every rule is deterministic — no model
judgment is involved.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Fail closed: an empty allowlist blocks every send until contacts are set.
# Comma-separated email addresses, compared case-insensitively.
CONTACTS = frozenset(
    c.strip().lower() for c in os.getenv("CONTACTS", "").split(",") if c.strip()
)

MAX_TOOL_CALLS = 8          # the ninth tool call is blocked


@dataclass
class Verdict:
    action: str              # "allow" | "block" | "needs_confirmation"
    reason: str = ""         # for "block"
    summary: str = ""        # for "needs_confirmation"


def allow() -> Verdict:
    return Verdict(action="allow")


def block(reason: str) -> Verdict:
    return Verdict(action="block", reason=reason)


def needs_confirmation(summary: str) -> Verdict:
    return Verdict(action="needs_confirmation", summary=summary)


def check(tool, args: dict, state) -> Verdict:
    """The four rules, in order: step limit, allowlist, provenance, confirmation."""
    if state.tool_calls >= MAX_TOOL_CALLS:
        return block("step limit")
    if not tool.mutating:
        return allow()
    if args["to"].strip().lower() not in CONTACTS:
        return block("recipient not in study group")
    if state.read_untrusted and appears_in_retrieved(args["to"], state):
        return block("recipient came from a document")
    return needs_confirmation(render(tool, args))


def appears_in_retrieved(addr: str, state) -> bool:
    """True when `addr` appears, case-insensitively, in any passage retrieved
    this turn — the provenance rule for a recipient taken from document text."""
    needle = addr.strip().lower()
    return any(needle in p.text.lower() for p in state.passages)


def render(tool, args: dict) -> str:
    fields = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{tool.name}({fields})"
