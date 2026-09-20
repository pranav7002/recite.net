"""A fake LLM that returns scripted responses in the OpenAI client's shape.

The agent loop takes it through the same parameter as the real client, so no
test ever calls Gemini.
"""
from __future__ import annotations

from types import SimpleNamespace


class FakeLLM:
    """Returns one scripted response per chat() call, in order."""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[list[dict]] = []          # messages sent to chat()

    def chat(self, messages, tools=None, temperature=None, response_format=None, **kw):
        self.calls.append(messages)
        if not self._script:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._script.pop(0)


def text(content: str):
    """A plain assistant response (no tool calls)."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def tool_call(*calls):
    """An assistant response with one or more tool calls.

    Each item is a (name, arguments) or (name, arguments, id) tuple.
    """
    tcs = []
    for i, c in enumerate(calls):
        name, arguments = c[0], c[1]
        cid = c[2] if len(c) > 2 else f"call_{i}"
        fn = SimpleNamespace(name=name, arguments=arguments)
        tcs.append(SimpleNamespace(id=cid, type="function", function=fn))
    msg = SimpleNamespace(content=None, tool_calls=tcs)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
