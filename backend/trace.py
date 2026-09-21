"""Per-request trace: timings land here as JSON log lines.

Day 2's endpoint writes the full request trace; for now this is where LLM
calls record wait time separately from call time, so latency numbers measure
the system, not Google's quota.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

log = logging.getLogger("recite.trace")

_capture: ContextVar[list[dict] | None] = ContextVar("trace_capture", default=None)


def record(**fields: object) -> None:
    fields.setdefault("ts", round(time.time(), 3))
    log.info(json.dumps(fields, default=str))
    sink = _capture.get()
    if sink is not None:
        sink.append(fields)


@contextmanager
def capture() -> Iterator[list[dict]]:
    """Collect every record() made inside the block, so a caller (the eval
    runner) can split a turn's wall-clock time into model time and limiter wait."""
    sink: list[dict] = []
    token = _capture.set(sink)
    try:
        yield sink
    finally:
        _capture.reset(token)


def timings(calls: list[dict]) -> dict[str, float]:
    """wait_s: seconds spent waiting on the rate limiter. model_s: seconds the
    API calls themselves took. Only model_s measures the system; wait_s
    measures the free tier."""
    return {"wait_s": round(sum(float(c.get("wait_s", 0)) for c in calls), 3),
            "model_s": round(sum(float(c.get("call_s", 0)) for c in calls), 3),
            "calls": len(calls)}
