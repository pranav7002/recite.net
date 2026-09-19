"""Per-request trace: timings land here as JSON log lines.

Day 2's endpoint writes the full request trace; for now this is where LLM
calls record wait time separately from call time, so latency numbers measure
the system, not Google's quota.
"""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("recite.trace")


def record(**fields: object) -> None:
    fields.setdefault("ts", round(time.time(), 3))
    log.info(json.dumps(fields, default=str))
