"""Parsing "(document, p. N)" citations out of an answer."""
from __future__ import annotations

import re

CITE = re.compile(r"\(([^,()]+),\s*p\.\s*(\d+)\)")


def cited(text: str) -> str:
    """Pages the answer itself cites; empty means the model cited nothing."""
    return ", ".join(f"{d.strip()}, p. {p}" for d, p in CITE.findall(text)) or "(none)"
