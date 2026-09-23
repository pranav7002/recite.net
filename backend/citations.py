"""Parsing "(document, p. N)" citations out of an answer."""
from __future__ import annotations

import re

# The document name may be quoted and may contain its own brackets, as in
# ("DS-2 (1).pdf", p. 9); one nested level of (...) is allowed. Quotes are not
# captured, so the name matches the stored document name.
_NAME = r'"?((?:[^,()"]|\([^()]*\))+?)"?'

# (name, first page). Extra pages ("p. 8, 9") or ranges ("pp. 12–17") are
# matched but not captured, so findall() still yields (name, page) pairs.
CITE = re.compile(rf"\(\s*{_NAME}\s*,\s*pp?\.\s*(\d+)[^()]*\)")

# (name, "14, 17") — every listed page, for callers that want all of them.
CITE_MULTI = re.compile(rf"\(\s*{_NAME}\s*,\s*pp?\.\s*([\d,\s]+)\)")


def cited(text: str) -> str:
    """Pages the answer itself cites; empty means the model cited nothing."""
    return ", ".join(f"{d.strip()}, p. {p}" for d, p in CITE.findall(text)) or "(none)"
