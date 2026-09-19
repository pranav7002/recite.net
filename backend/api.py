"""FastAPI endpoints: POST /ask and GET /health. Sessions live in memory for now."""
from __future__ import annotations

import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agent import answer
from backend.citations import CITE
from backend.llm import QuotaExhausted

app = FastAPI(title="recite.net")

_sessions: dict[str, list[dict]] = {}
_sessions_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=100)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    with _sessions_lock:
        history = list(_sessions.get(req.session_id, []))
    started = time.monotonic()
    try:
        result = answer(req.question, history=history)
    except QuotaExhausted as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    elapsed = time.monotonic() - started
    with _sessions_lock:
        _sessions.setdefault(req.session_id, []).append(
            {"question": req.question, "answer": result.text})
    return {
        "answer": result.text,
        "citations": [{"doc": d.strip(), "page": int(p)} for d, p in CITE.findall(result.text)],
        "pending_action": result.pending_action,
        "trace": {
            "session_id": req.session_id,
            "total_s": round(elapsed, 3),
            "steps": result.steps,
            "passages": len(result.passages),
            "retrieved": [{"doc": p.doc_name, "page": p.page, "score": round(p.score, 4)}
                          for p in result.passages],
        },
    }
