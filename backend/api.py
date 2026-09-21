"""FastAPI endpoints: ask, document upload, and pending-action confirm/cancel.

Sessions live in memory; pending actions live in the pending table, which
enforces the single-execution guarantee for confirmations.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import guardrails, ingest, store, tools, voice
from backend.agent import answer
from backend.citations import CITE
from backend.llm import QuotaExhausted

app = FastAPI(title="recite.net")

_sessions: dict[str, list[dict]] = {}
_sessions_lock = threading.Lock()

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 25_000_000
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs" / "uploads"


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
            "grounding": result.grounding,
            "retried": result.retried,
            "retrieved": [{"doc": p.doc_name, "page": p.page, "score": round(p.score, 4)}
                          for p in result.passages],
        },
    }


@app.post("/voice")
async def voice_turn(
    file: Annotated[UploadFile, File()],
    session_id: Annotated[str, Form()],
    t0: Annotated[float, Form()],
    config: Annotated[Literal["sequential", "stream", "stream_nocheck"], Form()] = "stream",
) -> StreamingResponse:
    """Spoken question in, spoken answer out as a server-sent event stream.

    t0 is the browser's timestamp of the last audio frame, so the browser can
    measure time-to-first-audio. config is sequential | stream | stream_nocheck.
    """
    audio = await file.read()
    with _sessions_lock:
        history = list(_sessions.get(session_id, []))

    def record(question: str, answer_text: str) -> None:
        with _sessions_lock:
            _sessions.setdefault(session_id, []).append(
                {"question": question, "answer": answer_text})

    gen = voice.voice_turn(audio, history, t0, config, record=record)
    return StreamingResponse(_sse(gen), media_type="text/event-stream")


async def _sse(gen):
    async for item in gen:
        yield f"data: {json.dumps(item, default=str)}\n\n"


@app.post("/documents")
def upload(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported file type: {suffix or '(none)'}. Upload a PDF, .txt or .md.")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (limit 25 MB).")
    path = _save_to_uploads(file.filename, data)
    try:
        chunks = ingest.index_document(path)
    except ingest.NoTextLayer:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422,
                            detail="This looks like a scanned PDF with no text layer.")
    except QuotaExhausted as e:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=429, detail=str(e)) from e
    return {"doc_id": chunks[0].doc_id,
            "pages": max(c.page for c in chunks),
            "chunks": len(chunks)}


@app.get("/documents")
def list_documents() -> list[dict]:
    return store.list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="No document with that id.")
    for f in UPLOADS_DIR.glob(f"{doc_id}*"):   # uploads are named <sha>.<ext>; doc_id is the sha prefix
        f.unlink(missing_ok=True)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/confirm/{pid}")
def confirm(pid: str) -> dict:
    row = store.confirm_pending(pid)
    if row is None:
        raise HTTPException(status_code=409,
                            detail="Nothing to confirm: already handled, cancelled, or expired.")
    tool_name, args = row
    if tool_name == "email_summary" and args["to"].lower() not in guardrails.CONTACTS:
        store.mark_pending(pid, "cancelled")
        raise HTTPException(status_code=409, detail="Recipient is no longer in your study group.")
    try:
        result = tools.run_tool(tool_name, args)
    except Exception:  # noqa: BLE001 — a failed write must not leave the action marked executed
        store.mark_pending(pid, "failed")
        raise HTTPException(status_code=500, detail="Action failed; nothing was sent.")
    return {"status": "executed", "tool": tool_name, "result": result}


@app.post("/cancel/{pid}")
def cancel(pid: str) -> dict:
    if not store.cancel_pending(pid):
        raise HTTPException(status_code=404, detail="No pending action with that id.")
    return {"status": "cancelled"}


def _save_to_uploads(filename: str, data: bytes) -> Path:
    sha = hashlib.sha256(data).hexdigest()
    suffix = Path(filename or "").suffix.lower()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOADS_DIR / f"{sha}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return path
