"""FastAPI endpoints: ask, document upload, and pending-action confirm/cancel.

Sessions live in memory; pending actions live in the pending table, which
enforces the single-execution guarantee for confirmations.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import guardrails, ingest, quiz, store, tools, voice
from backend.agent import answer
from backend.citations import CITE
from backend.llm import QuotaExhausted

log = logging.getLogger("recite.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the local speech models at startup so the first voice turn does not
    pay model loading. Skipped for SPEECH=gemini or WARM_SPEECH=0 (tests)."""
    if voice.SPEECH == "local" and os.getenv("WARM_SPEECH", "1") != "0":
        try:
            from backend import speech
            started = time.monotonic()
            speech.warm()
            log.info("speech models warm in %.1fs", time.monotonic() - started)
        except Exception as e:  # noqa: BLE001 — text endpoints must still start
            log.warning("could not warm speech models: %s", e)
    yield


app = FastAPI(title="recite.net", lifespan=lifespan)

_sessions: dict[str, list[dict]] = {}
_sessions_lock = threading.Lock()

# The pending action each voice session is waiting on, so a spoken "confirm" or
# "cancel" can be resolved in code. The pending table still enforces
# single execution and the 10-minute expiry.
_voice_pending: dict[str, str] = {}

CONFIRM_PHRASES = {"confirm", "confirm it", "yes confirm", "confirm send"}
CANCEL_PHRASES = {"cancel", "cancel it", "no cancel", "dont send", "do not send"}

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

    def remember_pending(action: dict) -> None:
        with _sessions_lock:
            _voice_pending[session_id] = action["id"]

    def intercept(transcript: str) -> str | None:
        command = spoken_command(transcript)
        if command is None:
            return None
        with _sessions_lock:
            pid = _voice_pending.pop(session_id, None)
        if pid is None:
            return "There's nothing waiting for confirmation."
        if command == "cancel":
            store.cancel_pending(pid)
            return "Cancelled. Nothing was sent."
        try:
            outcome = execute_confirm(pid)
        except ConfirmError as e:
            return e.spoken
        return f"Done. The summary for {outcome['args']['to']} is queued."

    gen = voice.voice_turn(audio, history, t0, config, record=record,
                           pending=remember_pending, intercept=intercept)
    return StreamingResponse(_sse(gen), media_type="text/event-stream")


async def _sse(gen):
    async for item in gen:
        yield f"data: {json.dumps(item, default=str)}\n\n"


class QuizNextRequest(BaseModel):
    unit: str = Field(min_length=1, max_length=100)


class QuizAnswerRequest(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=100)
    answer: str = Field(min_length=1, max_length=5000)
    question: str = Field(default="", max_length=1000)


@app.post("/quiz/next")
def quiz_next(req: QuizNextRequest) -> dict:
    try:
        item = quiz.next_question(req.unit)
    except quiz.UnknownUnit:
        raise HTTPException(status_code=404, detail=f"No document with id {req.unit!r}.")
    except QuotaExhausted as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    return {"chunk_id": item.chunk_id, "question": item.question,
            "doc": item.passage.doc_name, "page": item.passage.page}


@app.post("/quiz/answer")
def quiz_answer(req: QuizAnswerRequest) -> dict:
    try:
        item = quiz.item_from(req.chunk_id, req.question)
    except quiz.UnknownUnit:
        raise HTTPException(status_code=404, detail=f"No passage with id {req.chunk_id!r}.")
    try:
        result = quiz.grade(item, req.answer)
    except QuotaExhausted as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    return {"grade": result.grade, "missing": result.missing,
            "doc": item.passage.doc_name, "page": item.passage.page}


@app.post("/quiz/answer_voice")
def quiz_answer_voice(
    file: Annotated[UploadFile, File()],
    chunk_id: Annotated[str, Form(min_length=1, max_length=100)],
    question: Annotated[str, Form(max_length=1000)] = "",
) -> dict:
    """Spoken quiz answer: transcribe with the same STT as /voice, then grade."""
    try:
        item = quiz.item_from(chunk_id, question)
    except quiz.UnknownUnit:
        raise HTTPException(status_code=404, detail=f"No passage with id {chunk_id!r}.")
    transcript = voice.stt(file.file.read()).strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="I didn't hear an answer. Try again.")
    try:
        result = quiz.grade(item, transcript)
    except QuotaExhausted as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    return {"transcript": transcript, "grade": result.grade, "missing": result.missing,
            "doc": item.passage.doc_name, "page": item.passage.page}


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


def spoken_command(transcript: str) -> str | None:
    """'confirm' or 'cancel' when the whole utterance is one of the fixed
    phrases, else None. Exact match after normalising case and punctuation, so
    "confirm the mean is 50" is a question, not a confirmation."""
    words = " ".join(re.sub(r"[^a-z ]", "", transcript.lower().replace("'", "")).split())
    if words in CONFIRM_PHRASES:
        return "confirm"
    if words in CANCEL_PHRASES:
        return "cancel"
    return None


class ConfirmError(Exception):
    def __init__(self, status: int, detail: str, spoken: str) -> None:
        super().__init__(detail)
        self.status, self.detail, self.spoken = status, detail, spoken


def execute_confirm(pid: str) -> dict:
    """Run a pending action once. Shared by POST /confirm and spoken "confirm"."""
    row = store.confirm_pending(pid)
    if row is None:
        raise ConfirmError(409, "Nothing to confirm: already handled, cancelled, or expired.",
                           "That request expired or was already handled, so nothing was sent.")
    tool_name, args = row
    if tool_name == "email_summary" and args["to"].lower() not in guardrails.CONTACTS:
        store.mark_pending(pid, "cancelled")
        raise ConfirmError(409, "Recipient is no longer in your study group.",
                           "That recipient is no longer in your study group, so nothing was sent.")
    try:
        result = tools.run_tool(tool_name, args)
    except Exception:  # noqa: BLE001 — a failed write must not leave the action marked executed
        store.mark_pending(pid, "failed")
        raise ConfirmError(500, "Action failed; nothing was sent.",
                           "Sending failed, so nothing was sent.")
    return {"status": "executed", "tool": tool_name, "args": args, "result": result}


@app.post("/confirm/{pid}")
def confirm(pid: str) -> dict:
    try:
        outcome = execute_confirm(pid)
    except ConfirmError as e:
        raise HTTPException(status_code=e.status, detail=e.detail) from e
    return {"status": outcome["status"], "tool": outcome["tool"], "result": outcome["result"]}


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
