"""Thin wrapper over the OpenAI client pointed at Gemini — the one place that
talks to the API and deals with free-tier limits.

Chat goes through Gemini's OpenAI-compatible endpoint. Embeddings go through
the native google-genai client, because the compatibility layer does not
expose Gemini's output-dimension option. Everything else receives an LLM
instance as a parameter, which is what lets tests swap in a fake.
"""
from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from google.genai import types
from openai import InternalServerError, OpenAI, RateLimitError

from backend import store, trace

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("recite.llm")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
PACIFIC = ZoneInfo("America/Los_Angeles")       # daily quotas reset at midnight Pacific

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))
EMBED_RPM = float(os.getenv("EMBED_RPM", "5"))
EMBED_RPD = int(os.getenv("EMBED_RPD", "1000"))
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "768"))

TTS_VOICE = os.getenv("TTS_VOICE", "charon")
TTS_RPM = float(os.getenv("TTS_RPM", "15"))
TTS_RPD = int(os.getenv("TTS_RPD", "500"))
STT_RPM = float(os.getenv("STT_RPM", "15"))
STT_RPD = int(os.getenv("STT_RPD", "500"))


def _required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set — add it to .env")
    return value


def _api_key() -> str:
    return _required("GEMINI_API_KEY")


class QuotaExhausted(Exception):
    """Raised when a model's free-tier quota is used up."""


class Limiter:
    """Spaces calls 60/rpm seconds apart so a per-minute limit is never exceeded."""

    def __init__(self, rpm: float) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_call = 0.0

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            delay = max(self._next_call - now, 0.0)
            if delay > 0:
                log.info("rate limiter: sleeping %.1fs", delay)
                time.sleep(delay)
                now = time.monotonic()
            self._next_call = max(now, self._next_call) + self._interval
            return delay


class DailyCounter:
    """The quota table in Postgres: one row per (model, day)."""

    def __init__(self, model: str, rpd: int) -> None:
        self.model, self.rpd = model, rpd

    def take(self) -> None:
        count = store.quota_take(self.model, datetime.now(PACIFIC).date())
        if count > self.rpd:
            raise QuotaExhausted(f"{self.model}: {self.rpd} requests/day exhausted")


class LLM:
    def __init__(self, model: str, rpm: int, rpd: int) -> None:
        self.model = model
        self.limiter = Limiter(rpm)
        self.day_budget = DailyCounter(model, rpd)

    def chat(self, messages, tools=None, temperature=0.2, stream=False, **kw):
        self.day_budget.take()                      # raises before a wasted call
        waited = self.limiter.wait()
        for attempt in range(5):
            try:
                t = time.monotonic()
                resp = _chat_client().chat.completions.create(
                    model=self.model, messages=messages, tools=tools,
                    temperature=temperature, stream=stream, **kw,
                )
                trace.record(model=self.model, wait_s=round(waited, 3),
                             call_s=round(time.monotonic() - t, 3))
                return resp
            except (RateLimitError, InternalServerError):   # 429, or 5xx such as 503 overloaded
                time.sleep(min(60, 2 ** attempt) + random.random())   # backoff with jitter
        raise QuotaExhausted(f"{self.model}: rate limited or unavailable after 5 attempts")


answer_llm = LLM(
    _required("ANSWER_MODEL"),
    int(os.getenv("ANSWER_MODEL_RPM", "10")),
    int(os.getenv("ANSWER_MODEL_RPD", "1000")),
)
check_llm = LLM(
    _required("CHECK_MODEL"),
    int(os.getenv("CHECK_MODEL_RPM", "15")),
    int(os.getenv("CHECK_MODEL_RPD", "1000")),
)

# The model that drives the RLM arm's code-execution loop (backend/retrieval/
# rlm_arm.py). gemini-3.1-flash-lite (the answer model, 500/day) plans forever
# and never finalises a grounded answer through that loop; gemma-4-26b-a4b-it
# (14,400/day) does, once its <thought> leakage is stripped — see
# EXPERIMENTS.md. Defaulted rather than required: an unset RLM_MODEL should
# not break every other model's import.
rlm_llm = LLM(
    os.getenv("RLM_MODEL", "gemma-4-26b-a4b-it"),
    int(os.getenv("RLM_MODEL_RPM", "5")),
    int(os.getenv("RLM_MODEL_RPD", "14000")),
)


# ---- embeddings (native google-genai: the compat layer lacks output dims) ----

_chat_client_singleton = None
_genai_client_singleton = None


def _chat_client() -> OpenAI:
    global _chat_client_singleton
    if _chat_client_singleton is None:
        _chat_client_singleton = OpenAI(base_url=GEMINI_BASE_URL, api_key=_api_key())
    return _chat_client_singleton


def _genai_client():
    global _genai_client_singleton
    if _genai_client_singleton is None:
        from google import genai

        _genai_client_singleton = genai.Client(api_key=_api_key())
    return _genai_client_singleton


_embed_limiter = Limiter(EMBED_RPM)


def embed(
    texts: list[str],
    model: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    dimensions: int = EMBED_DIMENSIONS,
    task_type: str | None = None,
) -> list[list[float]]:
    """Embed texts in batches; each API call goes through the rate limiter,
    the daily quota counter, and the same retry-with-backoff as chat()."""
    if not texts:
        return []
    model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    client = _genai_client()
    out: list[list[float]] = []
    started = time.monotonic()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out.extend(_embed_batch(client, model, batch, dimensions, task_type))
    log.info(
        "embedded %d texts with %s (%d dim) in %.1fs",
        len(texts), model, dimensions, time.monotonic() - started,
    )
    return out


def _embed_batch(client, model: str, batch: list[str], dimensions: int,
                 task_type: str | None) -> list[list[float]]:
    DailyCounter(model, EMBED_RPD).take()          # raises before a wasted call
    waited = _embed_limiter.wait()
    for attempt in range(5):
        try:
            t = time.monotonic()
            config: dict = {"output_dimensionality": dimensions}
            if task_type:
                config["task_type"] = task_type
            response = client.models.embed_content(model=model, contents=batch, config=config)
            trace.record(model=model, op="embed", wait_s=round(waited, 3),
                         call_s=round(time.monotonic() - t, 3))
            return [list(e.values) for e in response.embeddings]
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 429:
                time.sleep(min(60, 2 ** attempt) + random.random())   # backoff with jitter
                continue
            raise
    raise QuotaExhausted(f"{model}: rate limited after 5 attempts")


def normalise(vectors: list[list[float]] | np.ndarray) -> np.ndarray:
    """L2-normalise each row so cosine similarity is a dot product."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


# ---- speech (native google-genai: transcription and TTS) ----

_tts_limiter = Limiter(TTS_RPM)
_stt_limiter = Limiter(STT_RPM)


def synthesize(text: str, model: str | None = None) -> tuple[bytes, int]:
    """Text to speech via Gemini. Returns (raw 16-bit mono PCM, sample rate)."""
    model = model or os.getenv("TTS_MODEL")
    if not model:
        raise RuntimeError("TTS_MODEL is not set — add it to .env")
    config = types.GenerateContentConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
            )
        ),
    )
    DailyCounter(model, TTS_RPD).take()
    waited = _tts_limiter.wait()
    for attempt in range(5):
        try:
            t = time.monotonic()
            resp = _genai_client().models.generate_content(model=model, contents=text, config=config)
            trace.record(model=model, op="tts", wait_s=round(waited, 3),
                         call_s=round(time.monotonic() - t, 3))
            blob = _audio_blob(resp)
            return blob.data, _sample_rate(blob.mime_type)
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 429:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            raise
    raise QuotaExhausted(f"{model}: rate limited after 5 attempts")


def transcribe(audio: bytes, mime_type: str = "audio/wav", model: str | None = None) -> str:
    """Speech to text via Gemini; returns the transcript."""
    model = model or os.getenv("STT_MODEL")
    if not model:
        raise RuntimeError("STT_MODEL is not set — add it to .env")
    DailyCounter(model, STT_RPD).take()
    waited = _stt_limiter.wait()
    for attempt in range(5):
        try:
            t = time.monotonic()
            resp = _genai_client().models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=audio, mime_type=mime_type)],
            )
            trace.record(model=model, op="stt", wait_s=round(waited, 3),
                         call_s=round(time.monotonic() - t, 3))
            return _transcript(resp)
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 429:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            raise
    raise QuotaExhausted(f"{model}: rate limited after 5 attempts")


def _audio_blob(resp) -> types.Blob:
    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data
    raise RuntimeError("TTS returned no audio")


def _sample_rate(mime_type: str) -> int:
    match = re.search(r"rate=(\d+)", mime_type or "")
    return int(match.group(1)) if match else 24000


def _transcript(resp) -> str:
    for part in resp.candidates[0].content.parts:
        if part.audio_transcription is not None:
            return part.audio_transcription.text or ""
    parts = resp.candidates[0].content.parts
    return (parts[0].text or "") if parts else ""
