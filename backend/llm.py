"""Gemini API client, rate limiter, and embedding helpers.

Synchronous on purpose: ingest runs it from a worker thread, so the rate
limiter's sleeps never block the event loop.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("recite.llm")

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
# Free-tier limits vary per project; these are conservative defaults. Put the
# real numbers from your AI Studio rate-limit page in .env.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))
EMBED_RPM = float(os.getenv("EMBED_RPM", "5"))
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "768"))

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set — add it to .env")
        _client = genai.Client(api_key=api_key)
    return _client


class RateLimiter:
    """Spaces calls out evenly so a per-minute limit is never exceeded."""

    def __init__(self, rpm: float) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_call - now
            if delay > 0:
                log.info("rate limiter: sleeping %.1fs", delay)
                time.sleep(delay)
                now = time.monotonic()
            self._next_call = max(now, self._next_call) + self._interval


_embed_limiter = RateLimiter(EMBED_RPM)


def embed(
    texts: list[str],
    model: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    dimensions: int = EMBED_DIMENSIONS,
) -> list[list[float]]:
    """Embed texts in batches; every API call passes through the rate limiter."""
    if not texts:
        return []
    model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    client = _get_client()
    out: list[list[float]] = []
    started = time.monotonic()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        _embed_limiter.wait()
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config={"output_dimensionality": dimensions},
        )
        out.extend(list(e.values) for e in response.embeddings)
    log.info(
        "embedded %d texts with %s (%d dim) in %.1fs",
        len(texts), model, dimensions, time.monotonic() - started,
    )
    return out


def normalise(vectors: list[list[float]] | np.ndarray) -> np.ndarray:
    """L2-normalise each row so cosine similarity is a dot product."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)
