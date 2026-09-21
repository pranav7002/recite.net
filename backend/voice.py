"""Cascaded voice pipeline: Gemini speech-to-text and text-to-speech around the
existing answer loop.

STT, the answer and TTS are blocking network calls, so the turn is an async
generator that offloads each stage to a worker thread and streams audio back
sentence by sentence. The headline metric is time-to-first-audio: the first
sentence is synthesised while the rest of the answer is still being produced.

Three configurations, selected by `config`, share one code path:

- ``sequential``      full answer -> one TTS call -> send the whole thing once
- ``stream``          one TTS call per sentence, after the grounding check
- ``stream_nocheck``  stream per sentence, but skip the grounding check, so the
                      latency cost of the check can be measured directly
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
import time
import wave
from collections.abc import AsyncIterator, Callable

from backend import agent, llm

# A sentence boundary needs a capital letter (or quote/bracket) to follow, so
# a full stop inside a citation such as "(notes.pdf, p. 3)" never splits — the
# same lesson as the grounding claim splitter (iteration 8 in the README).
SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'\[$\\])|\n{2,}")

TTS_SAMPLE_WIDTH = 2        # Gemini TTS returns L16: 16-bit PCM
TTS_CHANNELS = 1            # mono
VALID_CONFIGS = ("sequential", "stream", "stream_nocheck")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences for TTS streaming, one audio chunk per
    sentence so the first can play while the rest is still synthesised."""
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def _pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    """Wrap raw PCM in a WAV header so the browser can play it directly."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(TTS_CHANNELS)
        w.setsampwidth(TTS_SAMPLE_WIDTH)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def voice_turn(
    audio: bytes,
    history: list[dict] | None,
    t0: float,
    config: str = "stream",
    *,
    stt: Callable[[bytes], str] = llm.transcribe,
    tts: Callable[[str], tuple[bytes, int]] = llm.synthesize,
    answer: Callable[..., agent.TurnResult] = agent.answer,
    record: Callable[[str, str], None] | None = None,
) -> AsyncIterator[dict]:
    """Run one spoken turn, yielding SSE-ready dicts.

    t0 is the browser's timestamp of the last audio frame (ms). All stage marks
    are server times relative to request arrival; the browser measures the true
    time-to-first-audio from t0. `record`, when given, is called with the
    (question, answer) pair so the caller can update session history.
    """
    if config not in VALID_CONFIGS:
        raise ValueError(f"unknown voice config {config!r}; expected one of {VALID_CONFIGS}")

    arrival = time.monotonic()
    marks: dict[str, float] = {"t0_ms": t0}

    text = await asyncio.to_thread(stt, audio)
    marks["stt_ms"] = round((time.monotonic() - arrival) * 1000, 1)

    result = await asyncio.to_thread(answer, text, history=history,
                                     check_grounding=config != "stream_nocheck")
    marks["answer_ready_ms"] = round((time.monotonic() - arrival) * 1000, 1)
    if record is not None:
        record(text, result.text)

    if config == "sequential":
        pcm, rate = await asyncio.to_thread(tts, result.text)
        marks["first_audio_ms"] = round((time.monotonic() - arrival) * 1000, 1)
        yield {"seq": 0, "mime_type": "audio/wav",
               "audio_b64": base64.b64encode(_pcm_to_wav(pcm, rate)).decode(),
               "text": result.text}
    else:
        for i, sentence in enumerate(split_sentences(result.text)):
            pcm, rate = await asyncio.to_thread(tts, sentence)
            if i == 0:
                marks["first_audio_ms"] = round((time.monotonic() - arrival) * 1000, 1)
            yield {"seq": i, "mime_type": "audio/wav",
                   "audio_b64": base64.b64encode(_pcm_to_wav(pcm, rate)).decode(),
                   "text": sentence}

    yield {"done": True, "marks": marks, "trace": {
        "config": config, "grounding": result.grounding, "retried": result.retried,
        "steps": result.steps, "passages": len(result.passages),
        "pending_action": result.pending_action,
    }}
