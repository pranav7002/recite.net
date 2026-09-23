"""Cascaded voice pipeline: STT → answer loop → grounding → TTS, streamed back
sentence by sentence.

Speech runs **locally** (faster-whisper + Piper) by default, so no audio leaves
the machine and the only network hops in a turn are the model calls. Set
``SPEECH=gemini`` to use Gemini STT/TTS instead (for a comparison row).

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
import os
import re
import time
import wave
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from dotenv import load_dotenv

from backend import agent, llm, speech, trace
from backend.speakable import speakable

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# A sentence boundary needs a capital letter (or quote/bracket) to follow, so
# a full stop inside a citation such as "(notes.pdf, p. 3)" never splits — the
# same lesson as the grounding claim splitter (iteration 8 in the README).
SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'\[$\\])|\n{2,}")

TTS_SAMPLE_WIDTH = 2        # 16-bit PCM, mono
TTS_CHANNELS = 1
VALID_CONFIGS = ("sequential", "stream", "stream_nocheck")

SPEECH = os.getenv("SPEECH", "local")   # "local" (faster-whisper + Piper) | "gemini"


def stt(audio: bytes) -> str:
    if SPEECH == "gemini":
        return llm.transcribe(audio)
    return speech.transcribe(audio)


def tts(text: str) -> tuple[bytes, int]:
    if SPEECH == "gemini":
        return llm.synthesize(text)
    return speech.synthesize(text)


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


async def spoken_reply(reply_text: str, t0: float, tts_fn: Callable[[str], tuple[bytes, int]] = tts) -> AsyncIterator[dict]:
    """Speak a short, fixed reply (used for confirm/cancel and empty audio)."""
    pcm, rate = await asyncio.to_thread(tts_fn, reply_text)
    yield {"seq": 0, "mime_type": "audio/wav",
           "audio_b64": base64.b64encode(_pcm_to_wav(pcm, rate)).decode(),
           "text": reply_text}
    yield {"done": True, "marks": {"t0_ms": t0}, "trace": {"config": "reply"}}


async def voice_turn(
    audio: bytes,
    history: list[dict] | None,
    t0: float,
    config: str = "stream",
    *,
    text: str | None = None,
    stt: Callable[[bytes], str] = stt,
    tts: Callable[[str], tuple[bytes, int]] = tts,
    answer: Callable[..., agent.TurnResult] = agent.answer,
    record: Callable[[str, str], None] | None = None,
    pending: Callable[[dict], None] | None = None,
    intercept: Callable[[str], str | None] | None = None,
) -> AsyncIterator[dict]:
    """Run one spoken turn, yielding SSE-ready dicts.

    t0 is the browser's timestamp of the last audio frame (ms). All stage marks
    are server times relative to request arrival; the browser measures the true
    time-to-first-audio from t0. `record` is called with (question, answer);
    `pending` is called with the pending_action dict when the turn pauses for a
    confirmation, so the caller can remember it for a spoken "confirm"/"cancel".
    `intercept` sees the transcript before the answer loop; if it returns a
    reply, that reply is spoken and the turn ends without a model call (used
    for spoken confirm/cancel, which must be decided in code, not by a model).
    """
    if config not in VALID_CONFIGS:
        raise ValueError(f"unknown voice config {config!r}; expected one of {VALID_CONFIGS}")

    arrival = time.monotonic()
    marks: dict[str, float] = {"t0_ms": t0}

    transcript = (text if text is not None else await asyncio.to_thread(stt, audio)).strip()
    marks["stt_ms"] = round((time.monotonic() - arrival) * 1000, 1)

    # A blank or silent recording: return without spending a model call.
    if not transcript:
        yield {"done": True, "marks": marks, "text": "I didn't hear anything. Try again.",
               "trace": {"config": config, "empty": True}}
        return

    # Sent before the answer loop runs, so the browser can show what it heard
    # while the (often much slower) answer is still being generated.
    yield {"transcript": transcript}

    if intercept is not None:
        reply = await asyncio.to_thread(intercept, transcript)
        if reply is not None:
            pcm, rate = await asyncio.to_thread(tts, reply)
            marks["first_audio_ms"] = round((time.monotonic() - arrival) * 1000, 1)
            yield {"seq": 0, "mime_type": "audio/wav",
                   "audio_b64": base64.b64encode(_pcm_to_wav(pcm, rate)).decode(),
                   "text": reply}
            yield {"done": True, "marks": marks,
                   "trace": {"config": config, "intercepted": True, "transcript": transcript}}
            return

    # capture() collects every model call's wait/backoff/call time (to_thread
    # copies this context into the worker), so the latency runner can report
    # time-to-first-audio with the free-tier limiter waits taken out.
    with trace.capture() as calls:
        result = await asyncio.to_thread(answer, transcript, history=history,
                                         check_grounding=config != "stream_nocheck")
    marks["answer_ready_ms"] = round((time.monotonic() - arrival) * 1000, 1)
    t = trace.timings(calls)
    marks.update(wait_ms=round(t["wait_s"] * 1000, 1), backoff_ms=round(t["backoff_s"] * 1000, 1),
                 model_ms=round(t["model_s"] * 1000, 1), model_calls=t["calls"])
    if record is not None:
        record(transcript, result.text)
    if pending is not None and result.pending_action is not None:
        pending(result.pending_action)

    if config == "sequential":
        pcm, rate = await asyncio.to_thread(tts, speakable(result.text))
        marks["first_audio_ms"] = round((time.monotonic() - arrival) * 1000, 1)
        yield {"seq": 0, "mime_type": "audio/wav",
               "audio_b64": base64.b64encode(_pcm_to_wav(pcm, rate)).decode(),
               "text": result.text}
    else:
        for i, sentence in enumerate(split_sentences(result.text)):
            pcm, rate = await asyncio.to_thread(tts, speakable(sentence))
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
