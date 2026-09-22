"""Local speech: faster-whisper STT and Piper TTS, on-device so no audio leaves
the machine. Both models load lazily once and are cached for the process.

Run `python scripts/download_models.py` first to fetch the Piper voice; the
Whisper model is fetched automatically by faster-whisper on first use.
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")

_lock = threading.Lock()
_whisper = None
_piper = None


def _load_whisper():
    global _whisper
    with _lock:
        if _whisper is None:
            from faster_whisper import WhisperModel

            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                                    download_root=str(MODELS_DIR))
        return _whisper


def _load_piper():
    global _piper
    with _lock:
        if _piper is None:
            from piper import PiperVoice

            onnx = MODELS_DIR / f"{PIPER_VOICE}.onnx"
            config = MODELS_DIR / f"{PIPER_VOICE}.onnx.json"
            if not onnx.exists():
                raise RuntimeError(
                    f"Piper voice {PIPER_VOICE!r} not found in {MODELS_DIR}. "
                    "Run `python scripts/download_models.py`."
                )
            _piper = PiperVoice.load(str(onnx), str(config))
        return _piper


def transcribe(audio: bytes) -> str:
    """Speech to text with faster-whisper; audio is any format PyAV decodes."""
    model = _load_whisper()
    # faster-whisper decodes a path or a file-like object, not raw bytes.
    segments, _info = model.transcribe(io.BytesIO(audio))
    return "".join(s.text for s in segments).strip()


def synthesize(text: str) -> tuple[bytes, int]:
    """Text to speech with Piper. Returns (16-bit mono PCM, sample rate)."""
    voice = _load_piper()
    pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(text))
    return pcm, voice.config.sample_rate


def warm() -> None:
    """Load both models and run each once, so the first real turn does not pay
    model loading or first-inference setup (that would pollute time-to-first-audio)."""
    import numpy as np

    segments, _ = _load_whisper().transcribe(np.zeros(16000, dtype=np.float32))
    list(segments)                       # transcribe is lazy; consume it
    synthesize("Ready.")
