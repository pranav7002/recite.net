"""Voice pipeline tests: sentence splitting and the three streaming configs.

stt, tts and answer are injected as fakes, so no test calls Gemini or the store.
"""
from __future__ import annotations

import asyncio
import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend import agent, api, voice


def _run(agen):
    async def collect():
        return [item async for item in agen]
    return asyncio.run(collect())


def _fake_answer(grounding_calls, text="First answer. Second answer."):
    def answer(question, history=None, check_grounding=True):
        grounding_calls.append(check_grounding)
        return agent.TurnResult(text=text,
                                grounding="skipped" if not check_grounding else "supported")
    return answer


def _fake_tts():
    def tts(text):
        return (b"\x00\x01\x02\x03", 24000)
    return tts


def test_split_sentences_keeps_citation_attached():
    text = "The mean is 50. See (notes.pdf, p. 3) for the formula. That is it."
    assert voice.split_sentences(text) == [
        "The mean is 50.",
        "See (notes.pdf, p. 3) for the formula.",
        "That is it.",
    ]


def test_sequential_sends_one_audio_chunk():
    items = _run(voice.voice_turn(b"audio", None, 0.0, "sequential",
                                  stt=lambda b: "q", tts=_fake_tts(),
                                  answer=_fake_answer([])))
    audio_items = [i for i in items if "audio_b64" in i]
    assert len(audio_items) == 1
    assert audio_items[0]["text"] == "First answer. Second answer."
    assert items[-1]["done"] is True


def test_stream_sends_one_chunk_per_sentence():
    items = _run(voice.voice_turn(b"audio", None, 0.0, "stream",
                                  stt=lambda b: "q", tts=_fake_tts(),
                                  answer=_fake_answer([])))
    audio_items = [i for i in items if "audio_b64" in i]
    assert [i["text"] for i in audio_items] == ["First answer.", "Second answer."]
    assert [i["seq"] for i in audio_items] == [0, 1]


def test_stream_nocheck_skips_grounding():
    calls = []
    items = _run(voice.voice_turn(b"audio", None, 0.0, "stream_nocheck",
                                  stt=lambda b: "q", tts=_fake_tts(),
                                  answer=_fake_answer(calls)))
    assert calls == [False]
    assert items[-1]["trace"]["grounding"] == "skipped"


def test_stream_grounds_by_default():
    calls = []
    _run(voice.voice_turn(b"audio", None, 0.0, "stream",
                          stt=lambda b: "q", tts=_fake_tts(),
                          answer=_fake_answer(calls)))
    assert calls == [True]


def test_audio_is_wav_wrapped():
    items = _run(voice.voice_turn(b"audio", None, 0.0, "sequential",
                                  stt=lambda b: "q", tts=_fake_tts(),
                                  answer=_fake_answer([])))
    import base64
    raw = base64.b64decode(items[0]["audio_b64"])
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    with wave.open(io.BytesIO(raw), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1


def test_marks_recorded():
    items = _run(voice.voice_turn(b"audio", None, 1234.5, "sequential",
                                  stt=lambda b: "q", tts=_fake_tts(),
                                  answer=_fake_answer([])))
    marks = items[-1]["marks"]
    assert marks["t0_ms"] == 1234.5
    for key in ("stt_ms", "answer_ready_ms", "first_audio_ms"):
        assert marks[key] >= 0


def test_record_callback_receives_turn():
    recorded = []
    _run(voice.voice_turn(b"audio", None, 0.0, "stream",
                          stt=lambda b: "what is it", tts=_fake_tts(),
                          answer=_fake_answer([]),
                          record=lambda q, a: recorded.append((q, a))))
    assert recorded == [("what is it", "First answer. Second answer.")]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        _run(voice.voice_turn(b"audio", None, 0.0, "bogus",
                              stt=lambda b: "q", tts=_fake_tts(),
                              answer=_fake_answer([])))


def test_voice_endpoint_invalid_config_422():
    client = TestClient(api.app)
    resp = client.post("/voice", data={"session_id": "s", "t0": "0", "config": "bogus"},
                       files={"file": ("q.wav", b"\x00\x00", "audio/wav")})
    assert resp.status_code == 422


def test_voice_endpoint_streams_sse(monkeypatch):
    async def fake_voice_turn(audio, history, t0, config, record=None):
        yield {"seq": 0, "text": "hello"}
        yield {"done": True}

    monkeypatch.setattr(voice, "voice_turn", fake_voice_turn)
    client = TestClient(api.app)
    resp = client.post("/voice", data={"session_id": "s", "t0": "0", "config": "stream"},
                       files={"file": ("q.wav", b"\x00\x00", "audio/wav")})
    assert resp.status_code == 200
    assert "data: " in resp.text
    assert '"seq": 0' in resp.text
