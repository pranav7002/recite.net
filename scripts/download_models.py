"""Download the local speech models into data/models/.

- Piper voice (onnx + config) from HuggingFace.
- Whisper small.en (int8), which faster-whisper fetches on first use; we trigger
  that here so startup is not slow.
"""
from __future__ import annotations

from pathlib import Path

import requests

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
PIPER_VOICE = "en_US-lessac-medium"


def _piper_url(voice: str, ext: str) -> str:
    lang_region, speaker, quality = voice.split("-")[:3]
    lang = lang_region.split("_")[0]
    return (
        f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        f"{lang}/{lang_region}/{speaker}/{quality}/{voice}{ext}"
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".onnx", ".onnx.json"):
        dest = MODELS_DIR / f"{PIPER_VOICE}{ext}"
        if dest.exists():
            print(f"already present: {dest.name}")
            continue
        url = _piper_url(PIPER_VOICE, ext)
        print(f"downloading {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"wrote {dest.name} ({len(r.content)} bytes)")

    from faster_whisper import WhisperModel

    print("fetching Whisper small.en (int8)...")
    WhisperModel("small.en", device="cpu", compute_type="int8",
                 download_root=str(MODELS_DIR))
    print("done")


if __name__ == "__main__":
    main()
