"""
stt.py — Standalone Speech-to-Text demo using faster-whisper.

What this teaches:
  - Loading a Whisper model with quantisation (int8 = fast CPU inference)
  - Transcribing a WAV file and reading segment-level output
  - Measuring model load time vs. transcription time separately

Usage (from repo root):
  uv run --project backend python scripts/stt.py path/to/audio.wav
  uv run --project backend python scripts/stt.py          # uses built-in demo sine wave
"""

import sys
import wave
import struct
import math
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — change these or set via env
# ---------------------------------------------------------------------------
MODEL_SIZE = "small.en"   # tiny.en | base.en | small.en | medium.en | large-v3
DEVICE = "cpu"
COMPUTE_TYPE = "int8"     # int8 (fast) | float16 (GPU) | float32 (slow CPU)
LANGUAGE = "en"
VAD_FILTER = True


def generate_demo_wav(path: Path, frequency: float = 440.0, duration: float = 2.0, sample_rate: int = 16000) -> None:
    """Write a pure sine-wave WAV so the script runs without a real audio file."""
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        frames = []
        for i in range(int(sample_rate * duration)):
            sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            frames.append(struct.pack("<h", sample))
        f.writeframes(b"".join(frames))
    print(f"[demo] Generated sine-wave WAV → {path}")


def transcribe(audio_path: Path) -> None:
    from faster_whisper import WhisperModel  # type: ignore

    print(f"\n{'─'*50}")
    print(f"  Model : {MODEL_SIZE}  device={DEVICE}  compute={COMPUTE_TYPE}")
    print(f"  File  : {audio_path}")
    print(f"{'─'*50}")

    # 1. Load model — time it separately (cached after first run)
    t0 = time.perf_counter()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"  Model loaded in {load_ms} ms")

    # 2. Transcribe
    t1 = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language=LANGUAGE or None,
        beam_size=1,
        vad_filter=VAD_FILTER,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    transcribe_ms = round((time.perf_counter() - t1) * 1000, 1)

    print(f"  Language detected : {info.language} (prob={info.language_probability:.2f})")
    print(f"  Transcription time: {transcribe_ms} ms")
    print(f"\n  Transcript → \"{text or '[no speech detected]'}\"\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio = Path(sys.argv[1])
        if not audio.exists():
            print(f"Error: file not found → {audio}")
            sys.exit(1)
    else:
        audio = Path("/tmp/neurotalk_demo.wav")
        generate_demo_wav(audio)

    transcribe(audio)
