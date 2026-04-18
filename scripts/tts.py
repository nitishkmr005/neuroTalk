"""
tts.py — Standalone Text-to-Speech demo using the system TTS engine.

What this teaches:
  - Using the OS-native TTS (no API key, no model download)
  - macOS: `say` command  |  Linux: `espeak`  |  Windows: pyttsx3 pattern
  - How TTS fits into the voice agent pipeline: LLM text → speech output

Usage (from repo root):
  uv run --project backend python scripts/tts.py "Hello, how can I help you today?"
  uv run --project backend python scripts/tts.py   # uses default text
"""

import platform
import subprocess
import sys
import time

DEFAULT_TEXT = "Hello! I am your voice assistant. How can I help you today?"


def speak(text: str) -> None:
    system = platform.system()
    print(f"\n{'─'*50}")
    print(f"  Platform : {system}")
    print(f"  Text     : {text!r}")
    print(f"{'─'*50}")

    t0 = time.perf_counter()

    if system == "Darwin":
        # macOS built-in — supports many voices, no install needed
        subprocess.run(["say", text], check=True)

    elif system == "Linux":
        # espeak: sudo apt install espeak
        try:
            subprocess.run(["espeak", text], check=True)
        except FileNotFoundError:
            print("  [info] espeak not found. Install with: sudo apt install espeak")
            print(f"  [text] {text}")

    elif system == "Windows":
        # PowerShell built-in speech synthesizer
        ps_cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
        subprocess.run(["powershell", "-Command", ps_cmd], check=True)

    else:
        print(f"  [unsupported] Platform {system!r} — printing text instead:")
        print(f"  {text}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"\n  Speech rendered in {elapsed_ms} ms\n")


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TEXT
    speak(text)
