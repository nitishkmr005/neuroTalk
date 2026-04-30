# NeuroTalk

A fully local, real-time voice agent with selectable `WebRTC` and `WebSocket` transports — speak into your browser, get a spoken reply back. No cloud APIs. No API keys. Everything runs on your machine.

![NeuroTalk voice agent console](docs/images/neurotalk-console-preview.png)

---

## What is NeuroTalk?

NeuroTalk is a conversational voice agent that runs the full speech-to-speech pipeline locally:

```
Microphone → Denoise → STT (Whisper) → LLM (Ollama / llama-cpp) → TTS (Kokoro/Chatterbox) → Speaker
```

Audio streams over WebRTC (or WebSocket). The backend transcribes speech incrementally, applies deep neural noise suppression before the STT model sees the signal, uses a semantic **Smart Turn** gate to decide when you have actually finished your thought, then starts the LLM call and begins synthesizing audio as soon as the first sentence is ready — so the agent starts talking in parallel with its own thinking rather than waiting for a complete response.

## Key Features

- **Fully local** — STT, LLM, and TTS all run on your hardware; nothing leaves your machine
- **Low-latency streaming** — partial transcripts appear live as you speak; TTS playback starts mid-generation
- **Interrupt support** — speak over the agent at any time to cancel and start a new turn
- **Smart Turn turn-taking** — semantic utterance-completion model prevents premature replies on mid-sentence pauses
- **DeepFilterNet3 denoising** — server-side deep neural noise suppression applied before Whisper for cleaner transcripts
- **Hallucination filtering** — `log_prob_threshold` discards low-confidence Whisper segments before they reach the LLM
- **Swappable models** — change STT size, LLM, or TTS engine with a single env var
- **Learnable codebase** — each pipeline stage has a standalone script you can run independently

## Demo

https://github.com/nitishkmr005/neuroTalk/raw/main/docs/NeuroTalk_Social_Media_V3.mp4

In the preview above, `Response Generation: 14.16 s` reflects the time the LLM spends generating the reply token by token in real time.
Because text streaming and voice streaming are synchronized, TTS starts speaking as tokens arrive, so playback overlaps with generation instead of waiting for the full response.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 15 · TypeScript · Lora + DM Serif Display fonts |
| Backend | FastAPI · Python 3.11+ · uv |
| Transport (audio in) | **WebRTC / RTP** (Opus codec, `aiortc` + `PyAV`) · WebSocket PCM streaming |
| Transport (agent out) | **RTCDataChannel** (ordered JSON) · WebSocket JSON |
| Denoise | **DeepFilterNet3** (server-side, deep neural noise suppression before STT) |
| STT | faster-whisper (`small.en`, int8, CPU) — via CTranslate2 |
| Turn-taking | **Smart Turn v3.2** (Whisper-based ONNX semantic completion gate) |
| VAD | `Silero VAD` for streaming endpointing/barge-in · RMS fallback when disabled |
| LLM | Ollama (local) or llama-cpp (GGUF) — `llama3.2:3b` (default) |
| TTS | Kokoro 82M MLX (default) · Chatterbox Turbo · Qwen · VibeVoice |
| ICE / NAT traversal | STUN (`stun.l.google.com:19302`) · Vanilla ICE (full gather before offer) |
| Config | Pydantic Settings + `.env` |
| Logging | Loguru — colorful terminal + rotating JSON files |

## Quick Start

```bash
# 1. Install system dependency (macOS — required by aiortc for SRTP)
brew install libsrtp

# 2. Install project dependencies
make setup

# 3. Set up Ollama (LLM — local, no API key)
brew install ollama
ollama pull llama3.2:3b
ollama serve          # runs at http://localhost:11434

# 4. Run both services
make dev
# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
```

> **Linux:** replace `brew install libsrtp` with `apt-get install libsrtp2-dev` (Debian/Ubuntu).

## Transports

NeuroTalk supports two transport modes selectable in the UI:

| Mode | Audio path | Signalling |
|------|-----------|------------|
| **WebRTC** (default) | Browser mic → Opus RTP → UDP → aiortc → PCM 16kHz | RTCDataChannel (JSON) |
| **WebSocket** | Browser mic → Float32 PCM → WebSocket binary frames | Same WebSocket (JSON) |

**WebRTC** (default) sends Opus-compressed audio over UDP with browser-native echo cancellation, noise suppression, and auto-gain — use this unless UDP is blocked by a firewall.
**WebSocket** sends raw PCM over TCP with no echo cancellation and higher bandwidth; switch to it if WebRTC fails to connect.

## Pipeline overview

```
Browser mic
  │  AEC + NS + AGC (browser-native)
  │  Opus → RTP → UDP
  ▼
Server
  │  Opus decode → PCM 16 kHz
  ├─ Silero VAD  (speech_start → barge-in; speech_end → turn gate)
  │
  │  DeepFilterNet3 denoise
  │  faster-whisper STT  (log_prob_threshold -0.7)
  │  Smart Turn v3.2     (semantic completeness check)
  │
  ▼
  LLM (sentence streaming)
  TTS (per sentence, async)
  RTCDataChannel → browser playback
```

## Environment Variables

Copy `backend/.env.example` → `backend/.env` and adjust as needed.

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL_SIZE` | `small.en` | Whisper model (`tiny.en` → `large-v3`) |
| `STT_DEVICE` | `cpu` | `cpu` or `cuda` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.2:3b` | Any model pulled via `ollama pull` |
| `LLM_MAX_HISTORY_TURNS` | `6` | Conversation turns kept in context |
| `TTS_BACKEND` | `kokoro` | TTS engine — see below |
| `WELCOME_MESSAGE` | `Hello! I'm your Neurotalk voice assistant...` | Spoken greeting on session start; empty disables it |

### Denoise (DeepFilterNet3)

| Variable | Default | Description |
|----------|---------|-------------|
| `DENOISE_ENABLED` | `true` | Apply DeepFilterNet3 before STT. Set `false` to skip (faster, noisier) |

> **Setup:** `uv sync --group deepfilter && python scripts/download_deepfilter_model.py`

### Streaming / debounce

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_EMIT_INTERVAL_MS` | `700` | Minimum gap between partial STT emits |
| `STREAM_MIN_AUDIO_MS` | `500` | Minimum buffered audio before STT runs |
| `STREAM_LLM_MIN_CHARS` | `8` | Minimum transcript length before starting the LLM |
| `STREAM_LLM_SILENCE_MS` | `500` | Silence debounce fallback when VAD end does not fire cleanly |

### VAD (Silero)

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_VAD_ENABLED` | `true` | Enable dedicated streaming voice activity detection |
| `STREAM_VAD_THRESHOLD` | `0.6` | Speech probability threshold for VAD start detection |
| `STREAM_VAD_MIN_SILENCE_MS` | `500` | Required silence before VAD emits speech end |
| `STREAM_VAD_SPEECH_PAD_MS` | `250` | Extra speech padding kept around VAD boundaries |
| `STREAM_VAD_FRAME_SAMPLES` | `512` | Frame size fed into the streaming VAD at 16 kHz |

### Smart Turn (semantic turn-taking)

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_SMART_TURN_ENABLED` | `true` | Enable semantic turn-completion gating after VAD silence |
| `STREAM_SMART_TURN_THRESHOLD` | `0.65` | Completion probability threshold (0–1) |
| `STREAM_SMART_TURN_MAX_BUDGET_MS` | `600` | Max extra wait while polling for completion |
| `STREAM_SMART_TURN_INCOMPLETE_WAIT_MS` | `1500` | Extra grace period when Smart Turn says "not complete yet" |

> **Setup:** `uv sync --group smart_turn && python scripts/download_smart_turn_model.py`

## Switching LLM Models

NeuroTalk uses Ollama (or llama-cpp) for local LLM inference. Switching models is one line.

**Available models (fast → quality):**

| Model | `LLM_MODEL` value | Notes |
|-------|-------------------|-------|
| Llama 3.2 3B | `llama3.2:3b` | **Default.** Fast, low RAM (~2 GB). |
| Qwen3 4B | `qwen3:4b` | Fast, strong tool-calling (~3 GB). |
| Gemma 3 1B | `gemma3:1b` | Fastest, minimal memory (~1 GB). |
| Gemma 3 4B | `gemma3:4b` | Better quality (~3 GB RAM). |
| Llama 3.2 1B | `llama3.2:1b` | Similar speed to gemma3:1b. |
| Mistral 7B | `mistral` | Strong general model. |

```bash
# 1. Pull the model
ollama pull qwen3:4b

# 2. Set in backend/.env
LLM_MODEL=qwen3:4b

# 3. Restart backend
make backend
```

One-liner (no .env edit):
```bash
LLM_MODEL=qwen3:4b make backend
```

> To use a non-Ollama provider (OpenAI, Anthropic, etc.), update `backend/app/services/llm.py` to call the respective SDK instead of the Ollama client.

---

## Switching STT Models

NeuroTalk uses `faster-whisper` for speech recognition.

**Whisper model sizes:**

| `STT_MODEL_SIZE` | Speed | Accuracy | RAM |
|------------------|-------|----------|-----|
| `tiny.en` | ~4× faster | Lower | ~200 MB |
| `small.en` | **Default** | Good | ~500 MB |
| `medium.en` | Slower | Better | ~1.5 GB |
| `large-v3` | Slowest | Best | ~3 GB |

```bash
# Set in backend/.env
STT_MODEL_SIZE=tiny.en   # for speed
STT_MODEL_SIZE=large-v3  # for accuracy

make backend
```

**Using Google Speech Recognition (or other providers):**
Replace `backend/app/services/stt.py` with a client for the desired provider. The service must implement `transcribe(*, file_path, request_id, filename, audio_bytes) -> ServiceResult`. All other code stays the same.

---

## Switching TTS Models

Four TTS engines are available. Only one is installed at a time.

| Backend | Value | Notes |
|---------|-------|-------|
| Kokoro 82M MLX | `kokoro` | **Default.** Fast, natural. Apple Silicon only. |
| Chatterbox Turbo | `chatterbox` | Emotion tag support. Requires PyTorch. |
| Qwen TTS | `qwen` | Requires PyTorch. |
| VibeVoice | `vibevoice` | Requires PyTorch. |

**To switch:**

```bash
# 1. Set the backend in backend/.env
TTS_BACKEND=chatterbox   # or kokoro / qwen / vibevoice

# 2. Reinstall backend deps with the new model group
make backend-install TTS_BACKEND=chatterbox

# 3. Restart the backend
make backend
```

One-liner (no .env change needed):
```bash
make dev TTS_BACKEND=chatterbox
```

> **Note:** `kokoro` uses `mlx-audio` which requires Apple Silicon (macOS). For Linux/cloud deployment, use `chatterbox` or `qwen`.

## Project Structure

```
neuroTalk/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── main.py       # WebSocket route + app startup/warmup
│   │   ├── webrtc/       # WebRTC transport
│   │   │   ├── router.py     # POST /webrtc/offer, DELETE /webrtc/session/{id}
│   │   │   └── session.py    # RTCPeerConnection, RTP consumer, VAD, STT→LLM→TTS
│   │   ├── services/     # STT, LLM, TTS, VAD, Denoise, SmartTurn modules
│   │   │   ├── stt.py        # faster-whisper wrapper
│   │   │   ├── llm.py        # multi-provider LLM streaming
│   │   │   ├── tts.py        # Kokoro / Chatterbox / Qwen / VibeVoice
│   │   │   ├── vad.py        # Silero VAD streaming
│   │   │   ├── denoise.py    # DeepFilterNet3 noise suppression
│   │   │   └── smart_turn.py # ONNX semantic turn-completion detector
│   │   ├── prompts/      # System prompts
│   │   ├── utils/        # Shared utilities (emotion tag cleaning, etc.)
│   │   └── models.py     # Pydantic response models
│   ├── config/           # Settings + logging
│   ├── scripts/          # Model download helpers
│   └── logs/             # JSON log files (latest 5 kept)
├── frontend/             # Next.js app
│   └── components/
│       ├── voice-agent-console.tsx   # Main UI — WebRTC + WS mode toggle
│       └── webrtc-transport.ts       # RTCPeerConnection + RTCDataChannel client
├── scripts/              # Standalone learnable Python demos
│   ├── stt.py            # STT only
│   ├── llm_call.py       # LLM only
│   ├── tts.py            # TTS only
│   └── agent.py          # Full pipeline
├── docs/
│   └── blog.md           # End-to-end pipeline deep-dive
└── Makefile
```

## Learnable Scripts

Run each module independently to understand how it works:

```bash
# STT — transcribe a WAV file
uv run --project backend python scripts/stt.py path/to/audio.wav

# LLM — stream a response from Ollama
uv run --project backend python scripts/llm_call.py "Reset my password"

# TTS — speak text aloud
uv run --project backend python scripts/tts.py "Hello, how can I help?"

# Full pipeline — audio → transcript → LLM → speech
uv run --project backend python scripts/agent.py path/to/audio.wav
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start backend + frontend (with port cleanup) |
| `make backend` | Backend only (hot-reload) |
| `make frontend` | Frontend only |
| `make setup` | Install all dependencies |
| `make check` | Lint + type check |
| `make tts-envs` | Install isolated venvs for all TTS models |
| `make tts-report` | Run all TTS models and save comparison report to `scripts/speech/` |

## Deep-dive

For a step-by-step technical explanation of every stage — WebRTC, VAD, DeepFilterNet3, Smart Turn, STT, LLM, TTS, and interrupt handling — see [`docs/blog.md`](docs/blog.md).
