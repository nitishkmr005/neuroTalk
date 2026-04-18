# NeuroTalk

Real-time voice agent console — live speech transcription with AI-assisted responses.

Designed for two contexts: **customer-facing** (direct query answering) and **associate-facing** (live call assist with database/article lookup). A voice response layer with emotional expressiveness is on the roadmap.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 15 · TypeScript · Lora + DM Serif Display fonts |
| Backend | FastAPI · Python 3.11+ · uv |
| STT | faster-whisper (`small.en`, int8, CPU) |
| LLM | Ollama (local) — `gemma4` |
| TTS | Kokoro 82M MLX (default) · Chatterbox Turbo · Qwen · VibeVoice |
| Transport | WebSocket streaming |
| Config | Pydantic Settings + `.env` |
| Logging | Loguru — colorful terminal + rotating JSON files |

## Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Set up Ollama (LLM — local, no API key)
brew install ollama
ollama pull llama3.2
ollama serve          # runs at http://localhost:11434

# 3. Run both services
make dev
# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
```

## Environment Variables

Copy `backend/.env.example` → `backend/.env` and adjust as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL_SIZE` | `small.en` | Whisper model (`tiny.en` → `large-v3`) |
| `STT_DEVICE` | `cpu` | `cpu` or `cuda` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `gemma4:latest` | Any model pulled via `ollama pull` |
| `LLM_MAX_TOKENS` | `150` | Max tokens per LLM response |
| `TTS_BACKEND` | `kokoro` | TTS engine — see below |

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
│   │   ├── services/     # STT and LLM service modules
│   │   ├── prompts/      # System prompts
│   │   ├── agents/       # Agent orchestrators
│   │   ├── tools/        # Tool definitions
│   │   ├── utils/        # Shared utilities
│   │   └── modules/      # Reusable modules
│   ├── config/           # Settings + logging
│   └── logs/             # JSON log files (latest 5 kept)
├── frontend/             # Next.js app
├── scripts/              # Standalone learnable Python demos
│   ├── stt.py            # STT only
│   ├── llm_call.py       # LLM only
│   ├── tts.py            # TTS only
│   └── agent.py          # Full pipeline
├── docs/
│   └── blog.md           # Project explainer
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
