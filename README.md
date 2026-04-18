# NeuroTalk

Real-time voice agent console — live speech transcription with AI-assisted responses.

Designed for two contexts: **customer-facing** (direct query answering) and **associate-facing** (live call assist with database/article lookup). A voice response layer with emotional expressiveness is on the roadmap.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 15 · TypeScript · Lora + DM Serif Display fonts |
| Backend | FastAPI · Python 3.11+ · uv |
| STT | faster-whisper (`small.en`, int8, CPU) |
| LLM | Ollama (local) — `llama3.2` |
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
| `LLM_MODEL` | `llama3.2` | Any model pulled via `ollama pull` |
| `LLM_MAX_TOKENS` | `150` | Max tokens per LLM response |

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
