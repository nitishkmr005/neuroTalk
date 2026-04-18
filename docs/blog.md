# Building NeuroTalk: A Real-Time Voice Agent for Customer Service

> A deep-dive into how we built a live speech transcription + AI response system using faster-whisper, Ollama, FastAPI, and Next.js — and what we learned about latency, streaming, and agent design.

---

## The Problem

Customer service is still largely reactive. Associates listen to a customer, mentally process what they said, and search for an answer — while the customer waits. What if an AI could listen to the same conversation in real time, understand the query, and surface an answer before the associate even finishes typing?

That is the core idea behind **NeuroTalk**: a voice agent console that captures live audio, transcribes it in near-real-time, and generates an AI-assisted response — all within the duration of a natural pause in conversation.

The same interface works from two angles:
- **Associate-facing**: AI surfaces relevant policy, account info, or next steps during the call.
- **Customer-facing**: AI answers repetitive queries directly, reducing wait times.

---

## Architecture

```
Microphone
    │
    ▼ PCM16 audio chunks (WebSocket binary frames)
FastAPI WebSocket /ws/transcribe
    │
    ▼ faster-whisper (int8, CPU)
Partial transcript (every ~1.2 s)
    │  ◄── displayed live in the browser
    ▼ Final transcript (on stop)
Ollama LLM (llama3.2, local)
    │
    ▼ Streamed AI response tokens (WebSocket JSON)
Browser — displayed below transcript
```

Everything flows over a single WebSocket connection. The browser streams raw PCM16 audio up; the server streams partial transcripts and LLM tokens back down.

---

## Key Technical Choices

### 1. faster-whisper for STT

OpenAI Whisper is accurate but slow in its original form. `faster-whisper` is a reimplementation using CTranslate2 that runs 4–8× faster with `int8` quantisation on CPU.

**Config choices:**
- `small.en` — best accuracy/speed balance for English
- `beam_size=1` — greedy decoding, ~40% faster than beam=5
- `vad_filter=True` — skips silent segments, reduces hallucinations
- Buffering: emit a partial transcript every 1.2 s with at least 0.9 s of audio buffered

**Typical latency:** 200–800 ms per transcription pass depending on audio length.

### 2. WebSocket streaming (not polling)

HTTP polling would add 200–500 ms of unnecessary latency per update. A persistent WebSocket connection lets us:
- Stream binary PCM16 audio frames from browser to server continuously
- Push partial transcripts back as JSON without waiting for a request
- Stream LLM tokens back one-by-one as they are generated

The browser uses the Web Audio API (`ScriptProcessorNode`) to capture microphone audio at the native sample rate and convert it to PCM16 before sending.

### 3. Ollama for local LLM inference

We chose Ollama over cloud APIs for three reasons:
1. **Privacy** — audio transcripts never leave the machine
2. **Zero cost** — no API keys, no per-token billing
3. **Latency** — on Apple Silicon, llama3.2 delivers first token in ~100–200 ms

The `ollama` Python library provides an `AsyncClient` with native streaming support, which integrates cleanly with FastAPI's async WebSocket handler.

**System prompt design:** Short and explicit — "respond in 1-3 sentences, plain spoken language, no markdown." LLM responses are spoken aloud, not read, so bullet points and headers are actively harmful.

### 4. Pydantic Settings + dotenv

All configuration lives in `backend/.env` and is validated by `pydantic-settings`. This means:
- Type-safe config (int, bool, Path) with defaults
- No scattered `os.getenv()` calls
- Easy to override in production via environment variables

### 5. Loguru for structured logging

Two sinks:
- **Terminal** — colorful, human-readable, shows timestamps + function names
- **JSON files** — machine-readable, rotated at 10 MB, latest 5 kept. Each line is a structured log record with timestamp, level, module, and message. Useful for debugging latency regressions.

---

## Latency Breakdown (typical, Apple M-series CPU)

| Stage | Latency |
|-------|---------|
| Audio capture + WebSocket send | ~5 ms per chunk |
| Whisper transcription (0.9 s audio) | 200–600 ms |
| WebSocket round-trip | ~10 ms |
| Ollama TTFT (first token) | 100–250 ms |
| Full LLM response (3 sentences) | 500–1200 ms |
| **Perceived response start** | **~400 ms after speech stops** |

The key insight: we start streaming LLM tokens immediately after the final transcript arrives. The user sees the first words of the AI response within ~400 ms of stopping speech — well within the natural conversational pause window.

---

## What We Learned

**Streaming beats batching everywhere.** Both the STT partial updates and LLM token streaming make the system feel dramatically more responsive than a batch request-response cycle would.

**VAD filter is non-negotiable.** Without it, faster-whisper hallucinates text during silence ("Thank you for watching.", "Subtitles by..."). The VAD filter eliminates ~90% of hallucination artifacts.

**System prompt length matters for TTS.** We initially wrote a detailed system prompt. The LLM would respond with bullet points and headers — which sound terrible when spoken aloud. Explicit "no markdown" instruction fixed this entirely.

**Local LLMs are production-ready for constrained tasks.** For a specific, well-scoped task (answer in 3 sentences), llama3.2 is accurate enough and dramatically faster than GPT-4 for this use case.

---

## What's Next

- **Voice output** — TTS with emotional expressiveness (speed, pitch variation based on sentiment)
- **RAG** — Ground LLM responses in a product knowledge base or policy documents
- **Multi-speaker diarisation** — Separate customer vs. associate voice in the transcript
- **Emotion detection** — Detect caller frustration from acoustic features, surface to associate

---

## Running It Yourself

```bash
# Clone and set up
git clone <repo>
make setup

# Start Ollama
brew install ollama && ollama pull llama3.2 && ollama serve

# Run
make dev          # → http://localhost:3000
```

Or explore each module individually:

```bash
uv run --project backend python scripts/stt.py       # transcribe audio
uv run --project backend python scripts/llm_call.py  # call local LLM
uv run --project backend python scripts/agent.py     # full pipeline
```
