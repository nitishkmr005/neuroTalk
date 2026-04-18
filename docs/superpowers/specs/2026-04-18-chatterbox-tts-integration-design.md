# Chatterbox TTS Integration Design

**Date:** 2026-04-18
**Status:** Approved

## Goal

Integrate Chatterbox Turbo TTS into the NeuroTalk voice agent so the AI speaks its responses aloud. The mic stays live throughout. The user can interrupt AI speech mid-playback (barge-in). LLM responses include inline emotion tags that Chatterbox uses to add expressiveness.

---

## Architecture & Data Flow

```
Mic (always on)
  │
  ▼
WebSocket /ws/transcribe
  │
  ▼
STT (Whisper)
  │
  ▼
LLM (Ollama) — system prompt instructs emotion tag insertion
  │
  ▼
TTS Service (Chatterbox, in-process singleton)
  │
  ▼
base64 WAV → {"type": "tts_audio", "data": "...", "tts_ms": 1234}
  │
  ▼
Web Audio API → AudioBufferSourceNode → plays in browser
  │
  ▼
Barge-in detector (mic RMS > 0.15 for 2+ frames while speaking)
  │ triggers
  ▼
{"type": "interrupt"} → stop AudioBufferSourceNode → mode = listening
```

### WebSocket Message Contract

| Direction | Type | Payload |
|---|---|---|
| server → client | `tts_start` | `{}` |
| server → client | `tts_audio` | `{data: base64, sample_rate: int, tts_ms: float}` |
| server → client | `tts_done` | `{}` |
| client → server | `interrupt` | `{}` |

### Mode State Machine

```
listening → thinking → responding → speaking → listening (loop)
                                       ↑
                             barge-in → listening (interrupt)
```

---

## Backend Changes

### New: `app/services/tts.py`

- Lazy singleton: loads `ChatterboxTurboTTS` on first call, reuses across requests.
- Runs warmup synthesis on startup to avoid first-call latency spike.
- Exposes: `async def synthesize(text: str) -> tuple[bytes, int]` — returns `(wav_bytes, sample_rate)`.
- Emotion tags in `text` are passed through to Chatterbox as-is.

### Updated: `app/services/llm.py`

- After full LLM response is collected, strip emotion tags to produce `display_text`.
- `display_text` goes to the `llm_final` WebSocket message (clean chat bubble).
- Raw text with tags goes to TTS synthesis.

### Updated: `app/main.py` — `run_llm_stream`

Post-`llm_final` sequence:
1. Send `{"type": "tts_start"}`.
2. Call `synthesize(full_response)` — full text including emotion tags.
3. Encode WAV bytes as base64.
4. Send `{"type": "tts_audio", "data": base64_wav, "sample_rate": sr, "tts_ms": elapsed}`.
5. Send `{"type": "tts_done"}`.

Interrupt handling:
- New `interrupt_event: asyncio.Event` per session, checked before/during TTS synthesis.
- Incoming `{"type": "interrupt"}` message sets the event.
- If event is set before TTS starts, skip synthesis entirely.
- Also cancels any pending `llm_task` (same pattern as existing cancel logic).

### Updated: `app/prompts/system.py`

Prompt addition — instruct LLM to insert Chatterbox inline emotion tags where natural:
- Supported: `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[clears throat]`
- Rules: use sparingly, only where tone clearly warrants it, never in technical/factual sentences.
- Keep existing 1–3 sentence constraint and no-markdown rule.

### Updated: `backend/pyproject.toml`

No structural change — `chatterbox_model` dependency group already defined. `make backend` updated to sync with `--group chatterbox_model`.

### Updated: `Makefile`

`backend` target: `uv --directory backend sync --group chatterbox_model && uv run ...`

---

## Frontend Changes (`components/voice-agent-console.tsx`)

### New mode: `"speaking"`

Added to `Mode` type and `modeConfig`:
- Eyebrow: `"AI is responding"`
- Headline: `"Speaking the reply aloud."`
- Accent: `"voice-delivery"` (reuses existing class)
- Orb: slow steady pulse (CSS animation, distinct from mic-reactive pulse)

### Audio Playback

- On `tts_audio`: decode `data` from base64 → `ArrayBuffer` → `AudioContext.decodeAudioData()` → `AudioBufferSourceNode.start()`.
- Store source node in `ttsSourceRef` for instant `.stop()` on interrupt.
- On `tts_done`: mode → `"listening"`.

### Barge-in Detection

- In the existing `onaudioprocess` callback: when `mode === "speaking"` and RMS amplitude > `0.15` for 2+ consecutive frames, fire interrupt.
- Send `{"type": "interrupt"}` over WebSocket.
- Call `ttsSourceRef.current?.stop()`.
- Set mode → `"listening"`.
- Guard with `interruptSentRef` flag to prevent duplicate interrupt messages per barge-in.

### Updated: `StreamMessage` type

Add: `"tts_start" | "tts_audio" | "tts_done"` to the `type` union.
Add fields: `data?: string`, `tts_ms?: number`.

### Updated: Latency Cards

TTS card: `value` bound to new `ttsLatencyMs` state (set on `tts_audio` receipt).

### Updated: Orchestration Steps

Voice Playback row: `status: "pending"` → `"online"`, detail updated to describe Chatterbox Turbo.

---

## What Does Not Change

- WebSocket connection/lifecycle (open, close, reconnect).
- STT streaming and partial/final transcript logic.
- Chat thread rendering and message bubbles (emotion tags already stripped from display text).
- All existing debug/metrics panels.
- LLM streaming token-by-token display.

---

## Constraints & Risks

| Risk | Mitigation |
|---|---|
| Chatterbox + Whisper + Ollama in one process (~6-8 GB RAM) | Acceptable for dev; note in README |
| First synthesis cold start (~3-4s) | Warmup call at backend startup |
| Barge-in threshold too sensitive (false triggers) | Threshold 0.15 RMS + 2-frame debounce; tunable via env var later |
| Echo: AI audio picked up by mic | Browser `echoCancellation: true` already set in `getUserMedia` constraints |
