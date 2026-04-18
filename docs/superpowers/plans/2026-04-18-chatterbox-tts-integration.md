# Chatterbox TTS + Barge-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Chatterbox Turbo TTS into the NeuroTalk voice agent so the AI speaks its LLM responses aloud, with emotion tags for expressiveness and real-time barge-in interruption.

**Architecture:** Chatterbox loads as a lazy in-process singleton in FastAPI. After each LLM response, the backend synthesises WAV audio and sends it as a base64 `tts_audio` WebSocket message. The frontend decodes and plays it via Web Audio API. RMS amplitude monitoring on the always-on mic detects barge-in and sends an `interrupt` message to cancel in-flight synthesis.

**Tech Stack:** Python 3.12, FastAPI, Chatterbox Turbo (`chatterbox-tts`), PyTorch, asyncio, pytest, pytest-asyncio; TypeScript, Web Audio API (`AudioContext`, `AudioBufferSourceNode`).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `backend/app/utils/emotion.py` | `strip_emotion_tags(text) -> str` pure utility |
| Create | `backend/app/services/tts.py` | `TTSService` lazy singleton + `get_tts_service()` |
| Create | `backend/tests/__init__.py` | marks test package |
| Create | `backend/tests/test_emotion.py` | unit tests for emotion tag stripping |
| Create | `backend/tests/test_tts.py` | unit tests for TTSService with mocked model |
| Modify | `backend/app/prompts/system.py` | add emotion tag instructions to prompt |
| Modify | `backend/app/main.py` | interrupt_event, synthesize_and_send, interrupt handler, strip tags in partial/final |
| Modify | `backend/pyproject.toml` | add `dev` dependency group with pytest/pytest-asyncio |
| Modify | `Makefile` | `backend-install` syncs `chatterbox_model` group |
| Modify | `frontend/components/voice-agent-console.tsx` | `speaking` mode, TTS playback, barge-in detection, UI updates |

---

## Task 1: Add dev test dependencies to backend

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add dev group**

Open `backend/pyproject.toml`. After the existing `[dependency-groups]` block (after the `conflicts` closing bracket in `[tool.uv]`), add a `dev` group and pytest config. The final `[dependency-groups]` section should have a new entry:

```toml
[dependency-groups]
dev = [
  "pytest>=8.0.0",
  "pytest-asyncio>=0.23.0",
]
```

Also add pytest config at the end of the file:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Sync dev group**

```bash
uv --directory backend sync --group dev
```

Expected: resolves and installs pytest and pytest-asyncio. No errors.

- [ ] **Step 3: Verify pytest works**

```bash
uv --directory backend run pytest --collect-only
```

Expected output contains: `no tests ran` or similar — no import errors.

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "chore: add dev test dependencies (pytest, pytest-asyncio)"
```

---

## Task 2: Update system prompt with emotion tags

**Files:**
- Modify: `backend/app/prompts/system.py`

- [ ] **Step 1: Replace prompt content**

Replace the entire contents of `backend/app/prompts/system.py` with:

```python
VOICE_AGENT_PROMPT = (
    "You are a concise, helpful voice assistant for customer service. "
    "Respond in 1-3 sentences only. Be clear, natural, and conversational. "
    "Do not use markdown, bullet points, or lists — plain spoken language only. "
    "Where natural and appropriate, insert one of these inline emotion tags to add expressiveness: "
    "[laugh], [chuckle], [sigh], [gasp], [clears throat]. "
    "Use emotion tags sparingly — only when the tone clearly warrants it. "
    "Never place an emotion tag inside a technical or factual sentence."
)
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/prompts/system.py
git commit -m "feat: add Chatterbox emotion tag instructions to system prompt"
```

---

## Task 3: Add `strip_emotion_tags` utility (TDD)

**Files:**
- Create: `backend/app/utils/emotion.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_emotion.py`

- [ ] **Step 1: Create test package**

```bash
touch backend/tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `backend/tests/test_emotion.py`:

```python
from app.utils.emotion import strip_emotion_tags


def test_strips_laugh():
    assert strip_emotion_tags("[laugh] That is funny.") == "That is funny."


def test_strips_chuckle():
    assert strip_emotion_tags("Sure thing. [chuckle] Let me check that.") == "Sure thing. Let me check that."


def test_strips_sigh():
    assert strip_emotion_tags("[sigh] I understand your frustration.") == "I understand your frustration."


def test_strips_gasp():
    assert strip_emotion_tags("Oh! [gasp] I see the issue now.") == "Oh! I see the issue now."


def test_strips_clears_throat():
    assert strip_emotion_tags("[clears throat] Right, so the account shows...") == "Right, so the account shows..."


def test_strips_multiple_tags():
    assert strip_emotion_tags("[chuckle] Happy to help. [sigh] It can be tricky.") == "Happy to help. It can be tricky."


def test_no_tags_unchanged():
    assert strip_emotion_tags("Hello, how can I help you today?") == "Hello, how can I help you today?"


def test_empty_string():
    assert strip_emotion_tags("") == ""


def test_collapses_extra_spaces():
    assert strip_emotion_tags("Hello  [laugh]  world") == "Hello world"


def test_unknown_tag_stripped():
    assert strip_emotion_tags("[hesitates] Well, let me think.") == "Well, let me think."
```

- [ ] **Step 3: Run tests — expect failure**

```bash
uv --directory backend run pytest tests/test_emotion.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.utils.emotion'` or `ImportError`.

- [ ] **Step 4: Implement `strip_emotion_tags`**

Create `backend/app/utils/emotion.py`:

```python
from __future__ import annotations

import re

_TAG_PATTERN = re.compile(r"\[[^\]]+\]")


def strip_emotion_tags(text: str) -> str:
    stripped = _TAG_PATTERN.sub("", text)
    return " ".join(stripped.split())
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
uv --directory backend run pytest tests/test_emotion.py -v
```

Expected: 10 tests, all PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/app/utils/emotion.py backend/tests/__init__.py backend/tests/test_emotion.py
git commit -m "feat: add strip_emotion_tags utility"
```

---

## Task 4: Create TTS service (TDD)

**Files:**
- Create: `backend/app/services/tts.py`
- Create: `backend/tests/test_tts.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_tts.py`:

```python
from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.services.tts import TTSService


def make_mock_model(sample_rate: int = 24000, duration_samples: int = 24000) -> MagicMock:
    model = MagicMock()
    model.sr = sample_rate
    model.generate.return_value = torch.zeros(1, duration_samples)
    return model


@pytest.mark.asyncio
async def test_synthesize_returns_bytes_and_sample_rate():
    service = TTSService()
    service._model = make_mock_model()

    wav_bytes, sr = await service.synthesize("Hello world.")

    assert isinstance(wav_bytes, bytes)
    assert sr == 24000
    assert len(wav_bytes) > 44  # at least WAV header (44 bytes)


@pytest.mark.asyncio
async def test_synthesize_passes_text_with_emotion_tags_to_model():
    service = TTSService()
    mock_model = make_mock_model()
    service._model = mock_model

    await service.synthesize("Happy to help. [chuckle] Let me check that.")

    mock_model.generate.assert_called_once_with("Happy to help. [chuckle] Let me check that.")


@pytest.mark.asyncio
async def test_synthesize_output_is_valid_wav():
    service = TTSService()
    service._model = make_mock_model()

    wav_bytes, sr = await service.synthesize("Test audio.")

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getframerate() == sr
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


@pytest.mark.asyncio
async def test_synthesize_reuses_loaded_model():
    service = TTSService()
    mock_model = make_mock_model()
    service._model = mock_model

    await service.synthesize("First call.")
    await service.synthesize("Second call.")

    assert mock_model.generate.call_count == 2


def test_get_tts_service_returns_singleton():
    from app.services.tts import get_tts_service

    s1 = get_tts_service()
    s2 = get_tts_service()
    assert s1 is s2
```

- [ ] **Step 2: Run tests — expect failure**

```bash
uv --directory backend run pytest tests/test_tts.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.services.tts'`.

- [ ] **Step 3: Implement `TTSService`**

Create `backend/app/services/tts.py`:

```python
from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

import numpy as np
import torch
from loguru import logger

_WARMUP_TEXT = "Hello."


class TTSService:
    def __init__(self) -> None:
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    def _load_model(self) -> Any:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info("event=tts_load device={}", device)
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        model.generate(_WARMUP_TEXT)
        logger.info("event=tts_ready device={}", device)
        return model

    def _run_inference(self, text: str) -> tuple[bytes, int]:
        waveform = self._model.generate(text)
        samples = (
            waveform.detach().cpu().squeeze().numpy()
            if torch.is_tensor(waveform)
            else np.asarray(waveform).squeeze()
        )
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._model.sr)
            wf.writeframes(pcm16.tobytes())

        return buf.getvalue(), self._model.sr

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, self._load_model)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_inference, text)


_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
uv --directory backend run pytest tests/test_tts.py -v
```

Expected: 5 tests, all PASSED.

- [ ] **Step 5: Run full test suite**

```bash
uv --directory backend run pytest -v
```

Expected: all 15 tests PASSED (10 emotion + 5 TTS).

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/tts.py backend/tests/test_tts.py
git commit -m "feat: add TTSService with Chatterbox Turbo lazy singleton"
```

---

## Task 5: Wire TTS into `main.py`

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: Add new imports at the top of `main.py`**

After the existing imports block, add:

```python
import base64

from app.services.tts import get_tts_service
from app.utils.emotion import strip_emotion_tags
```

- [ ] **Step 2: Add `interrupt_event` to session state**

Inside `transcribe_stream`, after the existing `latest_llm_input = ""` line, add:

```python
interrupt_event = asyncio.Event()
```

- [ ] **Step 3: Add `synthesize_and_send` inner function**

Inside `transcribe_stream`, after the existing `send_json` inner function and before `run_llm_stream`, add:

```python
async def synthesize_and_send(text: str) -> None:
    if interrupt_event.is_set():
        return
    tts_t0 = perf_counter()
    await send_json({"type": "tts_start"})
    try:
        tts_service = get_tts_service()
        wav_bytes, sample_rate = await tts_service.synthesize(text)
    except Exception as tts_err:
        logger.warning("request_id={} event=tts_error error={}", request_id, tts_err)
        return
    if interrupt_event.is_set():
        return
    tts_ms = round((perf_counter() - tts_t0) * 1000, 2)
    logger.info("request_id={} event=tts_done tts_ms={}", request_id, tts_ms)
    wav_b64 = base64.b64encode(wav_bytes).decode()
    await send_json({"type": "tts_audio", "data": wav_b64, "sample_rate": sample_rate, "tts_ms": tts_ms})
    await send_json({"type": "tts_done"})
```

- [ ] **Step 4: Update `run_llm_stream` to strip tags and call TTS**

Replace the existing `run_llm_stream` function with the updated version. The changes are:
1. Clear `interrupt_event` at the start.
2. Strip emotion tags when sending `llm_partial` and `llm_final` (display only).
3. After successful LLM response, call `synthesize_and_send` with the raw full_response (tags included).
4. Remove the socket-close logic from `llm_final` — socket lifetime moves to `tts_done` on the frontend.

```python
async def run_llm_stream(text: str, trigger: str) -> None:
    nonlocal llm_task, pending_llm_call, latest_llm_input
    interrupt_event.clear()
    llm_t0 = perf_counter()
    call_ts = _iso()
    full_response = ""
    llm_ms = 0.0
    call_error: str | None = None
    latest_llm_input = text

    try:
        await send_json({"type": "llm_start"})
        async for token in stream_llm_response(text):
            full_response += token
            await send_json({"type": "llm_partial", "text": strip_emotion_tags(full_response)})
        llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
        logger.info("request_id={} event=llm_done llm_ms={}", request_id, llm_ms)
        display_text = strip_emotion_tags(full_response)
        await send_json({"type": "llm_final", "text": display_text, "llm_ms": llm_ms})
    except Exception as llm_err:
        llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
        call_error = str(llm_err)
        logger.warning("request_id={} event=llm_error error={}", request_id, llm_err)
        try:
            await send_json({"type": "llm_error", "message": "LLM unavailable — is Ollama running?"})
        except Exception:
            pass

    approx_tokens = round(len(full_response.split()) * 1.3) if full_response else 0
    session_log.llm_calls.append(
        LLMCallLog(
            timestamp=call_ts,
            trigger=trigger,
            latency_ms=llm_ms,
            model=settings.llm_model,
            host=settings.ollama_host,
            system_prompt_preview=settings.llm_system_prompt[:100],
            input_transcript=text,
            input_length_chars=len(text),
            output_response=full_response,
            output_preview=full_response[:200],
            output_length_chars=len(full_response),
            approx_tokens_out=approx_tokens,
            cancelled=False,
            error=call_error,
        )
    )

    if full_response and call_error is None:
        await synthesize_and_send(full_response)

    if pending_llm_call is None:
        llm_task = None
        return

    next_text, next_trigger = pending_llm_call
    pending_llm_call = None
    if next_text == latest_llm_input:
        llm_task = None
        return

    llm_task = asyncio.create_task(run_llm_stream(next_text, next_trigger))
```

- [ ] **Step 5: Add `interrupt` message handler in the main receive loop**

Inside the `while True:` loop, in the `if message.get("text") is not None:` block, after the `if event_type == "start":` block and before `if event_type == "stop":`, add:

```python
if event_type == "interrupt":
    logger.info("request_id={} event=interrupt_received", request_id)
    interrupt_event.set()
    if llm_task is not None and not llm_task.done():
        llm_task.cancel()
        with suppress(asyncio.CancelledError):
            await llm_task
        llm_task = None
    continue
```

- [ ] **Step 6: Verify backend starts without error**

```bash
uv --directory backend sync --group chatterbox_model
uv --directory backend run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Expected: Server starts. Log line `event=startup` appears. No import errors.

Stop the server with Ctrl-C.

- [ ] **Step 7: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: wire Chatterbox TTS into WebSocket handler with barge-in interrupt support"
```

---

## Task 6: Update Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Update `backend-install` target**

In `Makefile`, replace:

```makefile
backend-install:
	$(UV_BACKEND) sync
```

with:

```makefile
backend-install:
	$(UV_BACKEND) sync --group chatterbox_model
```

- [ ] **Step 2: Verify `make backend-install` works**

```bash
make backend-install
```

Expected: `uv sync` resolves with chatterbox_model group. No errors.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: include chatterbox_model group in backend-install"
```

---

## Task 7: Frontend — types, state, mode config

**Files:**
- Modify: `frontend/components/voice-agent-console.tsx`

All changes in this task are additions to the existing file. Do not remove any existing code in this task.

- [ ] **Step 1: Extend `Mode` type**

Find:
```typescript
type Mode = "listening" | "thinking" | "responding";
```
Replace with:
```typescript
type Mode = "listening" | "thinking" | "responding" | "speaking";
```

- [ ] **Step 2: Extend `StreamMessage` type**

Find:
```typescript
type StreamMessage = {
  type: "ready" | "partial" | "final" | "error" | "llm_start" | "llm_partial" | "llm_final" | "llm_error";
```
Replace with:
```typescript
type StreamMessage = {
  type: "ready" | "partial" | "final" | "error" | "llm_start" | "llm_partial" | "llm_final" | "llm_error" | "tts_start" | "tts_audio" | "tts_done";
```

Also add two new optional fields to `StreamMessage`, after the `llm_ms` field:
```typescript
  data?: string;
  tts_ms?: number;
```

- [ ] **Step 3: Add `"speaking"` entry to `modeConfig`**

Find the closing brace of the `responding` entry in `modeConfig` (the `},` before the `};`), and add after it:

```typescript
  speaking: {
    eyebrow: "AI voice active",
    headline: "Speaking the reply aloud.",
    summary:
      "Chatterbox Turbo is synthesising the AI reply into speech. Speak at any time to interrupt and take your turn.",
    accent: "voice-delivery",
  },
```

- [ ] **Step 4: Add new state and refs**

In the `VoiceAgentConsole` component body, after the `const [llmLatencyMs, ...]` line, add:

```typescript
const [ttsLatencyMs, setTtsLatencyMs] = useState<number | null>(null);
const ttsSourceRef = useRef<AudioBufferSourceNode | null>(null);
const interruptSentRef = useRef(false);
const bargeinFrameCountRef = useRef(0);
```

- [ ] **Step 5: Add barge-in constants**

At module level, after `const initialWaveLevels = ...`, add:

```typescript
const BARGE_IN_THRESHOLD = 0.15;
const BARGE_IN_FRAMES = 2;
```

- [ ] **Step 6: Commit**

```bash
git add frontend/components/voice-agent-console.tsx
git commit -m "feat(frontend): add speaking mode type, TTS message types, barge-in state"
```

---

## Task 8: Frontend — TTS playback, barge-in detection, socket lifecycle

**Files:**
- Modify: `frontend/components/voice-agent-console.tsx`

- [ ] **Step 1: Stop TTS on audio graph cleanup**

In `stopAudioGraph`, after `gainNodeRef.current = null;` and before `mediaStreamRef.current?.getTracks()...`, add:

```typescript
ttsSourceRef.current?.stop();
ttsSourceRef.current = null;
```

- [ ] **Step 2: Add barge-in detection in `onaudioprocess`**

In the `processorNode.onaudioprocess` handler, after the existing lines that compute `smoothedAmplitude` and update `waveLevels`, add:

```typescript
if (ttsSourceRef.current && !interruptSentRef.current) {
  if (rms > BARGE_IN_THRESHOLD) {
    bargeinFrameCountRef.current += 1;
    if (bargeinFrameCountRef.current >= BARGE_IN_FRAMES) {
      interruptSentRef.current = true;
      ttsSourceRef.current.stop();
      ttsSourceRef.current = null;
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "interrupt" }));
      }
      startTransition(() => { setMode("listening"); });
    }
  } else {
    bargeinFrameCountRef.current = 0;
  }
}
```

- [ ] **Step 3: Add `tts_start` handler in `socket.onmessage`**

After the `llm_error` handler block (after its `return;`), add:

```typescript
if (payload.type === "tts_start") {
  startTransition(() => { setMode("speaking"); });
  return;
}
```

- [ ] **Step 4: Add `tts_audio` handler**

After the `tts_start` handler block, add:

```typescript
if (payload.type === "tts_audio") {
  if (payload.tts_ms != null) setTtsLatencyMs(payload.tts_ms);
  const binaryString = atob(payload.data ?? "");
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  const audioCtx = audioContextRef.current ?? new AudioContext();
  if (!audioContextRef.current) audioContextRef.current = audioCtx;
  void audioCtx.decodeAudioData(bytes.buffer.slice(0), (buffer) => {
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    ttsSourceRef.current = source;
    interruptSentRef.current = false;
    bargeinFrameCountRef.current = 0;
    source.onended = () => { ttsSourceRef.current = null; };
    source.start();
  });
  return;
}
```

- [ ] **Step 5: Add `tts_done` handler**

After the `tts_audio` handler block, add:

```typescript
if (payload.type === "tts_done") {
  startTransition(() => { setMode(isRecordingRef.current ? "listening" : "responding"); });
  if (!isRecordingRef.current) {
    normalCloseRef.current = true;
    socket.close();
  }
  return;
}
```

- [ ] **Step 6: Remove socket close from `llm_final` handler**

Find the `llm_final` handler. It currently ends with:
```typescript
if (!isRecordingRef.current) {
  normalCloseRef.current = true;
  socket.close();
}
return;
```

Remove the `if (!isRecordingRef.current)` block (the 3 lines). The `llm_final` handler should now end with just `return;`. Socket lifetime is now controlled by `tts_done`.

- [ ] **Step 7: Reset `ttsLatencyMs` on new session**

In `startStreaming`, after `setLlmLatencyMs(null);`, add:

```typescript
setTtsLatencyMs(null);
```

- [ ] **Step 8: Commit**

```bash
git add frontend/components/voice-agent-console.tsx
git commit -m "feat(frontend): add TTS audio playback and barge-in interruption"
```

---

## Task 9: Frontend — UI updates (latency cards, pipeline status)

**Files:**
- Modify: `frontend/components/voice-agent-console.tsx`

- [ ] **Step 1: Update TTS latency card**

Find the `latencyCards` array. Find the TTS card:
```typescript
{
  title: "TTS",
  label: "Voice Playback",
  value: "--",
  detail: "Voice synthesis latency — coming soon.",
},
```
Replace with:
```typescript
{
  title: "TTS",
  label: "Voice Synthesis",
  value: formatSeconds(ttsLatencyMs),
  detail: "Chatterbox Turbo synthesis time for the last AI reply.",
},
```

- [ ] **Step 2: Update E2E latency card to include TTS**

Find the E2E latency card:
```typescript
{
  title: "E2E",
  label: "Turn Latency",
  value: formatSeconds(metrics?.total_ms != null && llmLatencyMs != null ? metrics.total_ms + llmLatencyMs : (metrics?.total_ms ?? null)),
  detail: "Combined STT + LLM pipeline time for the last session.",
},
```
Replace with:
```typescript
{
  title: "E2E",
  label: "Turn Latency",
  value: formatSeconds(
    metrics?.total_ms != null && llmLatencyMs != null && ttsLatencyMs != null
      ? metrics.total_ms + llmLatencyMs + ttsLatencyMs
      : metrics?.total_ms != null && llmLatencyMs != null
        ? metrics.total_ms + llmLatencyMs
        : metrics?.total_ms ?? null
  ),
  detail: "Combined STT + LLM + TTS pipeline time for the last session.",
},
```

- [ ] **Step 3: Update Voice Playback orchestration step**

Find the `orchestrationSteps` array. Find the Voice Playback entry:
```typescript
{ label: "Voice Playback", detail: "Voice synthesis — coming soon to complete the full voice loop.", status: "pending" },
```
Replace with:
```typescript
{ label: "Voice Playback", detail: "Chatterbox Turbo synthesises AI replies and streams audio to the browser.", status: "online" },
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
npm --prefix frontend run typecheck
```

Expected: no type errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/voice-agent-console.tsx
git commit -m "feat(frontend): update TTS latency card, E2E metric, pipeline status"
```

---

## Task 10: End-to-end smoke test

- [ ] **Step 1: Start services**

In three terminals:

```bash
# Terminal 1
ollama serve

# Terminal 2
make backend

# Terminal 3
make frontend
```

Expected: backend logs `event=startup`. Frontend available at `http://localhost:3000`.

- [ ] **Step 2: Verify first AI response speaks**

1. Open `http://localhost:3000`.
2. Click **Start Live Transcription**.
3. Say: "Hello, can you help me?"
4. Stop recording.
5. Expected sequence in UI:
   - Mode changes: `listening` → `thinking` → `responding` → `speaking`
   - Chat shows clean AI text (no `[laugh]` etc. visible)
   - Audio plays from browser speakers
   - TTS latency card fills with a value
   - E2E card updates
   - Voice Playback orchestration step shows `online`

- [ ] **Step 3: Verify barge-in**

1. Start a new recording.
2. Say a sentence. Wait for AI to start speaking.
3. While audio plays, speak clearly.
4. Expected: audio stops mid-playback, UI returns to `listening` mode, no echo.

- [ ] **Step 4: Run full backend test suite one last time**

```bash
uv --directory backend run pytest -v
```

Expected: all 15 tests PASSED.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: Chatterbox TTS voice agent — speaking mode, barge-in, emotion tags"
```
