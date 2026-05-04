# From Orb Click to AI Speaking — How the Backend Works

> **Who this is for:** If you are new to async programming and want to understand how clicking the orb triggers a chain of events that ends with the AI speaking, this document walks through every step with plain-language explanations.

---

## Quick Async Primer (Read This First)

Before diving in, here are the four async concepts you will see throughout this codebase:

| Concept | Plain English |
|---|---|
| `async def` | "This function can be paused mid-execution without blocking everything else." Like a chef who starts boiling water, then preps vegetables while waiting — instead of staring at the pot. |
| `await` | "Pause *this* function here until the result is ready, and let other things run in the meantime." You only write `await` inside an `async def`. |
| `asyncio.Task` | A background job. `asyncio.create_task(my_coroutine())` schedules it to run without waiting for it to finish right now. Think of it as handing a task to a sous-chef. |
| Event Loop | The single-threaded scheduler that decides whose turn it is to run. When your code hits `await`, control returns to the loop, which runs something else until the awaited thing is ready. |

**Why async here?** The app is I/O-bound: it waits on microphones, network sockets, GPU/CPU inference, and audio playback. Async lets the server handle all of these concurrently without spawning threads for each user.

---

## The Big Picture

```
[Browser]                          [FastAPI Backend]
   │                                      │
   │  1. Click orb                        │
   │──────────────────────────────────►   │
   │  2. Mic audio streams in real-time   │
   │──────────────────────────────────►   │  3. Denoise + STT (Whisper)
   │                                      │  4. VAD detects end of speech
   │  5. Partial transcript live          │
   │◄──────────────────────────────────   │
   │                                      │  6. LLM streams tokens
   │  7. LLM tokens arrive live           │
   │◄──────────────────────────────────   │  8. TTS synthesises per sentence
   │  9. Audio WAV chunk (base64)         │
   │◄──────────────────────────────────   │
   │  10. Browser plays audio             │
```

---

## Step-by-Step Flow

### Step 1 — User Clicks the Orb

**File:** `frontend/components/voice-agent-console.tsx`  
**Function:** `startStreaming()` (~line 579)

The orb is a `<button>` with an `onClick` handler. When clicked:

1. `navigator.mediaDevices.getUserMedia({ audio: true })` — asks the browser for microphone access and returns a `MediaStream`.
2. Depending on the selected transport mode, either:
   - **WebRTC path:** `WebRTCTransport.connect(stream)` in `frontend/components/webrtc-transport.ts`
   - **WebSocket path:** `new WebSocket(backendUrl + "/ws/transcribe")`

---

### Step 2 — Transport Connects to the Backend

#### WebRTC path (the default)

**File:** `frontend/components/webrtc-transport.ts`  
**Function:** `connect(stream)` (~line 61)

```
Browser                                  Backend
  │  POST /webrtc/offer  (SDP offer)       │
  │────────────────────────────────────►   │
  │  ◄────────────────────────────────── SDP answer
  │                                        │
  │  RTP audio frames (Opus, 48 kHz)       │
  │════════════════════════════════════►   │  ← this is the mic audio
  │                                        │
  │  Data channel "signaling" (JSON)       │
  │◄══════════════════════════════════════►│  ← control messages both ways
```

- **SDP** (Session Description Protocol) is just a text format that tells both sides what audio codecs and network addresses to use.
- **RTP** is a real-time streaming protocol — the actual raw audio packets.
- **Data channel** is a WebRTC side-channel for JSON messages (transcripts, LLM tokens, TTS audio).

**Backend files involved:**
- `backend/app/webrtc/router.py` — `webrtc_offer()` receives the SDP offer, creates a `WebRTCSession`, and returns the SDP answer.
- `backend/app/webrtc/session.py` — `WebRTCSession.__init__()` sets up the peer connection; `setup()` completes the SDP handshake.

#### WebSocket path (simpler)

**File:** `backend/app/main.py`  
**Function:** `transcribe_stream(websocket)` (~line 307)

The browser opens a persistent two-way WebSocket connection. Audio arrives as raw binary PCM-16 frames; control messages are JSON.

---

### Step 3 — Audio Arrives at the Backend

**File:** `backend/app/webrtc/session.py`  
**Function:** `_consume_audio(track)` (~line 240)

This is an `async def` that runs in an infinite `while` loop, pulling one audio frame at a time:

```python
frame = await asyncio.wait_for(track.recv(), timeout=5.0)
```

- `track.recv()` blocks (asynchronously) until the next Opus-encoded audio packet arrives from the browser.
- `asyncio.wait_for(..., timeout=5.0)` means "give up and try again if nothing arrives in 5 seconds."

The Opus frames arrive at **48 kHz** (the WebRTC standard). Whisper (STT) expects **16 kHz**, so each frame is resampled:

```python
resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
for resampled in resampler.resample(frame):
    pcm = resampled.to_ndarray().tobytes()
    self._pcm_buffer.extend(pcm)
```

`_pcm_buffer` is a `bytearray` — think of it as a growing list of raw audio samples.

---

### Step 4 — Voice Activity Detection (VAD)

**File:** `backend/app/services/vad.py`  
**Class:** `StreamingVAD`  
**Method:** `process_pcm16(pcm)`

VAD answers one question: **"Is someone speaking right now?"**

The backend uses [Silero VAD](https://github.com/snakers4/silero-vad), a small neural net that outputs a probability (0–1) for each audio chunk:

- If `speech_prob >= 0.6` → emit `"start"` event (speech detected)
- If `speech_prob < 0.45` for long enough → emit `"end"` event (silence)

Back in `_consume_audio`, these events are processed:

```python
for vad_event in self._vad_stream.process_pcm16(pcm):
    if vad_event.event == "start":
        # User started speaking — cancel any pending LLM silence timer
    elif vad_event.event == "end":
        # User stopped speaking — start the turn-finalization countdown
```

**Barge-in detection:** If the AI is currently speaking (`_is_agent_speaking = True`) and VAD detects the user speaking, `_handle_interrupt()` is called to cancel the AI's speech immediately.

---

### Step 5 — Smart Turn (Optional Semantic Gate)

**File:** `backend/app/services/smart_turn.py`  
**Called from:** `_silence_debounce_then_fire()` in `session.py`

When VAD detects end-of-speech, the backend does not immediately call the LLM. Instead:

1. It waits a short grace period (`stream_llm_silence_ms`, default ~500 ms).
2. If **Smart Turn** is enabled, it asks a small ONNX model: *"Does this utterance sound complete?"*
   - If yes → proceed to LLM.
   - If no → wait a bit longer (the user might still be thinking).

This prevents the AI from interrupting mid-sentence when the user pauses to think.

**Function:** `_silence_debounce_then_fire(text, trigger, vad_triggered)` (~line 550)

```python
await asyncio.sleep(wait_ms / 1000)   # wait for silence window
is_complete, _ = smart_turn.predict(bytes(self._pcm_buffer))
if is_complete:
    # fire the LLM
```

---

### Step 6 — Speech-to-Text (STT / Transcription)

**File:** `backend/app/services/stt.py`  
**Class:** `SpeechToTextService`  
**Method:** `transcribe(file_path, ...)`

**File:** `backend/app/services/denoise.py`  
**Used before STT:** `get_denoise_service().enhance(pcm_bytes, sample_rate)`

**File:** `backend/app/webrtc/session.py`  
**Method:** `_transcribe_buffer()` (~line 391)

Two things happen before Whisper sees the audio:
1. **Denoising** — DeepFilterNet3 removes background noise (fan, keyboard, room echo).
2. **WAV file creation** — the raw PCM bytes are written to a temporary `.wav` file.

Then Whisper runs:

```python
result = service.transcribe(file_path=temp_path, ...)
```

This is a CPU/GPU-bound operation (not I/O), so it runs in a thread pool to avoid blocking the event loop:

```python
result = await loop.run_in_executor(None, self._transcribe_buffer)
```

`run_in_executor` = "run this blocking function in a background thread, and `await` its result back in the async world."

**Partial vs. final transcripts:**

- **Partial** — emitted every ~700 ms while the user is still speaking (`_maybe_emit_stt()`). Lets the frontend show live "you said: ..." text.
- **Final** — emitted once after VAD end + Smart Turn confirm the turn is complete (`_finalize_speech_turn()`).

Both send a JSON message over the data channel:
```json
{ "type": "partial", "text": "Hello, I was wondering...", "timings_ms": {...} }
```

---

### Step 7 — LLM Scheduling and Streaming

**File:** `backend/app/webrtc/session.py`  
**Method:** `_schedule_llm(text)` (~line 605)  
**Method:** `_run_llm(llm_seq, text)` (~line 700)

**File:** `backend/app/services/llm.py`  
**Function:** `stream_llm_response(transcript, conversation_history)`

#### _schedule_llm — The Gatekeeper

Before firing the LLM, `_schedule_llm` applies guards:
- Too short? (< `stream_llm_min_chars`) → skip.
- Same as last question? → skip (prevents double-firing).
- A pause command like "hold on"? → skip.
- LLM already running? → cancel it, use newer text instead.

#### _run_llm — The Actual Call

`_run_llm` is where the LLM gets invoked. It:

1. Clears the interrupt flag so stale cancellations don't affect this new call.
2. Immediately starts a concurrent TTS pipeline task.
3. Streams tokens from the LLM one by one:

```python
async for token in stream_llm_response(text, conversation_history=...):
    full_response += token
    await self._send_json({"type": "llm_partial", "text": full_response})
```

`async for` is like a regular `for` loop, but each iteration can `await` the next value — here, each `token` is a word piece that arrives as the LLM generates it.

4. Detects sentence boundaries (`. ! ?`) and puts complete sentences into a queue:

```python
sentence = clean_for_tts(tail[: m.end()].strip())
await sent_queue.put(sentence)
```

**`llm_seq`** is a sequence number that increments with each LLM call. The frontend uses it to discard stale responses if the user interrupted mid-stream.

**Provider dispatch** (`backend/app/services/llm.py`):

| Provider | Function |
|---|---|
| Ollama (local) | `_stream_ollama()` |
| OpenAI | `_stream_openai()` |
| Anthropic | `_stream_anthropic()` |
| Gemini | `_stream_gemini()` |
| llama-cpp (local GGUF) | `_stream_llamacpp()` |

All providers yield tokens via `async for`, so the rest of the code is provider-agnostic.

---

### Step 8 — Text-to-Speech (TTS)

**File:** `backend/app/webrtc/session.py`  
**Method:** `_tts_sentence_pipeline(llm_seq, queue)` (~line 644)

**File:** `backend/app/services/tts.py`  
**Class:** `TTSService`  
**Method:** `synthesize(text, voice, speed)`

The TTS pipeline runs **concurrently** with the LLM. As soon as the LLM produces a full sentence, it lands in `sent_queue`. The TTS pipeline picks it up immediately:

```python
while True:
    sentence = await queue.get()   # wait for next sentence
    if sentence is None:           # None = "all done"
        break
    wav_bytes, sr = await tts_service.synthesize(sentence, ...)
```

`await queue.get()` suspends this coroutine until a sentence is available — no busy-waiting, no polling.

TTS backends:
- **Kokoro MLX** — Apple Silicon optimised (fast on M-series Macs).
- **Chatterbox** — alternative, supports more voice styles.

The WAV bytes are base64-encoded and sent to the browser:

```python
wav_b64 = base64.b64encode(wav_bytes).decode()
await self._send_json({
    "type": "tts_audio",
    "data": wav_b64,
    "sample_rate": sr,
    "sentence_text": sentence,
})
```

---

### Step 9 — Browser Plays the Audio

**File:** `frontend/components/voice-agent-console.tsx`  
**Function:** `playNextTtsChunk()` (~line 471)

The browser receives `tts_audio` messages, decodes the base64 WAV, and plays it using the Web Audio API:

1. Decode base64 → `ArrayBuffer`
2. `audioContext.decodeAudioData(buffer)` → `AudioBuffer`
3. `audioContext.createBufferSource()` → connect → `audioContext.destination`
4. `source.start()` → audio plays
5. `source.onended` → call `playNextTtsChunk()` again for the next sentence

The text is revealed word-by-word as audio plays, so the on-screen transcript is in sync with what the AI is saying.

---

### Step 10 — Turn Completes, Loop Resets

When all TTS chunks are sent, the backend sends:
```json
{ "type": "tts_done", "llm_seq": 1 }
```

The conversation history is saved:
```python
self._conversation_history.append({"role": "user", "content": text})
self._conversation_history.append({"role": "assistant", "content": full_response})
```

The audio buffer is cleared:
```python
self._pcm_buffer.clear()
self._vad_stream.reset()
```

The session stays open. VAD is watching the microphone. The system is ready for the next utterance.

---

## Full Timeline (WebRTC path)

```
t=0ms      User clicks orb
t=50ms     WebRTC peer connection established
t=100ms    Data channel opens → backend sends {"type": "ready"}
t=100ms    Welcome message TTS starts streaming
t=Xms      User starts speaking (VAD fires "start")
t=X+700ms  Partial transcript sent (every 700ms)
t=Yms      User stops speaking (VAD fires "end")
t=Y+50ms   Smart Turn checks if utterance is complete
t=Y+200ms  Final transcript sent
t=Y+200ms  LLM call starts
t=Y+2000ms First LLM sentence complete → TTS starts for sentence 1
t=Y+2400ms tts_audio chunk 1 sent → browser plays
t=Y+3000ms tts_audio chunk 2 sent → browser plays
...
t=Zms      tts_done sent → turn complete, back to listening
```

---

## Key Files Quick Reference

| File | Role | Key Function/Class |
|---|---|---|
| `frontend/components/voice-agent-console.tsx` | UI, orb button, audio playback | `startStreaming()`, `playNextTtsChunk()` |
| `frontend/components/webrtc-transport.ts` | WebRTC client transport | `WebRTCTransport.connect()` |
| `backend/app/webrtc/router.py` | HTTP endpoint for SDP handshake | `webrtc_offer()` |
| `backend/app/webrtc/session.py` | Full pipeline orchestration | `WebRTCSession`, `_consume_audio()`, `_run_llm()` |
| `backend/app/main.py` | FastAPI app, WebSocket endpoint | `transcribe_stream()` |
| `backend/app/services/denoise.py` | Background noise removal | `DenoiseService.enhance()` |
| `backend/app/services/stt.py` | Whisper transcription | `SpeechToTextService.transcribe()` |
| `backend/app/services/vad.py` | Voice activity detection | `StreamingVAD.process_pcm16()` |
| `backend/app/services/smart_turn.py` | Semantic turn-end detection | `SmartTurnService.predict()` |
| `backend/app/services/llm.py` | LLM streaming, provider dispatch | `stream_llm_response()` |
| `backend/app/services/tts.py` | Speech synthesis | `TTSService.synthesize()` |
| `backend/config/settings.py` | All configuration | `Settings` (Pydantic BaseSettings) |

---

## Message Schema (Data Channel / WebSocket)

### Browser → Backend

| Message | When sent | Key fields |
|---|---|---|
| `start` | Recording begins | `sample_rate`, `voice` |
| binary bytes | Every audio frame | raw PCM-16 |
| `interrupt` | User barge-in button | — |
| `stop` | Recording ends | — |
| `tts_voice` | Voice changed in UI | `voice` |
| `tts_speed` | Speed changed in UI | `speed` |

### Backend → Browser

| Message | When sent | Key fields |
|---|---|---|
| `ready` | Data channel open | `request_id` |
| `partial` | Live transcript update | `text`, `timings_ms` |
| `final` | Turn transcription done | `text`, `timings_ms` |
| `llm_start` | LLM inference begins | `llm_seq`, `user_text` |
| `llm_partial` | Token arrives | `llm_seq`, `text` |
| `llm_final` | Full response ready | `llm_seq`, `text`, `llm_ms` |
| `tts_start` | First TTS chunk starting | `llm_seq` |
| `tts_audio` | One sentence of audio | `llm_seq`, `data` (base64 WAV), `sentence_text` |
| `tts_done` | All audio sent | `llm_seq` |
| `tts_interrupted` | Barge-in cancelled audio | `llm_seq` |

---

## Concurrency Map

The hardest thing to picture with async is what runs in parallel. Here is the concurrent structure during a typical turn:

```
Event loop tick ──────────────────────────────────────────────────────►
                                                                        
_consume_audio()     ███████████████████████████████████████████████   ← always running
                                                                        
_run_llm()                           ████████████████████████          ← starts after final STT
                                                                        
_tts_sentence_pipeline()                      ████████████████████     ← starts on first sentence
                                                                        
_send_json() calls             ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪     ← tiny, frequent
```

Key insight: `_run_llm` and `_tts_sentence_pipeline` run concurrently. The LLM fills the sentence queue; TTS drains it. This is how the AI can start speaking before it has finished "thinking."
