# NeuroTalk: Real-Time Voice Agent, No Cloud

You say something into the microphone. Before you finish, a transcript is on screen. Before you stop, a local LLM is already drafting a reply. Before the reply finishes generating, the first sentence is being spoken back at you. Interrupt it by talking and it stops mid-word.

That is NeuroTalk. No OpenAI key, no Deepgram, no ElevenLabs. One FastAPI process, one WebSocket, one Next.js tab. This post is about how the pipeline is wired and the tricks that make the wait disappear.

---

## What's on the wire

One WebSocket at `/ws/transcribe` carries everything.

Upstream (browser → server):
- `{"type":"start","sample_rate":48000}` once the socket opens
- Binary PCM16 frames, continuously, from a `ScriptProcessorNode` tapping `getUserMedia`
- `{"type":"interrupt"}` when the user starts speaking over the agent
- `{"type":"stop"}` when the user clicks the button

Downstream (server → browser):
- `partial` — transcript so far, emitted every ~800 ms
- `llm_start`, `llm_partial`, `llm_final` — streamed tokens from Ollama
- `tts_start`, `tts_audio` (base64 WAV), `tts_done` — spoken reply
- `final` — the authoritative transcript when the user stops

No polling, no REST round-trips mid-turn. The browser just pushes mono PCM16 at its native sample rate and reacts to whatever comes back.

---

## STT: faster-whisper on CPU, tuned for throughput

`backend/app/services/stt.py` wraps `faster-whisper` with three non-default choices:

```python
self._model = WhisperModel(
    self.settings.stt_model_size,   # "small" (English-leaning)
    device=self.settings.stt_device, # "cpu"
    compute_type=self.settings.stt_compute_type,  # "int8"
)
...
segments, info = model.transcribe(
    str(file_path),
    beam_size=self.settings.stt_beam_size,  # 1 — greedy
    vad_filter=self.settings.stt_vad_filter, # True
    language=self.settings.stt_language or None,
)
```

Why those:

- `compute_type="int8"` — ~4× faster than fp16 on CPU, accuracy drop is imperceptible on conversational audio.
- `beam_size=1` — greedy decoding. On voice-agent prompts ("reset my password", "what's my balance") beam search does nothing useful.
- `vad_filter=True` — without this, Whisper hallucinates *"Thank you for watching"* and *"Subtitles by the Amara community"* during silence. You will see this exactly once before you turn it on forever.

### The buffering trick

Whisper needs a file, not a stream. The server accumulates PCM16 into a `bytearray`, rewrites it to a WAV on each tick, and re-transcribes the whole buffer (`transcribe_stream_buffer` in `app/main.py`). The partial is only sent if the text actually changed:

```python
if current_text != last_text_sent:
    await send_json({"type": "partial", **result_payload})
    last_text_sent = current_text
    schedule_llm_stream(current_text, "debounced_partial")
```

So the user sees *"reset"* → *"reset my"* → *"reset my password"* appearing live — and each stable update can already start the LLM.

Tuning knobs in `config/settings.py`:

| Setting | Default | What it controls |
|---|---|---|
| `stream_emit_interval_ms` | 800 | Min gap between partial transcriptions |
| `stream_min_audio_ms` | 600 | Don't bother transcribing < 0.6 s of buffered audio |
| `stream_llm_min_chars` | 8 | Don't fire the LLM on single words |

---

## LLM: Ollama, streaming, with a short leash

`backend/app/services/llm.py` is twenty lines:

```python
stream = await client.chat(
    model=settings.llm_model,          # gemma3:1b by default
    messages=[
        {"role": "system", "content": settings.llm_system_prompt},
        {"role": "user",   "content": transcript},
    ],
    stream=True,
)
async for chunk in stream:
    token: str = chunk.message.content
    if token:
        yield token
```

Two things do most of the work here.

**The model choice.** `gemma3:1b` runs in about 1 GB of RAM and gives first token in ~100–200 ms on Apple Silicon. That's the entire reason the agent feels live. Swap to `gemma3:4b` or `mistral` and quality goes up, but you watch the latency floor climb with it.

**The system prompt.** It's deliberately brutal (`backend/app/prompts/system.py`):

```
You are a concise, helpful voice assistant for customer service.
Respond in 1-3 sentences only. Be clear, natural, and conversational.
Do not use markdown, bullet points, or lists — plain spoken language only.
Where natural and appropriate, insert one of these inline emotion tags to add expressiveness:
[laugh], [chuckle], [sigh], [gasp], [clears throat].
```

The "no markdown" clause is not cosmetic. The output is about to be read by a TTS engine. A bullet point gets pronounced as *"dash"*. Asterisks become *"asterisk asterisk"*. Headers sound deranged. If you're building a voice agent on top of a text-trained LLM, this instruction is load-bearing.

Emotion tags are stripped from the text shown in the chat window (`strip_emotion_tags`) but left in the TTS input so the synthesizer can react.

---

## Three latency tricks that matter more than the model

Most of the perceived speedup is not the model. It's the plumbing.

### 1. Fire the LLM on a stable partial, not on "stop"

When the user says *"hey what's the status of order 1 2 3 4"*, the final transcript arrives on `stop`. That's too late. `schedule_llm_stream` fires on each new partial that:
- is at least `stream_llm_min_chars` long,
- is different from the last LLM input, and
- isn't a pause command like *"hold on"* (more on that in a second).

If a later partial arrives while an earlier LLM call is still generating, it's queued as `pending_llm_call`. When the current call finishes, the queued one runs — unless its text is now equal to `latest_llm_input`, in which case it's dropped. Net effect: the LLM is always working on the most recent complete thought, without thrashing.

### 2. Start TTS on the first sentence, not the full reply

Inside the token streaming loop:

```python
async for token in stream_llm_response(text):
    full_response += token
    await send_json({"type": "llm_partial", "text": strip_emotion_tags(full_response)})
    if first_sent_task is None and not interrupt_event.is_set():
        m = _SENT_BOUNDARY.search(full_response)
        if m and m.end() >= 20:
            first_sent_end = m.end()
            snippet = full_response[:first_sent_end].strip()
            first_sent_task = asyncio.ensure_future(tts_svc.synthesize(snippet))
```

As soon as a `.`, `!` or `?` lands and the sentence is at least 20 characters, TTS synthesis of that sentence starts *in parallel* with the remainder of the LLM stream. When the LLM finishes, the two audio chunks are concatenated by stripping the 44-byte WAV header from the second part and rewriting the container.

The user hears sentence one before the model has finished writing sentence two.

### 3. The pause-command filter

```python
_PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|"
    r"just a (?:moment|second|sec)|give me a (?:second|moment|sec)|"
    r"hang on|please wait|just wait|ok wait)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
```

If the whole transcript is one of these, the LLM call is skipped. Without the filter, saying *"hold on"* to a colleague while the agent is listening triggers a pointless generation.

---

## Barge-in: interrupting the agent by talking

Any voice agent that can't be cut off feels like a hold line. The frontend (`voice-agent-console.tsx`) runs RMS amplitude on every audio frame from the mic. When TTS is playing and the amplitude crosses `BARGE_IN_THRESHOLD` for `BARGE_IN_FRAMES` consecutive frames:

```ts
if (rms > BARGE_IN_THRESHOLD) {
  bargeinFrameCountRef.current += 1;
  if (bargeinFrameCountRef.current >= BARGE_IN_FRAMES) {
    interruptSentRef.current = true;
    ttsSourceRef.current.stop();                        // kill local audio
    socket.send(JSON.stringify({ type: "interrupt" })); // kill server work
  }
}
```

On the server, `interrupt` sets an `asyncio.Event`, cancels the live LLM task, drops any pending queued call, and every TTS send-path checks `interrupt_event.is_set()` before pushing bytes. The agent shuts up, the mic is already hot, and the next user utterance becomes the new turn.

The threshold needs care. Too low and the agent cuts itself off on its own echo — hence `echoCancellation: true, noiseSuppression: true` in the `getUserMedia` constraints.

---

## TTS: pluggable, Kokoro by default

`backend/app/services/tts.py` is a thin switch:

```python
model = self._load_kokoro() if self._backend == "kokoro" else self._load_chatterbox()
```

Four backends are installable as uv dependency groups — only one at a time:

| Backend | Hardware | Why pick it |
|---|---|---|
| Kokoro 82M (mlx-audio) | Apple Silicon only | Default. ~100 ms synth for a short sentence. |
| Chatterbox Turbo | Any (PyTorch) | Emotion tags, cross-platform. |
| Qwen TTS | Any (PyTorch) | Multilingual. |
| VibeVoice | Any (PyTorch) | Experimental. |

Switching is one env var plus a reinstall:

```bash
make backend-install TTS_BACKEND=chatterbox
TTS_BACKEND=chatterbox make backend
```

The reason they're separate groups and not all installed is size. Chatterbox drags in PyTorch. If you're on a Mac and you're fine with Kokoro, you don't need 2 GB of wheels.

---

## What gets logged

Two sinks, different jobs (`config/logging.py`):

- **Terminal (Loguru, colorized)** — runtime tail for development. Not persisted.
- **`backend/logs/*.json`** — rotated, latest 5 kept. Every entry has `event=`, `request_id=`, and structured fields.

On top of that, each WebSocket session writes a `SessionLog` on disconnect. It captures every STT partial, every LLM call (input, output, latency, cancellation), and the final. When you want to answer *"why did that turn feel slow"* three days later, this is the file you open, not the terminal scrollback.

Rule: never `print()`. Not once. Structured logs with `event=x latency_ms=...` are the only way to grep latency regressions.

---

## Running it

```bash
# install everything (backend via uv, frontend via npm)
make setup

# local LLM
brew install ollama
ollama pull gemma3:1b
ollama serve   # :11434

# run both services
make dev
# → http://localhost:3000
```

If you want to understand one piece at a time, `scripts/` has standalone demos that bypass the WebSocket entirely:

```bash
uv run --project backend python scripts/stt.py path/to/audio.wav
uv run --project backend python scripts/llm_call.py "reset my password"
uv run --project backend python scripts/tts.py "Hello, how can I help?"
uv run --project backend python scripts/agent.py path/to/audio.wav
```

Each one prints its own timing breakdown. Read `stt.py` first — it's the shortest path to understanding why a 900 ms audio clip transcribes in 300 ms.

---

## What's worth stealing

If you build your own version, the three things that change how it *feels*:

1. **Transcribe on partials, not just on stop.** Trigger the LLM off stable intermediate transcripts. Dedup on the last-seen input so you don't replay the same turn.
2. **Pipeline TTS with LLM streaming.** Send the first complete sentence to the synthesizer the instant its terminator shows up. The user hears audio before the model is done writing.
3. **Barge-in is not optional.** An agent you can't interrupt is worse than an agent that responds slowly.

Everything else — the model, the backend, the UI — is tradeable. These three behaviors are what turn the pipeline from a demo into something you'd actually keep the tab open for.
