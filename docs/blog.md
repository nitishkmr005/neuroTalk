# NeuroTalk: How a Local Voice Agent Works

If you are new to voice agents, the cleanest mental model is this:

1. The microphone captures audio.
2. Speech-to-text turns audio into text.
3. A language model reads that text and writes a reply.
4. Text-to-speech turns the reply back into audio.
5. The app manages timing, interruption, and streaming so the whole thing feels conversational.

NeuroTalk follows exactly that pipeline. In this repo, speech recognition is handled by Whisper through `faster-whisper`, reply generation is handled by an Ollama-served LLM such as `gemma3:1b`, and speech synthesis is handled by Kokoro or Chatterbox in `backend/app/services/tts.py`. The repo also declares Qwen TTS, VibeVoice, and OmniVoice as optional backends in dependency groups, but the current `TTSService` only loads Kokoro or Chatterbox. Sources for the training claims and model descriptions are listed at the end, and every non-code claim below is tied to those sources.

---

## The pipeline in one example

Imagine the user says:

> "Can you help me reset my password?"

NeuroTalk processes that utterance like this:

1. The browser streams PCM audio frames over a WebSocket.
2. The backend repeatedly retranscribes the buffered audio and emits partial text such as `can you`, then `can you help`, then `can you help me reset my password`.
3. Once the partial text looks stable enough, the backend starts the LLM call before the user fully finishes.
4. As the LLM streams its answer, the backend looks for the first sentence boundary.
5. As soon as that first sentence is complete, TTS starts synthesizing it in parallel with the rest of the LLM output.
6. If the user speaks over the reply, the app cancels the running LLM/TTS work and starts listening for the next turn.

That is the high-level behavior. The rest of the article breaks down how each model is trained and how each stage is inferenced in this codebase.

---

## Step 1: audio enters through one WebSocket

NeuroTalk uses one WebSocket route, `/ws/transcribe`, to move the whole conversation loop. The browser sends audio bytes plus control messages such as `start`, `interrupt`, and `stop`. The backend sends transcript updates, streamed LLM text, and TTS audio back to the client.

The live loop is in [`backend/app/main.py`](../backend/app/main.py):

```python
@app.websocket("/ws/transcribe")
async def transcribe_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    ...
    while True:
        message = await websocket.receive()
        ...
        if message.get("bytes") is None or sample_rate is None:
            continue

        pcm_buffer.extend(message["bytes"])
        chunk_count += 1
        buffered_audio_ms = len(pcm_buffer) / 2 / sample_rate * 1000
```

What matters here:

- `pcm_buffer` stores the raw incoming speech for the current turn.
- `sample_rate` tells the backend how to interpret those PCM bytes.
- The backend keeps extending the same buffer so STT can re-run on progressively larger audio.

That buffering strategy is why the app can show live partial transcripts instead of waiting for a full upload.

---

## Step 2: STT with Whisper via `faster-whisper`

### How Whisper is trained

Whisper was trained by OpenAI on 680,000 hours of weakly supervised multilingual and multitask audio-text data. The core idea is simple: give the model audio and train it to predict the transcript tokens. Because the training corpus is large and diverse, the model generalizes better than narrow speech recognizers that were trained on smaller, more curated datasets.

Beginner example:

- Training input: audio of someone saying "my order number is 4521"
- Training target: `my order number is 4521`

After enough examples like that, the model learns a statistical mapping from acoustic patterns to text tokens.

### How STT inference works in NeuroTalk

The STT service lives in [`backend/app/services/stt.py`](../backend/app/services/stt.py):

```python
self._model = WhisperModel(
    self.settings.stt_model_size,
    device=self.settings.stt_device,
    compute_type=self.settings.stt_compute_type,
)

segments, info = model.transcribe(
    str(file_path),
    beam_size=self.settings.stt_beam_size,
    language=self.settings.stt_language or None,
    vad_filter=self.settings.stt_vad_filter,
)
```

Parameter breakdown:

- `stt_model_size`: which Whisper checkpoint to load. The repo default is `"small"`.
- `device`: where inference runs. The repo default is `"cpu"`.
- `compute_type`: numeric precision. The repo default is `"int8"` for lower memory use and faster CPU inference.
- `beam_size`: decoding width. The repo default is `1`, which means greedy decoding rather than a wider beam search.
- `language`: optional language hint. Empty means auto-detect.
- `vad_filter`: whether to filter non-speech and silence before transcription.

The defaults are defined in [`backend/config/settings.py`](../backend/config/settings.py):

```python
stt_model_size: str = "small"
stt_device: str = "cpu"
stt_compute_type: str = "int8"
stt_beam_size: int = 1
stt_vad_filter: bool = True
```

In the streaming loop, NeuroTalk does not feed raw live audio directly into Whisper token by token. Instead, it buffers audio, writes a WAV, retranscribes, and only emits a new partial when the text changes:

```python
should_emit = (
    buffered_audio_ms >= settings.stream_min_audio_ms
    and ((now - last_emit_at) * 1000) >= settings.stream_emit_interval_ms
)

result_payload = transcribe_stream_buffer(...)
current_text = str(result_payload["text"])
if current_text != last_text_sent:
    await send_json({"type": "partial", **result_payload})
    last_text_sent = current_text
```

Parameter breakdown:

- `stream_min_audio_ms`: minimum buffered speech before trying STT. Default `600`.
- `stream_emit_interval_ms`: how often the backend is allowed to emit a fresh partial. Default `800`.

This creates the familiar voice-agent behavior:

- at first: `reset`
- then: `reset my`
- then: `reset my password`

That rolling hypothesis behavior is one of the key differences between batch transcription and interactive transcription.

Two implementation details are worth calling out:

1. NeuroTalk uses `faster-whisper`, which is Whisper implemented on top of CTranslate2. SYSTRAN documents that this gives faster inference and supports reduced precision such as `int8`.
2. NeuroTalk enables `vad_filter=True`, which matters in practice because silence and room noise can otherwise turn into spurious transcript fragments that trigger unnecessary LLM calls.

---

## Step 3: deciding when to call the LLM

A voice agent feels slow if it waits for the user to fully stop, then starts thinking. NeuroTalk avoids that by starting LLM work from a stable partial transcript instead of waiting only for the final transcript.

The debounce logic is in [`backend/app/main.py`](../backend/app/main.py):

```python
if silence_debounce_task is not None and not silence_debounce_task.done():
    silence_debounce_task.cancel()
silence_debounce_task = asyncio.create_task(
    _silence_debounce_then_fire(current_text, "debounced_partial")
)
```

And the delay is controlled by:

```python
stream_llm_min_chars: int = 8
stream_llm_silence_ms: int = 900
```

Parameter breakdown:

- `stream_llm_min_chars`: do not call the LLM for transcript fragments that are too short to be meaningful.
- `stream_llm_silence_ms`: wait until the partial text stays unchanged long enough to count as a likely pause.

This is not a model-training idea. It is orchestration. But it matters because the perceived speed of a voice agent depends heavily on when the app decides the transcript is "good enough" to start generating a reply.

---

## Step 4: LLM inference through Ollama

### How the LLM is trained

The default language model in this repo is `gemma3:1b`. Google describes Gemma 3 as a family of lightweight open models with pre-trained and instruction-tuned variants and at least 128K context. The Gemma 3 technical report states that the models were trained with distillation and then improved with post-training methods for chat, instruction following, math, and multilingual behavior.

At a beginner level, the training picture is:

- pretraining teaches the model to predict the next token from previous text
- instruction tuning and post-training teach it to act more like an assistant

Example:

- Input text: `can you help me reset my password`
- Expected behavior after post-training: a short, assistant-style response rather than random continuation text

### How LLM inference works in NeuroTalk

The whole LLM service is intentionally small. It lives in [`backend/app/services/llm.py`](../backend/app/services/llm.py):

```python
async def stream_llm_response(transcript: str) -> AsyncGenerator[str, None]:
    settings = get_settings()
    client = AsyncClient(host=settings.ollama_host)

    stream = await client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": settings.llm_system_prompt},
            {"role": "user", "content": transcript},
        ],
        stream=True,
    )

    async for chunk in stream:
        token: str = chunk.message.content
        if token:
            yield token
```

Parameter breakdown:

- `ollama_host`: where the local Ollama server is running. Default `http://localhost:11434`.
- `llm_model`: which local model to use. Default `gemma3:1b`.
- `llm_system_prompt`: the behavioral instruction that shapes reply style.
- `stream=True`: tells Ollama to return incremental chunks rather than one final completion.

The corresponding defaults in [`backend/config/settings.py`](../backend/config/settings.py) are:

```python
ollama_host: str = "http://localhost:11434"
llm_model: str = "gemma3:1b"
llm_max_tokens: int = 100
llm_system_prompt: str = VOICE_AGENT_PROMPT
```

The important conceptual point is that this is not an end-to-end speech model. The LLM never sees raw audio. It only sees the transcript text produced by STT.

So the actual chain is:

`audio -> transcript -> chat completion`

That boundary explains a lot of failures:

- if STT gets the user intent wrong, the LLM may answer the wrong question
- if the system prompt is poorly designed, the reply may be too long or too markdown-heavy for TTS

This repo explicitly controls that second problem with a voice-oriented system prompt in [`backend/app/prompts/system.py`](../backend/app/prompts/system.py).

---

## Step 5: starting TTS before the full reply is done

One of the best latency tricks in NeuroTalk is that it does not wait for the entire LLM answer before starting speech synthesis. It starts TTS as soon as the first sentence is complete.

That logic is in [`backend/app/main.py`](../backend/app/main.py):

```python
async for token in stream_llm_response(text):
    full_response += token
    await send_json({"type": "llm_partial", "text": strip_emotion_tags(full_response)})
    if first_sent_task is None and not interrupt_event.is_set():
        m = _SENT_BOUNDARY.search(full_response)
        if m and m.end() >= 20:
            first_sent_end = m.end()
            snippet = full_response[:first_sent_end].strip()
            tts_svc = get_tts_service()
            first_sent_task = asyncio.ensure_future(tts_svc.synthesize(snippet))
```

Parameter breakdown:

- `_SENT_BOUNDARY`: regular expression that looks for `.`, `!`, or `?`.
- `m.end() >= 20`: do not fire TTS for trivially short fragments.
- `first_sent_task`: background synthesis task for the first sentence.

This overlap matters because the user hears the reply sooner even though the model is still generating later text.

---

## Step 6: TTS inference in this repo

### The current backends

The live `TTSService` in [`backend/app/services/tts.py`](../backend/app/services/tts.py) supports two code paths:

```python
def _load_model(self) -> Any:
    model = self._load_kokoro() if self._backend == "kokoro" else self._load_chatterbox()
```

That means:

- `tts_backend == "kokoro"` loads Kokoro
- anything else currently falls through to Chatterbox

The settings file lists supported backend names as `kokoro | chatterbox | qwen | vibevoice | omnivoice`, and `backend/pyproject.toml` defines dependency groups for all of them. But today, only Kokoro and Chatterbox are actually loaded in code.

### Kokoro

#### How Kokoro is trained

The original Kokoro model card describes it as an 82M open-weight TTS model, and the MLX checkpoint used here is a conversion of that original model. The card also points to a StyleTTS2-derived lineage. At a beginner level, you can think of Kokoro as a text-to-speech model trained on aligned text-audio pairs so it learns pronunciation, prosody, and voice identity.

Example:

- Input text during training: `your refund will arrive in three business days`
- Target: an audio recording of a speaker saying that sentence naturally

#### How Kokoro inference works in this repo

Kokoro loading:

```python
from mlx_audio.tts.utils import load_model
model = load_model(_KOKORO_MODEL_ID)
for _ in model.generate(_WARMUP_TEXT, voice=_KOKORO_VOICE, speed=_KOKORO_SPEED, lang_code=_KOKORO_LANG):
    pass
```

Kokoro synthesis:

```python
for result in self._model.generate(clean, voice=_KOKORO_VOICE, speed=_KOKORO_SPEED, lang_code=_KOKORO_LANG):
    final_audio = result.audio
    sample_rate = getattr(result, "sample_rate", 24000)
```

Parameter breakdown:

- `_KOKORO_MODEL_ID`: the MLX model checkpoint, `mlx-community/Kokoro-82M-bf16`
- `voice`: speaker preset, default `af_heart`
- `speed`: speaking rate, default `1.0`
- `lang_code`: language control code, default `"a"`

Then NeuroTalk converts the generated floating-point waveform to PCM16 WAV bytes:

```python
samples = np.asarray(final_audio).squeeze()
pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
```

One implementation detail matters a lot: Kokoro strips emotion tags before synthesis in this repo:

```python
from app.utils.emotion import strip_emotion_tags
clean = strip_emotion_tags(text)
```

So Kokoro is being used here as a clean text-to-audio engine, not as the expressive tag-preserving path.

### Chatterbox Turbo

#### How Chatterbox is trained

Resemble AI describes Chatterbox Turbo as a streamlined 350M-parameter TTS model and says its speech-token-to-mel decoder was distilled from 10 steps to 1. The project also documents native support for paralinguistic tags such as `[laugh]` and `[chuckle]`, which indicates an expressive synthesis design rather than plain neutral readout only.

#### How Chatterbox inference works in this repo

Loading:

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
model = ChatterboxTurboTTS.from_pretrained(device=device)
model.generate(_WARMUP_TEXT)
```

Parameter breakdown:

- `device`: picks Apple Metal, CUDA, or CPU depending on the machine.
- `from_pretrained(...)`: loads pretrained weights rather than training locally.
- `_WARMUP_TEXT`: forces one initial generation so the model is ready before user-facing traffic.

Synthesis:

```python
waveform = self._model.generate(text)
samples = (
    waveform.detach().cpu().squeeze().numpy()
    if torch.is_tensor(waveform)
    else np.asarray(waveform).squeeze()
)
```

Unlike the Kokoro path, the Chatterbox path keeps the input text intact, including supported expressive tags. That fits the project prompt design, where the LLM is allowed to emit a small set of inline emotion markers.

---

## Optional backends declared in the repo

The repo also defines dependency groups for Qwen TTS, VibeVoice, and OmniVoice in [`backend/pyproject.toml`](../backend/pyproject.toml). They are not loaded by the current `TTSService`, but they still matter because they show the intended expansion path of the project.

### Qwen TTS

The Qwen3-TTS technical report describes a multilingual, controllable, streaming TTS family trained on more than 5 million hours of speech data across 10 languages. The paper says the system uses a dual-track LM architecture plus specialized speech tokenizers for streaming generation.

The beginner picture is:

- training: text and control signals are paired with speech targets
- inference: the model predicts speech tokens and a decoder reconstructs waveform audio

Because this repo currently only declares `qwen-tts` as a dependency group and does not yet wire it into `TTSService`, there is no live inference snippet to show from the codebase yet.

### VibeVoice

Microsoft describes VibeVoice as a voice-model family using continuous speech tokenizers at 7.5 Hz plus a next-token diffusion framework. The technical report focuses on long-form multi-speaker synthesis.

The beginner picture is:

- training: text context, semantic planning, and acoustic supervision are learned together
- inference: the model predicts compressed speech representations and a diffusion component produces detailed speech

Again, this repo declares the dependency group but does not yet expose a VibeVoice inference path in `backend/app/services/tts.py`.

### OmniVoice

The OmniVoice paper describes a zero-shot multilingual TTS model for more than 600 languages. It uses a diffusion language model-style discrete non-autoregressive architecture and was trained on a 581k-hour multilingual dataset curated from open-source data.

The beginner picture is:

- training: text is supervised against multilingual acoustic-token targets
- inference: text maps to acoustic tokens, then to waveform audio, with a design that aims to be faster than older multi-stage pipelines

As with Qwen and VibeVoice, NeuroTalk currently declares OmniVoice as an optional backend but does not yet route inference to it.

---

## Step 7: interruption and turn-taking

Voice agents feel broken if they cannot be interrupted. NeuroTalk treats interruption as a core interaction, not a bonus feature.

When the frontend decides the user is speaking over the agent, it sends `interrupt`. On the backend:

```python
if event_type == "interrupt":
    interrupt_event.set()
    pending_llm_call = None
    ...
    if llm_task is not None and not llm_task.done():
        llm_task.cancel()
```

And before TTS sends audio, the backend checks the same interrupt flag:

```python
async def synthesize_and_send(...):
    if interrupt_event.is_set():
        return
```

That means interruption is implemented as a shared cancellation signal across the whole voice loop:

- stop any queued LLM work
- cancel running generation
- suppress outgoing TTS audio
- prepare for the next user turn

This is one of the main reasons the system feels like a conversation instead of a push-to-talk demo.

---

## Why the sequence matters

The main lesson from NeuroTalk is not just which models are used. It is how they are sequenced.

1. STT is trained to map audio to text, and NeuroTalk runs it repeatedly on a growing buffer.
2. The LLM is trained as a text model, and NeuroTalk feeds it stable partial transcripts instead of waiting for the full stop event.
3. TTS is trained to map text to speech, and NeuroTalk starts it on the first sentence before the full reply is complete.
4. The app manages interruption and timing so the boundaries between the models feel invisible to the user.

That is the real shape of a practical voice agent: not one magical model, but multiple specialized models stitched together with careful streaming logic.

---

## Sources

### Repo files

- [`backend/app/main.py`](../backend/app/main.py)
- [`backend/app/services/stt.py`](../backend/app/services/stt.py)
- [`backend/app/services/llm.py`](../backend/app/services/llm.py)
- [`backend/app/services/tts.py`](../backend/app/services/tts.py)
- [`backend/config/settings.py`](../backend/config/settings.py)
- [`backend/app/prompts/system.py`](../backend/app/prompts/system.py)
- [`backend/pyproject.toml`](../backend/pyproject.toml)

### External references

- OpenAI, "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper): https://arxiv.org/abs/2212.04356
- SYSTRAN, `faster-whisper` README: https://github.com/SYSTRAN/faster-whisper
- OpenNMT, CTranslate2 README: https://github.com/OpenNMT/CTranslate2
- Google, Gemma 3 model card: https://ai.google.dev/gemma/docs/core/model_card_3
- Gemma Team, "Gemma 3 Technical Report": https://arxiv.org/abs/2503.19786
- Ollama API docs: https://docs.ollama.com/api
- hexgrad, Kokoro model card: https://huggingface.co/hexgrad/Kokoro-82M
- MLX Community, Kokoro MLX model card: https://huggingface.co/mlx-community/Kokoro-82M-bf16
- Resemble AI, Chatterbox repository: https://github.com/resemble-ai/chatterbox
- Qwen Team, "Qwen3-TTS Technical Report": https://arxiv.org/abs/2601.15621
- Microsoft, VibeVoice repository: https://github.com/microsoft/VibeVoice
- Microsoft Research, "VibeVoice Technical Report": https://arxiv.org/abs/2508.19205
- "OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models": https://arxiv.org/abs/2604.00688
