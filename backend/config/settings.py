from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prompts.system import VOICE_AGENT_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. APP — server identity, networking, and logging
    # ─────────────────────────────────────────────────────────────────────────
    app_name: str = "NeuroTalk STT Backend"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"          # DEBUG | INFO | WARNING | ERROR
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DENOISE — DeepFilterNet3 noise suppression, applied before STT
    # ─────────────────────────────────────────────────────────────────────────
    denoise_enabled: bool = True
    denoise_model_dir: Path = Path("models/deepfilter")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. STT — faster-whisper speech-to-text
    # ─────────────────────────────────────────────────────────────────────────
    stt_model_size: str = "small.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"

    # Analogy: Wider search vs. fastest path
    # beam_size=1 is like asking Google Maps for ONE route and taking it immediately.
    # beam_size=5 is like asking for 5 routes, comparing them, then picking the best.
    # For voice chat where speed matters more than perfection, beam_size=1 is ideal.
    #
    # beam_size=1 (greedy):  "Hello" → done in 120ms
    # beam_size=5 (careful): "Hello" → 5 candidates compared → done in 400ms
    stt_beam_size: int = 1

    stt_vad_filter: bool = True
    stt_language: str = "en"

    # ─────────────────────────────────────────────────────────────────────────
    # 4. STREAMING VAD — Silero voice-activity detector for endpointing
    # ─────────────────────────────────────────────────────────────────────────

    stream_vad_enabled: bool = True

    # Analogy: The nightclub bouncer
    # The bouncer only lets people in if they look like a genuine guest (not tourists
    # wandering by). stream_vad_threshold is the minimum "confidence of speech" score
    # the Silero model must output before it declares "yes, someone is talking."
    #
    # Silero outputs a probability 0.0–1.0 for each 32ms audio frame:
    #   Fan noise       → prob ≈ 0.05  (below 0.6 → bouncer says "not speech, ignored")
    #   Keyboard clicks → prob ≈ 0.25  (below 0.6 → ignored)
    #   Faint whisper   → prob ≈ 0.55  (below 0.6 → borderline; missed at this threshold)
    #   Normal speech   → prob ≈ 0.90  (above 0.6 → bouncer says "come in, speech detected")
    #
    # ↑ 0.8 → misses quiet or far-away speakers (bouncer is very strict)
    # ↓ 0.3 → fan and keyboard trigger VAD (bouncer lets everyone in)
    stream_vad_threshold: float = 0.6

    # Analogy: The motion-sensor light
    # A motion sensor light doesn't turn off the instant you stop moving. It waits
    # for the room to be still for a set duration before switching off, so it doesn't
    # flicker every time you pause mid-gesture.
    # stream_vad_min_silence_ms = the "stillness window" before VAD declares end-of-speech.
    #
    # User: "How are you..."  [150ms pause]  "...doing today?"
    #        ─── speech ───   ─── gap ───    ─── speech ───
    # VAD:  [start]           still on ✓     still on ✓    (150ms < 500ms → same turn)
    #
    # User: "How are you doing today?"  [600ms silence]
    #        ──────── speech ─────────  ─── silence ───
    # VAD:  [start]                     500ms passes → [end] ✓  (600ms > 500ms → turn ends)
    #
    # ↑ 1000ms → waits longer; natural mid-sentence pauses don't split the turn (sluggish)
    # ↓ 200ms  → snappy but splits "I want to... order a pizza" into two separate turns
    stream_vad_min_silence_ms: int = 500

    # Analogy: The fabric safety margin
    # When cutting a piece of cloth along a drawn line, tailors cut 5mm wider than
    # the line to avoid accidentally clipping the edge. stream_vad_speech_pad_ms adds
    # a buffer of audio before and after the detected speech boundary.
    #
    # Without padding (pad=0ms):
    #   Detected:  [----speech----]
    #   Actual:  [--speech--]           ← first syllable clipped, last syllable clipped
    #
    # With padding (pad=250ms):
    #   Detected:  [----speech----]
    #   Buffered: [silence|----speech----|silence]  ← includes run-up and tail
    #
    # ↑ 500ms → retains a full half-second of silence on each side (wastes STT compute)
    # ↓  50ms → tight crop; first/last syllables may be clipped in Whisper
    stream_vad_speech_pad_ms: int = 250

    # Silero frame size in samples at 16 kHz. 512 samples = 32ms per VAD check.
    # Silero only supports 256, 512, or 768 — do not change without checking the model docs.
    stream_vad_frame_samples: int = 512

    # ─────────────────────────────────────────────────────────────────────────
    # 5. STREAMING / DEBOUNCE — controls when partials and LLM calls fire
    # ─────────────────────────────────────────────────────────────────────────

    # Analogy: The sports ticker update rate
    # A live sports ticker doesn't print a new score every millisecond — it updates
    # every few seconds so it's readable. stream_emit_interval_ms controls how often
    # a new partial transcript is sent to the browser while the user is still speaking.
    #
    # User speaks continuously for 3 seconds:
    #   t=0ms    Audio buffering starts
    #   t=700ms  Partial sent → browser shows: "How are you"
    #   t=1400ms Partial sent → browser shows: "How are you doing"
    #   t=2100ms Partial sent → browser shows: "How are you doing today"
    #   t=2800ms Partial sent → browser shows: "How are you doing today my friend"
    #
    # ↑ 2000ms → UI feels frozen between updates (score board stuck)
    # ↓  200ms → fast flicker; may overwhelm slow browser connections
    stream_emit_interval_ms: int = 700

    # Analogy: The doctor's minimum observation window
    # A doctor won't diagnose from 0.1 seconds of symptoms — they need enough signal.
    # stream_min_audio_ms is the minimum audio that must be buffered before we even
    # attempt a Whisper transcription. Below this, the result would just be noise or silence.
    #
    # Buffer at t=200ms (200ms < 500ms) → skip STT this tick, keep accumulating
    # Buffer at t=500ms (500ms ≥ 500ms) → enough audio, run Whisper now
    # Buffer at t=700ms (700ms ≥ 500ms) → run Whisper again (combined with interval check)
    #
    # ↑ 1000ms → first partial appears very late (doctor wants too much data)
    # ↓  100ms → Whisper runs on near-empty buffers; results are blank or hallucinated
    stream_min_audio_ms: int = 500

    # Analogy: The search bar minimum query length
    # Google ignores a single letter "a" — it needs at least a few characters to
    # return anything useful. stream_llm_min_chars prevents the LLM from being called
    # on noise artefacts that Whisper mistakenly transcribed as a word or two.
    #
    # Whisper hears fan noise → transcribes as "um"  (2 chars < 8 → LLM skipped ✓)
    # Whisper hears cough    → transcribes as "hey"  (3 chars < 8 → LLM skipped ✓)
    # User says "Stop"       → transcribes as "Stop" (4 chars < 8 → LLM skipped ✓)
    # User says "What time?" → transcribes as "What time?" (10 chars ≥ 8 → LLM fires ✓)
    #
    # ↑ 20 → misses short commands like "Yes", "Stop", "Repeat that"
    # ↓  2 → LLM fires on every noise artefact Whisper hallucinates
    stream_llm_min_chars: int = 8

    # Analogy: The elevator door
    # An elevator door doesn't close the instant the last person steps away. It waits
    # a fixed period. If someone runs back during that window, the timer resets. Only
    # when the full silence window passes with no new speech does the door close (LLM fires).
    #
    # User speaks: "How are you..."
    #                    ↓
    #              [Start 500ms timer]
    #                    ↓
    # User keeps speaking: "...doing today?"
    #                    ↓
    #              [Cancel & reset timer]  ← new audio = door re-opens
    #              [Start fresh 500ms countdown]
    #                    ↓
    # User stops speaking
    #                    ↓
    #              [500ms of silence passes — timer completes]
    #                    ↓
    #              ✓ Fire LLM call with full transcript
    #
    # ↑ 1500ms → very patient; rarely splits one thought into two but adds noticeable lag
    # ↓  200ms → ultra-responsive but may fire mid-sentence on a natural breath pause
    stream_llm_silence_ms: int = 500

    # ─────────────────────────────────────────────────────────────────────────
    # 6. SMART TURN DETECTION — ONNX semantic model for intent-based endpointing
    # ─────────────────────────────────────────────────────────────────────────

    stream_smart_turn_enabled: bool = True

    # Analogy: The confidence threshold before placing a trade
    # A stock trader only places a buy order when they are at least 65% confident.
    # Below that, the signal is too weak — they wait for more information.
    # stream_smart_turn_threshold is the minimum probability the ONNX model must
    # output for "utterance is complete" before the LLM is allowed to fire.
    #
    # User says: "I want to..."           → model outputs 0.30 (below 0.65 → wait, incomplete)
    # User says: "I want to know about Python" → model outputs 0.91 (above 0.65 → fire LLM ✓)
    # User says: "What's the weather"    → model outputs 0.72 (above 0.65 → fire LLM ✓)
    #
    # ↑ 0.90 → model rarely fires; user waits for the full silence budget on most questions
    # ↓ 0.40 → model fires too eagerly; cuts off "I want to... [pause] ...order a pizza"
    stream_smart_turn_threshold: float = 0.65

    stream_smart_turn_model_path: str = "models/smart_turn/smart-turn-v3.2-cpu.onnx"

    # Analogy: The chef tasting the sauce
    # A chef doesn't taste the sauce every millisecond — they stir, wait a beat,
    # taste again. stream_smart_turn_base_wait_ms is the pause between each
    # "is the utterance complete?" check. The model gets more audio context each wait.
    #
    # VAD fires "end of speech" at t=0
    #   t=0ms    Smart Turn check 1 → "I want to..." → 0.30 (incomplete, wait)
    #   t=200ms  Smart Turn check 2 → "I want to..." → 0.35 (still incomplete, wait)
    #   t=400ms  Smart Turn check 3 → "I want to..." → 0.45 (still incomplete, wait)
    #   t=600ms  Budget exhausted → fall back to silence-timer path
    #
    # ↑ 500ms → model only gets 1-2 chances within the budget (slow stirring, may miss the moment)
    # ↓  50ms → rapid polling; wastes CPU on nearly identical audio frames
    stream_smart_turn_base_wait_ms: int = 200

    # Analogy: The chess clock for Smart Turn
    # A chess player has a total time budget per move. Once the clock runs out, they
    # must play whatever move they have. stream_smart_turn_max_budget_ms is the total
    # time Smart Turn has to make its call. If it can't decide confidently within
    # 600ms, the silence-timer result is used anyway.
    #
    # Budget=600ms, base_wait=200ms → Smart Turn gets at most 3 checks:
    #   Check 1 at t=200ms
    #   Check 2 at t=400ms
    #   Check 3 at t=600ms → budget exhausted, move on
    #
    # ↑ 2000ms → Smart Turn has 10 attempts; very thorough but adds up to 2s extra latency
    # ↓  200ms → only 1 check; Smart Turn barely has a chance to be useful
    stream_smart_turn_max_budget_ms: int = 600

    # Analogy: The teacher's extra patience window
    # When a student raises their hand mid-thought and pauses, a good teacher doesn't
    # immediately call on someone else — they wait a bit longer, sensing the student
    # isn't done. stream_smart_turn_incomplete_wait_ms is the extra patience given when
    # Smart Turn consistently says "this utterance isn't complete yet."
    #
    # Normal path (model says complete):
    #   VAD end → Smart Turn check → 0.85 ≥ 0.65 → LLM fires immediately ✓
    #
    # Incomplete path (model says not done):
    #   VAD end → Smart Turn checks all say < 0.65 → budget exhausted
    #           → [extra 1500ms wait] → user may resume speaking in that window
    #           → if VAD fires again (user resumed): debounce resets, starts over
    #           → if still silent after 1500ms: LLM fires with what we have
    #
    # ↑ 3000ms → very patient; excellent for slow deliberate speakers; frustrating for fast ones
    # ↓  300ms → barely any extra wait; same as if Smart Turn weren't enabled
    stream_smart_turn_incomplete_wait_ms: int = 1500

    # ─────────────────────────────────────────────────────────────────────────
    # 7. LLM — language model provider and context
    # ─────────────────────────────────────────────────────────────────────────
    llm_provider: str = "llama-cpp"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Analogy: The waiter's memory notepad
    # A waiter writes down orders for the current table. After N tables, they flip
    # to a fresh page and only keep the most recent notes. llm_max_history_turns
    # controls how many back-and-forth exchanges the LLM can "see" when answering.
    #
    # Turn 1: User: "What is Python?"         Assistant: "Python is a language..."
    # Turn 2: User: "Give me an example."      Assistant: "Here's a hello world..."
    # Turn 3: User: "What about loops?"        Assistant: "In Python, you use for..."
    # ...
    # Turn 7: User: "Go back to your first answer"
    #         → With max_history_turns=6: LLM sees turns 2–7 but NOT turn 1 (forgotten)
    #         → With max_history_turns=10: LLM still has turn 1 in context ✓
    #
    # Each "turn" = 1 user message + 1 assistant reply = 2 entries in the list.
    # max_history_turns=6 keeps 12 messages in context (6 user + 6 assistant).
    #
    # ↑ 20 → rich long-term memory; each request sends more tokens → higher latency
    # ↓  2 → model forgets context after 2 exchanges; feels like talking to a goldfish
    llm_max_history_turns: int = 6

    llm_system_prompt: str = VOICE_AGENT_PROMPT
    llm_llamacpp_model_path: Path = Path("models/llm/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    llm_llamacpp_n_ctx: int = 2000
    llm_llamacpp_n_gpu_layers: int = -1
    llm_llamacpp_n_batch: int = 2048
    llm_llamacpp_flash_attn: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # 8. WEB SEARCH — live search tool injected into LLM context
    # ─────────────────────────────────────────────────────────────────────────
    web_search_enabled: bool = False
    web_search_max_results: int = 3
    web_search_timeout_s: float = 5.0

    # ─────────────────────────────────────────────────────────────────────────
    # 9. TTS — text-to-speech synthesis
    # ─────────────────────────────────────────────────────────────────────────
    tts_backend: str = "kokoro"
    tts_kokoro_voice: str = "af_heart"
    tts_kokoro_speed: float = 1.0
    welcome_message: str = "Hello! I'm your Neurotalk voice assistant. How can I assist you today?"

    # ─────────────────────────────────────────────────────────────────────────
    # 10. MODEL DIRECTORIES — all models loaded from local disk at startup
    # ─────────────────────────────────────────────────────────────────────────
    stt_model_dir: Path = Path("models/stt")
    vad_model_path: Path = Path("models/vad/silero_vad.jit")
    tts_kokoro_model_dir: Path = Path("models/kokoro")
    tts_chatterbox_model_dir: Path = Path("models/chatterbox")
    stream_smart_turn_extractor_dir: Path = Path("models/smart_turn/whisper-base")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. STORAGE — local filesystem paths
    # ─────────────────────────────────────────────────────────────────────────
    temp_dir: Path = Path(".cache/audio")

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
