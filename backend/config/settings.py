from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prompts.system import VOICE_AGENT_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. APP — server identity, networking, and logging
    #    Loaded first; governs how the process binds and what it emits.
    #
    # app_host          "0.0.0.0" listens on all interfaces (needed for Docker /
    #                   LAN access). Use "127.0.0.1" to restrict to loopback only.
    # app_port          Change if 8000 is taken by another process.
    # log_level         ↑ WARNING/ERROR → silent in prod, hides useful traces.
    #                   ↓ DEBUG         → floods every audio frame; very noisy.
    #                   INFO is the right default.
    # cors_origins_raw  Comma-separated browser origins allowed to connect.
    #                   Wrong value → browser blocks every API call silently.
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
    #    First stage in the audio pipeline; cleans the signal Whisper sees.
    #
    # denoise_enabled   When False (or deepfilternet not installed) audio goes
    #                   to Whisper unmodified — faster but noisier transcripts.
    # denoise_model_dir Local directory with DeepFilterNet3 weights.
    #                   Run scripts/download_deepfilter_model.py to populate.
    # ─────────────────────────────────────────────────────────────────────────
    denoise_enabled: bool = True
    denoise_model_dir: Path = Path("models/deepfilter")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. STT — faster-whisper speech-to-text
    #    Runs on denoised audio and produces the transcript.
    #
    # stt_model_size    tiny.en | base.en | small.en | medium.en | large-v3.
    #                   ↑ larger → more accurate, slower first-word latency.
    #                   ↓ smaller → near-instant, but misses mumbled words.
    # stt_device        cpu | cuda | mps. MPS = Apple Silicon GPU.
    # stt_compute_type  int8 (fast CPU) | float16 (GPU) | float32 (slow CPU).
    # stt_beam_size     ↑ high (3–5) → more accurate beam search, slower.
    #                   ↓ 1          → greedy decode, fastest possible.
    # stt_vad_filter    Drops silent frames before Whisper sees them.
    #                   ↑ True  → large speed win, may drop very faint speech.
    #                   ↓ False → Whisper processes every frame; slower.
    # stt_language      Force language (e.g. "en"). Empty = auto-detect.
    #                   ↑ forced → skips language-detect overhead, more stable.
    #                   ↓ auto   → slight latency penalty per chunk.
    # ─────────────────────────────────────────────────────────────────────────
    stt_model_size: str = "small.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_vad_filter: bool = True
    stt_language: str = "en"

    # ─────────────────────────────────────────────────────────────────────────
    # 4. STREAMING VAD — Silero voice-activity detector for endpointing
    #    Runs continuously on incoming audio; signals when speech starts/stops.
    #    Fires before the debounce logic that decides when to call the LLM.
    #
    # stream_vad_enabled        Master switch. False → silence-timer-only
    #                           endpointing (less accurate).
    # stream_vad_threshold      ↑ high (0.8+) → misses quiet/distant speech.
    #                           ↓ low  (0.3−) → false positives on background
    #                           noise; LLM fires mid-noise.
    # stream_vad_min_silence_ms ↑ high → waits longer to confirm end-of-speech;
    #                           feels sluggish but avoids cutting off sentences.
    #                           ↓ low  → snappy but may split one utterance
    #                           into two turns.
    # stream_vad_speech_pad_ms  ↑ high → retains more context around edges;
    #                           wastes bandwidth on silence.
    #                           ↓ low  → tight crop; first/last syllables may
    #                           be clipped.
    # stream_vad_frame_samples  Silero frame size. 512 = 32 ms @ 16 kHz.
    #                           ↑ larger → coarser detection, lower CPU use.
    #                           ↓ smaller → finer, but not supported by Silero.
    # ─────────────────────────────────────────────────────────────────────────
    stream_vad_enabled: bool = True
    stream_vad_threshold: float = 0.6
    stream_vad_min_silence_ms: int = 500
    stream_vad_speech_pad_ms: int = 250
    stream_vad_frame_samples: int = 512

    # ─────────────────────────────────────────────────────────────────────────
    # 5. STREAMING / DEBOUNCE — controls when partials and LLM calls fire
    #    After VAD confirms silence, these thresholds gate the LLM trigger.
    #
    # stream_emit_interval_ms   ↑ high → infrequent partial updates; UI feels
    #                           laggy but less chatter over the WebSocket.
    #                           ↓ low  → fast partials; can overwhelm slow
    #                           clients or cause flicker.
    # stream_min_audio_ms       ↑ high → more audio buffered before emitting;
    #                           reduces empty/noisy results.
    #                           ↓ low  → fast first partial, but may be blank.
    # stream_llm_min_chars      ↑ high → LLM only fires on longer phrases;
    #                           misses short commands ("Stop", "Yes").
    #                           ↓ low  → fires on every noise fragment.
    # stream_llm_silence_ms     ↑ high → more patient; avoids splitting one
    #                           utterance into two requests.
    #                           ↓ low  → very responsive but may cut off slow
    #                           speakers mid-sentence.
    # ─────────────────────────────────────────────────────────────────────────
    stream_emit_interval_ms: int = 700
    stream_min_audio_ms: int = 500
    stream_llm_min_chars: int = 8
    stream_llm_silence_ms: int = 500

    # ─────────────────────────────────────────────────────────────────────────
    # 6. SMART TURN DETECTION — ONNX semantic model for intent-based endpointing
    #    Runs after silence is detected; confirms the user finished their turn
    #    before allowing the LLM to reply. Falls back to silence-only when
    #    disabled or the model file is absent.
    #    Requires: uv sync --group smart_turn
    #              models/smart-turn-v3.2-cpu.onnx (see scripts/download_smart_turn_model.py)
    #
    # stream_smart_turn_enabled          Master switch.
    # stream_smart_turn_threshold        ↑ high (0.9+) → rarely triggers; user
    #                                    often has to wait for the full silence
    #                                    budget before LLM fires.
    #                                    ↓ low  (0.4−) → triggers too eagerly;
    #                                    cuts off incomplete sentences.
    # stream_smart_turn_base_wait_ms     ↑ high → slower polling; model gets
    #                                    more audio before each check.
    #                                    ↓ low  → rapid polling; wastes CPU.
    # stream_smart_turn_max_budget_ms    ↑ high → longer extra wait beyond
    #                                    silence timeout before giving up.
    #                                    ↓ low  → falls back to silence-only
    #                                    quickly; Smart Turn barely helps.
    # stream_smart_turn_incomplete_wait_ms  Extra wait added when model
    #                                    consistently says "not complete".
    #                                    ↑ high → very patient; helps slow
    #                                    deliberate speakers.
    #                                    ↓ low  → cuts off if model keeps
    #                                    returning incomplete.
    # ─────────────────────────────────────────────────────────────────────────
    stream_smart_turn_enabled: bool = True
    stream_smart_turn_threshold: float = 0.65
    stream_smart_turn_model_path: str = "models/smart_turn/smart-turn-v3.2-cpu.onnx"
    stream_smart_turn_base_wait_ms: int = 200
    stream_smart_turn_max_budget_ms: int = 600
    stream_smart_turn_incomplete_wait_ms: int = 1500

    # ─────────────────────────────────────────────────────────────────────────
    # 7. LLM — language model provider and context
    #    Receives the confirmed transcript and generates a reply.
    #
    # llm_provider          ollama | openai | anthropic | gemini | llama-cpp.
    # ollama_host           Base URL for a local Ollama server.
    # llm_model             Model tag for Ollama/OpenAI/Anthropic/Gemini providers.
    #                         qwen3:4b  — fast, strong tool-calling, low memory
    #                         qwen3:8b  — higher quality, ~2x slower
    #                         gemma3:1b — fastest, minimal memory, lower quality
    # openai/anthropic/     API keys — read from .env; needed only for the
    # gemini_api_key        selected provider.
    # llm_max_history_turns ↑ high → richer multi-turn context; more tokens
    #                       per request, higher latency.
    #                       ↓ low  → forgets earlier context quickly; cheap
    #                       and fast but breaks long conversations.
    # llm_system_prompt     Injected at the top of every request.
    # llm_llamacpp_model_path  Path to the GGUF file. Run
    #                          scripts/download_models.py --only-llm to fetch.
    # llm_llamacpp_n_ctx    ↑ high → handles longer conversations; uses more
    #                       RAM/VRAM proportionally.
    #                       ↓ low  → truncates long contexts silently.
    # llm_llamacpp_n_batch  ↑ high → faster prompt prefill (parallel tokens);
    #                       more peak memory during prefill.
    #                       ↓ low  → slower prefill; negligible memory impact.
    # llm_llamacpp_n_gpu_layers  ↑ -1 / all → fastest (full Metal/CUDA offload).
    #                            ↓ 0         → CPU-only; much slower on large models.
    # llm_llamacpp_flash_attn   True → lower memory, faster attention on GPU.
    #                           False → compatible with older drivers; slightly
    #                           slower and uses more VRAM.
    # ─────────────────────────────────────────────────────────────────────────
    llm_provider: str = "llama-cpp"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    llm_max_history_turns: int = 6
    llm_system_prompt: str = VOICE_AGENT_PROMPT
    llm_llamacpp_model_path: Path = Path("models/llm/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    llm_llamacpp_n_ctx: int = 2000
    llm_llamacpp_n_gpu_layers: int = -1
    llm_llamacpp_n_batch: int = 2048
    llm_llamacpp_flash_attn: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # 8. WEB SEARCH — live search tool injected into LLM context
    #    Called as an LLM tool during the LLM inference step when enabled.
    #    Requires: uv sync --group search.
    #
    # web_search_enabled      False → LLM answers from training knowledge only.
    # web_search_max_results  ↑ high → more context for the LLM; slower fetch
    #                         and larger prompt.
    #                         ↓ low  → fast but may miss the best result.
    # web_search_timeout_s    ↑ high → tolerates slow networks; stalls response
    #                         if search is the bottleneck.
    #                         ↓ low  → fails fast on slow links; LLM falls back
    #                         to training knowledge.
    # ─────────────────────────────────────────────────────────────────────────
    web_search_enabled: bool = False
    web_search_max_results: int = 3
    web_search_timeout_s: float = 5.0

    # ─────────────────────────────────────────────────────────────────────────
    # 9. TTS — text-to-speech synthesis
    #    Last step in the pipeline; converts the LLM reply to audio.
    #
    # tts_backend       kokoro | chatterbox. Must match the installed uv group
    #                   (e.g. uv sync --group kokoro_model).
    # tts_kokoro_voice  Default Kokoro voice. GET /tts/voices lists all.
    #                   Overridden per-session via the tts_voice WebSocket msg.
    # tts_kokoro_speed  ↑ high (1.5–2.0) → faster speech; harder to follow for
    #                   non-native speakers.
    #                   ↓ low  (0.5–0.8) → slower, clearer; may feel unnatural.
    #                   Overridden per-session via the tts_speed WebSocket msg.
    # welcome_message   Spoken greeting on session start. Set "" to disable.
    # ─────────────────────────────────────────────────────────────────────────
    tts_backend: str = "kokoro"
    tts_kokoro_voice: str = "af_heart"
    tts_kokoro_speed: float = 1.0
    welcome_message: str = "Hello! I'm your Neurotalk voice assistant. How can I assist you today?"

    # ─────────────────────────────────────────────────────────────────────────
    # 10. MODEL DIRECTORIES — all models loaded from local disk at startup
    #
    # stt_model_dir                  CTranslate2 Whisper model (config.json +
    #                                model.bin). Passed directly to WhisperModel.
    # vad_model_path                 Silero VAD TorchScript model (.jit).
    # tts_kokoro_model_dir           Kokoro MLX model dir (config.json +
    #                                kokoro-v1_0.safetensors + voices/).
    # tts_chatterbox_model_dir       Local Chatterbox model cache dir. If present,
    #                                loaded offline; otherwise downloads from HF.
    # stream_smart_turn_extractor_dir  Whisper feature extractor weights for
    #                                  Smart Turn (only loaded when enabled).
    # ─────────────────────────────────────────────────────────────────────────
    stt_model_dir: Path = Path("models/stt")
    vad_model_path: Path = Path("models/vad/silero_vad.jit")
    tts_kokoro_model_dir: Path = Path("models/kokoro")
    tts_chatterbox_model_dir: Path = Path("models/chatterbox")
    stream_smart_turn_extractor_dir: Path = Path("models/smart_turn/whisper-base")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. STORAGE — local filesystem paths
    #
    # temp_dir  Scratch directory for temporary audio files. Created on startup
    #           if absent. Safe to wipe between runs.
    # ─────────────────────────────────────────────────────────────────────────
    temp_dir: Path = Path(".cache/audio")

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
