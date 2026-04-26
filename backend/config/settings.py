from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prompts.system import VOICE_AGENT_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─────────────────────────────────────────────────────────────────────────
    # APP — server identity, networking, and logging
    #
    # app_host / app_port   Change if another process owns 8000, or to bind
    #                       only on loopback (127.0.0.1) for local-only use.
    # log_level             DEBUG floods every audio frame. INFO is the right
    #                       default. WARNING/ERROR for production silence.
    # cors_origins_raw      Comma-separated browser origins allowed to connect.
    #                       Add your deployed frontend URL here; wrong value
    #                       causes the browser to block all API calls.
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
    # STT — faster-whisper speech-to-text
    #
    # stt_model_size    tiny.en | base.en | small.en | medium.en | large-v3.
    #                   Larger = more accurate, slower first-word latency.
    # stt_device        cpu | cuda | mps. MPS is Apple Silicon GPU.
    # stt_compute_type  int8 (fast CPU) | float16 (GPU) | float32 (slow CPU).
    # stt_beam_size     1 is greedy-fastest; raise to 3–5 for accuracy.
    # stt_vad_filter    Drops silent frames before Whisper sees them — large
    #                   speed win at the cost of very faint speech.
    # stt_language      Force language (e.g. "en"). Leave empty for auto-detect.
    # ─────────────────────────────────────────────────────────────────────────
    stt_model_size: str = "small.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_vad_filter: bool = True
    stt_language: str = "en"

    # ─────────────────────────────────────────────────────────────────────────
    # STREAMING / DEBOUNCE — controls when partials and LLM calls fire
    #
    # stream_emit_interval_ms   How often to emit a partial STT result (ms).
    #                           Lower = faster LLM trigger but more chatter.
    # stream_min_audio_ms       Minimum audio buffered before emitting (ms).
    #                           Lower = faster, more empty/noisy results.
    # stream_llm_min_chars      Minimum transcript length before firing LLM.
    #                           Guards against one-word false starts.
    # stream_llm_silence_ms     Silence window before firing the LLM (ms).
    #                           Lower = more responsive; higher = avoids
    #                           splitting one utterance into two replies.
    # ─────────────────────────────────────────────────────────────────────────
    stream_emit_interval_ms: int = 1200
    stream_min_audio_ms: int = 900
    stream_llm_min_chars: int = 8
    stream_llm_silence_ms: int = 950

    # ─────────────────────────────────────────────────────────────────────────
    # STREAMING VAD — dedicated voice-activity detector for endpointing
    #
    # stream_vad_enabled        Master switch. Disable to fall back to
    #                           silence-timer-only endpointing.
    # stream_vad_threshold      Silero probability threshold (0–1). Higher =
    #                           less sensitive; lower = triggers on noise.
    # stream_vad_min_silence_ms Silence duration to confirm end-of-speech.
    # stream_vad_speech_pad_ms  Padding kept around detected speech edges.
    # stream_vad_frame_samples  Silero frame size. 512 = 32 ms @ 16 kHz.
    # ─────────────────────────────────────────────────────────────────────────
    stream_vad_enabled: bool = True
    stream_vad_threshold: float = 0.6
    stream_vad_min_silence_ms: int = 800
    stream_vad_speech_pad_ms: int = 250
    stream_vad_frame_samples: int = 512

    # ─────────────────────────────────────────────────────────────────────────
    # TTS — text-to-speech synthesis
    #
    # tts_backend       kokoro | chatterbox. Must match the installed uv group
    #                   (e.g. uv sync --group kokoro_model).
    # tts_kokoro_voice  Default Kokoro voice name. GET /tts/voices lists all.
    #                   Overridden per-session via the tts_voice WebSocket msg.
    # tts_kokoro_speed  Playback speed multiplier (0.5–2.0). 1.0 = natural.
    #                   Overridden per-session via the tts_speed WebSocket msg.
    # welcome_message   Spoken greeting on session start. Set to "" to disable.
    # ─────────────────────────────────────────────────────────────────────────
    tts_backend: str = "kokoro"
    tts_kokoro_voice: str = "af_heart"
    tts_kokoro_speed: float = 1.0
    welcome_message: str = "Hello! I'm your Neurotalk voice assistant. How can I assist you today?"

    # ─────────────────────────────────────────────────────────────────────────
    # LLM — language model provider and context
    #
    # llm_provider          ollama | openai | anthropic | gemini | llama-cpp.
    # ollama_host           Base URL for a local Ollama server.
    # llm_model             Model tag for Ollama/OpenAI/Anthropic/Gemini providers.
    #                         qwen3:4b  — fast, strong tool-calling, low memory
    #                         qwen3:8b  — higher quality, ~2x slower
    #                         gemma3:1b — fastest, minimal memory, lower quality
    # openai/anthropic/     API keys — read from .env; only needed for the
    # gemini_api_key        selected provider.
    # llm_max_history_turns Number of user+assistant turn pairs kept in context.
    # llm_system_prompt     System prompt injected at the top of every request.
    # llm_llamacpp_model_path  Path to the GGUF model file for llama-cpp provider.
    #                          Run scripts/download_models.py --only-llm to fetch.
    # llm_llamacpp_n_ctx    Context window size in tokens (default 4096).
    # llm_llamacpp_n_gpu_layers  GPU layers offloaded. -1 = all (Metal on Apple
    #                            Silicon). Set 0 to run entirely on CPU.
    # ─────────────────────────────────────────────────────────────────────────
    llm_provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    llm_max_history_turns: int = 6
    llm_system_prompt: str = VOICE_AGENT_PROMPT
    llm_llamacpp_model_path: Path = Path("models/llm/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    llm_llamacpp_n_ctx: int = 4096
    llm_llamacpp_n_gpu_layers: int = -1

    # ─────────────────────────────────────────────────────────────────────────
    # WEB SEARCH — live search tool injected into LLM context
    #
    # web_search_enabled      Requires: uv sync --group search.
    # web_search_max_results  Cap on results fetched per query.
    # web_search_timeout_s    Per-request HTTP timeout. Raise if on a slow link.
    # ─────────────────────────────────────────────────────────────────────────
    web_search_enabled: bool = False
    web_search_max_results: int = 3
    web_search_timeout_s: float = 5.0

    # ─────────────────────────────────────────────────────────────────────────
    # SMART TURN DETECTION — ONNX model for intent-based endpointing
    #
    # stream_smart_turn_enabled     Master switch. Falls back to silence-only
    #                               debounce when disabled or model is absent.
    #                               Requires: uv sync --group smart_turn
    #                               and models/smart-turn-v3.2-cpu.onnx.
    #                               See scripts/download_smart_turn_model.py.
    # stream_smart_turn_threshold   Confidence threshold (0–1) to accept a
    #                               turn-complete prediction.
    # stream_smart_turn_model_path  Path to the ONNX model file.
    # stream_smart_turn_base_wait_ms  Polling interval while awaiting model
    #                               confirmation (ms).
    # stream_smart_turn_max_budget_ms  Max extra wait beyond silence timeout.
    # ─────────────────────────────────────────────────────────────────────────
    stream_smart_turn_enabled: bool = False
    stream_smart_turn_threshold: float = 0.5
    stream_smart_turn_model_path: str = "models/smart_turn/smart-turn-v3.2-cpu.onnx"
    stream_smart_turn_base_wait_ms: int = 200
    stream_smart_turn_max_budget_ms: int = 1000

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL DIRECTORIES — all models loaded from local disk; never downloaded
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
    # STORAGE — local filesystem paths
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
