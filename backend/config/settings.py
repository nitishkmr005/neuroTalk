from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prompts.system import VOICE_AGENT_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "NeuroTalk STT Backend"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"          # DEBUG | INFO | WARNING | ERROR
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS",
    )

    # ── STT — faster-whisper ──────────────────────────────────────────────────
    # Model size: tiny.en | base.en | small.en | medium.en | large-v3
    stt_model_size: str = "small.en"
    # Device: cpu | cuda | mps
    stt_device: str = "cpu"
    # Compute type: int8 (fast CPU) | float16 (GPU) | float32 (slow CPU)
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_vad_filter: bool = True
    # Force language (e.g. "en"). Leave empty for auto-detect.
    stt_language: str = "en"

    # ── Streaming / Debounce ──────────────────────────────────────────────────
    # How often to emit a partial STT result (ms). Lower = faster LLM trigger.
    stream_emit_interval_ms: int = 1200
    # Minimum audio buffer before emitting (ms). Lower = faster, more empty results.
    stream_min_audio_ms: int = 900
    # Minimum transcript length before firing the LLM (chars).
    stream_llm_min_chars: int = 8
    # Silence window before firing the LLM (ms). Lower = more responsive,
    # higher = avoids splitting one utterance into two replies.
    stream_llm_silence_ms: int = 950
    # Dedicated streaming VAD for endpointing and barge-in.
    stream_vad_enabled: bool = True
    stream_vad_threshold: float = 0.6
    stream_vad_min_silence_ms: int = 800
    stream_vad_speech_pad_ms: int = 250
    stream_vad_frame_samples: int = 512

    # ── TTS ───────────────────────────────────────────────────────────────────
    # Backend: kokoro | chatterbox | qwen | vibevoice | omnivoice
    # Must match the installed uv dependency group (e.g. uv sync --group kokoro_model).
    tts_backend: str = "kokoro"
    # Spoken greeting on session start. Set to "" to disable.
    welcome_message: str = "Hello! I'm your Neurotalk voice assistant. How can I assist you today?"

    # ── LLM — Ollama ─────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    # Recommended models (ollama pull <model>):
    #   qwen3:4b   — fast, strong tool-calling, low memory  (recommended)
    #   qwen3:8b   — higher quality, ~2x slower
    #   gemma3:1b  — fastest, minimal memory, lower quality
    #   gemma4:latest — high quality, large (9.6 GB)
    llm_model: str = "llama3.2:3b" #"gemma3:1b"
    # Number of user+assistant turn pairs to keep in context.
    llm_max_history_turns: int = 6
    llm_system_prompt: str = VOICE_AGENT_PROMPT

    # ── Storage ───────────────────────────────────────────────────────────────
    temp_dir: Path = Path(".cache/audio")

    @property
    def cors_origins(self) -> list[str]:
        """
        Parse the comma-separated CORS_ORIGINS env var into a list of origin strings.

        Args:
            None

        Returns:
            List of stripped origin URL strings.

        Library:
            pydantic-settings (BaseSettings) — value sourced from .env or environment.
        """
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the global Settings singleton, loaded once from .env.

    Args:
        None

    Returns:
        Cached Settings instance.

    Library:
        functools.lru_cache, pydantic-settings (BaseSettings).
    """
    return Settings()
