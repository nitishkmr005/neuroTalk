from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prompts.system import VOICE_AGENT_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NeuroTalk STT Backend"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS",
    )

    stt_model_size: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_vad_filter: bool = True
    stt_language: str = ""
    stream_emit_interval_ms: int = 1200
    stream_min_audio_ms: int = 900
    stream_llm_min_chars: int = 8

    temp_dir: Path = Path(".cache/audio")

    # TTS backend — controls which dependency group is installed
    # Supported: chatterbox | qwen | vibevoice | omnivoice
    tts_backend: str = "kokoro"

    # LLM — Ollama
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "gemma4:latest"
    llm_max_tokens: int = 150
    llm_system_prompt: str = VOICE_AGENT_PROMPT

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
