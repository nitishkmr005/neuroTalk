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
    # How often to emit a partial STT result (ms). Lower = faster LLM trigger.
    stream_emit_interval_ms: int = 800
    # Minimum audio buffer before emitting (ms). Lower = faster, but more empty results.
    stream_min_audio_ms: int = 600
    stream_llm_min_chars: int = 8

    temp_dir: Path = Path(".cache/audio")

    # TTS backend — controls which dependency group is installed
    # Supported: kokoro | chatterbox | qwen | vibevoice | omnivoice
    tts_backend: str = "kokoro"
    # Spoken greeting when a session starts. Set to "" to disable.
    welcome_message: str = "Hello! I'm your Neurotalk voice assistant. How can I assist you today ?"

    # LLM — Ollama
    ollama_host: str = "http://localhost:11434"
    # gemma3:1b  — fast, low memory  |  gemma4:latest  — higher quality
    llm_model: str = "gemma3:1b"
    llm_max_tokens: int = 100
    llm_system_prompt: str = VOICE_AGENT_PROMPT

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
