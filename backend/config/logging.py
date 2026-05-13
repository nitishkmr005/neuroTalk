import os
import sys

from loguru import logger

from config.settings import get_settings

# Suppress tqdm progress bars from mlx_audio (Chatterbox token generation bars
# go to stderr and pollute structured log output on every TTS call).
os.environ.setdefault("TQDM_DISABLE", "1")
# Suppress the "model of type chatterbox_turbo to instantiate model of type ''"
# warning from HuggingFace transformers when loading the Chatterbox tokenizer.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Log events that indicate a model is being loaded or invoked.
# Matching messages are rendered in bold green so they stand out in the terminal.
_MODEL_EVENTS: frozenset[str] = frozenset({
    # STT
    "event=model_load_started",
    "event=model_load_finished",
    "event=stt_warmup_done",
    # LLM (llama-cpp)
    "event=llamacpp_load",
    "event=llamacpp_ready",
    "event=llamacpp_warmup_done",
    # TTS
    "event=tts_load",
    "event=tts_ready",
    "event=tts_warmup_done",
    # VAD / DeepFilter / Smart-turn
    "event=vad_warmup_done",
    "event=denoise_warmup_done",
    "event=smart_turn_loaded",
    # Meeting recorder
    "event=meeting_transcribe_done",
    "event=meeting_stt_dispatch",
    "event=meeting_llm_dispatch",
})


def _fmt(record: dict) -> str:
    is_model = any(e in record["message"] for e in _MODEL_EVENTS)
    msg_markup = "<green><b>{message}</b></green>" if is_model else "<level>{message}</level>"
    return (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        + msg_markup
        + "\n{exception}"
    )


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        colorize=True,
        format=_fmt,
    )
