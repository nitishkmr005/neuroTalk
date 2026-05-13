from __future__ import annotations

import asyncio
import io
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from config.settings import get_settings

_WARMUP_TEXT = "Hello."

_KOKORO_LANG = "a"
_ESPEAK_CANDIDATES = [
    "/opt/homebrew/share/espeak-ng-data",
    "/usr/local/share/espeak-ng-data",
    "/usr/share/espeak-ng-data",
]


class TTSService:
    def __init__(self) -> None:
        self._model: Any = None
        self._load_lock = asyncio.Lock()
        self._backend: str = get_settings().tts_backend

    def _load_model(self) -> Any:
        logger.info("event=tts_load backend={}", self._backend)
        model = self._load_kokoro() if self._backend == "kokoro" else self._load_chatterbox()
        logger.info("event=tts_ready backend={}", self._backend)
        return model

    def _resolve_voice(self, voice: str) -> str:
        # Allowlist: only bare filenames (no path separators or dots beyond the stem).
        if "/" in voice or "\\" in voice or ".." in voice:
            voice = get_settings().tts_kokoro_voice
        voice_path = get_settings().tts_kokoro_model_dir / "voices" / f"{voice}.safetensors"
        return str(voice_path) if voice_path.exists() else voice

    def _load_kokoro(self) -> Any:
        if "ESPEAK_DATA_PATH" not in os.environ:
            for candidate in _ESPEAK_CANDIDATES:
                if Path(candidate).is_dir():
                    os.environ["ESPEAK_DATA_PATH"] = candidate
                    break
        settings = get_settings()
        from mlx_audio.tts.utils import load_model
        model = load_model(settings.tts_kokoro_model_dir)
        for _ in model.generate(
            _WARMUP_TEXT,
            voice=self._resolve_voice(settings.tts_kokoro_voice),
            speed=settings.tts_kokoro_speed,
            lang_code=_KOKORO_LANG,
        ):
            pass
        return model

    def _load_chatterbox(self) -> Any:
        from mlx_audio.tts.utils import load_model as mlx_load_model
        local_dir = get_settings().tts_chatterbox_model_dir
        if not local_dir.exists():
            raise RuntimeError(
                f"Chatterbox model not found at {local_dir}. "
                "Run: make chatterbox-model"
            )
        model = mlx_load_model(local_dir)
        for _ in model.generate(_WARMUP_TEXT):
            pass
        return model

    def _run_inference(self, text: str, voice: str | None = None, speed: float | None = None) -> tuple[bytes, int]:
        return self._run_kokoro(text, voice, speed) if self._backend == "kokoro" else self._run_chatterbox(text)

    def _run_kokoro(self, text: str, voice: str | None = None, speed: float | None = None) -> tuple[bytes, int]:
        settings = get_settings()
        from app.utils.emotion import strip_emotion_tags
        clean = strip_emotion_tags(text)
        v = self._resolve_voice(voice or settings.tts_kokoro_voice)
        s = speed if speed is not None else settings.tts_kokoro_speed
        final_audio = None
        sample_rate = 24000
        for result in self._model.generate(clean, voice=v, speed=s, lang_code=_KOKORO_LANG):
            final_audio = result.audio
            sample_rate = getattr(result, "sample_rate", 24000)
        if final_audio is None:
            raise RuntimeError("Kokoro returned no audio")
        samples = np.asarray(final_audio).squeeze()
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue(), sample_rate

    def _run_chatterbox(self, text: str) -> tuple[bytes, int]:
        from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import ChatterboxTurboTTS
        is_turbo = isinstance(self._model, ChatterboxTurboTTS)
        if is_turbo:
            # Turbo: normalize tag variants and strip unknowns before T3 sees the text.
            # Unrecognized [tags] are read aloud verbatim; canonical special tokens are not.
            from app.utils.emotion import normalize_for_turbo
            synthesis_text = normalize_for_turbo(text)
            kwargs: dict = {}
        else:
            # Standard Chatterbox: strip tags from text, drive emotion via exaggeration float.
            from app.utils.emotion import extract_exaggeration, strip_emotion_tags
            synthesis_text = strip_emotion_tags(text)
            kwargs = {"exaggeration": extract_exaggeration(text)}
        logger.debug(
            "event=chatterbox_infer is_turbo={} kwargs={} text_preview={!r}",
            is_turbo, kwargs, synthesis_text[:60],
        )
        chunks: list[np.ndarray] = []
        sample_rate = self._model.sample_rate
        for result in self._model.generate(synthesis_text, **kwargs):
            chunks.append(np.asarray(result.audio).squeeze())
        if not chunks:
            raise RuntimeError("Chatterbox returned no audio")
        samples = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue(), sample_rate

    async def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> tuple[bytes, int]:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, self._load_model)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_inference, text, voice, speed)


_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


def get_available_voices() -> list[str]:
    voices_dir = get_settings().tts_kokoro_model_dir / "voices"
    return sorted(p.stem for p in voices_dir.glob("*.safetensors"))
