from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

import numpy as np
import torch
from loguru import logger

_WARMUP_TEXT = "Hello."


class TTSService:
    def __init__(self) -> None:
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    def _load_model(self) -> Any:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info("event=tts_load device={}", device)
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        model.generate(_WARMUP_TEXT)
        logger.info("event=tts_ready device={}", device)
        return model

    def _run_inference(self, text: str) -> tuple[bytes, int]:
        waveform = self._model.generate(text)
        samples = (
            waveform.detach().cpu().squeeze().numpy()
            if torch.is_tensor(waveform)
            else np.asarray(waveform).squeeze()
        )
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._model.sr)
            wf.writeframes(pcm16.tobytes())

        return buf.getvalue(), self._model.sr

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, self._load_model)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_inference, text)


_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
