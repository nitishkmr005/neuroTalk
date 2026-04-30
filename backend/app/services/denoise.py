"""DeepFilterNet3 noise suppression applied to raw PCM before STT.

Applied server-side on the accumulated PCM buffer, giving Whisper cleaner
input and reducing hallucinations in noisy environments.

Setup
-----
1. Install dependencies::

       uv sync --group deepfilter

2. Download model weights::

       python scripts/download_deepfilter_model.py

3. Enable in .env or settings::

       DENOISE_ENABLED=true

When the package is not installed or the model directory is absent, every
``enhance()`` call returns the original bytes unchanged — no pipeline change.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np
from loguru import logger

from config.settings import get_settings


class DenoiseService:
    """Wraps DeepFilterNet3 for real-time PCM noise suppression.

    Attributes:
        is_loaded: ``True`` when the model is ready for inference.
    """

    def __init__(self) -> None:
        self._model = None
        self._df_state = None
        self._enhance_fn = None
        self._df_sr: int = 48_000
        self._loaded = False
        self._load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _load(self) -> None:
        settings = get_settings()
        if not settings.denoise_enabled:
            return
        try:
            from df.enhance import enhance, init_df

            self._model, self._df_state, _ = init_df(
                model_base_dir=str(settings.denoise_model_dir)
            )
            self._enhance_fn = enhance
            self._df_sr = self._df_state.sr()
            self._loaded = True
            logger.info(
                "event=denoise_loaded model=DeepFilterNet3 sr={} model_dir={}",
                self._df_sr,
                settings.denoise_model_dir,
            )
        except ImportError as err:
            logger.warning(
                "event=denoise_import_error error={} — run: uv sync --group deepfilter",
                err,
            )
        except Exception as err:
            logger.warning("event=denoise_load_error error={}", err)

    def enhance(self, pcm_bytes: bytes, in_sr: int = 16_000) -> bytes:
        """Denoise raw PCM-16 audio and return cleaned PCM-16 bytes.

        Args:
            pcm_bytes: Raw 16-bit little-endian PCM at ``in_sr`` Hz.
            in_sr: Sample rate of ``pcm_bytes`` (default 16 000 Hz).

        Returns:
            Denoised PCM-16 bytes at the same ``in_sr``.  Returns
            ``pcm_bytes`` unchanged when the model is not loaded.
        """
        if not self._loaded or not pcm_bytes:
            return pcm_bytes

        import torch

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32_768.0

        # Resample to model SR (48 kHz) if needed.
        if in_sr != self._df_sr:
            from scipy.signal import resample_poly
            g = gcd(self._df_sr, in_sr)
            audio = resample_poly(audio, self._df_sr // g, in_sr // g).astype(np.float32)

        # DeepFilterNet expects (channels, samples).
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        enhanced_tensor = self._enhance_fn(self._model, self._df_state, audio_tensor)
        enhanced = enhanced_tensor.squeeze(0).numpy()

        # Resample back to the original SR.
        if in_sr != self._df_sr:
            from scipy.signal import resample_poly
            g = gcd(self._df_sr, in_sr)
            enhanced = resample_poly(enhanced, in_sr // g, self._df_sr // g).astype(np.float32)

        enhanced = np.clip(enhanced, -1.0, 1.0)
        return (enhanced * 32_768.0).astype(np.int16).tobytes()


@lru_cache(maxsize=1)
def get_denoise_service() -> DenoiseService:
    """Return the global DenoiseService singleton."""
    return DenoiseService()
