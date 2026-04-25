"""Smart Turn v3.2: semantic utterance-completion detector.

Uses a Whisper-based ONNX model to predict whether the user has finished
speaking.  Acts as a second gate alongside the silence debounce — when
enabled, the LLM is only triggered once the model is confident the
utterance is complete, preventing premature responses on mid-sentence pauses.

Setup
-----
1. Install dependencies::

       uv sync --group smart_turn

2. Download the model file::

       python scripts/download_smart_turn_model.py

3. Enable in .env or settings::

       STREAM_SMART_TURN_ENABLED=true

When the model file is absent or dependencies are not installed the service
falls back transparently — every ``predict()`` call returns ``(True, 1.0)``
so the pipeline behaves exactly as before smart turn was added.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import get_settings


class SmartTurnService:
    """Predicts utterance-completion probability from raw PCM-16 audio.

    Attributes:
        is_loaded: ``True`` when the ONNX session and feature extractor are ready.
    """

    def __init__(self) -> None:
        self._session = None
        self._extractor = None
        self._loaded = False
        self._load()

    @property
    def is_loaded(self) -> bool:
        """Whether the model is ready for inference."""
        return self._loaded

    def _load(self) -> None:
        """Load the ONNX session and Whisper feature extractor.

        Sets ``_loaded = True`` on success.  Any missing file or import is
        logged as a warning and the service remains in pass-through mode.
        """
        settings = get_settings()
        model_path = Path(settings.stream_smart_turn_model_path)
        if not model_path.exists():
            logger.warning(
                "event=smart_turn_model_missing path={} — falling back to silence debounce",
                model_path,
            )
            return
        try:
            import onnxruntime as rt
            from transformers import WhisperFeatureExtractor

            opts = rt.SessionOptions()
            opts.log_severity_level = 3  # suppress ONNX Runtime info logs
            self._session = rt.InferenceSession(str(model_path), sess_options=opts)
            self._extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
            self._loaded = True
            logger.info("event=smart_turn_loaded path={}", model_path)
        except ImportError as err:
            logger.warning(
                "event=smart_turn_import_error error={} — run: uv sync --group smart_turn",
                err,
            )

    def predict(self, pcm_bytes: bytes) -> tuple[bool, float]:
        """Predict whether the user has finished their utterance.

        Args:
            pcm_bytes: Raw 16-bit little-endian PCM audio sampled at 16 kHz.
                       Uses the last 8 seconds (128 000 samples) at most.

        Returns:
            Tuple of ``(is_complete, probability)`` where ``is_complete`` is
            ``True`` when ``probability >= settings.stream_smart_turn_threshold``.
            Returns ``(True, 1.0)`` in pass-through mode (model not loaded).
        """
        if not self._loaded:
            return True, 1.0

        settings = get_settings()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32_768.0
        if len(audio) > 128_000:
            audio = audio[-128_000:]

        features = self._extractor(
            audio, sampling_rate=16_000, return_tensors="np"
        )
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        prob = float(
            self._session.run(
                [output_name], {input_name: features["input_features"]}
            )[0][0][0]
        )
        return prob >= settings.stream_smart_turn_threshold, prob


@lru_cache(maxsize=1)
def get_smart_turn_service() -> SmartTurnService:
    """Return the global SmartTurnService singleton.

    Returns:
        Cached ``SmartTurnService`` instance (loaded once on first call).
    """
    return SmartTurnService()
