from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from faster_whisper import WhisperModel
from loguru import logger

from app.models import DebugInfo, LatencyMetrics
from config.settings import Settings, get_settings


@dataclass
class ServiceResult:
    text: str
    timings_ms: LatencyMetrics
    debug: DebugInfo


class SpeechToTextService:
    def __init__(
        self,
        settings: Settings,
        *,
        model_id: str | None = None,          # model name or flat-file path
        model_size_label: str | None = None,  # for logging/debug only
        beam_size: int | None = None,
        download_root: Path | None = None,    # HF cache root; None = default HF cache
    ) -> None:
        self.settings = settings
        self._model_id = model_id if model_id is not None else str(settings.stt_model_dir)
        self._model_size_label = model_size_label or settings.stt_model_size
        self._beam_size = beam_size if beam_size is not None else settings.stt_beam_size
        self._download_root = download_root
        self._model: WhisperModel | None = None

    def _load_model(self) -> tuple[WhisperModel, float]:
        if self._model is not None:
            return self._model, 0.0

        logger.info(
            "event=model_load_started model_id={} device={} compute_type={}",
            self._model_id,
            self.settings.stt_device,
            self.settings.stt_compute_type,
        )
        started_at = perf_counter()
        extra: dict = {}
        if self._download_root is not None:
            extra["download_root"] = str(self._download_root)
        self._model = WhisperModel(
            self._model_id,
            device=self.settings.stt_device,
            compute_type=self.settings.stt_compute_type,
            **extra,
        )
        model_load_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info("event=model_load_finished model_load_ms={}", model_load_ms)
        return self._model, model_load_ms

    def transcribe(self, *, file_path: Path, request_id: str, filename: str, audio_bytes: int) -> ServiceResult:
        model, model_load_ms = self._load_model()

        logger.info("request_id={} event=transcribe_started path={}", request_id, file_path)
        started_at = perf_counter()
        segments, info = model.transcribe(
            str(file_path),
            beam_size=self._beam_size,
            language=self.settings.stt_language or None,
            vad_filter=self.settings.stt_vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_speech_threshold=0.6,
            log_prob_threshold=-0.7,
            condition_on_previous_text=False,
        )
        segment_list = list(segments)
        transcribe_ms = round((perf_counter() - started_at) * 1000, 2)
        text = " ".join(segment.text.strip() for segment in segment_list if segment.text.strip()).strip()

        logger.info(
            "request_id={} event=transcribe_finished language={} segments={} text_length={} transcribe_ms={}",
            request_id,
            info.language,
            len(segment_list),
            len(text),
            transcribe_ms,
        )

        return ServiceResult(
            text=text,
            timings_ms=LatencyMetrics(
                model_load_ms=model_load_ms,
                transcribe_ms=transcribe_ms,
            ),
            debug=DebugInfo(
                request_id=request_id,
                filename=filename,
                audio_bytes=audio_bytes,
                detected_language=info.language,
                segments=len(segment_list),
                model_size=self._model_size_label,
                device=self.settings.stt_device,
                compute_type=self.settings.stt_compute_type,
            ),
        )


@lru_cache(maxsize=1)
def get_stt_service() -> SpeechToTextService:
    return SpeechToTextService(get_settings())


@lru_cache(maxsize=1)
def get_meeting_stt_service() -> SpeechToTextService:
    """STT service using a larger Whisper model for meeting transcription.

    Always uses the model name (e.g. "large-v3-turbo") so faster-whisper
    resolves the HuggingFace cache structure inside meeting_stt_model_dir.
    If the directory doesn't exist, faster-whisper downloads to its default
    HF cache (~/.cache/huggingface/hub/).
    """
    settings = get_settings()
    model_dir = settings.meeting_stt_model_dir
    return SpeechToTextService(
        settings,
        model_id=settings.meeting_stt_model_size,   # name, not path — HF resolves cache
        model_size_label=settings.meeting_stt_model_size,
        beam_size=settings.meeting_stt_beam_size,
        download_root=model_dir if model_dir.exists() else None,
    )
