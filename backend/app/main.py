import asyncio
import base64
from pathlib import Path
from time import perf_counter
from uuid import uuid4

# aioice deliberately excludes 127.0.0.1 from host candidates (RFC purist).
# That breaks browser↔server ICE on the same machine: the browser sends
# connectivity checks to 127.0.0.1 but the server has no matching candidate.
# Patch the function so loopback is included for localhost development.
import aioice.ice as _aioice_ice
_orig_get_host_addresses = _aioice_ice.get_host_addresses
def _get_host_addresses_with_loopback(use_ipv4: bool, use_ipv6: bool) -> list[str]:
    addresses = _orig_get_host_addresses(use_ipv4, use_ipv6)
    if use_ipv4 and "127.0.0.1" not in addresses:
        addresses.insert(0, "127.0.0.1")
    return addresses
_aioice_ice.get_host_addresses = _get_host_addresses_with_loopback

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from app.models import HealthResponse, TranscriptionResponse
from app.services.denoise import get_denoise_service
from app.services.llm import warmup_llamacpp
from app.services.stt import get_stt_service
from app.services.tts import get_available_voices, get_tts_service
from app.services.vad import get_vad_service
from app.webrtc.router import router as webrtc_router
from config.logging import setup_logging
from config.settings import get_settings

setup_logging()
settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(webrtc_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _warmup_models() -> None:
    """Pre-load STT, VAD, and TTS models at startup to avoid cold-start latency.

    Runs each model's load/inference path once so the first real request is
    served from a warm model. Failures are logged as warnings and do not block
    startup.
    """
    loop = asyncio.get_event_loop()

    stt_t0 = perf_counter()
    try:
        await loop.run_in_executor(None, get_stt_service()._load_model)
        logger.info("event=stt_warmup_done ms={}", round((perf_counter() - stt_t0) * 1000))
    except Exception as err:
        logger.warning("event=stt_warmup_failed error={}", err)

    if settings.stream_vad_enabled:
        vad_t0 = perf_counter()
        try:
            await loop.run_in_executor(None, get_vad_service()._load_model)
            logger.info("event=vad_warmup_done ms={}", round((perf_counter() - vad_t0) * 1000))
        except Exception as err:
            logger.warning("event=vad_warmup_failed error={}", err)

    tts_t0 = perf_counter()
    try:
        await get_tts_service().synthesize("Hello.")
        logger.info("event=tts_warmup_done ms={}", round((perf_counter() - tts_t0) * 1000))
    except Exception as err:
        logger.warning("event=tts_warmup_failed error={}", err)

    if settings.llm_provider == "llama-cpp":
        llm_t0 = perf_counter()
        try:
            await warmup_llamacpp(settings)
            logger.info("event=llamacpp_warmup_done ms={}", round((perf_counter() - llm_t0) * 1000))
        except Exception as err:
            logger.warning("event=llamacpp_warmup_failed error={}", err)

    if settings.denoise_enabled:
        denoise_t0 = perf_counter()
        try:
            await loop.run_in_executor(None, get_denoise_service)
            logger.info("event=denoise_warmup_done ms={}", round((perf_counter() - denoise_t0) * 1000))
        except Exception as err:
            logger.warning("event=denoise_warmup_failed error={}", err)


def _loop_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # aioice schedules STUN retry timers via loop.call_later(). When the UDP
    # transport is torn down first (sock=None, loop=None), the timer fires into
    # a dead socket. aioice 0.10.x does not cancel these handles in
    # connection_lost(), so the crash is unavoidable at the library level.
    # The session is already closed when this fires — suppress the noise.
    if "Transaction.__retry" in context.get("message", ""):
        return
    loop.default_exception_handler(context)


@app.on_event("startup")
async def startup_event() -> None:
    """FastAPI startup hook: create temp directory and kick off model warmup."""
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    asyncio.get_event_loop().set_exception_handler(_loop_exception_handler)
    logger.info(
        "event=startup app_name={} host={} port={} temp_dir={} cors_origins={}",
        settings.app_name,
        settings.app_host,
        settings.app_port,
        settings.temp_dir,
        settings.cors_origins,
    )
    asyncio.create_task(_warmup_models())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a simple liveness probe response.

    Returns:
        HealthResponse with status ``"ok"``.
    """
    return HealthResponse()


@app.get("/tts/voices")
async def list_tts_voices() -> dict:
    """Return available TTS voice IDs and the configured default for the Kokoro backend."""
    return {"voices": get_available_voices(), "default_voice": settings.tts_kokoro_voice}


class _TTSPreviewRequest(BaseModel):
    text: str = "Hello, this is a voice preview."
    voice: str | None = None
    speed: float | None = None


@app.post("/tts/preview")
async def preview_tts(body: _TTSPreviewRequest) -> Response:
    """Synthesise a short text sample and return the WAV audio."""
    voice = body.voice or settings.tts_kokoro_voice
    speed = body.speed if body.speed is not None else settings.tts_kokoro_speed
    wav_bytes, _ = await get_tts_service().synthesize(body.text, voice=voice, speed=speed)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/tts/welcome")
async def get_welcome_audio() -> dict:
    """Return the welcome message text and pre-synthesised WAV as base64.

    The frontend fetches this at page load so the audio is ready to play
    the instant the user clicks the orb — no per-click synthesis latency.
    """
    welcome = settings.welcome_message
    if not welcome:
        return {"text": "", "audio": None, "sample_rate": 24000}
    wav_bytes, sr = await get_tts_service().synthesize(
        welcome, voice=settings.tts_kokoro_voice, speed=settings.tts_kokoro_speed
    )
    return {"text": welcome, "audio": base64.b64encode(wav_bytes).decode(), "sample_rate": sr}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscriptionResponse:
    """Transcribe an uploaded audio file and return the text with timing metadata.

    Accepts any audio format supported by faster-whisper (webm, wav, mp4, etc.).
    The file is written to a temporary path, transcribed, then deleted.

    Args:
        audio: Uploaded audio file from a multipart/form-data request.

    Returns:
        TranscriptionResponse with ``text``, ``timings_ms``, and ``debug`` fields.

    Raises:
        HTTPException: 400 if the uploaded file is empty.
    """
    request_id = uuid4().hex[:8]
    started_at = perf_counter()
    logger.info(
        "request_id={} event=request_received filename={} content_type={}",
        request_id,
        audio.filename,
        audio.content_type,
    )

    filename = audio.filename or "recording.webm"
    suffix = Path(filename).suffix or ".webm"
    temp_path = settings.temp_dir / f"{request_id}{suffix}"

    try:
        read_started_at = perf_counter()
        content = await audio.read()
        request_read_ms = round((perf_counter() - read_started_at) * 1000, 2)

        if not content:
            raise HTTPException(status_code=400, detail="Audio file is empty.")

        write_started_at = perf_counter()
        temp_path.write_bytes(content)
        file_write_ms = round((perf_counter() - write_started_at) * 1000, 2)

        service = get_stt_service()
        result = service.transcribe(
            file_path=temp_path,
            request_id=request_id,
            filename=filename,
            audio_bytes=len(content),
        )
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        result.timings_ms.request_read_ms = request_read_ms
        result.timings_ms.file_write_ms = file_write_ms
        result.timings_ms.total_ms = total_ms

        logger.info(
            "request_id={} event=request_finished total_ms={} request_read_ms={} file_write_ms={} transcribe_ms={}",
            request_id,
            total_ms,
            request_read_ms,
            file_write_ms,
            result.timings_ms.transcribe_ms,
        )
        return TranscriptionResponse(
            text=result.text,
            timings_ms=result.timings_ms,
            debug=result.debug,
        )
    finally:
        temp_path.unlink(missing_ok=True)

