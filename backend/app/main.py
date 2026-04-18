import json
import wave
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.models import HealthResponse, TranscriptionResponse
from app.services.stt import get_stt_service
from config.logging import setup_logging
from config.settings import get_settings

setup_logging()
settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "event=startup app_name={} host={} port={} temp_dir={} cors_origins={}",
        settings.app_name,
        settings.app_host,
        settings.app_port,
        settings.temp_dir,
        settings.cors_origins,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def write_pcm16_wav(*, pcm_bytes: bytes, sample_rate: int, file_path: Path) -> float:
    started_at = perf_counter()
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return round((perf_counter() - started_at) * 1000, 2)


def transcribe_stream_buffer(
    *,
    request_id: str,
    sample_rate: int,
    pcm_buffer: bytearray,
    chunk_count: int,
) -> dict[str, object]:
    started_at = perf_counter()
    temp_path = settings.temp_dir / f"{request_id}_stream.wav"
    try:
        file_write_ms = write_pcm16_wav(pcm_bytes=bytes(pcm_buffer), sample_rate=sample_rate, file_path=temp_path)
        service = get_stt_service()
        result = service.transcribe(
            file_path=temp_path,
            request_id=request_id,
            filename=f"stream_{request_id}.wav",
            audio_bytes=len(pcm_buffer),
        )
        buffered_audio_ms = round(len(pcm_buffer) / 2 / sample_rate * 1000, 2)
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        result.timings_ms.file_write_ms = file_write_ms
        result.timings_ms.total_ms = total_ms
        result.timings_ms.buffered_audio_ms = buffered_audio_ms
        result.debug.sample_rate = sample_rate
        result.debug.chunks_received = chunk_count

        return {
            "text": result.text,
            "timings_ms": result.timings_ms.model_dump(),
            "debug": result.debug.model_dump(),
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscriptionResponse:
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


@app.websocket("/ws/transcribe")
async def transcribe_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    request_id = uuid4().hex[:8]
    logger.info("request_id={} event=ws_connected client={}", request_id, websocket.client)

    sample_rate: int | None = None
    pcm_buffer = bytearray()
    chunk_count = 0
    last_emit_at = 0.0
    last_text_sent = ""

    await websocket.send_json({"type": "ready", "request_id": request_id})

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            if message.get("text") is not None:
                payload = json.loads(message["text"])
                event_type = payload.get("type")

                if event_type == "start":
                    sample_rate = int(payload.get("sample_rate", 16000))
                    logger.info("request_id={} event=stream_started sample_rate={}", request_id, sample_rate)
                    continue

                if event_type == "stop":
                    if sample_rate and pcm_buffer:
                        result_payload = transcribe_stream_buffer(
                            request_id=request_id,
                            sample_rate=sample_rate,
                            pcm_buffer=pcm_buffer,
                            chunk_count=chunk_count,
                        )
                        await websocket.send_json({"type": "final", **result_payload})
                    else:
                        await websocket.send_json(
                            {
                                "type": "final",
                                "text": "",
                                "timings_ms": {
                                    "request_read_ms": 0,
                                    "file_write_ms": 0,
                                    "model_load_ms": 0,
                                    "transcribe_ms": 0,
                                    "total_ms": 0,
                                    "buffered_audio_ms": 0,
                                    "client_roundtrip_ms": None,
                                },
                                "debug": {
                                    "request_id": request_id,
                                    "filename": "stream.wav",
                                    "audio_bytes": 0,
                                    "detected_language": None,
                                    "segments": 0,
                                    "model_size": settings.stt_model_size,
                                    "device": settings.stt_device,
                                    "compute_type": settings.stt_compute_type,
                                    "sample_rate": sample_rate,
                                    "chunks_received": chunk_count,
                                },
                            }
                        )
                    break

            if message.get("bytes") is None or sample_rate is None:
                continue

            pcm_buffer.extend(message["bytes"])
            chunk_count += 1
            buffered_audio_ms = len(pcm_buffer) / 2 / sample_rate * 1000
            now = perf_counter()
            should_emit = (
                buffered_audio_ms >= settings.stream_min_audio_ms
                and ((now - last_emit_at) * 1000) >= settings.stream_emit_interval_ms
            )

            if not should_emit:
                continue

            result_payload = transcribe_stream_buffer(
                request_id=request_id,
                sample_rate=sample_rate,
                pcm_buffer=pcm_buffer,
                chunk_count=chunk_count,
            )
            current_text = str(result_payload["text"])
            if current_text != last_text_sent:
                await websocket.send_json({"type": "partial", **result_payload})
                last_text_sent = current_text
            last_emit_at = perf_counter()
    except WebSocketDisconnect:
        logger.info("request_id={} event=ws_disconnected chunks_received={}", request_id, chunk_count)
    except Exception as error:
        logger.exception("request_id={} event=ws_failed error={}", request_id, error)
        await websocket.send_json({"type": "error", "message": "Streaming transcription failed."})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            logger.debug("request_id={} event=ws_close_skipped", request_id)
