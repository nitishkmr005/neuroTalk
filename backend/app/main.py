import asyncio
import base64
from contextlib import suppress
import json
import re
import wave
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.models import HealthResponse, TranscriptionResponse
from app.services.llm import stream_llm_response
from app.services.stt import get_stt_service
from app.services.tts import get_tts_service
from app.utils.emotion import strip_emotion_tags
from app.utils.session_logger import LLMCallLog, STTRunLog, SessionLog, _iso, write_session_log
from config.logging import setup_logging
from config.settings import get_settings

setup_logging()
settings = get_settings()

_PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|just a (?:moment|second|sec)|"
    r"give me a (?:second|moment|sec)|hang on|please wait|just wait|ok wait)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _warmup_models() -> None:
    loop = asyncio.get_event_loop()

    stt_t0 = perf_counter()
    try:
        await loop.run_in_executor(None, get_stt_service()._load_model)
        logger.info("event=stt_warmup_done ms={}", round((perf_counter() - stt_t0) * 1000))
    except Exception as err:
        logger.warning("event=stt_warmup_failed error={}", err)

    tts_t0 = perf_counter()
    try:
        await get_tts_service().synthesize("Hello.")
        logger.info("event=tts_warmup_done ms={}", round((perf_counter() - tts_t0) * 1000))
    except Exception as err:
        logger.warning("event=tts_warmup_failed error={}", err)


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
    asyncio.create_task(_warmup_models())


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
    send_lock = asyncio.Lock()
    llm_task: asyncio.Task[None] | None = None
    pending_llm_call: tuple[str, str] | None = None
    latest_llm_input = ""
    interrupt_event = asyncio.Event()
    silence_debounce_task: asyncio.Task[None] | None = None
    conversation_history: list[dict[str, str]] = []  # grows each completed turn

    # ── Session log scaffolding ───────────────────────────────────────────────
    session_log = SessionLog(
        session_id=request_id,
        stt_model=settings.stt_model_size,
        stt_device=settings.stt_device,
        stt_compute_type=settings.stt_compute_type,
        stt_vad_filter=settings.stt_vad_filter,
        stt_beam_size=settings.stt_beam_size,
        llm_model=settings.llm_model,
        llm_host=settings.ollama_host,
        llm_system_prompt_preview=settings.llm_system_prompt[:100],
    )

    async def send_json(payload: dict[str, object]) -> None:
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

    _SENT_BOUNDARY = re.compile(r"[.!?](?:\s|$)")

    async def synthesize_and_send(full_text: str, first_sent_task: asyncio.Future | None, first_sent_end: int) -> None:
        if interrupt_event.is_set():
            return
        tts_t0 = perf_counter()
        await send_json({"type": "tts_start"})
        try:
            tts_service = get_tts_service()

            if first_sent_task is not None:
                # Sentence 1 synthesis was started during LLM streaming — await it now
                wav1, sr1 = await first_sent_task
                remainder = full_text[first_sent_end:].strip()
                if remainder and not interrupt_event.is_set():
                    wav2, sr2 = await tts_service.synthesize(remainder)
                    # Concatenate: strip 44-byte WAV header from part 2, rewrite combined
                    combined_pcm = wav1[44:] + wav2[44:]
                    import io as _io
                    buf = _io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr1)
                        wf.writeframes(combined_pcm)
                    wav_bytes, sr = buf.getvalue(), sr1
                else:
                    wav_bytes, sr = wav1, sr1
            else:
                wav_bytes, sr = await tts_service.synthesize(full_text)

        except Exception as tts_err:
            logger.warning("request_id={} event=tts_error error={}", request_id, tts_err)
            return
        if interrupt_event.is_set():
            return
        tts_ms = round((perf_counter() - tts_t0) * 1000, 2)
        logger.info("request_id={} event=tts_done tts_ms={}", request_id, tts_ms)
        wav_b64 = base64.b64encode(wav_bytes).decode()
        await send_json({"type": "tts_audio", "data": wav_b64, "sample_rate": sr, "tts_ms": tts_ms})
        await send_json({"type": "tts_done"})

    async def run_llm_stream(text: str, trigger: str) -> None:
        nonlocal llm_task, pending_llm_call, latest_llm_input, last_text_sent, chunk_count, last_emit_at
        # Clear audio buffer — this utterance is captured; next audio is a new turn
        pcm_buffer.clear()
        last_text_sent = ""
        chunk_count = 0
        last_emit_at = 0.0
        interrupt_event.clear()
        llm_t0 = perf_counter()
        call_ts = _iso()
        full_response = ""
        llm_ms = 0.0
        call_error: str | None = None
        latest_llm_input = text

        first_sent_task: asyncio.Future | None = None
        first_sent_end = 0

        try:
            await send_json({"type": "llm_start", "user_text": text})
            async for token in stream_llm_response(text, conversation_history=list(conversation_history)):
                full_response += token
                await send_json({"type": "llm_partial", "text": strip_emotion_tags(full_response)})
                # Fire sentence-1 TTS as soon as first sentence boundary is detected
                if first_sent_task is None and not interrupt_event.is_set():
                    m = _SENT_BOUNDARY.search(full_response)
                    if m and m.end() >= 20:
                        first_sent_end = m.end()
                        snippet = full_response[:first_sent_end].strip()
                        tts_svc = get_tts_service()
                        first_sent_task = asyncio.ensure_future(tts_svc.synthesize(snippet))
                        logger.info("request_id={} event=tts_early_start snippet_len={}", request_id, len(snippet))
            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            logger.info("request_id={} event=llm_done llm_ms={}", request_id, llm_ms)
            display_text = strip_emotion_tags(full_response)
            await send_json({"type": "llm_final", "text": display_text, "llm_ms": llm_ms})
        except Exception as llm_err:
            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            call_error = str(llm_err)
            logger.warning("request_id={} event=llm_error error={}", request_id, llm_err)
            try:
                await send_json({"type": "llm_error", "message": "LLM unavailable — is Ollama running?"})
            except Exception:
                pass

        approx_tokens = round(len(full_response.split()) * 1.3) if full_response else 0
        session_log.llm_calls.append(
            LLMCallLog(
                timestamp=call_ts,
                trigger=trigger,
                latency_ms=llm_ms,
                model=settings.llm_model,
                host=settings.ollama_host,
                system_prompt_preview=settings.llm_system_prompt[:100],
                input_transcript=text,
                input_length_chars=len(text),
                output_response=full_response,
                output_preview=full_response[:200],
                output_length_chars=len(full_response),
                approx_tokens_out=approx_tokens,
                cancelled=False,
                error=call_error,
            )
        )

        if first_sent_task is not None and (interrupt_event.is_set() or call_error is not None):
            first_sent_task.cancel()
            first_sent_task = None

        if full_response and call_error is None:
            conversation_history.append({"role": "user", "content": text})
            conversation_history.append({"role": "assistant", "content": full_response})
            max_msgs = settings.llm_max_history_turns * 2
            if len(conversation_history) > max_msgs:
                conversation_history[:] = conversation_history[-max_msgs:]
            await synthesize_and_send(full_response, first_sent_task, first_sent_end)

        if pending_llm_call is None:
            llm_task = None
            return

        next_text, next_trigger = pending_llm_call
        pending_llm_call = None
        if next_text == latest_llm_input:
            llm_task = None
            return

        llm_task = asyncio.create_task(run_llm_stream(next_text, next_trigger))

    def schedule_llm_stream(text: str, trigger: str) -> None:
        nonlocal llm_task, pending_llm_call
        normalized_text = text.strip()
        if len(normalized_text) < settings.stream_llm_min_chars:
            return
        if normalized_text == latest_llm_input:
            return
        if _PAUSE_PATTERN.match(normalized_text):
            logger.info("request_id={} event=pause_command_detected text={}", request_id, normalized_text)
            return

        # If an LLM is already running for older text, cancel it — the user kept talking
        # and we have a more complete utterance now. Avoids "two replies for one query".
        if llm_task is not None and not llm_task.done():
            logger.info("request_id={} event=llm_cancel_for_newer_text", request_id)
            interrupt_event.set()
            pending_llm_call = None
            llm_task.cancel()
            llm_task = None

        pending_llm_call = (normalized_text, trigger)
        if llm_task is None or llm_task.done():
            next_text, next_trigger = pending_llm_call
            pending_llm_call = None
            llm_task = asyncio.create_task(run_llm_stream(next_text, next_trigger))

    async def _silence_debounce_then_fire(text: str, trigger: str) -> None:
        try:
            await asyncio.sleep(settings.stream_llm_silence_ms / 1000)
        except asyncio.CancelledError:
            return
        schedule_llm_stream(text, trigger)

    async def run_welcome() -> None:
        welcome = settings.welcome_message
        if not welcome:
            return
        await send_json({"type": "tts_start"})
        try:
            wav_bytes, sr = await get_tts_service().synthesize(welcome)
        except Exception as err:
            logger.warning("request_id={} event=welcome_tts_error error={}", request_id, err)
            return
        wav_b64 = base64.b64encode(wav_bytes).decode()
        await send_json({"type": "tts_audio", "data": wav_b64, "sample_rate": sr, "tts_ms": 0})
        await send_json({"type": "tts_done"})

    await send_json({"type": "ready", "request_id": request_id})
    asyncio.create_task(run_welcome())

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

                if event_type == "interrupt":
                    logger.info("request_id={} event=interrupt_received", request_id)
                    interrupt_event.set()
                    pending_llm_call = None
                    if silence_debounce_task is not None and not silence_debounce_task.done():
                        silence_debounce_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await silence_debounce_task
                        silence_debounce_task = None
                    if llm_task is not None and not llm_task.done():
                        llm_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await llm_task
                        llm_task = None
                    continue

                if event_type == "stop":
                    # Cancel pending silence-debounce so it doesn't double-fire after the
                    # final transcript triggers its own LLM call.
                    if silence_debounce_task is not None and not silence_debounce_task.done():
                        silence_debounce_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await silence_debounce_task
                        silence_debounce_task = None
                    if sample_rate and pcm_buffer:
                        audio_duration_ms = round(len(pcm_buffer) / 2 / sample_rate * 1000, 2)
                        audio_file_path = str(settings.temp_dir / f"{request_id}_stream.wav")

                        result_payload = transcribe_stream_buffer(
                            request_id=request_id,
                            sample_rate=sample_rate,
                            pcm_buffer=pcm_buffer,
                            chunk_count=chunk_count,
                        )
                        await send_json({"type": "final", **result_payload})

                        timings = result_payload.get("timings_ms", {})
                        debug = result_payload.get("debug", {})
                        final_text = str(result_payload.get("text", "")).strip()

                        session_log.stt_runs.append(
                            STTRunLog(
                                timestamp=_iso(),
                                trigger="final",
                                latency_ms=timings.get("transcribe_ms", 0),
                                audio_file_path=audio_file_path,
                                audio_bytes=len(pcm_buffer),
                                audio_duration_ms=audio_duration_ms,
                                sample_rate=sample_rate,
                                transcript=final_text,
                                transcript_length_chars=len(final_text),
                                language_detected=debug.get("detected_language"),
                                segments=debug.get("segments", 0),
                            )
                        )

                        if final_text and session_log.stt_partial_run_count > 0 and final_text.strip() != latest_llm_input:
                            schedule_llm_stream(final_text, "final")
                    else:
                        await send_json(
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

                    while llm_task is not None:
                        current_task = llm_task
                        await current_task
                        if llm_task is current_task:
                            llm_task = None
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
                await send_json({"type": "partial", **result_payload})
                last_text_sent = current_text
                session_log.stt_partial_run_count += 1

                timings = result_payload.get("timings_ms", {})
                debug = result_payload.get("debug", {})
                session_log.stt_runs.append(
                    STTRunLog(
                        timestamp=_iso(),
                        trigger="partial",
                        latency_ms=timings.get("transcribe_ms", 0),
                        audio_file_path=str(settings.temp_dir / f"{request_id}_stream.wav"),
                        audio_bytes=len(pcm_buffer),
                        audio_duration_ms=round(len(pcm_buffer) / 2 / sample_rate * 1000, 2),
                        sample_rate=sample_rate,
                        transcript=current_text,
                        transcript_length_chars=len(current_text),
                        language_detected=debug.get("detected_language"),
                        segments=debug.get("segments", 0),
                    )
                )
                # Debounce: only fire LLM once partial text has been stable for
                # `stream_llm_silence_ms` (i.e. user paused). Reset timer on every new partial.
                if silence_debounce_task is not None and not silence_debounce_task.done():
                    silence_debounce_task.cancel()
                silence_debounce_task = asyncio.create_task(
                    _silence_debounce_then_fire(current_text, "debounced_partial")
                )

            last_emit_at = perf_counter()

    except WebSocketDisconnect:
        logger.info("request_id={} event=ws_disconnected chunks_received={}", request_id, chunk_count)
    except Exception as error:
        session_log.error = str(error)
        logger.exception("request_id={} event=ws_failed error={}", request_id, error)
        await send_json({"type": "error", "message": "Streaming transcription failed."})
    finally:
        if silence_debounce_task is not None and not silence_debounce_task.done():
            silence_debounce_task.cancel()
            with suppress(asyncio.CancelledError):
                await silence_debounce_task
        if llm_task is not None and not llm_task.done():
            llm_task.cancel()
            with suppress(asyncio.CancelledError):
                await llm_task
        write_session_log(session_log)
        try:
            await websocket.close()
        except RuntimeError:
            logger.debug("request_id={} event=ws_close_skipped", request_id)
