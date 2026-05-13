"""Meeting recorder feature: transcription, summarization, and file saving.

This module is self-contained and can be removed without touching other code:
1. Delete  backend/app/meeting/
2. Remove  ``app.include_router(meeting_router)``  from main.py
3. Delete  frontend/components/meeting-recorder.tsx
4. Remove  MeetingRecorder import + usage from voice-agent-console.tsx
5. Remove  Meeting Recorder section from frontend/app/globals.css
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/meeting", tags=["meeting"])

_RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"

_SUMMARIZE_SYSTEM = (
    "You are a professional meeting summarizer. "
    "Given a meeting transcript, produce a structured summary with these sections:\n"
    "**Key Topics** — bullet-point the main subjects discussed.\n"
    "**Decisions Made** — list any conclusions or agreements reached.\n"
    "**Action Items** — list tasks with owners if mentioned.\n"
    "**Next Steps** — what happens after this meeting.\n\n"
    "Be concise. Skip any section that has nothing to report."
)

_MAX_TRANSCRIPT_CHARS = 24_000


# ── Models ────────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str


class SaveRequest(BaseModel):
    session: str | None = None   # timestamp folder name, e.g. "2024_01_15_14_30_00"
    transcript: str | None = None
    summary: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_dir(session: str | None) -> Path:
    """Return (and create) the recordings subfolder for a session."""
    name = session or datetime.now().strftime("%Y%m%d_%H%M%S")
    d = _RECORDINGS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _dispatch(messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
    """Forward messages to the configured LLM, using meeting-specific model if set."""
    from app.services.llm import (
        _stream_anthropic,
        _stream_gemini,
        _stream_llamacpp,
        _stream_ollama,
        _stream_openai,
    )
    from config.settings import get_settings

    settings = get_settings()

    # Use meeting-specific provider + model when configured
    overrides: dict = {}
    if settings.meeting_llm_provider:
        overrides["llm_provider"] = settings.meeting_llm_provider
    if settings.meeting_llm_model and settings.meeting_llm_provider != "llama-cpp":
        overrides["llm_model"] = settings.meeting_llm_model
    elif settings.meeting_llm_llamacpp_model_path and Path(settings.meeting_llm_llamacpp_model_path).exists():
        overrides["llm_llamacpp_model_path"] = settings.meeting_llm_llamacpp_model_path
    if overrides:
        settings = settings.model_copy(update=overrides)

    provider = settings.llm_provider
    model_label = (
        str(settings.llm_llamacpp_model_path) if provider == "llama-cpp" else settings.llm_model
    )
    logger.info("event=meeting_llm_dispatch provider={} model={}", provider, model_label)

    if provider == "ollama":
        async for t in _stream_ollama(messages, settings):
            yield t
    elif provider == "openai":
        async for t in _stream_openai(messages, settings):
            yield t
    elif provider == "anthropic":
        async for t in _stream_anthropic(messages, settings):
            yield t
    elif provider == "gemini":
        async for t in _stream_gemini(messages, settings):
            yield t
    elif provider == "llama-cpp":
        async for t in _stream_llamacpp(messages, settings):
            yield t
    else:
        raise ValueError(f"Unknown llm_provider {provider!r}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_meeting_segment(audio: UploadFile = File(...)) -> dict:
    """Transcribe a meeting audio segment using the high-quality meeting STT model.

    Uses ``large-v3-turbo`` (beam_size=5) instead of the real-time STT model
    so accuracy is prioritised over latency for batch meeting segments.

    Args:
        audio: Uploaded WebM/WAV audio segment.

    Returns:
        ``{ "text": "..." }``
    """
    from app.services.stt import get_meeting_stt_service
    from config.settings import get_settings

    settings = get_settings()
    logger.info(
        "event=meeting_stt_dispatch model={} beam_size={}",
        settings.meeting_stt_model_size,
        settings.meeting_stt_beam_size,
    )
    request_id = uuid4().hex[:8]
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio segment is empty.")

    filename = audio.filename or "segment.webm"
    suffix = Path(filename).suffix or ".webm"
    temp_path = settings.temp_dir / f"meeting_{request_id}{suffix}"
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        temp_path.write_bytes(content)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: get_meeting_stt_service().transcribe(
                file_path=temp_path,
                request_id=request_id,
                filename=filename,
                audio_bytes=len(content),
            ),
        )
        logger.info(
            "event=meeting_transcribe_done request_id={} ms={} text_len={}",
            request_id,
            result.timings_ms.transcribe_ms,
            len(result.text),
        )
        return {"text": result.text}
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/summarize")
async def summarize(body: SummarizeRequest) -> StreamingResponse:
    """Stream a meeting summary using the meeting-specific LLM model.

    Args:
        body: ``{ "text": "<full transcript>" }``

    Returns:
        StreamingResponse of plain-text tokens.
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Transcript text is empty.")

    truncated = len(text) > _MAX_TRANSCRIPT_CHARS
    if truncated:
        text = text[:_MAX_TRANSCRIPT_CHARS]
        logger.warning("event=summarize_truncated chars={}", len(body.text))

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Transcript:\n\n{text}"},
    ]
    logger.info("event=summarize_start chars={} truncated={}", len(text), truncated)

    async def token_stream() -> AsyncGenerator[bytes, None]:
        try:
            async for token in _dispatch(messages):
                yield token.encode()
        except Exception as err:
            logger.error("event=summarize_error error={}", err)
            yield b"\n\n[Summarization failed]"

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")


@router.post("/save")
async def save_meeting(body: SaveRequest) -> dict:
    """Save transcript and/or summary into a session-timestamped folder.

    Files are written to ``backend/recordings/<session>/``.

    Args:
        body: ``{ "session": "20240115_143000", "transcript": "...", "summary": "..." }``

    Returns:
        ``{ "saved": [paths], "session": "<folder name>" }``
    """
    d = _session_dir(body.session)
    saved: list[str] = []

    if body.transcript:
        path = d / "transcript.txt"
        path.write_text(body.transcript, encoding="utf-8")
        saved.append(str(path))
        logger.info("event=meeting_saved type=transcript session={}", d.name)

    if body.summary:
        path = d / "summary.txt"
        path.write_text(body.summary, encoding="utf-8")
        saved.append(str(path))
        logger.info("event=meeting_saved type=summary session={}", d.name)

    return {"saved": saved, "session": d.name}


@router.post("/save-audio")
async def save_audio(
    audio: UploadFile = File(...),
    session: str = Form(default=""),
) -> dict:
    """Save recorded meeting audio into the session folder.

    Args:
        audio: WebM audio file.
        session: Session folder name (same value used in /save).

    Returns:
        ``{ "path": "<absolute path>", "session": "<folder name>" }``
    """
    d = _session_dir(session or None)
    path = d / "recording.webm"
    content = await audio.read()
    path.write_bytes(content)
    logger.info("event=meeting_audio_saved session={} size_bytes={}", d.name, len(content))
    return {"path": str(path), "session": d.name}
