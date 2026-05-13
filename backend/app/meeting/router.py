"""Meeting recorder feature: /meeting/summarize streaming endpoint.

This module is self-contained and can be removed without touching other code:
1. Delete  backend/app/meeting/
2. Remove  ``app.include_router(meeting_router)``  from main.py
3. Delete  frontend/components/meeting-recorder.tsx
4. Remove  MeetingRecorder import + usage from voice-agent-console.tsx
5. Remove  Meeting Recorder section from frontend/app/globals.css
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

_RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"

router = APIRouter(prefix="/meeting", tags=["meeting"])

_SUMMARIZE_SYSTEM = (
    "You are a professional meeting summarizer. "
    "Given a meeting transcript, produce a structured summary with these sections:\n"
    "**Key Topics** — bullet-point the main subjects discussed.\n"
    "**Decisions Made** — list any conclusions or agreements reached.\n"
    "**Action Items** — list tasks with owners if mentioned.\n"
    "**Next Steps** — what happens after this meeting.\n\n"
    "Be concise. Skip any section that has nothing to report."
)

# Longest transcript we'll forward to the LLM. Keeps us within typical
# local-model context windows (~4 K–8 K tokens ≈ 16 K–32 K chars).
_MAX_TRANSCRIPT_CHARS = 24_000


class SummarizeRequest(BaseModel):
    text: str


class SaveRequest(BaseModel):
    transcript: str | None = None
    summary: str | None = None


async def _dispatch(messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
    """Forward a pre-built message list to the configured LLM provider."""
    from app.services.llm import (
        _stream_anthropic,
        _stream_gemini,
        _stream_llamacpp,
        _stream_ollama,
        _stream_openai,
    )
    from config.settings import get_settings

    settings = get_settings()
    provider = settings.llm_provider
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


@router.post("/summarize")
async def summarize(body: SummarizeRequest) -> StreamingResponse:
    """Stream a meeting summary for the provided transcript.

    Sends the transcript (truncated to ``_MAX_TRANSCRIPT_CHARS`` if needed)
    to the configured LLM with a summarisation system prompt. Tokens are
    streamed back as ``text/plain`` so the browser can display them
    incrementally via a ``ReadableStream`` fetch.

    Args:
        body: ``{ "text": "<full transcript>" }``

    Returns:
        StreamingResponse of plain-text tokens.

    Raises:
        HTTPException: 400 if ``text`` is empty.
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Transcript text is empty.")

    truncated = len(text) > _MAX_TRANSCRIPT_CHARS
    if truncated:
        text = text[:_MAX_TRANSCRIPT_CHARS]
        logger.warning(
            "event=summarize_truncated original_chars={} limit={}",
            len(body.text),
            _MAX_TRANSCRIPT_CHARS,
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Transcript:\n\n{text}"},
    ]

    logger.info(
        "event=summarize_start chars={} truncated={}",
        len(text),
        truncated,
    )

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
    """Save transcript and/or summary as text files in backend/recordings/.

    Args:
        body: ``{ "transcript": "...", "summary": "..." }`` — either field optional.

    Returns:
        ``{ "saved": [list of absolute file paths written] }``
    """
    _RECORDINGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved: list[str] = []

    if body.transcript:
        path = _RECORDINGS_DIR / f"transcript_{ts}.txt"
        path.write_text(body.transcript, encoding="utf-8")
        saved.append(str(path))
        logger.info("event=meeting_saved type=transcript path={}", path)

    if body.summary:
        path = _RECORDINGS_DIR / f"summary_{ts}.txt"
        path.write_text(body.summary, encoding="utf-8")
        saved.append(str(path))
        logger.info("event=meeting_saved type=summary path={}", path)

    return {"saved": saved}
