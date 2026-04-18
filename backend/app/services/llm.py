from typing import AsyncGenerator

from loguru import logger
from ollama import AsyncClient

from config.settings import get_settings


async def stream_llm_response(transcript: str) -> AsyncGenerator[str, None]:
    settings = get_settings()
    client = AsyncClient(host=settings.ollama_host)

    logger.info("event=llm_start model={} host={} input_preview={!r}", settings.llm_model, settings.ollama_host, transcript[:60])

    stream = await client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": settings.llm_system_prompt},
            {"role": "user", "content": transcript},
        ],
        stream=True,
    )

    async for chunk in stream:
        token: str = chunk.message.content
        if token:
            yield token
