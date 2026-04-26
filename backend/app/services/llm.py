"""LLM streaming service with multi-provider support and optional web search."""

from __future__ import annotations

from typing import AsyncGenerator

from loguru import logger

from config.settings import get_settings


# ── Message helpers ───────────────────────────────────────────────────────────

def _build_messages(
    transcript: str,
    system_prompt: str,
    conversation_history: list[dict[str, str]],
    search_context: str,
) -> list[dict[str, str]]:
    """Build an OpenAI-compatible message list.

    Injects web-search results as a prefixed block in the user message when
    ``search_context`` is non-empty, keeping the system prompt unchanged.

    Args:
        transcript: Current user utterance.
        system_prompt: LLM system prompt string.
        conversation_history: Prior turns as ``[{"role": ..., "content": ...}]``.
        search_context: Pre-formatted search results string, or ``""`` if none.

    Returns:
        Message list ready for any OpenAI-compatible chat API.
    """
    user_content = transcript
    if search_context:
        user_content = (
            f"[Web Search Results]\n{search_context}\n\n"
            f"[User Query]\n{transcript}"
        )
    return [
        {"role": "system", "content": system_prompt},
        *(conversation_history or []),
        {"role": "user", "content": user_content},
    ]


async def _fetch_search_context(transcript: str) -> str:
    """Run a web search and return results as a formatted string.

    Args:
        transcript: User utterance used as the search query.

    Returns:
        Numbered list of ``title / snippet / url`` entries, or ``""`` on failure.
    """
    from app.services.search import web_search

    results = await web_search(transcript)
    if not results:
        return ""
    lines = [
        f"{i}. {r['title']}\n   {r['snippet']}\n   {r['url']}"
        for i, r in enumerate(results, 1)
    ]
    return "\n".join(lines)


# ── Provider-specific streamers ───────────────────────────────────────────────

async def _stream_ollama(
    messages: list[dict[str, str]], settings
) -> AsyncGenerator[str, None]:
    """Stream tokens from a local Ollama instance.

    Args:
        messages: OpenAI-compatible message list.
        settings: Application settings (uses ``ollama_host`` and ``llm_model``).

    Yields:
        Token strings as they arrive from the model.
    """
    from ollama import AsyncClient

    client = AsyncClient(host=settings.ollama_host)
    stream = await client.chat(model=settings.llm_model, messages=messages, stream=True)
    async for chunk in stream:
        token: str = chunk.message.content
        if token:
            yield token


async def _stream_openai(
    messages: list[dict[str, str]], settings
) -> AsyncGenerator[str, None]:
    """Stream tokens from the OpenAI Chat Completions API.

    Requires ``openai`` package (``uv sync --group openai_llm``) and a valid
    ``OPENAI_API_KEY`` in the environment or ``.env``.

    Args:
        messages: OpenAI-compatible message list.
        settings: Application settings (uses ``openai_api_key`` and ``llm_model``).

    Yields:
        Token strings as they arrive from the model.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _stream_anthropic(
    messages: list[dict[str, str]], settings
) -> AsyncGenerator[str, None]:
    """Stream tokens from the Anthropic Messages API.

    Requires ``anthropic`` package (``uv sync --group anthropic_llm``) and a
    valid ``ANTHROPIC_API_KEY``.  The system message is extracted and passed
    via the dedicated ``system`` parameter rather than in the messages list.

    Args:
        messages: OpenAI-compatible message list (system role handled separately).
        settings: Application settings (uses ``anthropic_api_key`` and ``llm_model``).

    Yields:
        Token strings as they arrive from the model.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    non_system = [m for m in messages if m["role"] != "system"]
    async with client.messages.stream(
        model=settings.llm_model,
        system=system,
        messages=non_system,
        max_tokens=1024,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _stream_gemini(
    messages: list[dict[str, str]], settings
) -> AsyncGenerator[str, None]:
    """Stream tokens from the Google Gemini API.

    Requires ``google-generativeai`` package (``uv sync --group gemini_llm``) and
    a valid ``GEMINI_API_KEY``.  Converts OpenAI-style messages to Gemini's
    ``user`` / ``model`` format and passes the system prompt via
    ``system_instruction``.

    Args:
        messages: OpenAI-compatible message list.
        settings: Application settings (uses ``gemini_api_key`` and ``llm_model``).

    Yields:
        Token strings as they arrive from the model.
    """
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    # Gemini uses "user" / "model" roles; drop system (passed via system_instruction)
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
        for m in messages
        if m["role"] != "system"
    ]
    model = genai.GenerativeModel(settings.llm_model, system_instruction=system)
    response = await model.generate_content_async(contents, stream=True)
    async for chunk in response:
        if chunk.text:
            yield chunk.text


# ── llama-cpp singleton ───────────────────────────────────────────────────────

_llama_instance = None


def _get_llama(settings):
    """Load the Llama model once and cache it for the lifetime of the process."""
    global _llama_instance
    if _llama_instance is None:
        from llama_cpp import Llama

        model_path = str(settings.llm_llamacpp_model_path)
        logger.info(
            "event=llamacpp_load model_path={} n_ctx={} n_gpu_layers={}",
            model_path,
            settings.llm_llamacpp_n_ctx,
            settings.llm_llamacpp_n_gpu_layers,
        )
        _llama_instance = Llama(
            model_path=model_path,
            n_ctx=settings.llm_llamacpp_n_ctx,
            n_gpu_layers=settings.llm_llamacpp_n_gpu_layers,
            verbose=False,
        )
        logger.info("event=llamacpp_ready")
    return _llama_instance


async def _stream_llamacpp(
    messages: list[dict[str, str]], settings
) -> AsyncGenerator[str, None]:
    """Stream tokens from a local GGUF model via llama-cpp-python.

    Inference is synchronous, so it runs in a thread-pool executor while tokens
    are forwarded to an asyncio queue so callers can ``async for`` over them.

    Args:
        messages: OpenAI-compatible message list.
        settings: Application settings (uses ``llm_llamacpp_*`` fields).

    Yields:
        Token strings as they are produced by the model.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    llm = _get_llama(settings)

    def _run() -> None:
        try:
            for chunk in llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                stream=True,
            ):
                delta: str = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    executor = ThreadPoolExecutor(max_workers=1)
    future = loop.run_in_executor(executor, _run)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token

    await future
    executor.shutdown(wait=False)


# ── Public interface ──────────────────────────────────────────────────────────

async def stream_llm_response(
    transcript: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream LLM response tokens for a user utterance.

    Orchestrates the full pipeline:
    1. Optionally runs a web search and injects results as context.
    2. Builds the message list (system + history + user).
    3. Dispatches to the configured provider and yields tokens.

    Supported providers (set ``LLM_PROVIDER`` in ``.env``):
    - ``ollama``     — local Ollama instance (default)
    - ``openai``     — OpenAI API (requires ``uv sync --group openai_llm``)
    - ``anthropic``  — Anthropic API (requires ``uv sync --group anthropic_llm``)
    - ``gemini``     — Google Gemini API (requires ``uv sync --group gemini_llm``)
    - ``llama-cpp``  — local GGUF model via llama-cpp-python (requires
                       ``uv sync --group llama_cpp_llm`` and a GGUF file at
                       ``llm_llamacpp_model_path``; run
                       ``python scripts/download_models.py --only-llm`` to fetch)

    Args:
        transcript: The user's transcribed speech to send as the current message.
        conversation_history: Prior ``{"role": ..., "content": ...}`` turn pairs.

    Yields:
        Token strings from the model as they stream.

    Raises:
        ValueError: If ``settings.llm_provider`` is not a recognised provider.
    """
    settings = get_settings()

    search_context = ""
    if settings.web_search_enabled:
        search_context = await _fetch_search_context(transcript)
        if search_context:
            logger.info(
                "event=web_search_injected chars={} query={!r}",
                len(search_context),
                transcript[:60],
            )

    messages = _build_messages(
        transcript,
        settings.llm_system_prompt,
        conversation_history or [],
        search_context,
    )

    logger.info(
        "event=llm_start provider={} model={} history_turns={} input_preview={!r}",
        settings.llm_provider,
        settings.llm_model,
        len(conversation_history or []) // 2,
        transcript[:60],
    )

    if settings.llm_provider == "ollama":
        async for token in _stream_ollama(messages, settings):
            yield token
    elif settings.llm_provider == "openai":
        async for token in _stream_openai(messages, settings):
            yield token
    elif settings.llm_provider == "anthropic":
        async for token in _stream_anthropic(messages, settings):
            yield token
    elif settings.llm_provider == "gemini":
        async for token in _stream_gemini(messages, settings):
            yield token
    elif settings.llm_provider == "llama-cpp":
        async for token in _stream_llamacpp(messages, settings):
            yield token
    else:
        raise ValueError(
            f"Unknown llm_provider {settings.llm_provider!r}. "
            "Choose: ollama | openai | anthropic | gemini | llama-cpp"
        )
