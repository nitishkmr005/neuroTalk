"""DuckDuckGo web search helper for LLM context augmentation."""

from __future__ import annotations

import asyncio

from loguru import logger

from config.settings import get_settings


async def web_search(query: str) -> list[dict[str, str]]:
    """Search DuckDuckGo and return structured results.

    Runs the blocking DDGS call in a thread-pool executor and enforces a
    configurable timeout so a slow/absent network never stalls the pipeline.

    Args:
        query: Natural-language search query derived from the user transcript.

    Returns:
        List of dicts with keys ``title``, ``snippet``, and ``url``.
        Returns an empty list on timeout or any error.
    """
    settings = get_settings()
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("event=web_search_unavailable reason=duckduckgo_search_not_installed")
        return []

    def _run() -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=settings.web_search_max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
        return results

    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=settings.web_search_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("event=web_search_timeout query={!r}", query)
        return []
    except Exception as err:
        logger.warning("event=web_search_error error={}", err)
        return []
