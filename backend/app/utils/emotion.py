from __future__ import annotations

import re

_TAG_PATTERN = re.compile(r"\[[^\]]+\]")

# Maps LLM emotion tags to Chatterbox exaggeration values (0.0–1.0).
# Higher = more expressive / dramatic delivery.
_EMOTION_EXAGGERATION: dict[str, float] = {
    "[gasp]": 0.8,
    "[laugh]": 0.7,
    "[chuckle]": 0.6,
    "[sigh]": 0.55,
    "[clear throat]": 0.35,
}
_BASE_EXAGGERATION = 0.3

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

_MARKDOWN_PATTERN = re.compile(
    r"#{1,6}\s?"
    r"|(\*{1,3}|_{1,3})"
    r"|`{1,3}"
    r"|^\s*[-*•]\s+"
    r"|^\s*\d+\.\s+",
    re.MULTILINE,
)


def extract_exaggeration(text: str) -> float:
    """Return a Chatterbox exaggeration value based on the dominant emotion tag in text.

    Args:
        text: LLM response string that may contain bracketed emotion tags.

    Returns:
        Float in [0.3, 0.8]; higher = more expressive delivery.
        Falls back to 0.3 (neutral) when no recognised tag is found.
    """
    lower = text.lower()
    return max(
        (_EMOTION_EXAGGERATION[tag] for tag in _EMOTION_EXAGGERATION if tag in lower),
        default=_BASE_EXAGGERATION,
    )


def strip_emotion_tags(text: str) -> str:
    """
    Remove inline emotion tags (e.g. [laugh], [sigh]) from a text string.

    Args:
        text: Raw LLM response string that may contain bracketed emotion tags.

    Returns:
        Cleaned string with all bracketed tags removed and whitespace normalised.

    Library:
        re (standard library) — uses _TAG_PATTERN to match and strip tags.
    """
    stripped = _TAG_PATTERN.sub("", text)
    return " ".join(stripped.split())


def clean_for_tts(text: str) -> str:
    """
    Strip markdown formatting and emojis so TTS receives plain speakable text.

    Args:
        text: Raw LLM response string potentially containing markdown and emojis.

    Returns:
        Plain text string safe for TTS — no asterisks, hashes, emojis, or
        symbols that would be read aloud verbatim.

    Library:
        re (standard library) — uses _MARKDOWN_PATTERN and _EMOJI_PATTERN.
    """
    text = _EMOJI_PATTERN.sub("", text)
    text = _MARKDOWN_PATTERN.sub("", text)
    return " ".join(text.split())
