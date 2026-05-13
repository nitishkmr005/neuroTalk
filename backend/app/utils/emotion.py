from __future__ import annotations

import re

_TAG_PATTERN = re.compile(r"\[[^\]]+\]")

# Maps LLM emotion tags to Chatterbox exaggeration values (0.0–1.0).
# Higher = more expressive / dramatic delivery.
_EMOTION_EXAGGERATION: dict[str, float] = {
    "[gasp]": 0.8,
    "[laugh]": 0.7,
    "[surprised]": 0.65,
    "[chuckle]": 0.6,
    "[happy]": 0.55,
    "[sigh]": 0.55,
    "[clear throat]": 0.35,
}
_BASE_EXAGGERATION = 0.3

# All paralinguistic tags that Chatterbox Turbo recognizes as special tokens.
# Tags outside this set are stripped by normalize_for_turbo() to prevent literal read-back.
VALID_TURBO_TAGS: frozenset[str] = frozenset({
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[clear throat]",
    "[happy]", "[surprised]", "[whispering]", "[crying]",
    "[dramatic]", "[shush]", "[cough]", "[groan]", "[sniff]",
    "[angry]", "[fear]", "[sarcastic]", "[narration]", "[advertisement]",
})

# Common LLM tag variants → canonical Chatterbox Turbo special-token form.
_TAG_VARIANTS: dict[str, str] = {
    "[laughs]": "[laugh]",
    "[laughing]": "[laugh]",
    "[chuckles]": "[chuckle]",
    "[chuckling]": "[chuckle]",
    "[sighs]": "[sigh]",
    "[sighing]": "[sigh]",
    "[exhales]": "[sigh]",
    "[gasps]": "[gasp]",
    "[gasping]": "[gasp]",
    "[clears throat]": "[clear throat]",
    "[throat clear]": "[clear throat]",
    "[throat clearing]": "[clear throat]",
    "[hmm]": "[clear throat]",
    "[whisper]": "[whispering]",
    "[whispers]": "[whispering]",
    "[cries]": "[crying]",
    "[sobbing]": "[crying]",
    "[tearing up]": "[crying]",
    "[excited]": "[happy]",
    "[happily]": "[happy]",
    "[joyful]": "[happy]",
    "[wow]": "[surprised]",
    "[shocking]": "[surprised]",
    "[surprising]": "[surprised]",
    # Frustration → sigh (closest audible empathy sound)
    "[frustrated]": "[sigh]",
    "[frustration]": "[sigh]",
    "[empathy]": "[sigh]",
    "[understanding]": "[sigh]",
}

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


def normalize_for_turbo(text: str) -> str:
    """Map LLM tag variants to Chatterbox Turbo special-token forms; strip unrecognized tags.

    Ensures only valid special tokens reach the T3 model — unrecognized [tags]
    would otherwise be read aloud verbatim by the TTS.

    Args:
        text: LLM response string potentially containing emotion tags.

    Returns:
        Text with valid canonical tags preserved and unknown tags removed.
    """
    for variant, canonical in _TAG_VARIANTS.items():
        text = re.sub(re.escape(variant), canonical, text, flags=re.IGNORECASE)

    def _keep_or_strip(m: re.Match) -> str:
        return m.group(0) if m.group(0).lower() in VALID_TURBO_TAGS else ""

    text = _TAG_PATTERN.sub(_keep_or_strip, text)
    return " ".join(text.split())


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
