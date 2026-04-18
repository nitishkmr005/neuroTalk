from __future__ import annotations

import re

_TAG_PATTERN = re.compile(r"\[[^\]]+\]")


def strip_emotion_tags(text: str) -> str:
    stripped = _TAG_PATTERN.sub("", text)
    return " ".join(stripped.split())
