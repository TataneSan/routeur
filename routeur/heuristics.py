from __future__ import annotations

import re
from typing import Any

from .schema import MAX_LEVEL, MIN_LEVEL
from .tasks import infer_task

CODE_RE = re.compile(r"```|def |class |import |SELECT |CREATE TABLE|Traceback|TypeError|Exception", re.I)
MATH_RE = re.compile(r"\b(prove|proof|derive|equation|integral|probability|olympiad|aime|math)\b", re.I)
LONG_CONTEXT_RE = re.compile(r"\b(analyze|compare|critique|summarize|extract|audit)\b", re.I)
HIGH_STAKES_RE = re.compile(r"\b(legal|medical|diagnosis|tax|contract|security|vulnerability|production)\b", re.I)
CREATIVE_RE = re.compile(r"\b(write|rewrite|translate|brainstorm|email|caption|tweet)\b", re.I)


def _token_count(text: str, tokenizer: Any | None = None) -> int:
    """Return a token count. Uses the tokenizer when available, else words."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # pragma: no cover - tokenizer may be unavailable
            pass
    return len(text.split())


def heuristic_level(prompt: str, tokenizer: Any | None = None) -> int:
    text = prompt.strip()
    words = _token_count(text, tokenizer)
    level = MIN_LEVEL

    # Word thresholds are calibrated to approximate token counts
    # (1 word ~ 1.3 tokens). This avoids forcing every long prompt
    # to the most expensive frontier model.
    if words > 60 or CREATIVE_RE.search(text):
        level = max(level, 2)
    if words > 300 or LONG_CONTEXT_RE.search(text):
        level = max(level, 3)
    if CODE_RE.search(text) or MATH_RE.search(text):
        level = max(level, 4)
    if words > 2500 or HIGH_STAKES_RE.search(text):
        level = max(level, 5)

    # Capability is a stronger signal than keywords alone for code and
    # specialist tasks. Keep routine writing cheap, but avoid sending a
    # multi-file engineering or reasoning request to a tiny model.
    task = infer_task(text)
    if task in {"frontend", "backend", "coding", "data", "research"}:
        level = max(level, 3)
    if task == "reasoning":
        level = max(level, 4)
    if task == "safety":
        level = max(level, 5)

    return min(MAX_LEVEL, max(MIN_LEVEL, level))
