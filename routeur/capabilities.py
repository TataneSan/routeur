from __future__ import annotations

import re
from collections.abc import Iterable

CAPABILITIES = (
    "vision",
    "coding",
    "reasoning",
    "web_search",
    "tool_use",
    "long_context",
    "multilingual",
    "creative",
    "safety",
)

RISKS = ("low", "medium", "high")


def normalize_capabilities(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    valid = set(CAPABILITIES)
    return sorted({str(value).strip().lower().replace("-", "_") for value in values} & valid)


def infer_capabilities(prompt: str, task: str) -> list[str]:
    text = re.sub(r"\s+", " ", prompt.lower()).strip()
    capabilities: set[str] = set()
    if task in {"frontend", "backend", "coding", "data", "agentic"}:
        capabilities.add("coding")
    if task in {"reasoning", "research", "data", "safety"}:
        capabilities.add("reasoning")
    if task == "research" or any(token in text for token in ("browse", "search the web", "sources", "citations", "latest", "current")):
        capabilities.add("web_search")
    if task == "agentic" or any(token in text for token in ("tool call", "use tools", "mcp", "browser automation", "execute commands")):
        capabilities.add("tool_use")
    if task == "vision" or any(token in text for token in ("image", "photo", "diagram", "ocr", "video", "screenshot")):
        capabilities.add("vision")
    if any(token in text for token in ("creative", "story", "poem", "copywriting", "brand voice")):
        capabilities.add("creative")
    if task == "safety":
        capabilities.add("safety")
    if len(prompt) > 6000 or any(token in text for token in ("long context", "entire repository", "whole codebase", "full document")):
        capabilities.add("long_context")
    if any(token in text for token in ("translate", "multilingual", "french and english", "français", "traduire")):
        capabilities.add("multilingual")
    return normalize_capabilities(capabilities)
