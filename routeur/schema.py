from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tasks import infer_task, normalize_task

MIN_LEVEL = 1
MAX_LEVEL = 5
LEVELS = tuple(range(MIN_LEVEL, MAX_LEVEL + 1))


class SchemaError(ValueError):
    pass


def normalize_level(value: Any, *, field: str = "level") -> int:
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field} must be an integer between 1 and 5") from exc
    if level not in LEVELS:
        raise SchemaError(f"{field} must be between 1 and 5, got {level}")
    return level


@dataclass(frozen=True)
class RouterExample:
    prompt: str
    level: int
    task: str = "general"
    source: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "RouterExample":
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            raise SchemaError("prompt is required")
        level = normalize_level(row.get("level"))
        task = normalize_task(row.get("task")) if row.get("task") else infer_task(prompt)
        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise SchemaError("metadata must be an object when provided")
        source = row.get("source")
        return cls(prompt=prompt, level=level, task=task, source=str(source) if source else None, metadata=metadata)

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {"prompt": self.prompt, "level": self.level, "task": self.task}
        if self.source:
            row["source"] = self.source
        if self.metadata:
            row["metadata"] = self.metadata
        return row


@dataclass(frozen=True)
class LevelResult:
    level: int
    quality: float
    cost_usd: float | None = None
    latency_ms: float | None = None
    ok: bool | None = None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "LevelResult":
        level = normalize_level(row.get("level"), field="results.level")
        quality = float(row.get("quality", 0.0))
        if quality < 0:
            raise SchemaError("quality must be non-negative")
        cost = row.get("cost_usd")
        latency = row.get("latency_ms")
        ok = row.get("ok")
        return cls(
            level=level,
            quality=quality,
            cost_usd=float(cost) if cost is not None else None,
            latency_ms=float(latency) if latency is not None else None,
            ok=bool(ok) if ok is not None else None,
        )
