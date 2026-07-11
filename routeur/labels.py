from __future__ import annotations

from typing import Any

from .schema import LEVELS, LevelResult, SchemaError
from .tasks import infer_task


DEFAULT_LEVEL_COSTS_USD = {
    1: 0.00002,
    2: 0.00008,
    3: 0.00030,
    4: 0.00120,
    5: 0.00450,
}


def choose_oracle_level(
    results: list[LevelResult],
    *,
    min_quality: float,
    max_latency_ms: float | None = None,
    level_costs: dict[int, float] | None = None,
) -> int:
    if not results:
        raise SchemaError("results must contain at least one level result")

    costs = level_costs or DEFAULT_LEVEL_COSTS_USD
    by_level = {result.level: result for result in results}
    candidates: list[tuple[float, int]] = []
    for level in LEVELS:
        result = by_level.get(level)
        if result is None:
            continue
        if result.ok is False:
            continue
        if result.quality < min_quality:
            continue
        if max_latency_ms is not None and result.latency_ms is not None and result.latency_ms > max_latency_ms:
            continue
        candidates.append((result.cost_usd if result.cost_usd is not None else costs[level], level))

    if candidates:
        return min(candidates)[1]

    best = max(results, key=lambda item: (item.quality, -item.level))
    return best.level


def label_trace_row(
    row: dict[str, Any],
    *,
    min_quality: float,
    max_latency_ms: float | None = None,
    level_costs: dict[int, float] | None = None,
) -> dict[str, Any]:
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        raise SchemaError("prompt is required")
    raw_results = row.get("results")
    if not isinstance(raw_results, list):
        raise SchemaError("results must be a list")
    results = [LevelResult.from_json(item) for item in raw_results]
    level = choose_oracle_level(
        results,
        min_quality=min_quality,
        max_latency_ms=max_latency_ms,
        level_costs=level_costs,
    )
    return {
        "prompt": prompt,
        "level": level,
        "task": infer_task(prompt),
        "source": row.get("source", "trace"),
        "metadata": {
            "min_quality": min_quality,
            "max_latency_ms": max_latency_ms,
            "chosen_from_results": True,
        },
    }
