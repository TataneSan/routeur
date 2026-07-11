from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable

from .labels import DEFAULT_LEVEL_COSTS_USD
from .schema import normalize_level


# Shared weights used to score router checkpoints and compare runs.
# Keep them in one place so training, calibration and experiment selection
# optimize the exact same surface.
SCORING_WEIGHTS: dict[str, float] = {
    "level_macro_f1": 1.50,
    "task_macro_f1": 0.60,
    "capability_micro_f1": 0.25,
    "risk_accuracy": 0.15,
    "preference_accuracy": 0.20,
    "severe_underroute_rate": -2.50,
    "underroute_rate": -0.40,
    "overroute_rate": -0.30,
    "level_ece": -0.25,
    "external_router_accuracy": 0.35,
    "throughput_log": 0.01,
    "cost_ratio_vs_oracle": -0.10,
}


def routing_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    level_costs: dict[int, float] | None = None,
) -> dict[str, float | dict[str, int]]:
    true_levels = [normalize_level(level, field="y_true") for level in y_true]
    pred_levels = [normalize_level(level, field="y_pred") for level in y_pred]
    if len(true_levels) != len(pred_levels):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_levels:
        raise ValueError("at least one example is required")

    costs = level_costs or DEFAULT_LEVEL_COSTS_USD
    diffs = [pred - true for true, pred in zip(true_levels, pred_levels, strict=True)]
    exact = [diff == 0 for diff in diffs]
    adjacent = [abs(diff) <= 1 for diff in diffs]
    under = [diff < 0 for diff in diffs]
    severe_under = [diff <= -2 for diff in diffs]
    oracle_cost = [costs[level] for level in true_levels]
    routed_cost = [costs[level] for level in pred_levels]
    max_cost = costs[max(costs)]

    confusion = Counter(f"{true}->{pred}" for true, pred in zip(true_levels, pred_levels, strict=True))
    return {
        "count": float(len(true_levels)),
        "accuracy": mean(exact),
        "adjacent_accuracy": mean(adjacent),
        "mean_abs_error": mean(abs(diff) for diff in diffs),
        "underroute_rate": mean(under),
        "severe_underroute_rate": mean(severe_under),
        "overroute_rate": mean(diff > 0 for diff in diffs),
        "avg_oracle_cost_usd": mean(oracle_cost),
        "avg_routed_cost_usd": mean(routed_cost),
        "avg_always_level_5_cost_usd": max_cost,
        "savings_vs_always_level_5": 1.0 - (mean(routed_cost) / max_cost),
        "cost_ratio_vs_oracle": mean(routed_cost) / mean(oracle_cost),
        "confusion": dict(sorted(confusion.items())),
    }
