from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CandidateEstimate:
    model: str
    mean_score: float
    score_std: float
    lower_confidence_bound: float
    mean_cost: float
    observations: int


def candidate_estimates(rows: Iterable[dict[str, Any]], *, z_value: float = 1.645) -> list[CandidateEstimate]:
    """Aggregate repeated scored generations into stable query-model estimates."""
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if not bool(row.get("parse_success", True)):
            continue
        grouped[str(row["model"])].append((float(row["score"]), float(row["cost"])))
    estimates: list[CandidateEstimate] = []
    for model, values in grouped.items():
        scores = [score for score, _cost in values]
        costs = [cost for _score, cost in values]
        mean_score = statistics.mean(scores)
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        standard_error = score_std / math.sqrt(len(scores))
        estimates.append(
            CandidateEstimate(
                model=model,
                mean_score=mean_score,
                score_std=score_std,
                lower_confidence_bound=mean_score - z_value * standard_error,
                mean_cost=statistics.mean(costs),
                observations=len(scores),
            )
        )
    return estimates


def choose_cheapest_sufficient(
    estimates: Iterable[CandidateEstimate],
    *,
    quality_threshold: float,
    model_costs: dict[str, float],
) -> tuple[CandidateEstimate, bool]:
    """Choose the cheapest statistically sufficient model, or best fallback.

    The boolean is true only when at least one model's lower confidence bound
    clears the quality threshold. This prevents a single lucky generation from
    becoming a gold routing label.
    """
    choices = list(estimates)
    if not choices:
        raise ValueError("Cannot build an oracle label without valid model observations")
    sufficient = [item for item in choices if item.lower_confidence_bound >= quality_threshold]
    if sufficient:
        return min(sufficient, key=lambda item: (model_costs[item.model], -item.mean_score)), True
    return max(choices, key=lambda item: (item.lower_confidence_bound, -model_costs[item.model])), False


def cost_rank_levels(model_costs: dict[str, float]) -> dict[str, int]:
    """Map an arbitrary candidate set monotonically onto five cost levels."""
    ordered = sorted(model_costs, key=lambda model: (model_costs[model], model))
    if len(ordered) == 1:
        return {ordered[0]: 1}
    return {
        model: 1 + round(index * 4 / (len(ordered) - 1))
        for index, model in enumerate(ordered)
    }


def bootstrap_level_stability(
    rows: Iterable[dict[str, Any]],
    *,
    selected_level: int,
    quality_threshold: float,
    model_costs: dict[str, float],
    levels: dict[str, int],
    z_value: float = 1.645,
    iterations: int = 200,
    seed: int = 42,
) -> float:
    """Estimate how often repeated observations reproduce the oracle level."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("parse_success", True)):
            grouped[str(row["model"])].append(row)
    rng = random.Random(seed)
    matches = 0
    for _ in range(iterations):
        sample: list[dict[str, Any]] = []
        for values in grouped.values():
            sample.extend(rng.choice(values) for _ in range(len(values)))
        estimates = candidate_estimates(sample, z_value=z_value)
        selected, _sufficient = choose_cheapest_sufficient(
            estimates,
            quality_threshold=quality_threshold,
            model_costs=model_costs,
        )
        matches += int(levels[selected.model] == selected_level)
    return matches / iterations
