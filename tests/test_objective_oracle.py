from routeur.objective_oracle import (
    CandidateEstimate,
    bootstrap_level_stability,
    choose_cheapest_sufficient,
    cost_rank_levels,
)


def test_cost_rank_levels_are_monotonic_and_bounded():
    levels = cost_rank_levels({"cheap": 1.0, "middle": 2.0, "expensive": 3.0})
    assert levels == {"cheap": 1, "middle": 3, "expensive": 5}


def test_oracle_selects_cheapest_candidate_whose_lcb_passes():
    candidates = [
        CandidateEstimate("cheap", 0.85, 0.1, 0.79, 1.0, 10),
        CandidateEstimate("middle", 0.88, 0.1, 0.82, 2.0, 10),
        CandidateEstimate("expensive", 0.95, 0.1, 0.90, 3.0, 10),
    ]
    selected, sufficient = choose_cheapest_sufficient(
        candidates,
        quality_threshold=0.8,
        model_costs={"cheap": 1.0, "middle": 2.0, "expensive": 3.0},
    )
    assert sufficient
    assert selected.model == "middle"


def test_bootstrap_stability_is_one_for_unambiguous_scores():
    rows = [
        {"model": model, "score": score, "cost": cost, "parse_success": True}
        for model, score, cost in (("cheap", 1.0, 1.0), ("expensive", 1.0, 2.0))
        for _ in range(5)
    ]
    assert bootstrap_level_stability(
        rows,
        selected_level=1,
        quality_threshold=0.8,
        model_costs={"cheap": 1.0, "expensive": 2.0},
        levels={"cheap": 1, "expensive": 5},
        iterations=20,
    ) == 1.0
