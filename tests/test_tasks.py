from routeur.policy import ModelPolicy
from routeur.capabilities import infer_capabilities
from routeur.tasks import difficulty_to_level, infer_task


def test_task_inference_separates_frontend_and_backend():
    assert infer_task("Build a responsive React UI with Tailwind CSS") == "frontend"
    assert infer_task("Design a PostgreSQL REST API with idempotent retries") == "backend"


def test_task_inference_detects_vision_and_agentic_requirements():
    assert infer_task("Analyze this image and extract the visible table") == "vision"
    assert infer_task("Use browser tools autonomously and verify every step") == "agentic"
    assert "vision" in infer_capabilities("Analyze this image", "vision")
    assert "tool_use" in infer_capabilities("Use browser tools autonomously", "agentic")
    assert "creative" not in infer_capabilities("Translate hello to French", "writing")


def test_difficulty_mapping_is_conservative():
    assert difficulty_to_level(1) == 1
    assert difficulty_to_level(4) == 3
    assert difficulty_to_level(7) == 5


def test_policy_prefers_specialists_at_frontier_level():
    model, alternatives = ModelPolicy().choose("frontend", 5)
    assert model is not None
    assert model.split("/")[0] in {"anthropic", "zai", "qwen", "openai", "google"}
    assert alternatives

    model, _ = ModelPolicy().choose("backend", 5)
    assert model is not None


def test_direct_policy_always_returns_a_concrete_model_for_simple_prompts():
    selected, alternatives = ModelPolicy().choose_direct("general", 1, confidence=0.95)
    assert selected is not None
    assert selected["id"]
    assert alternatives


def test_arena_catalog_and_fallbacks_have_unique_callable_ids():
    policy = ModelPolicy()
    model_ids = [str(model["id"]) for model in policy.models]
    assert len(model_ids) == len(set(model_ids))
    selected, alternatives = policy.choose_direct("agentic", 5, confidence=0.9)
    routed_ids = [str(selected["id"]), *(str(model["id"]) for model in alternatives)]
    assert len(routed_ids) == len(set(routed_ids))
    assert not str(selected["id"]).startswith("lmarena/")


def test_policy_eligible_includes_models_with_min_level_one():
    policy = ModelPolicy()
    eligible = policy._eligible(1)
    # Models without an explicit min_level default to 1 and are eligible.
    assert any(int(model.get("min_level", 1)) == 1 for model in eligible)


def test_policy_rank_direct_prefers_lower_cost_for_same_quality():
    policy = ModelPolicy()
    ranked = policy.rank_direct("general", 1, confidence=0.95)
    assert ranked
    # The top ranked model should be callable and have the lowest min_level.
    assert ranked[0].get("callable", True)
    assert int(ranked[0].get("min_level", 1)) <= 2


def test_policy_rank_direct_respects_target_tier():
    policy = ModelPolicy()
    # Confidence and risk should not alter the target tier inside the policy.
    ranked_low_conf = policy.rank_direct("general", 2, confidence=0.2)
    ranked_high_conf = policy.rank_direct("general", 2, confidence=0.9)
    assert int(ranked_low_conf[0].get("min_level", 1)) == int(ranked_high_conf[0].get("min_level", 1))
    ranked_risk = policy.rank_direct("general", 2, confidence=0.9, risk="high")
    ranked_safe = policy.rank_direct("general", 2, confidence=0.9, risk="low")
    assert int(ranked_risk[0].get("min_level", 1)) == int(ranked_safe[0].get("min_level", 1))
    # Higher target tier should not select a model with a lower min_level.
    ranked_tier3 = policy.rank_direct("general", 3, confidence=0.9)
    ranked_tier4 = policy.rank_direct("general", 4, confidence=0.9)
    assert int(ranked_tier4[0].get("min_level", 1)) >= int(ranked_tier3[0].get("min_level", 1))
