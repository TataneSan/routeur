from routeur.heuristics import heuristic_level
from routeur.router import HeuristicRouter, load_router


def test_heuristic_router_escalates_high_stakes_prompt():
    assert heuristic_level("Review this production security vulnerability patch") == 5


def test_heuristic_router_keeps_simple_prompt_cheap():
    assert heuristic_level("Translate hello to French") <= 2


def test_heuristic_router_route_batch_returns_same_as_route():
    router = HeuristicRouter()
    prompts = ["Translate hello to French", "Review this production security vulnerability patch"]
    batch = router.route_batch(prompts)
    single = [router.route(prompt) for prompt in prompts]
    assert len(batch) == len(prompts)
    assert [decision.level for decision in batch] == [decision.level for decision in single]
    assert [decision.model for decision in batch] == [decision.model for decision in single]


def test_load_router_heuristic_without_model_dir():
    router = load_router(None)
    decision = router.route("What is 2 + 2?")
    assert decision.model is not None
    assert 1 <= decision.level <= 5


import asyncio  # noqa: E402


def test_heuristic_router_async_methods():
    router = HeuristicRouter()

    async def _run() -> None:
        decision = await router.route_async("What is 2 + 2?")
        assert decision.model is not None
        assert 1 <= decision.level <= 5
        batch = await router.route_batch_async(["hello", "code"])
        assert len(batch) == 2

    asyncio.run(_run())


def test_heuristic_level_uses_tokenizer_when_provided():
    class DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG001
            return [0] * 500  # pretend every prompt is 500 tokens

    # With a tokenizer that reports 500 tokens, the long-context threshold fires.
    assert heuristic_level("Some prompt", tokenizer=DummyTokenizer()) >= 3


def test_prompt_router_route_delegates_to_route_batch():
    from routeur.router import PromptRouter, RouteDecision

    class DummyRouter(PromptRouter):
        def route_batch(self, prompts: list[str]) -> list[RouteDecision]:
            return [RouteDecision(level=1, confidence=1.0, raw_level=1) for _ in prompts]

    router = DummyRouter()
    decision = router.route("hello")
    assert decision.level == 1


def test_router_telemetry_callback_is_called():
    from routeur.router import HeuristicRouter

    calls: list[tuple[str, Any]] = []
    router = HeuristicRouter(telemetry_callback=lambda prompt, decision: calls.append((prompt, decision)))
    decision = router.route("hello")
    assert len(calls) == 1
    assert calls[0][0] == "hello"
    assert calls[0][1] is decision


def test_router_telemetry_callback_is_called_for_batch():
    from routeur.router import HeuristicRouter

    calls: list[tuple[str, Any]] = []
    router = HeuristicRouter(telemetry_callback=lambda prompt, decision: calls.append((prompt, decision)))
    decisions = router.route_batch(["hello", "world"])
    assert len(calls) == 2
    assert calls[0][0] == "hello"
    assert calls[1][0] == "world"
    assert calls[0][1] is decisions[0]
    assert calls[1][1] is decisions[1]
