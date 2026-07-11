from __future__ import annotations

import asyncio
import functools
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capabilities import CAPABILITIES, RISKS, infer_capabilities
from .heuristics import heuristic_level
from .io import read_json
from .policy import ModelPolicy
from .schema import MAX_LEVEL, MIN_LEVEL, normalize_level
from .tasks import TASKS, infer_task, normalize_task


@dataclass(frozen=True)
class RouteDecision:
    # `model` is the primary routing decision. `level` remains as a derived
    # compatibility field for existing metrics and downstream integrations.
    level: int
    confidence: float
    raw_level: int
    model: str | None = None
    model_display_name: str | None = None
    model_score: float | None = None
    task: str = "general"
    task_confidence: float = 0.0
    risk: str = "low"
    risk_confidence: float = 0.0
    required_capabilities: list[str] | None = None
    capability_scores: dict[str, float] | None = None
    model_candidates: list[str] | None = None
    probabilities: list[float] | None = None
    reason: str = "model"

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model": self.model,
            "model_display_name": self.model_display_name,
            "model_score": self.model_score,
            "level": self.level,
            "confidence": self.confidence,
            "raw_level": self.raw_level,
            "task": self.task,
            "task_confidence": self.task_confidence,
            "risk": self.risk,
            "risk_confidence": self.risk_confidence,
            "reason": self.reason,
        }
        if self.model_candidates is not None:
            row["model_candidates"] = self.model_candidates
        if self.required_capabilities is not None:
            row["required_capabilities"] = self.required_capabilities
        if self.capability_scores is not None:
            row["capability_scores"] = self.capability_scores
        if self.probabilities is not None:
            row["probabilities"] = self.probabilities
        return row


class PromptRouter:
    _telemetry_callback: Any | None = None

    def _emit_telemetry(self, prompt: str, decision: RouteDecision) -> None:
        if self._telemetry_callback is not None:
            self._telemetry_callback(prompt, decision)

    def route(self, prompt: str) -> RouteDecision:
        """Route a single prompt.

        Subclasses may override this for efficiency, but the default delegates
        to :meth:`route_batch` so that only batch logic needs to be maintained.
        """
        return self.route_batch([prompt])[0]

    def route_batch(self, prompts: list[str]) -> list[RouteDecision]:
        """Route multiple prompts efficiently."""
        return [self.route(prompt) for prompt in prompts]

    async def route_async(self, prompt: str) -> RouteDecision:
        """Asynchronous wrapper that runs the synchronous router in a thread.

        Keeps the event loop unblocked when the router is embedded in an async
        service (FastAPI, etc.).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.route, prompt)

    async def route_batch_async(self, prompts: list[str]) -> list[RouteDecision]:
        """Async batch wrapper."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.route_batch, prompts)


class HeuristicRouter(PromptRouter):
    def __init__(
        self,
        *,
        policy_path: str | Path | None = None,
        telemetry_callback: Any | None = None,
    ) -> None:
        self.policy = ModelPolicy(policy_path)
        self._telemetry_callback = telemetry_callback

    def route(self, prompt: str) -> RouteDecision:
        level = heuristic_level(prompt)
        task = infer_task(prompt)
        risk = "high" if task == "safety" or level == 5 else ("medium" if level >= 3 else "low")
        capabilities = infer_capabilities(prompt, task)
        # The heuristic lane has no calibrated probability distribution. Its
        # difficulty is already conservative, so do not treat the fixed
        # fallback confidence as classifier uncertainty and over-escalate easy
        # prompts.
        selected, alternatives = self.policy.choose_direct(
            task,
            level,
            confidence=1.0,
            capability_scores={capability: 1.0 for capability in capabilities},
            risk=risk,
        )
        decision = RouteDecision(
            model=str(selected["id"]) if selected else None,
            model_display_name=str(selected.get("display_name", selected["id"])) if selected else None,
            model_score=float(selected["score"]) if selected and selected.get("score") is not None else None,
            level=level,
            raw_level=level,
            confidence=0.35,
            task=task,
            task_confidence=0.35,
            risk=risk,
            risk_confidence=0.35,
            required_capabilities=capabilities,
            capability_scores={capability: 1.0 for capability in capabilities},
            model_candidates=[str(item["id"]) for item in alternatives],
            reason="heuristic",
        )
        self._emit_telemetry(prompt, decision)
        return decision


class TransformerRouter(PromptRouter):
    def __init__(
        self,
        model_dir: str | Path,
        *,
        confidence_threshold: float | None = None,
        safety_bump: int | None = None,
        max_length: int | None = None,
        policy_path: str | Path | None = None,
        temperature: float | None = None,
        compile: bool = False,  # noqa: A002
        cache_size: int = 1024,
        telemetry_callback: Any | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install routeur[serve] to use TransformerRouter") from exc

        self.torch = torch
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "router_config.json"
        config = read_json(config_path) if config_path.exists() else {}
        self.confidence_threshold = float(
            confidence_threshold if confidence_threshold is not None else config.get("confidence_threshold", 0.55)
        )
        self.safety_bump = int(safety_bump if safety_bump is not None else config.get("safety_bump", 1))
        self.max_length = int(max_length if max_length is not None else config.get("max_length", 512))
        self.input_prefix = str(config.get("input_prefix", ""))
        self.truncation_strategy = str(config.get("truncation_strategy", "right"))
        self.temperature = float(temperature if temperature is not None else config.get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        self.policy = ModelPolicy(policy_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.multitask = (self.model_dir / "router_model.json").exists()
        if self.multitask:
            from .modeling import MultiTaskRouterModel

            self.model = MultiTaskRouterModel.from_pretrained(self.model_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        if self.torch.cuda.is_available():
            self.device = self.torch.device("cuda")
        elif self.torch.backends.mps.is_available():
            self.device = self.torch.device("mps")
        else:
            self.device = self.torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        if compile and hasattr(self.torch, "compile"):
            self.model = self.torch.compile(self.model, dynamic=True)
        self._cache: dict[str, RouteDecision] = {}
        self._cache_size = max(1, cache_size)
        self._cache_order: list[str] = []
        self._cache_lock = threading.Lock()
        self._telemetry_callback = telemetry_callback

    def _encode(self, prompts: list[str]) -> dict[str, Any]:
        if self.truncation_strategy == "head_tail":
            from .tokenization import encode_head_tail_batch

            return encode_head_tail_batch(
                self.tokenizer,
                prompts,
                max_length=self.max_length,
                input_prefix=self.input_prefix,
                padding=True,
                return_tensors="pt",
            )
        return self.tokenizer(
            [self.input_prefix + prompt for prompt in prompts],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

    def route_batch(self, prompts: list[str]) -> list[RouteDecision]:
        if not prompts:
            return []
        # Check cache first; only encode/cache prompts that are not already
        # present so repeated prompts are essentially free.
        cached_decisions: list[RouteDecision | None] = [None] * len(prompts)
        missing_indices: list[int] = []
        missing_prompts: list[str] = []
        for index, prompt in enumerate(prompts):
            cached = self._cache_get(prompt)
            if cached is not None:
                cached_decisions[index] = cached
            else:
                missing_indices.append(index)
                missing_prompts.append(prompt)
        if missing_prompts:
            inputs = self._encode(missing_prompts)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            for batch_index, (position, prompt) in enumerate(zip(missing_indices, missing_prompts, strict=True)):
                decision = self._decide(prompt, outputs, batch_index)
                self._cache_prompt(prompt, decision)
                cached_decisions[position] = decision
        if self._telemetry_callback is not None:
            for prompt, decision in zip(prompts, cached_decisions, strict=True):
                self._telemetry_callback(prompt, decision)
        return cached_decisions  # type: ignore[return-value]

    def _cache_prompt(self, prompt: str, decision: RouteDecision) -> None:
        with self._cache_lock:
            if prompt in self._cache:
                self._cache_order.remove(prompt)
            elif len(self._cache) >= self._cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            self._cache[prompt] = decision
            self._cache_order.append(prompt)

    def _cache_get(self, prompt: str) -> RouteDecision | None:
        with self._cache_lock:
            return self._cache.get(prompt)

    def _decide(self, prompt: str, outputs: Any, index: int) -> RouteDecision:
        if self.multitask:
            level_logits = outputs["level_logits"][index]
            task_logits = outputs["task_logits"][index]
            risk_logits = outputs.get("risk_logits")
            capability_logits = outputs.get("capability_logits")
        else:
            level_logits = outputs.logits[index]
            task_logits = None
            risk_logits = None
            capability_logits = None
        level_probs_tensor = self.torch.softmax(level_logits / max(self.temperature, 1e-6), dim=-1)
        if task_logits is not None:
            task_probs_tensor = self.torch.softmax(task_logits, dim=-1)
        else:
            task_probs_tensor = None
        if risk_logits is not None:
            risk_probs_tensor = self.torch.softmax(risk_logits[index], dim=-1)
        else:
            risk_probs_tensor = None
        if capability_logits is not None:
            capability_probs_tensor = self.torch.sigmoid(capability_logits[index])
        else:
            capability_probs_tensor = None
        probabilities = [float(value) for value in level_probs_tensor.tolist()]
        raw_level = int(max(range(len(probabilities)), key=probabilities.__getitem__)) + 1
        confidence = probabilities[raw_level - 1]
        task_probabilities: list[float] | None = None
        if task_probs_tensor is None:
            task = infer_task(prompt)
            task_confidence = 0.35
        else:
            task_probabilities = [float(value) for value in task_probs_tensor.tolist()]
            task_index = max(range(len(task_probabilities)), key=task_probabilities.__getitem__)
            task = TASKS[task_index] if task_index < len(TASKS) else "general"
            task_confidence = task_probabilities[task_index]
        if risk_probs_tensor is None:
            risk = "high" if task == "safety" or raw_level == 5 else ("medium" if raw_level >= 3 else "low")
            risk_confidence = 0.35
        else:
            risk_probabilities = [float(value) for value in risk_probs_tensor.tolist()]
            risk_index = max(range(len(risk_probabilities)), key=risk_probabilities.__getitem__)
            risk = RISKS[risk_index] if risk_index < len(RISKS) else "medium"
            risk_confidence = risk_probabilities[risk_index]
        heuristic_capabilities = set(infer_capabilities(prompt, task))
        if capability_probs_tensor is None:
            capability_scores = {capability: 1.0 for capability in heuristic_capabilities}
        else:
            capability_scores = {
                CAPABILITIES[index]: float(value)
                for index, value in enumerate(capability_probs_tensor.tolist())
                if index < len(CAPABILITIES)
            }
            for capability in heuristic_capabilities:
                capability_scores[capability] = max(0.8, capability_scores.get(capability, 0.0))
        required_capabilities = sorted(
            capability for capability, score in capability_scores.items() if score >= 0.45
        )
        # A learned classifier must not erase explicit high-stakes signals.
        # Keep the model's specialty prediction for normal traffic, but force
        # the safety lane when the conservative lexical guard sees it.
        heuristic_task = infer_task(prompt)
        if heuristic_task == "safety":
            task = "safety"
            task_confidence = max(task_confidence, 0.8)
            risk = "high"
            risk_confidence = max(risk_confidence, 0.8)
            capability_scores["safety"] = max(0.8, capability_scores.get("safety", 0.0))
            required_capabilities = sorted(set(required_capabilities) | {"safety"})
            routed_level = MAX_LEVEL
        else:
            routed_level = raw_level
        reason = "model"
        # Low-confidence level-1/2 predictions are often harmless routine
        # prompts; do not pay a full tier just because their probability is
        # diffuse. Only genuinely high-risk rows retain a conservative bump.
        effective_threshold = min(self.confidence_threshold, 0.55) if raw_level <= 2 else self.confidence_threshold
        # Escalation on uncertainty is reserved for genuinely high-risk rows.
        # A global confidence bump made ordinary prompts over-route massively;
        # explicit safety prompts are already forced to the strongest lane.
        if routed_level < MAX_LEVEL and risk == "high" and confidence < effective_threshold:
            routed_level = min(MAX_LEVEL, routed_level + self.safety_bump)
            reason = "model_low_confidence_bump"
        if heuristic_task == "safety":
            reason = "safety_guard"
        selected, alternatives = self.policy.choose_direct(
            task,
            normalize_level(routed_level),
            confidence=confidence,
            safety=heuristic_task == "safety",
            task_scores=(
                {TASKS[index]: value for index, value in enumerate(task_probabilities or []) if index < len(TASKS)}
                if task_probabilities is not None
                else None
            ),
            capability_scores=capability_scores,
            risk=risk,
        )
        return RouteDecision(
            model=str(selected["id"]) if selected else None,
            model_display_name=str(selected.get("display_name", selected["id"])) if selected else None,
            model_score=float(selected["score"]) if selected and selected.get("score") is not None else None,
            level=normalize_level(routed_level),
            raw_level=normalize_level(raw_level),
            confidence=confidence,
            task=normalize_task(task),
            task_confidence=task_confidence,
            risk=risk,
            risk_confidence=risk_confidence,
            required_capabilities=required_capabilities,
            capability_scores={key: round(value, 5) for key, value in capability_scores.items()},
            model_candidates=[str(item["id"]) for item in alternatives],
            probabilities=probabilities,
            reason=reason,
        )


class FastRouter(PromptRouter):
    """CPU-first multi-task router designed for single-prompt latency below 10 ms."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        policy_path: str | Path | None = None,
        cache_size: int = 1024,
        telemetry_callback: Any | None = None,
    ) -> None:
        from .fast_model import FastModelArtifact

        self.artifact = FastModelArtifact(model_dir)
        self.policy = ModelPolicy(policy_path, candidate_limit=20)
        self.confidence_threshold = float(self.artifact.config.get("confidence_threshold", 0.0))
        self.safety_bump = int(self.artifact.config.get("safety_bump", 0))
        self._cache: dict[str, RouteDecision] = {}
        self._cache_size = max(1, int(cache_size))
        self._cache_order: list[str] = []
        self._cache_lock = threading.Lock()
        self._telemetry_callback = telemetry_callback

    def _cache_get(self, prompt: str) -> RouteDecision | None:
        with self._cache_lock:
            return self._cache.get(prompt)

    def _cache_prompt(self, prompt: str, decision: RouteDecision) -> None:
        with self._cache_lock:
            if prompt in self._cache:
                self._cache_order.remove(prompt)
            elif len(self._cache) >= self._cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            self._cache[prompt] = decision
            self._cache_order.append(prompt)

    def route_batch(self, prompts: list[str]) -> list[RouteDecision]:
        if not prompts:
            return []
        results: list[RouteDecision | None] = [None] * len(prompts)
        missing_positions: list[int] = []
        missing_prompts: list[str] = []
        for position, prompt in enumerate(prompts):
            cached = self._cache_get(prompt)
            if cached is None:
                missing_positions.append(position)
                missing_prompts.append(prompt)
            else:
                results[position] = cached
        if missing_prompts:
            predictions = self.artifact.predict(missing_prompts)
            for index, (position, prompt) in enumerate(zip(missing_positions, missing_prompts, strict=True)):
                decision = self._decide(prompt, predictions, index)
                self._cache_prompt(prompt, decision)
                results[position] = decision
        decisions = results  # keep a named value for telemetry and type narrowing
        if self._telemetry_callback is not None:
            for prompt, decision in zip(prompts, decisions, strict=True):
                self._telemetry_callback(prompt, decision)
        return decisions  # type: ignore[return-value]

    def _decide(self, prompt: str, predictions: Any, index: int) -> RouteDecision:
        level_probabilities = predictions.level_probabilities[index].tolist()
        raw_level_index = max(range(len(level_probabilities)), key=level_probabilities.__getitem__)
        raw_level = int(self.artifact.levels[raw_level_index])
        confidence = float(level_probabilities[raw_level_index])

        task_probabilities = predictions.task_probabilities[index].tolist()
        task_index = max(range(len(task_probabilities)), key=task_probabilities.__getitem__)
        task = self.artifact.tasks[task_index]
        task_confidence = float(task_probabilities[task_index])

        risk_probabilities = predictions.risk_probabilities[index].tolist()
        risk_index = max(range(len(risk_probabilities)), key=risk_probabilities.__getitem__)
        risk = self.artifact.risks[risk_index]
        risk_confidence = float(risk_probabilities[risk_index])

        capability_probabilities = predictions.capability_probabilities[index].tolist()
        capability_scores = {
            capability: float(capability_probabilities[position])
            for position, capability in enumerate(self.artifact.capabilities)
        }
        heuristic_capabilities = set(infer_capabilities(prompt, task))
        for capability in heuristic_capabilities:
            capability_scores[capability] = max(0.8, capability_scores.get(capability, 0.0))
        required_capabilities = sorted(name for name, score in capability_scores.items() if score >= 0.45)

        heuristic_task = infer_task(prompt)
        routed_level = raw_level
        reason = "fast_model"
        if heuristic_task == "safety":
            task = "safety"
            task_confidence = max(task_confidence, 0.8)
            risk = "high"
            risk_confidence = max(risk_confidence, 0.8)
            capability_scores["safety"] = max(0.8, capability_scores.get("safety", 0.0))
            required_capabilities = sorted(set(required_capabilities) | {"safety"})
            routed_level = MAX_LEVEL
            reason = "safety_guard"
        elif risk == "high" and confidence < self.confidence_threshold:
            routed_level = min(MAX_LEVEL, routed_level + self.safety_bump)
            reason = "fast_model_low_confidence_bump"

        task_scores = {
            self.artifact.tasks[position]: float(value)
            for position, value in enumerate(task_probabilities)
        }
        selected, alternatives = self.policy.choose_direct(
            task,
            normalize_level(routed_level),
            confidence=confidence,
            safety=heuristic_task == "safety",
            task_scores=task_scores,
            capability_scores=capability_scores,
            risk=risk,
        )
        return RouteDecision(
            model=str(selected["id"]) if selected else None,
            model_display_name=str(selected.get("display_name", selected["id"])) if selected else None,
            model_score=float(selected["score"]) if selected and selected.get("score") is not None else None,
            level=normalize_level(routed_level),
            raw_level=normalize_level(raw_level),
            confidence=confidence,
            task=normalize_task(task),
            task_confidence=task_confidence,
            risk=risk,
            risk_confidence=risk_confidence,
            required_capabilities=required_capabilities,
            capability_scores={key: round(value, 5) for key, value in capability_scores.items()},
            model_candidates=[str(item["id"]) for item in alternatives],
            probabilities=[float(value) for value in level_probabilities],
            reason=reason,
        )


def load_router(model_dir: str | Path | None, *, policy_path: str | Path | None = None) -> PromptRouter:
    if model_dir is None:
        return HeuristicRouter(policy_path=policy_path)
    if (Path(model_dir) / "fast_router.json").exists():
        return FastRouter(model_dir, policy_path=policy_path)
    return TransformerRouter(model_dir, policy_path=policy_path)
