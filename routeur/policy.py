from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .capabilities import CAPABILITIES
from .tasks import TASKS, normalize_task


CAPABILITY_SPECIALTIES: dict[str, tuple[str, ...]] = {
    "vision": ("vision", "frontend"),
    "coding": ("coding", "backend", "frontend", "data"),
    "reasoning": ("reasoning", "data"),
    "web_search": ("research",),
    "tool_use": ("agentic", "backend"),
    "long_context": ("research", "general"),
    "multilingual": ("writing", "general"),
    "creative": ("writing", "frontend"),
    "safety": ("safety", "reasoning"),
}


class ModelPolicy:
    def __init__(self, config_path: str | Path | None = None, *, candidate_limit: int | None = None) -> None:
        config_root = Path(__file__).resolve().parent.parent / "configs"
        path = Path(config_path) if config_path else (config_root / "models_lmarena.json" if (config_root / "models_lmarena.json").exists() else config_root / "models.json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"models": {}}
        self.models: list[dict[str, Any]] = [dict(model) for model in data.get("models", []) if model.get("enabled", True)]
        self.candidate_limit = max(6, int(candidate_limit)) if candidate_limit is not None else None
        self._shortlists: dict[tuple[int, str], list[dict[str, Any]]] = {}
        if self.candidate_limit is not None:
            for level in range(1, 6):
                for task in TASKS:
                    candidates = self._eligible(level)
                    candidates.sort(
                        key=lambda model: (
                            -self._specialty_score(model, task),
                            float(model.get("relative_cost", 1.0)),
                        )
                    )
                    self._shortlists[(level, task)] = candidates[: self.candidate_limit]

    @staticmethod
    def _clamp_level(level: int) -> int:
        return max(1, min(5, int(level)))

    @staticmethod
    def _specialty_score(model: Mapping[str, Any], task: str) -> float:
        specialties = model.get("specialties", {})
        return float(specialties.get(task, specialties.get("general", 0.2)))

    @classmethod
    def _capability_score(cls, model: Mapping[str, Any], capability: str) -> float:
        explicit = model.get("capabilities", {})
        if isinstance(explicit, dict) and capability in explicit:
            return float(explicit[capability])
        specialties = CAPABILITY_SPECIALTIES.get(capability, ("general",))
        return max(cls._specialty_score(model, task) for task in specialties)

    def _eligible(self, difficulty_hint: int) -> list[dict[str, Any]]:
        """Return models that can serve this difficulty, with a safe fallback.

        Tier-1 models are eligible for trivial prompts so the router can
        actually realize cost savings. If no model can serve the requested
        level, fall back to the full catalog.
        """
        target = self._clamp_level(difficulty_hint)
        eligible = [
            model for model in self.models
            if int(model.get("min_level", 1)) <= target
        ]
        if eligible:
            return eligible
        return list(self.models)

    def rank_direct(
        self,
        task: str,
        difficulty_hint: int = 3,
        *,
        confidence: float = 1.0,
        safety: bool = False,
        task_scores: Mapping[str, float] | None = None,
        capability_scores: Mapping[str, float] | None = None,
        risk: str = "low",
    ) -> list[dict[str, Any]]:
        """Rank concrete model IDs for a prompt.

        ``difficulty_hint`` is an internal budget signal only.  The public
        decision is the concrete model.  A task probability mixture makes the
        policy less brittle when the classifier is uncertain between, for
        example, coding and backend work.
        """
        primary_task = normalize_task(task)
        mixture = {
            normalize_task(name): max(0.0, float(weight))
            for name, weight in (task_scores or {primary_task: 1.0}).items()
        }
        total = sum(mixture.values()) or 1.0
        mixture = {name: weight / total for name, weight in mixture.items()}
        target = self._clamp_level(difficulty_hint)
        # The policy ranks models for the tier requested by the router.  Any
        # safety or confidence escalation is the responsibility of the caller
        # (e.g. TransformerRouter), so we avoid double-bumping here.

        candidates = self._eligible(target)
        if self.candidate_limit is not None:
            # The learned task distribution is usually sharp. A cached union
            # of the strongest per-task candidates retains provider diversity
            # while avoiding a 450-model full-catalog scan on every request.
            active_tasks = sorted(mixture, key=mixture.get, reverse=True)[:3]
            capability_tasks = {
                specialty
                for capability, score in (capability_scores or {}).items()
                if float(score) >= 0.45
                for specialty in CAPABILITY_SPECIALTIES.get(capability, ())
            }
            pool: dict[str, dict[str, Any]] = {}
            for name in (*active_tasks, *sorted(capability_tasks)):
                for model in self._shortlists.get((target, normalize_task(name)), []):
                    pool[str(model.get("id"))] = model
            candidates = list(pool.values()) or candidates[: self.candidate_limit]

        ranked: list[tuple[float, dict[str, Any]]] = []
        for model in candidates:
            min_level = int(model.get("min_level", 1))
            ideal_level = int(model.get("ideal_level", min_level))
            specialty_score = sum(
                weight * self._specialty_score(model, name)
                for name, weight in mixture.items()
            )
            required = {
                capability: max(0.0, float(weight))
                for capability, weight in (capability_scores or {}).items()
                if capability in CAPABILITIES and float(weight) >= 0.15
            }
            if required:
                capability_score = sum(
                    weight * self._capability_score(model, capability)
                    for capability, weight in required.items()
                ) / max(1e-6, sum(required.values()))
                quality_score = 0.72 * specialty_score + 0.28 * capability_score
            else:
                quality_score = specialty_score
            coverage = model.get("benchmark_coverage", {})
            if isinstance(coverage, dict):
                coverage_score = sum(
                    weight * float(coverage.get(name, coverage.get("general", 0.0)))
                    for name, weight in mixture.items()
                )
                quality_score *= 0.92 + 0.08 * max(0.0, min(1.0, coverage_score))
            level_fit = 1.0 - min(1.0, abs(target - ideal_level) * 0.03)
            cost = float(model.get("relative_cost", 1.0))
            # Specialty and Arena quality dominate. Cost selects among close
            # alternatives and keeps routine traffic on smaller models.
            cost_weight = 0.012 if target <= 2 else 0.006
            capability_penalty = max(0, min_level - target) * 0.04
            score = quality_score * level_fit - cost_weight * cost - capability_penalty
            ranked.append((score, {**model, "score": round(score, 5)}))
        ranked.sort(key=lambda item: (-item[0], float(item[1].get("relative_cost", 1.0))))
        return [item[1] for item in ranked]

    def rank(self, task: str, level: int) -> list[dict[str, Any]]:
        """Compatibility alias for callers that still speak in levels."""
        return self.rank_direct(task, level)

    def choose_direct(
        self,
        task: str,
        difficulty_hint: int = 3,
        *,
        confidence: float = 1.0,
        safety: bool = False,
        task_scores: Mapping[str, float] | None = None,
        capability_scores: Mapping[str, float] | None = None,
        risk: str = "low",
        max_alternatives: int = 5,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        ranked = self.rank_direct(
            task,
            difficulty_hint,
            confidence=confidence,
            safety=safety,
            task_scores=task_scores,
            capability_scores=capability_scores,
            risk=risk,
        )
        if not ranked:
            return None, []
        callable_ranked = [
            item
            for item in ranked
            if bool(item.get("callable", not str(item.get("id", "")).startswith("lmarena/")))
        ]
        selected = callable_ranked[0] if callable_ranked else ranked[0]
        alternatives: list[dict[str, Any]] = []
        seen_providers = {str(selected.get("id", "")).split("/", 1)[0]}
        remaining = [item for item in callable_ranked if item is not selected]
        for item in remaining:
            provider = str(item.get("id", "")).split("/", 1)[0]
            if provider not in seen_providers:
                alternatives.append(item)
                seen_providers.add(provider)
                if len(alternatives) >= max_alternatives:
                    return selected, alternatives
        for item in remaining:
            if item not in alternatives:
                alternatives.append(item)
                if len(alternatives) >= max_alternatives:
                    break
        if len(alternatives) < max_alternatives:
            for item in ranked:
                if item is not selected and item not in alternatives:
                    alternatives.append(item)
                    if len(alternatives) >= max_alternatives:
                        break
        return selected, alternatives

    def choose(self, task: str, level: int, *, max_alternatives: int = 5) -> tuple[str | None, list[str]]:
        selected, alternatives = self.choose_direct(task, level, max_alternatives=max_alternatives)
        if selected is None:
            return None, []
        return str(selected["id"]), [str(item["id"]) for item in alternatives]

    @staticmethod
    def validate() -> None:
        missing = set(TASKS)
        for model in ModelPolicy().models:
            missing -= set(model.get("specialties", {}))
        if missing:
            raise ValueError(f"model policy has no explicit specialty coverage: {sorted(missing)}")
