from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .tasks import difficulty_to_level, infer_task


def _first_text(row: dict[str, Any]) -> str:
    for key in ("prompt", "text", "instruction", "input", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = row.get("messages") or row.get("conversations") or row.get("conversation")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", message.get("from", ""))).lower()
            value = message.get("content", message.get("value", ""))
            if role in {"user", "human", "prompter"} and isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()


def _row(prompt: str, level: int, task: str, source: str, **metadata: Any) -> dict[str, Any] | None:
    prompt = " ".join(prompt.split()).strip()
    if len(prompt) < 12 or len(prompt) > 24000:
        return None
    return {
        "prompt": prompt,
        "level": max(1, min(5, int(level))),
        "task": task,
        "source": source,
        "metadata": metadata,
    }


def build_hf_examples(
    *,
    max_per_source: int = 10000,
    seed: int = 42,
    include_oasst: bool = True,
    include_wildchat: bool = False,
    include_router_data: bool = True,
) -> list[dict[str, Any]]:
    """Download a compact, deterministic, weakly-labelled HF mixture.

    The difficulty dataset provides the level label. No Robots provides human
    task categories. OASST adds multilingual and conversational coverage; its
    level is deliberately a conservative proxy and should not replace traces
    graded against the actual production models.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []

    difficulty = load_dataset(
        "agentlans/prompt-difficulty-model-ratings",
        split="train",
        streaming=True,
    )
    difficulty = difficulty.shuffle(seed=seed, buffer_size=min(10000, max_per_source * 2))
    for idx, item in enumerate(difficulty):
        prompt = _first_text(dict(item))
        row = _row(
            prompt,
            difficulty_to_level(item.get("label")),
            infer_task(prompt),
            "hf:agentlans/prompt-difficulty-model-ratings",
            difficulty_label=item.get("label"),
        )
        if row:
            candidates.append(row)
        if idx + 1 >= max_per_source * 2:
            break

    no_robots = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
    no_robots = no_robots.shuffle(seed=seed + 1, buffer_size=2000)
    for idx, item in enumerate(no_robots):
        item = dict(item)
        prompt = _first_text(item)
        category = str(item.get("category", ""))
        category_level = {
            "classification": 1,
            "extraction": 1,
            "summarization": 2,
            "rewriting": 2,
            "creation": 2,
            "brainstorming": 3,
            "open questions": 3,
            "programming": 4,
        }.get(category.lower(), 2)
        row = _row(
            prompt,
            category_level,
            infer_task(prompt, category=category),
            "hf:HuggingFaceH4/no_robots",
            category=category,
        )
        if row:
            candidates.append(row)
        if idx + 1 >= max_per_source:
            break

    if include_oasst:
        oasst = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        oasst = oasst.shuffle(seed=seed + 2, buffer_size=10000)
        for idx, item in enumerate(oasst):
            item = dict(item)
            if str(item.get("role", "")).lower() not in {"prompter", "user", "human"}:
                continue
            prompt = _first_text(item)
            words = len(prompt.split())
            task = infer_task(prompt)
            level = 3
            if words < 24 and task in {"general", "writing"}:
                level = 1 if words < 12 else 2
            elif task in {"reasoning", "safety", "research"} or words > 180:
                level = 4
            row = _row(prompt, level, task, "hf:OpenAssistant/oasst1", language=item.get("lang"))
            if row:
                candidates.append(row)
            if idx + 1 >= max_per_source:
                break

    if include_wildchat:
        wildchat = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        wildchat = wildchat.shuffle(seed=seed + 3, buffer_size=20000)
        for idx, item in enumerate(wildchat):
            item = dict(item)
            prompt = _first_text(item)
            if not prompt:
                continue
            task = infer_task(prompt)
            turns = int(item.get("turn", 1) or 1)
            words = len(prompt.split())
            level = 2
            if task in {"coding", "frontend", "backend", "data"} or turns >= 3 or words > 180:
                level = 3
            if task in {"reasoning", "research", "safety"} or words > 500:
                level = 4
            row = _row(
                prompt,
                level,
                task,
                "hf:allenai/WildChat-1M",
                language=item.get("language"),
                turns=turns,
            )
            if row:
                candidates.append(row)
            if idx + 1 >= max_per_source:
                break

    if include_router_data:
        supra = load_dataset("SupraLabs/Prompt-Routing-Dataset", split="train", streaming=True)
        for idx, item in enumerate(supra):
            item = dict(item)
            prompt = _first_text(item)
            coding_task = str(item.get("coding_task", "false")).lower() in {"1", "true", "yes"}
            math_task = str(item.get("math_task", "false")).lower() in {"1", "true", "yes"}
            reasoning_task = str(item.get("requires_reasoning", "false")).lower() in {"1", "true", "yes"}
            if coding_task:
                task = infer_task(prompt, category="programming")
            elif math_task or reasoning_task:
                task = "reasoning"
            else:
                task = infer_task(prompt, category=str(item.get("primary_domain", "")))
            row = _row(
                prompt,
                int(item.get("complexity_score", 3)),
                task,
                "hf:SupraLabs/Prompt-Routing-Dataset",
                routing_choice=item.get("routing_choice"),
                domain=item.get("primary_domain"),
            )
            if row:
                candidates.append(row)
            if idx + 1 >= max_per_source:
                break

        prompt_router = load_dataset("somukandula/prompt-router-dataset", split="train", streaming=True)
        prompt_router = prompt_router.shuffle(seed=seed + 4, buffer_size=2000)
        for idx, item in enumerate(prompt_router):
            item = dict(item)
            prompt = _first_text(item)
            label = str(item.get("label", ""))
            if label == "vision_model":
                task, level = "vision", 3
            elif label == "code_model":
                inferred = infer_task(prompt, category="programming")
                task, level = inferred, 3
            elif label == "strong_general":
                task, level = infer_task(prompt), 4
            else:
                task, level = infer_task(prompt), 1
            row = _row(
                prompt,
                level,
                task,
                "hf:somukandula/prompt-router-dataset",
                routing_label=label,
            )
            if row:
                candidates.append(row)
            if idx + 1 >= max_per_source:
                break

        complexity = load_dataset("anasnassar/llm-query-complexity-benchmark", split="train", streaming=True)
        complexity = complexity.shuffle(seed=seed + 5, buffer_size=5000)
        for idx, item in enumerate(complexity):
            item = dict(item)
            prompt = _first_text(item)
            ground_truth = str(item.get("ground_truth", "MEDIUM")).upper()
            level = {"LOW": 2, "MEDIUM": 3, "HIGH": 4}.get(ground_truth, 3)
            task = infer_task(prompt, category=str(item.get("domain", "")))
            if task == "safety" and ground_truth == "HIGH":
                level = 5
            row = _row(
                prompt,
                level,
                task,
                "hf:anasnassar/llm-query-complexity-benchmark",
                complexity=ground_truth,
                domain=item.get("domain"),
                subject=item.get("subject"),
            )
            if row:
                candidates.append(row)
            if idx + 1 >= max_per_source:
                break

        routellm = load_dataset("routellm/gpt4_dataset", split="train", streaming=True)
        routellm = routellm.shuffle(seed=seed + 6, buffer_size=10000)
        for idx, item in enumerate(routellm):
            item = dict(item)
            prompt = _first_text(item)
            try:
                score = int(item.get("mixtral_score", 3))
            except (TypeError, ValueError):
                score = 3
            level = 3 if score >= 4 else (4 if score == 3 else 5)
            row = _row(
                prompt,
                level,
                infer_task(prompt),
                "hf:routellm/gpt4_dataset",
                mixtral_score=score,
            )
            if row:
                candidates.append(row)
            if idx + 1 >= max_per_source:
                break

        model_config_path = Path(__file__).resolve().parent.parent / "configs" / "models_lmarena.json"
        if model_config_path.exists():
            model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
            feature_lookup: dict[str, list[float]] = {}
            from .tasks import TASKS

            for model in model_config.get("models", []):
                features = [float(model.get("specialties", {}).get(task, 0.0)) for task in TASKS]
                names = {
                    str(model.get("display_name", "")).lower(),
                    str(model.get("id", "")).split("/", 1)[-1].lower(),
                }
                for name in names:
                    if name:
                        feature_lookup[name] = features
            preferences = load_dataset("lmarena-ai/arena-human-preference-55k", split="train", streaming=True)
            preferences = preferences.shuffle(seed=seed + 7, buffer_size=10000)
            accepted = 0
            for item in preferences:
                item = dict(item)
                if int(item.get("winner_tie", 0) or 0):
                    continue
                model_a = str(item.get("model_a", "")).lower()
                model_b = str(item.get("model_b", "")).lower()
                features_a = feature_lookup.get(model_a)
                features_b = feature_lookup.get(model_b)
                if features_a is None or features_b is None:
                    continue
                if max(abs(a - b) for a, b in zip(features_a, features_b, strict=True)) < 0.01:
                    continue
                try:
                    turns = json.loads(str(item.get("prompt", "[]")))
                except json.JSONDecodeError:
                    turns = [str(item.get("prompt", ""))]
                prompt = str(turns[-1] if isinstance(turns, list) and turns else "").strip()
                winner_a = int(item.get("winner_model_a", 0) or 0) == 1
                winner = features_a if winner_a else features_b
                loser = features_b if winner_a else features_a
                task = infer_task(prompt)
                words = len(prompt.split())
                level = 2 if words < 30 and task in {"general", "writing"} else 3
                if task in {"reasoning", "research", "safety"} or words > 180:
                    level = 4
                row = _row(
                    prompt,
                    level,
                    task,
                    "hf:lmarena-ai/arena-human-preference-55k",
                    preference_winner=winner,
                    preference_loser=loser,
                    winner_model=model_a if winner_a else model_b,
                    loser_model=model_b if winner_a else model_a,
                )
                if row:
                    candidates.append(row)
                    accepted += 1
                if accepted >= max_per_source:
                    break

    # Keep the mixture balanced by level and task, then deduplicate prompts.
    buckets: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[(int(row["level"]), str(row["task"]), str(row["source"]))].append(row)
    balanced: list[dict[str, Any]] = []
    per_bucket = max(50, max_per_source // 25)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        balanced.extend(bucket_rows[:per_bucket])
    rng.shuffle(balanced)

    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in balanced:
        digest = _hash_prompt(str(row["prompt"]))
        if digest in seen:
            continue
        seen.add(digest)
        output.append(row)
    return output


def merge_examples(*collections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for collection in collections:
        for row in collection:
            prompt = str(row.get("prompt", "")).strip()
            digest = _hash_prompt(prompt)
            if prompt and digest not in seen:
                seen.add(digest)
                merged.append(dict(row))
    return merged
