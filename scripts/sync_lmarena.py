from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

CONFIGS = ("text", "webdev", "agent", "search", "vision")
SPECIALTY_CATEGORIES = {
    "general": ((("text", "overall"), 1.0),),
    "frontend": (
        (("webdev", "overall"), 0.35),
        (("webdev", "webdev-react"), 0.30),
        (("webdev", "webdev-html"), 0.20),
        (("webdev", "image_to_webdev"), 0.15),
    ),
    "backend": (
        (("text", "coding"), 0.35),
        (("text", "industry_software_and_it_services"), 0.30),
        (("agent", "overall"), 0.35),
    ),
    "coding": (
        (("text", "coding"), 0.55),
        (("agent", "overall"), 0.25),
        (("webdev", "overall"), 0.20),
    ),
    "reasoning": (
        (("text", "math"), 0.35),
        (("text", "expert"), 0.25),
        (("text", "hard_prompts"), 0.20),
        (("text", "industry_mathematical"), 0.20),
    ),
    "research": ((("search", "overall"), 0.65), (("text", "expert"), 0.35)),
    "writing": (
        (("text", "creative_writing"), 0.55),
        (("text", "industry_writing_and_literature_and_language"), 0.25),
        (("text", "overall"), 0.20),
    ),
    "data": (
        (("text", "industry_software_and_it_services"), 0.30),
        (("text", "industry_mathematical"), 0.25),
        (("text", "math"), 0.20),
        (("text", "coding"), 0.25),
    ),
    "safety": (
        (("text", "industry_legal_and_government"), 0.30),
        (("text", "industry_medicine_and_healthcare"), 0.30),
        (("text", "expert"), 0.20),
        (("text", "hard_prompts"), 0.20),
    ),
    "vision": (
        (("vision", "overall"), 0.40),
        (("vision", "ocr"), 0.20),
        (("vision", "diagram"), 0.25),
        (("webdev", "image_to_webdev"), 0.15),
    ),
    "agentic": (
        (("agent", "overall"), 0.60),
        (("text", "coding"), 0.25),
        (("search", "overall"), 0.15),
    ),
}

CAPABILITY_SPECIALTIES = {
    "vision": ("vision",),
    "coding": ("coding", "backend", "frontend", "data"),
    "reasoning": ("reasoning", "data"),
    "web_search": ("research",),
    "tool_use": ("agentic", "backend"),
    "long_context": ("research", "general"),
    "multilingual": ("general", "writing"),
    "creative": ("writing",),
    "safety": ("safety", "reasoning"),
}


def _slug(value: str) -> str:
    value = value.lower().replace("(", "-").replace(")", "")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value


ORGANIZATION_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "alibaba": "qwen",
    "zhipu-ai": "zai",
    "z-ai": "zai",
    "x-ai": "xai",
    "deepseek": "deepseek",
    "moonshot-ai": "moonshot",
    "moonshot": "moonshot",
    "baidu": "baidu",
    "bytedance": "bytedance",
    "amazon": "amazon",
    "nvidia": "nvidia",
    "mistral": "mistral",
    "tencent": "tencent",
    "stepfun": "stepfun",
    "cohere": "cohere",
    "meta": "meta",
    "allenai": "allenai",
    "ibm": "ibm",
    "microsoft": "microsoft",
    "ant-group": "antgroup",
    "meituan": "meituan",
    "inception-ai": "inception",
    "poolside": "poolside",
}


def api_model_id(display_name: str, organization: str | None = None) -> str:
    """Stable provider-neutral IDs; providers can override these aliases later."""
    name = display_name.strip()
    slug = _slug(name)
    # Arena annotates some harness/configuration variants that are not part of
    # the provider model ID. Keep the display name in arena_scores, but route
    # through the stable API family name.
    if slug.startswith("glm-") and slug.endswith("-max"):
        slug = slug.removesuffix("-max")
    slug = slug.removesuffix("-codex-harness")
    if slug.startswith("claude-"):
        return f"anthropic/{slug}"
    if slug.startswith("gpt-"):
        return f"openai/{slug}"
    if slug.startswith("gemini-"):
        return f"google/{slug}"
    if slug.startswith("qwen"):
        return f"qwen/{slug}"
    if slug.startswith("glm-"):
        return f"zai/{slug}"
    if slug.startswith("grok-"):
        return f"xai/{slug}"
    if slug.startswith("mimo-"):
        return f"mimo/{slug}"
    if slug.startswith("minimax-"):
        return f"minimax/{slug}"
    if slug.startswith("deepseek-"):
        return f"deepseek/{slug}"
    if slug.startswith("kimi-"):
        return f"moonshot/{slug}"
    organization_slug = _slug(str(organization or ""))
    provider = ORGANIZATION_PROVIDERS.get(organization_slug)
    if provider:
        return f"{provider}/{slug}"
    return f"lmarena/{slug}"


def _rank_score(row: dict[str, Any], category_size: int) -> float:
    try:
        rank = float(row["rank"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, 1.0 - (rank - 1.0) / max(1.0, category_size - 1.0))


def _cost_prior(name: str) -> float:
    lower = name.lower()
    if any(token in lower for token in ("mini", "flash", "haiku", "instant", "small", "8b", "11b", "27b", "32b", "35b")):
        return 0.7
    if any(token in lower for token in ("opus", "thinking", "max", "high", "o3", "o4", "pro-preview")):
        return 4.5
    if any(token in lower for token in ("sonnet", "pro", "large", "70b", "72b", "122b")):
        return 2.0
    return 1.2


def sync(*, output: Path, snapshot_output: Path, max_models: int) -> dict[str, Any]:
    from datasets import load_dataset

    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_snapshot: list[dict[str, Any]] = []
    category_sizes: dict[tuple[str, str], int] = {}
    for config in CONFIGS:
        dataset = load_dataset("lmarena-ai/leaderboard-dataset", config, split="latest")
        rows = [dict(row) for row in dataset]
        for category in {str(row.get("category", "overall")) for row in rows}:
            category_sizes[(config, category)] = sum(1 for row in rows if row.get("category") == category)
        for row in rows:
            name = str(row.get("model_name", "")).strip()
            category = str(row.get("category", "overall"))
            if not name:
                continue
            record = {
                "config": config,
                "category": category,
                "model_name": name,
                "organization": row.get("organization"),
                "rating": row.get("rating", row.get("score")),
                "rating_lower": row.get("rating_lower", row.get("score_ci_lower")),
                "rating_upper": row.get("rating_upper", row.get("score_ci_upper")),
                "rank": row.get("rank"),
                "votes": row.get("vote_count", row.get("observation_count")),
                "publish_date": row.get("leaderboard_publish_date"),
                "rank_score": _rank_score(row, category_sizes[(config, category)]),
            }
            try:
                votes = max(0.0, float(record["votes"] or 0.0))
            except (TypeError, ValueError):
                votes = 0.0
            reliability = min(1.0, math.log1p(votes) / math.log1p(20000.0))
            record["reliability"] = reliability
            record["robust_rank_score"] = float(record["rank_score"]) * (0.85 + 0.15 * reliability)
            rows_by_key[(name, config, category)] = record
            raw_snapshot.append(record)

    names = sorted({name for name, _, _ in rows_by_key})
    model_entries: list[dict[str, Any]] = []
    for name in names:
        organization = str(
            next(
                (
                    row["organization"]
                    for (model, _, _), row in rows_by_key.items()
                    if model == name and row.get("organization")
                ),
                "unknown",
            )
        )
        scores: dict[str, float] = {}
        coverage: dict[str, float] = {}
        evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for specialty, weighted_categories in SPECIALTY_CATEGORIES.items():
            if isinstance(weighted_categories[0], str):
                weighted_categories = (weighted_categories,)  # type: ignore[assignment]
            total = 0.0
            weight_total = 0.0
            possible_weight = sum(float(weight) for _key, weight in weighted_categories)  # type: ignore[misc]
            for (config, category), weight in weighted_categories:  # type: ignore[misc]
                row = rows_by_key.get((name, config, category))
                if row is None:
                    continue
                total += float(row["robust_rank_score"]) * float(weight)
                weight_total += float(weight)
                evidence[specialty].append({"config": config, "category": category, "rank": row["rank"], "rating": row["rating"], "votes": row["votes"], "reliability": row["reliability"]})
            coverage[specialty] = round(weight_total / max(1e-9, possible_weight), 6)
            if weight_total:
                raw_score = total / weight_total
                scores[specialty] = round(raw_score * (0.75 + 0.25 * coverage[specialty]), 6)
            else:
                scores[specialty] = round(0.55 * scores.get("general", 0.0), 6)
        peak = max(scores.values())
        if peak >= 0.78:
            ideal_level = 5
        elif peak >= 0.52:
            ideal_level = 4
        elif peak >= 0.25:
            ideal_level = 3
        elif peak >= 0.10:
            ideal_level = 2
        else:
            ideal_level = 1
        capabilities = {
            capability: round(max(scores.get(task, 0.0) for task in tasks), 6)
            for capability, tasks in CAPABILITY_SPECIALTIES.items()
        }
        model_id = api_model_id(name, organization)
        model_entries.append(
            {
                "id": model_id,
                "display_name": name,
                "provider": organization,
                "callable": not model_id.startswith("lmarena/"),
                "enabled": True,
                "ideal_level": ideal_level,
                "min_level": max(1, ideal_level - 1),
                "relative_cost": _cost_prior(name),
                "specialties": scores,
                "benchmark_coverage": coverage,
                "capabilities": capabilities,
                "arena_scores": dict(evidence),
                "source": "lmarena-ai/leaderboard-dataset",
            }
        )

    # Several Arena rows are harness aliases of the same callable API model.
    # Merge them after ID normalization so rankings and fallback lists do not
    # contain duplicate GPT/Claude/GLM families with identical API IDs.
    merged_by_id: dict[str, dict[str, Any]] = {}
    for entry in model_entries:
        model_id = str(entry["id"])
        current = merged_by_id.get(model_id)
        if current is None:
            entry["arena_aliases"] = [str(entry["display_name"])]
            merged_by_id[model_id] = entry
            continue
        current_peak = max(float(value) for value in current["specialties"].values())
        entry_peak = max(float(value) for value in entry["specialties"].values())
        current["arena_aliases"] = sorted(
            set(current.get("arena_aliases", [])) | {str(entry["display_name"])}
        )
        if entry_peak > current_peak:
            current["display_name"] = entry["display_name"]
            current["provider"] = entry["provider"]
        for field in ("specialties", "benchmark_coverage", "capabilities"):
            for key, value in entry[field].items():
                current[field][key] = max(float(current[field].get(key, 0.0)), float(value))
        for specialty, rows in entry["arena_scores"].items():
            current["arena_scores"].setdefault(specialty, []).extend(rows)
        current["relative_cost"] = max(float(current["relative_cost"]), float(entry["relative_cost"]))
        peak = max(float(value) for value in current["specialties"].values())
        current["ideal_level"] = 5 if peak >= 0.78 else 4 if peak >= 0.52 else 3 if peak >= 0.25 else 2 if peak >= 0.10 else 1
        current["min_level"] = max(1, int(current["ideal_level"]) - 1)
    model_entries = list(merged_by_id.values())

    # Preserve specialists even when their general-chat rank is mediocre.
    # A general-only truncation silently drops strong vision/search/code models.
    per_specialty = max(8, max_models // max(1, len(SPECIALTY_CATEGORIES) * 2))
    selected_ids: set[str] = set()
    for specialty in SPECIALTY_CATEGORIES:
        specialists = sorted(
            model_entries,
            key=lambda model: (-float(model["specialties"].get(specialty, 0.0)), str(model["id"])),
        )
        selected_ids.update(str(model["id"]) for model in specialists[:per_specialty])
    prioritized = sorted(
        model_entries,
        key=lambda model: (
            str(model["id"]) not in selected_ids,
            -max(float(value) for value in model["specialties"].values()),
            -float(model["specialties"].get("general", 0.0)),
            str(model["id"]),
        ),
    )
    model_entries = prioritized[:max_models]
    payload = {
        "schema_version": 2,
        "generated_at": date.today().isoformat(),
        "source": "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset",
        "leaderboard": "Arena/LMArena",
        "models": model_entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot_output.write_text(json.dumps(raw_snapshot, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"models": len(model_entries), "scores": len(raw_snapshot), "output": str(output), "snapshot": str(snapshot_output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/models_lmarena.json"))
    parser.add_argument("--snapshot-output", type=Path, default=Path("data/lmarena_snapshot.json"))
    parser.add_argument("--max-models", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(sync(output=args.output, snapshot_output=args.snapshot_output, max_models=args.max_models), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
