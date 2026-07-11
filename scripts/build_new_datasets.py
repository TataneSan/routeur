#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from routeur.io import read_jsonl, write_jsonl
from routeur.tasks import infer_task


def download(repo: str, filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            local_dir=cache_dir / repo.replace("/", "--"),
        )
    )


def mmr_level(weak_score: float, strong_score: float) -> int:
    """Map observed 0.6B-vs-30B quality into the five production tiers."""
    gap = strong_score - weak_score
    if weak_score >= 8 and gap <= 1:
        return 1
    if weak_score >= 7 and gap <= 2:
        return 2
    if strong_score < 6 or gap >= 6:
        return 5
    if gap >= 4 or weak_score < 4:
        return 4
    return 3


def conversation_prompt(messages: list[dict[str, Any]], turn_index: int) -> str:
    user_positions = [index for index, message in enumerate(messages) if str(message.get("role", "")).lower() == "user"]
    if turn_index >= len(user_positions):
        return ""
    end = user_positions[turn_index]
    visible = messages[: end + 1]
    if turn_index == 0:
        return str(visible[-1].get("content", "")).strip()
    return "\n".join(
        f"{str(message.get('role', 'user')).upper()}: {str(message.get('content', '')).strip()}"
        for message in visible
        if str(message.get("content", "")).strip()
    )


def build_mmr(cache_dir: Path, *, max_rows: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repo = "JiaqiXue/mmr-routing-20k"
    conversations_path = download(repo, "data/conversations.jsonl", cache_dir)
    features_path = download(repo, "data/features/qwen06b_20k.jsonl", cache_dir)
    conversations = {row["conversation_hash"]: row for row in read_jsonl(conversations_path)}
    candidates: list[dict[str, Any]] = []
    for row in read_jsonl(features_path):
        conversation = conversations.get(row.get("conversation_hash"))
        if not conversation:
            continue
        turn_index = int(row.get("turn_idx", 0))
        prompt = conversation_prompt(conversation.get("conversation") or [], turn_index)
        if len(prompt) < 12:
            continue
        weak_score = float(row.get("weak_score", 0.0))
        strong_score = float(row.get("strong_score", 0.0))
        level = mmr_level(weak_score, strong_score)
        task = infer_task(prompt)
        candidates.append(
            {
                "prompt": prompt,
                "level": level,
                "task": task,
                "source": "hf:JiaqiXue/mmr-routing-20k",
                "metadata": {
                    "conversation_hash": row.get("conversation_hash"),
                    "turn_index": turn_index,
                    "language": conversation.get("language"),
                    "weak_model": "Qwen3-0.6B",
                    "strong_model": "Qwen3-30B-A3B-Instruct-2507",
                    "weak_score": weak_score,
                    "strong_score": strong_score,
                    "score_gap": strong_score - weak_score,
                    "label_method": "observed_model_quality_gap_v1",
                },
            }
        )
    rng = random.Random(seed)
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_level[int(row["level"])].append(row)
    for rows in by_level.values():
        rng.shuffle(rows)
    # Preserve the natural distribution but cap any one tier at 45% so the
    # observed model comparisons cannot swamp teacher-gold supervision.
    selected: list[dict[str, Any]] = []
    cap = max(1, int(max_rows * 0.45))
    for level in sorted(by_level):
        selected.extend(by_level[level][:cap])
    rng.shuffle(selected)
    selected = selected[:max_rows]
    selected_keys = {" ".join(str(row["prompt"]).lower().split()) for row in selected}
    holdout = [
        row for row in candidates
        if " ".join(str(row["prompt"]).lower().split()) not in selected_keys
    ]
    rng.shuffle(holdout)
    return selected, holdout


def build_twinrouterbench(cache_dir: Path) -> list[dict[str, Any]]:
    path = download("Amorph/TwinRouterBench", "question_bank.jsonl", cache_dir)
    rows: list[dict[str, Any]] = []
    for item in read_jsonl(path):
        messages = item.get("messages") or []
        prompt = "\n".join(
            f"{str(message.get('role', 'user')).upper()}: {str(message.get('content', '')).strip()}"
            for message in messages
            if str(message.get("content", "")).strip()
        )
        tier_id = int(item.get("target_tier_id", 1))
        rows.append(
            {
                "prompt": prompt,
                "level": {0: 2, 1: 3, 2: 4, 3: 5}.get(tier_id, 3),
                "task": "agentic",
                "source": "benchmark:Amorph/TwinRouterBench",
                "metadata": {
                    "id": item.get("id"),
                    "benchmark": item.get("benchmark"),
                    "scenario": item.get("scenario"),
                    "target_tier": item.get("target_tier"),
                    "target_tier_id": tier_id,
                    "weak_label": True,
                },
            }
        )
    return rows


def build_coderouter_ood(cache_dir: Path) -> list[dict[str, Any]]:
    repo = "Lance1573/CodeRouterBench"
    tasks_path = download(repo, "ood176_tasks.jsonl", cache_dir)
    results_path = download(repo, "ood176_results_long.csv", cache_dir)
    models_path = download(repo, "models.json", cache_dir)
    model_data = json.loads(models_path.read_text(encoding="utf-8"))["models"]
    tier_by_model = {item["model"]: item["tier"] for item in model_data}
    level_by_tier = {"low": 2, "mid": 3, "high": 4, "premium": 5}
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            results[str(row["task_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for task in read_jsonl(tasks_path):
        outcomes = results.get(str(task["task_id"]), [])
        successful = [
            row for row in outcomes
            if str(row.get("resolved", "")).strip().lower() in {"1", "true", "yes", "pass"}
        ]
        if successful:
            chosen = min(successful, key=lambda row: float(row.get("cost_usd") or float("inf")))
            level = level_by_tier.get(tier_by_model.get(chosen.get("model"), "high"), 4)
        else:
            chosen = None
            level = 5
        prompt = str(task.get("prompt", "")).strip()
        if not prompt:
            continue
        rows.append(
            {
                "prompt": prompt,
                "level": level,
                "task": "coding",
                "source": "benchmark:Lance1573/CodeRouterBench:ood176",
                "metadata": {
                    "task_id": task.get("task_id"),
                    "dimension": task.get("dimension"),
                    "difficulty": task.get("difficulty"),
                    "oracle_model": chosen.get("model") if chosen else None,
                    "oracle_cost_usd": float(chosen.get("cost_usd") or 0.0) if chosen else None,
                },
            }
        )
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = " ".join(str(row["prompt"]).lower().split())
        unique.setdefault(key, row)
    return list(unique.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build recent public routing train and benchmark data.")
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/routeur-datasets"))
    parser.add_argument("--max-mmr-rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_dataset)
    mmr_rows, mmr_holdout_candidates = build_mmr(
        args.cache_dir, max_rows=args.max_mmr_rows, seed=args.seed
    )
    merged = deduplicate([*base_rows, *mmr_rows])
    training_keys = {" ".join(str(row["prompt"]).lower().split()) for row in merged}
    mmr_holdout = [
        row for row in mmr_holdout_candidates
        if " ".join(str(row["prompt"]).lower().split()) not in training_keys
    ][:3000]
    write_jsonl(args.output_dataset, merged)
    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    twin_rows = build_twinrouterbench(args.cache_dir)
    code_rows = build_coderouter_ood(args.cache_dir)
    write_jsonl(args.benchmark_dir / "twinrouterbench.jsonl", twin_rows)
    write_jsonl(args.benchmark_dir / "coderouterbench_ood176.jsonl", code_rows)
    write_jsonl(args.benchmark_dir / "mmr_holdout.jsonl", mmr_holdout)
    report = {
        "base_rows": len(base_rows),
        "mmr_rows": len(mmr_rows),
        "merged_rows": len(merged),
        "mmr_levels": dict(sorted(Counter(row["level"] for row in mmr_rows).items())),
        "mmr_holdout_rows": len(mmr_holdout),
        "twinrouterbench_rows": len(twin_rows),
        "coderouterbench_ood_rows": len(code_rows),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
