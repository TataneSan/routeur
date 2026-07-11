#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from routeur.capabilities import infer_capabilities
from routeur.objective_oracle import (
    bootstrap_level_stability,
    candidate_estimates,
    choose_cheapest_sufficient,
    cost_rank_levels,
)
from routeur.tasks import infer_task


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def build_labels(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    z_value: float,
    fixed_model_costs: dict[str, float] | None = None,
    bootstrap_samples: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    costs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_query[(str(row["dataset"]), str(row["query_id"]))].append(row)
        if bool(row.get("parse_success", True)):
            costs[str(row["model"])].append(float(row["cost"]))
    model_costs = fixed_model_costs or {model: statistics.median(values) for model, values in costs.items()}
    levels = cost_rank_levels(model_costs)
    labels: list[dict[str, Any]] = []
    sufficient_count = 0
    for (dataset, query_id), query_rows in sorted(per_query.items()):
        estimates = candidate_estimates(query_rows, z_value=z_value)
        selected, sufficient = choose_cheapest_sufficient(
            estimates,
            quality_threshold=threshold,
            model_costs=model_costs,
        )
        sufficient_count += int(sufficient)
        stable_seed = int.from_bytes(hashlib.sha256(f"{dataset}:{query_id}".encode()).digest()[:8], "big")
        label_stability = bootstrap_level_stability(
            query_rows,
            selected_level=levels[selected.model],
            quality_threshold=threshold,
            model_costs=model_costs,
            levels=levels,
            z_value=z_value,
            iterations=bootstrap_samples,
            seed=stable_seed,
        )
        prompt_variants = sorted(
            {
                str(row.get("input_question") or row.get("original_question") or "").strip()
                for row in query_rows
                if str(row.get("input_question") or row.get("original_question") or "").strip()
            }
        )
        for variant_index, prompt in enumerate(prompt_variants):
            task = infer_task(prompt)
            required_capabilities = infer_capabilities(prompt, task)
            confidence = (
                0.95
                if sufficient and selected.lower_confidence_bound >= threshold + 0.05
                else (0.8 if sufficient else 0.65)
            )
            labels.append({
                "prompt": prompt,
                "level": levels[selected.model],
                "task": task,
                "source": f"dars_{dataset}_objective_oracle",
                "metadata": {
                    "dataset": dataset,
                    "grader": "dars_repeated_trace_oracle",
                    "grader_confidence": confidence,
                    "query_id": query_id,
                    "prompt_variant_index": variant_index,
                    "prompt_variants": len(prompt_variants),
                    "oracle_model": selected.model,
                    "oracle_model_cost": model_costs[selected.model],
                    "expected_score": selected.mean_score,
                    "score_std": selected.score_std,
                    "lower_confidence_bound": selected.lower_confidence_bound,
                    "oracle_level_stability": label_stability,
                    "observations": selected.observations,
                    "quality_threshold": threshold,
                    "statistically_sufficient": sufficient,
                    "candidate_estimates": [item.__dict__ for item in estimates],
                    "required_capabilities": required_capabilities,
                },
            })
    summary = {
        "queries": len(per_query),
        "prompt_variants": len(labels),
        "statistically_sufficient_queries": sufficient_count,
        "quality_threshold": threshold,
        "z_value": z_value,
        "model_median_costs": dict(sorted(model_costs.items(), key=lambda item: item[1])),
        "model_cost_levels": levels,
    }
    return labels, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build repeated-trace, cost-aware DARS oracle labels.")
    parser.add_argument("--input", type=Path, required=True, help="DARS snapshot directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-threshold", type=float, default=0.8)
    parser.add_argument("--z-value", type=float, default=1.645, help="One-sided confidence multiplier.")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--min-stability", type=float, default=0.998)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        paths = sorted(args.input.glob(f"*/{split}_scored_generations.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No DARS {split} files below {args.input}")
        split_rows[split] = read_rows(paths)
    all_costs: dict[str, list[float]] = defaultdict(list)
    for rows in split_rows.values():
        for row in rows:
            if bool(row.get("parse_success", True)):
                all_costs[str(row["model"])].append(float(row["cost"]))
    fixed_model_costs = {model: statistics.median(values) for model, values in all_costs.items()}
    combined_summary: dict[str, Any] = {}
    for split, rows in split_rows.items():
        labels, summary = build_labels(
            rows,
            threshold=args.quality_threshold,
            z_value=args.z_value,
            fixed_model_costs=fixed_model_costs,
            bootstrap_samples=args.bootstrap_samples,
        )
        output = args.output_dir / f"dars_objective_{split}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for label in labels:
                handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
        stable = [row for row in labels if float(row["metadata"]["oracle_level_stability"]) >= args.min_stability]
        stable_output = args.output_dir / f"dars_objective_{split}_stable.jsonl"
        with stable_output.open("w", encoding="utf-8") as handle:
            for label in stable:
                handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
        summary["stable_prompt_variants"] = len(stable)
        summary["min_stability"] = args.min_stability
        combined_summary[split] = summary
        print(f"wrote {len(labels)} {split} labels ({len(stable)} stable) to {output}")
    (args.output_dir / "dars_objective_summary.json").write_text(
        json.dumps(combined_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
