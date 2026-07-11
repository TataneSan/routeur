from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .heuristics import heuristic_level
from .io import read_jsonl, write_jsonl
from .labels import label_trace_row
from .metrics import routing_metrics
from .router import load_router
from .schema import RouterExample
from .synthetic import generate_seed_examples


def cmd_label(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    labelled = [
        label_trace_row(row, min_quality=args.min_quality, max_latency_ms=args.max_latency_ms)
        for row in rows
    ]
    write_jsonl(args.output, labelled)
    print(f"wrote {len(labelled)} labelled examples to {args.output}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    router = load_router(args.model_dir, policy_path=args.policy)
    decision = router.route(args.prompt)
    print(json.dumps(decision.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    rows = [RouterExample.from_json(row) for row in read_jsonl(args.dataset)]
    if args.model_dir:
        router = load_router(args.model_dir, policy_path=args.policy)
        decisions = [router.route(row.prompt) for row in rows]
    else:
        router = load_router(None, policy_path=args.policy)
        decisions = [router.route(row.prompt) for row in rows]
    predictions = [decision.level for decision in decisions]
    metrics = routing_metrics([row.level for row in rows], predictions)
    task_correct = [decision.task == row.task for decision, row in zip(decisions, rows, strict=True)]
    metrics["task_accuracy"] = sum(task_correct) / len(task_correct)
    metrics["model_counts"] = dict(sorted({model: sum(decision.model == model for decision in decisions) for model in {decision.model for decision in decisions}}.items()))
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    rows = generate_seed_examples(args.count, seed=args.seed)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} seed examples to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routeur",
        description="Create labels, train data, evaluate and route prompts to concrete models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    label = sub.add_parser("label", help="Convert model evaluation traces to router training labels.")
    label.add_argument("--input", required=True, type=Path, help="JSONL traces with prompt and per-level results.")
    label.add_argument("--output", required=True, type=Path, help="Output JSONL with prompt and level.")
    label.add_argument("--min-quality", required=True, type=float, help="Minimum acceptable quality score.")
    label.add_argument("--max-latency-ms", type=float, default=None, help="Optional latency ceiling.")
    label.set_defaults(func=cmd_label)

    route = sub.add_parser("route", help="Route one prompt.")
    route.add_argument("--prompt", required=True)
    route.add_argument("--model-dir", type=Path, default=None, help="Trained model directory. Omit for heuristic router.")
    route.add_argument("--policy", type=Path, default=None, help="Optional model policy JSON file.")
    route.set_defaults(func=cmd_route)

    evaluate = sub.add_parser("eval", help="Evaluate router against labelled JSONL data.")
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--model-dir", type=Path, default=None, help="Trained model directory. Omit for heuristic baseline.")
    evaluate.add_argument("--policy", type=Path, default=None, help="Optional model policy JSON file.")
    evaluate.set_defaults(func=cmd_eval)

    seed = sub.add_parser("seed", help="Generate a deterministic starter dataset for smoke tests.")
    seed.add_argument("--output", required=True, type=Path)
    seed.add_argument("--count", type=int, default=200)
    seed.add_argument("--seed", type=int, default=42)
    seed.set_defaults(func=cmd_seed)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"routeur: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
