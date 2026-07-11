from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routeur.io import write_jsonl
from routeur.synthetic import TASK_TEMPLATES, generate_specialty_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Build balanced synthetic prompts for selected rare tasks.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--per-task", type=int, default=250)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    tasks = {value.strip() for value in args.tasks.split(",") if value.strip()}
    unknown = tasks - set(TASK_TEMPLATES)
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")
    generated = generate_specialty_examples(args.per_task * len(TASK_TEMPLATES), seed=args.seed)
    counts = {task: 0 for task in tasks}
    selected = []
    for row in generated:
        task = str(row["task"])
        if task in tasks and counts[task] < args.per_task:
            selected.append(row)
            counts[task] += 1
    write_jsonl(args.output, selected)
    print(f"wrote {len(selected)} prompts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
