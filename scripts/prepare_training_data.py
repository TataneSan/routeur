from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routeur.io import read_jsonl, write_jsonl
from routeur.synthetic import generate_specialty_examples
from routeur.hf_data import merge_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge HF, GLM-labelled and synthetic router data.")
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--annotated", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-count", type=int, default=12000)
    parser.add_argument("--max-per-bucket", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotated = [
        row
        for path in args.annotated
        if path.exists()
        for row in read_jsonl(path)
    ]
    weak = read_jsonl(args.weak)
    synthetic = generate_specialty_examples(args.synthetic_count, seed=args.seed)
    rows = merge_examples(annotated, synthetic, weak)
    buckets: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(int(row["level"]), str(row.get("task", "general")))].append(row)
    rng = random.Random(args.seed)
    balanced: list[dict[str, object]] = []
    for bucket in buckets.values():
        gold = [row for row in bucket if (row.get("metadata") or {}).get("grader")]
        strong_weak = [
            row for row in bucket
            if not (row.get("metadata") or {}).get("grader")
            and str(row.get("source", "")).startswith(
                ("hf:agentlans/", "hf:SupraLabs/", "hf:anasnassar/", "hf:somukandula/", "hf:routellm/")
            )
        ]
        remaining = [row for row in bucket if row not in gold and row not in strong_weak]
        rng.shuffle(gold)
        rng.shuffle(strong_weak)
        rng.shuffle(remaining)
        # Never discard teacher-labelled examples. Fill the remaining budget
        # with the strongest weak labels, then the noisier mixture.
        budget = max(args.max_per_bucket, len(gold))
        prioritized = gold + strong_weak + remaining
        balanced.extend(prioritized[:budget])
    rng.shuffle(balanced)
    write_jsonl(args.output, balanced)
    counts = defaultdict(int)
    for row in balanced:
        counts[f"{row['level']}:{row.get('task', 'general')}"] += 1
    teacher_rows = sum(bool((row.get("metadata") or {}).get("grader")) for row in balanced)
    print(json.dumps({"rows": len(balanced), "teacher_rows": teacher_rows, "buckets": dict(sorted(counts.items()))}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
