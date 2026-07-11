from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routeur.io import read_jsonl, write_jsonl


def _digest(prompt: str) -> str:
    return hashlib.sha256(" ".join(prompt.lower().split()).encode("utf-8")).hexdigest()


def select(rows: list[dict[str, Any]], *, count: int, excluded: set[str], seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set(excluded)
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        digest = _digest(prompt)
        if not prompt or digest in seen:
            continue
        seen.add(digest)
        source_family = str(row.get("source", "unknown")).split(":", 1)[-1]
        buckets[(int(row.get("level", 3)), str(row.get("task", "general")), source_family)].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    keys = list(buckets)
    rng.shuffle(keys)
    output: list[dict[str, Any]] = []
    while keys and len(output) < count:
        next_keys: list[tuple[int, str, str]] = []
        for key in keys:
            values = buckets[key]
            if values:
                output.append(values.pop())
                if len(output) >= count:
                    break
            if values:
                next_keys.append(key)
        keys = next_keys
        rng.shuffle(keys)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a diverse balanced sample for LLM annotation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument("--tasks", default=None, help="Optional comma-separated task allowlist.")
    args = parser.parse_args()
    excluded = {
        _digest(str(row.get("prompt", "")))
        for path in args.exclude
        if path.exists()
        for row in read_jsonl(path)
    }
    allowed_tasks = {value.strip() for value in args.tasks.split(",")} if args.tasks else None
    candidates = [
        row for row in read_jsonl(args.input)
        if int(row.get("level", 3)) >= args.min_level
        and (allowed_tasks is None or str(row.get("task", "general")) in allowed_tasks)
    ]
    rows = select(candidates, count=args.count, excluded=excluded, seed=args.seed)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} diverse prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
