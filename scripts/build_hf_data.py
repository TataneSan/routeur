from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routeur.hf_data import build_hf_examples
from routeur.io import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the weak/specialized Hugging Face router mixture.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-source", type=int, default=15000)
    parser.add_argument("--include-wildchat", action="store_true")
    parser.add_argument("--without-router-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = build_hf_examples(
        max_per_source=args.max_per_source,
        seed=args.seed,
        include_oasst=True,
        include_wildchat=args.include_wildchat,
        include_router_data=not args.without_router_data,
    )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
