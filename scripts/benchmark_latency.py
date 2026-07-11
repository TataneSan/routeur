#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from routeur.router import load_router


PROMPTS = (
    "Translate this sentence into French: The release is ready.",
    "Summarize the following customer note in one sentence and preserve all dates.",
    "Debug this Python asyncio timeout and propose a production-safe patch.",
    "Prove that there are infinitely many prime numbers using a contradiction.",
    "Review this OAuth callback for account-takeover vulnerabilities.",
    "Create a responsive React pricing card with accessible keyboard interactions.",
    "Search the web for the latest stable PostgreSQL release and cite the official notes.",
    "Analyze the attached screenshot and extract every visible invoice line.",
)


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure uncached single-prompt router latency.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-p95-ms", type=float, default=10.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    router = load_router(args.model_dir)
    warmups = [f"warmup-{index}: {PROMPTS[index % len(PROMPTS)]}" for index in range(args.warmup)]
    for prompt in warmups:
        router.route(prompt)
    timings: list[float] = []
    for index in range(args.iterations):
        prompt = f"request-{index}: {PROMPTS[index % len(PROMPTS)]}"
        started = time.perf_counter_ns()
        router.route(prompt)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    result = {
        "iterations": len(timings),
        "mean_ms": statistics.mean(timings),
        "p50_ms": percentile(timings, 50),
        "p95_ms": percentile(timings, 95),
        "p99_ms": percentile(timings, 99),
        "max_ms": max(timings),
        "target_p95_ms": args.max_p95_ms,
        "passed": percentile(timings, 95) < args.max_p95_ms,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] or args.no_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
