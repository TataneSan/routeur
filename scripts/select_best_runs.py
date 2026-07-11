from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.continuous_retrain import promote_artifact
from scripts.run_h100_experiments import objective


def fetch_metrics(
    run_name: str,
    destination: Path,
    *,
    wait_timeout: float = 0.0,
    poll_interval: float = 30.0,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, wait_timeout)
    command = [
        "modal",
        "volume",
        "get",
        "routeur-artifacts",
        f"/runs/{run_name}/metrics.json",
        str(destination),
        "--force",
    ]
    while True:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0 and destination.exists():
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"metrics are not available for {run_name}")
        time.sleep(max(1.0, poll_interval))
    metrics = json.loads(destination.read_text(encoding="utf-8"))
    if not metrics.get("raw_metrics"):
        progress_path = destination.with_name(destination.name.replace(".metrics.json", ".progress.jsonl"))
        progress_result = subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "routeur-artifacts",
                f"/runs/{run_name}/progress.jsonl",
                str(progress_path),
                "--force",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if progress_result.returncode == 0 and progress_path.exists():
            events: list[dict[str, Any]] = []
            for line in progress_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("status") == "epoch" and isinstance(event.get("business"), dict):
                    events.append(event)
            if events:
                target_f1 = float(metrics.get("level_macro_f1", 0.0))
                target_task = float(metrics.get("task_accuracy", 0.0))
                best_event = min(
                    events,
                    key=lambda event: abs(float(event.get("level_macro_f1", 0.0)) - target_f1)
                    + abs(float(event.get("task_accuracy", 0.0)) - target_task),
                )
                metrics["raw_metrics"] = best_event["business"]
                metrics["raw_metrics_source"] = "best_epoch_progress"
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare completed Modal router runs and promote the winner.")
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-timeout", type=float, default=0.0)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    args = parser.parse_args()

    comparison_dir = ROOT / "artifacts" / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_name in dict.fromkeys(args.runs):
        metrics_path = comparison_dir / f"{run_name}.metrics.json"
        metrics = fetch_metrics(
            run_name,
            metrics_path,
            wait_timeout=args.wait_timeout,
            poll_interval=args.poll_interval,
        )
        rows.append({"run_name": run_name, "objective": objective(metrics), "metrics": metrics})
    rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    winner = rows[0]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "winner": winner["run_name"],
        "runs": rows,
    }
    output = args.output or comparison_dir / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner["run_name"], "objective": winner["objective"]}, indent=2))
    if not args.no_promote:
        promote_artifact(str(winner["run_name"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
