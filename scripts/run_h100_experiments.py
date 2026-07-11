from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routeur.metrics import SCORING_WEIGHTS
from scripts.continuous_retrain import promote_artifact


EXPERIMENTS = (
    {
        "suffix": "xlmr-large",
        "base_model": "FacebookAI/xlm-roberta-large",
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1.2e-5,
        "amp_mode": "auto",
    },
    {
        "suffix": "mdeberta-v3-base",
        "base_model": "microsoft/mdeberta-v3-base",
        "batch_size": 24,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.2e-5,
        "amp_mode": "fp16",
    },
    {
        "suffix": "xlmr-base",
        "base_model": "xlm-roberta-base",
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "amp_mode": "auto",
    },
    {
        "suffix": "bge-m3",
        "base_model": "BAAI/bge-m3",
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1.0e-5,
        "amp_mode": "auto",
    },
    {
        "suffix": "multilingual-e5-large-instruct",
        "base_model": "intfloat/multilingual-e5-large-instruct",
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1.0e-5,
        "amp_mode": "auto",
    },
)


def objective(metrics: dict[str, Any]) -> float:
    business = metrics.get("raw_metrics") or metrics.get("metrics", {})
    throughput = max(1.0, float(metrics.get("inference_examples_per_second", 1.0)))
    return (
        SCORING_WEIGHTS["level_macro_f1"] * float(metrics.get("level_macro_f1", 0.0))
        + SCORING_WEIGHTS["task_macro_f1"] * float(metrics.get("task_macro_f1", 0.0))
        + SCORING_WEIGHTS["capability_micro_f1"] * float(metrics.get("capability_micro_f1", 0.0))
        + SCORING_WEIGHTS["risk_accuracy"] * float(metrics.get("risk_accuracy", 0.0))
        + SCORING_WEIGHTS["preference_accuracy"] * float(metrics.get("preference_accuracy", 0.0))
        + SCORING_WEIGHTS["external_router_accuracy"] * float((metrics.get("external_router_benchmark") or {}).get("accuracy", 0.0))
        + SCORING_WEIGHTS["severe_underroute_rate"] * float(business.get("severe_underroute_rate", 1.0))
        + SCORING_WEIGHTS["underroute_rate"] * float(business.get("underroute_rate", 1.0))
        + SCORING_WEIGHTS["level_ece"] * float(metrics.get("level_ece", 1.0))
        + SCORING_WEIGHTS["overroute_rate"] * float(business.get("overroute_rate", 1.0))
        + SCORING_WEIGHTS["throughput_log"] * math.log(throughput)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and compare router encoders on dedicated Modal H100s.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    prefix = args.prefix or "router-v3-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    logs_dir = ROOT / "artifacts" / "experiment-logs" / prefix
    logs_dir.mkdir(parents=True, exist_ok=True)
    remote_dataset = f"/datasets/{args.dataset.name}"
    subprocess.run(
        ["modal", "volume", "put", "routeur-artifacts", str(args.dataset), remote_dataset, "--force"],
        cwd=ROOT,
        check=True,
    )

    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for experiment in EXPERIMENTS:
        run_name = f"{prefix}-{experiment['suffix']}"
        command = [
            "modal", "run", "modal_train.py",
            "--dataset", remote_dataset,
            "--profile", "h100",
            "--base-model", str(experiment["base_model"]),
            "--run-name", run_name,
            "--epochs", str(args.epochs),
            "--batch-size", str(experiment["batch_size"]),
            "--gradient-accumulation-steps", str(experiment["gradient_accumulation_steps"]),
            "--learning-rate", str(experiment["learning_rate"]),
            "--max-length", str(args.max_length),
            "--class-weight-power", str(args.class_weight_power),
            "--amp-mode", str(experiment["amp_mode"]),
        ]
        handle = (logs_dir / f"{run_name}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((run_name, process, handle))
        print(f"started {run_name} pid={process.pid}", flush=True)

    completed: list[str] = []
    for run_name, process, handle in processes:
        return_code = process.wait()
        handle.close()
        if return_code == 0:
            completed.append(run_name)
        else:
            print(f"failed {run_name} exit={return_code}", file=sys.stderr, flush=True)
    if not completed:
        raise SystemExit("all H100 experiments failed")

    comparisons: list[dict[str, Any]] = []
    for run_name in completed:
        destination = logs_dir / f"{run_name}.metrics.json"
        subprocess.run(
            ["modal", "volume", "get", "routeur-artifacts", f"/runs/{run_name}/metrics.json", str(destination), "--force"],
            cwd=ROOT,
            check=True,
        )
        metrics = json.loads(destination.read_text(encoding="utf-8"))
        comparisons.append({"run_name": run_name, "objective": objective(metrics), "metrics": metrics})
    comparisons.sort(key=lambda row: float(row["objective"]), reverse=True)
    winner = comparisons[0]
    comparison_path = logs_dir / "comparison.json"
    comparison_path.write_text(json.dumps({"winner": winner["run_name"], "runs": comparisons}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner["run_name"], "objective": winner["objective"]}, indent=2), flush=True)
    if not args.no_promote:
        promote_artifact(str(winner["run_name"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
