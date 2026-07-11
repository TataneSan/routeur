from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PIPELINE_VERSION = 4


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(command), flush=True)
    command_env = os.environ.copy() if env is None else env.copy()
    existing_pythonpath = command_env.get("PYTHONPATH")
    command_env["PYTHONPATH"] = str(ROOT) + ((os.pathsep + existing_pythonpath) if existing_pythonpath else "")
    subprocess.run(command, cwd=ROOT, env=command_env, check=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_annotation_rows(rows: list[dict[str, Any]], count: int, seed: int = 42) -> list[dict[str, Any]]:
    import random

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(int(row.get("level", 3)), str(row.get("task", "general")))].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and buckets:
        for key in list(buckets):
            values = buckets[key]
            if values:
                selected.append(values.pop())
                if len(selected) >= count:
                    break
            if not values:
                buckets.pop(key, None)
    return selected


def promote_artifact(run_name: str) -> dict[str, Any]:
    artifact_root = ROOT / "artifacts" / run_name
    current_run_path = ROOT / "artifacts" / "current_run.json"
    existing_metrics_path = artifact_root / "metrics.json"
    existing_model = artifact_root / "model"
    if current_run_path.exists() and existing_metrics_path.exists() and (existing_model / "heads.pt").exists():
        current_run = json.loads(current_run_path.read_text(encoding="utf-8"))
        if current_run.get("run_name") == run_name:
            return json.loads(existing_metrics_path.read_text(encoding="utf-8"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_root / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    run(["modal", "volume", "get", "routeur-artifacts", f"/runs/{run_name}/metrics.json", str(metrics_path), "--force"])
    model_dir = artifact_root / "model"
    if model_dir.exists():
        if model_dir.is_dir():
            for child in model_dir.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
            model_dir.rmdir()
        else:
            model_dir.unlink()
    run(["modal", "volume", "get", "routeur-artifacts", f"/runs/{run_name}/model", str(artifact_root), "--force"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    business = metrics.get("metrics") or {}
    config_path = artifact_root / "model" / "router_config.json"
    # Legacy v3 runs used a globally conservative calibration that could bump
    # most low-confidence prompts and report >60% over-routing. Keep those
    # otherwise useful checkpoints deployable while new runs use the balanced
    # calibrator and the high-risk-only runtime guard.
    if (
        config_path.exists()
        and not metrics.get("raw_metrics")
        and float(business.get("overroute_rate", 0.0)) > 0.60
    ):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["confidence_threshold"] = 0.0
        config["safety_bump"] = 0
        config["calibration_guard"] = "legacy_overroute_disabled"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics["runtime_calibration_guard"] = "legacy_overroute_disabled"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    current = ROOT / "artifacts" / "current"
    temporary = ROOT / "artifacts" / f".current-{run_name}"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(artifact_root / "model", target_is_directory=True)
    temporary.replace(current)
    (ROOT / "artifacts" / "current_run.json").write_text(
        json.dumps({"run_name": run_name, "metrics": metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Arena, grow data, annotate, train on Modal, and promote the model.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-per-source", type=int, default=25000)
    parser.add_argument("--include-wildchat", action="store_true")
    parser.add_argument("--annotation-count", type=int, default=6000)
    parser.add_argument("--hard-annotation-count", type=int, default=1500)
    parser.add_argument("--rare-annotation-count", type=int, default=1500)
    parser.add_argument("--annotation-workers", type=int, default=4)
    parser.add_argument("--synthetic-count", type=int, default=18000)
    parser.add_argument("--max-per-bucket", type=int, default=2500)
    parser.add_argument("--profile", default="h100")
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    state_path = ROOT / "data" / "continuous_retrain_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    run([sys.executable, "scripts/sync_lmarena.py", "--max-models", "500"])
    snapshot = ROOT / "data" / "lmarena_snapshot.json"
    current_digest = digest(snapshot)
    if (
        not args.force
        and state.get("snapshot_sha256") == current_digest
        and state.get("pipeline_version") == PIPELINE_VERSION
    ):
        print("LMArena snapshot unchanged; no Modal run needed.")
        return 0

    from routeur.hf_data import build_hf_examples
    from routeur.io import write_jsonl

    weak_path = ROOT / "data" / "router_hf_continuous_weak.jsonl"
    weak_rows = build_hf_examples(
        max_per_source=args.max_per_source,
        seed=42,
        include_oasst=True,
        include_wildchat=args.include_wildchat,
    )
    write_jsonl(weak_path, weak_rows)
    snapshot_tag = current_digest[:12]
    annotation_input = ROOT / "data" / f"router_annotation_{snapshot_tag}_input.jsonl"
    selected_rows = select_annotation_rows(weak_rows, args.annotation_count)
    selected_prompts = {str(row["prompt"]).strip().lower() for row in selected_rows}
    hard_candidates = [
        row for row in weak_rows
        if int(row.get("level", 3)) >= 4
        and str(row["prompt"]).strip().lower() not in selected_prompts
    ]
    hard_rows = select_annotation_rows(hard_candidates, args.hard_annotation_count, seed=43)
    selected_rows.extend(hard_rows)
    selected_prompts.update(str(row["prompt"]).strip().lower() for row in hard_rows)
    rare_tasks = {"frontend", "backend", "safety", "vision", "agentic"}
    rare_candidates = [
        row for row in weak_rows
        if str(row.get("task", "general")) in rare_tasks
        and str(row["prompt"]).strip().lower() not in selected_prompts
    ]
    selected_rows.extend(select_annotation_rows(rare_candidates, args.rare_annotation_count, seed=44))
    write_jsonl(annotation_input, selected_rows)
    annotated_path = ROOT / "data" / f"router_annotation_{snapshot_tag}_glm_v2.jsonl"
    api_key = os.environ.get("EDELUX_API_KEY")
    if api_key:
        run(
            [
                sys.executable,
                "scripts/annotate_prompts.py",
                "--input",
                str(annotation_input),
                "--output",
                str(annotated_path),
                "--model",
                "glm-5.2",
                "--fallback-model",
                "Mistral:codestral-latest",
                "--batch-size",
                "24",
                "--workers",
                str(args.annotation_workers),
                "--resume",
            ],
            env=os.environ.copy(),
        )
    else:
        print("EDELUX_API_KEY not set; continuing with HF weak labels only.", flush=True)
        annotated_path = None

    train_path = ROOT / "data" / "router_train_continuous.jsonl"
    prepare_command = [
        sys.executable,
        "scripts/prepare_training_data.py",
        "--weak",
        str(weak_path),
        "--output",
        str(train_path),
        "--synthetic-count",
        str(args.synthetic_count),
        "--max-per-bucket",
        str(args.max_per_bucket),
    ]
    if annotated_path:
        prepare_command.extend(["--annotated", str(annotated_path)])
    run(prepare_command)

    run_name = "router-continuous-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run(
        [
            "modal",
            "run",
            "modal_train.py",
            "--dataset",
            str(train_path),
            "--profile",
            args.profile,
            "--base-model",
            args.base_model,
            "--run-name",
            run_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--class-weight-power",
            "0.35",
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
        ],
        env=os.environ.copy(),
    )
    metrics = promote_artifact(run_name) if not args.no_promote else {}
    state_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": PIPELINE_VERSION,
                "snapshot_sha256": current_digest,
                "run_name": run_name,
                "metrics": metrics,
                "weak_examples": len(weak_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
