#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from routeur.capabilities import CAPABILITIES, RISKS, infer_capabilities, normalize_capabilities
from routeur.fast_model import build_vectorizer, prepare_prompt
from routeur.io import read_jsonl
from routeur.metrics import routing_metrics
from routeur.schema import RouterExample
from routeur.tasks import TASKS, normalize_task


def normalized_prompt(row: dict[str, Any]) -> str:
    return " ".join(str(row["prompt"]).strip().lower().split())


def validation_group(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("query_id"):
        return f"{metadata.get('dataset', row.get('source', 'unknown'))}:{metadata['query_id']}"
    return normalized_prompt(row)


def split_rows(rows: list[dict[str, Any]], *, validation_ratio: float, seed: int):
    gold = [
        row
        for row in rows
        if (row.get("metadata") or {}).get("grader")
        and float((row.get("metadata") or {}).get("grader_confidence", 0.0)) >= 0.65
    ]
    if len(gold) < 300:
        train, validation = train_test_split(
            rows,
            test_size=validation_ratio,
            random_state=seed,
            stratify=[row["level"] for row in rows],
        )
        return train, validation, "mixed_fallback"
    groups = [validation_group(row) for row in gold]
    if len(set(groups)) < len(groups):
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=seed)
        _train_positions, validation_positions = next(splitter.split(gold, groups=groups))
        validation = [gold[int(position)] for position in validation_positions]
        validation_groups = {validation_group(row) for row in validation}
        train = [row for row in rows if validation_group(row) not in validation_groups]
        return train, validation, "held_out_teacher_gold_grouped"
    composite = [f"{row['level']}:{normalize_task(row.get('task'))}" for row in gold]
    counts = Counter(composite)
    strata = [label if counts[label] >= 5 else f"{row['level']}:other" for row, label in zip(gold, composite)]
    strata_counts = Counter(strata)
    validation_size = max(1, int(round(len(gold) * validation_ratio)))
    if min(strata_counts.values()) < 2 or len(strata_counts) > validation_size:
        strata = [str(row["level"]) for row in gold]
    _gold_train, validation = train_test_split(
        gold,
        test_size=validation_ratio,
        random_state=seed,
        stratify=strata,
    )
    validation_prompts = {normalized_prompt(row) for row in validation}
    train = [row for row in rows if normalized_prompt(row) not in validation_prompts]
    return train, validation, "held_out_teacher_gold"


def supervision(row: dict[str, Any]):
    task = normalize_task(row.get("task"))
    metadata = row.get("metadata") or {}
    risk = str(metadata.get("risk", "")).lower()
    if risk not in RISKS:
        risk = "high" if task == "safety" or int(row["level"]) == 5 else ("low" if int(row["level"]) <= 2 else "medium")
    capabilities = normalize_capabilities(metadata.get("required_capabilities"))
    if not capabilities:
        capabilities = infer_capabilities(str(row["prompt"]), task)
    source = str(row.get("source", ""))
    confidence = float(metadata.get("grader_confidence", 0.0))
    if metadata.get("grader"):
        weight = 1.5 + 0.5 * max(0.0, min(1.0, confidence))
    elif source.startswith("hf:agentlans/") or source.startswith("hf:SupraLabs/"):
        weight = 0.9
    elif source.startswith("hf:anasnassar/") or source.startswith("hf:somukandula/"):
        weight = 0.8
    elif source.startswith("synthetic_"):
        weight = 0.55
    else:
        weight = 0.7
    return task, risk, capabilities, weight


def train_head(features, labels, sample_weights, *, alpha: float, seed: int, max_iter: int):
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-3,
        class_weight="balanced",
        average=False,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features, labels, sample_weight=sample_weights)
    return model


def ordered_parameters(model, classes: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    positions = {value: index for index, value in enumerate(model.classes_.tolist())}
    if set(positions) != set(classes):
        raise ValueError(f"Classifier classes {sorted(positions)} do not match {classes}")
    order = [positions[value] for value in classes]
    return model.coef_[order].astype(np.float32), model.intercept_[order].astype(np.float32)


def train_capability_heads(features, targets: np.ndarray, weights: np.ndarray, *, alpha: float, seed: int, max_iter: int):
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    for index, capability in enumerate(CAPABILITIES):
        model = train_head(
            features,
            targets[:, index],
            weights,
            alpha=alpha,
            seed=seed + index + 1,
            max_iter=max_iter,
        )
        if model.classes_.tolist() != [0, 1]:
            raise ValueError(f"Capability {capability} needs both positive and negative examples")
        coefficients.append(model.coef_[0].astype(np.float32))
        intercepts.append(float(model.intercept_[0]))
    return np.stack(coefficients), np.asarray(intercepts, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the portable sub-10ms linear router.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-features", type=int, default=65536)
    parser.add_argument("--max-chars", type=int, default=4096)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=3e-5)
    parser.add_argument("--max-iter", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [RouterExample.from_json(row).to_json() for row in read_jsonl(args.dataset)]
    train_rows, validation_rows, validation_kind = split_rows(
        rows, validation_ratio=args.validation_ratio, seed=args.seed
    )
    train_supervision = [supervision(row) for row in train_rows]
    prompts = [prepare_prompt(row["prompt"], max_chars=args.max_chars) for row in train_rows]
    validation_prompts = [prepare_prompt(row["prompt"], max_chars=args.max_chars) for row in validation_rows]
    vectorizer = build_vectorizer(n_features=args.n_features, word_ngrams=(1, 2), char_ngrams=(3, 5))

    started = time.perf_counter()
    features = vectorizer.transform(prompts)
    validation_features = vectorizer.transform(validation_prompts)
    weights = np.asarray([item[3] for item in train_supervision], dtype=np.float64)
    level_labels = np.asarray([row["level"] for row in train_rows])
    task_labels = np.asarray([item[0] for item in train_supervision])
    risk_labels = np.asarray([item[1] for item in train_supervision])
    capability_targets = np.asarray(
        [[int(capability in item[2]) for capability in CAPABILITIES] for item in train_supervision],
        dtype=np.int8,
    )

    level_model = train_head(features, level_labels, weights, alpha=args.alpha, seed=args.seed, max_iter=args.max_iter)
    task_model = train_head(features, task_labels, weights, alpha=args.alpha, seed=args.seed + 101, max_iter=args.max_iter)
    risk_model = train_head(features, risk_labels, weights, alpha=args.alpha, seed=args.seed + 202, max_iter=args.max_iter)
    capability_coef, capability_intercept = train_capability_heads(
        features, capability_targets, weights, alpha=args.alpha, seed=args.seed + 303, max_iter=args.max_iter
    )

    level_predictions = level_model.predict(validation_features)
    task_predictions = task_model.predict(validation_features)
    risk_predictions = risk_model.predict(validation_features)
    validation_levels = np.asarray([row["level"] for row in validation_rows])
    validation_supervision = [supervision(row) for row in validation_rows]
    metrics = routing_metrics(validation_levels.tolist(), level_predictions.tolist())
    metrics.update(
        {
            "level_macro_f1": float(f1_score(validation_levels, level_predictions, average="macro")),
            "task_accuracy": float(accuracy_score([item[0] for item in validation_supervision], task_predictions)),
            "task_macro_f1": float(f1_score([item[0] for item in validation_supervision], task_predictions, average="macro")),
            "risk_accuracy": float(accuracy_score([item[1] for item in validation_supervision], risk_predictions)),
        }
    )

    level_coef, level_intercept = ordered_parameters(level_model, [1, 2, 3, 4, 5])
    task_coef, task_intercept = ordered_parameters(task_model, list(TASKS))
    risk_coef, risk_intercept = ordered_parameters(risk_model, list(RISKS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "fast_router.npz",
        level_coef=level_coef,
        level_intercept=level_intercept,
        task_coef=task_coef,
        task_intercept=task_intercept,
        risk_coef=risk_coef,
        risk_intercept=risk_intercept,
        capability_coef=capability_coef,
        capability_intercept=capability_intercept,
    )
    config = {
        "architecture": "hashed_word_char_linear_multitask_v1",
        "levels": [1, 2, 3, 4, 5],
        "tasks": list(TASKS),
        "risks": list(RISKS),
        "capabilities": list(CAPABILITIES),
        "n_features": args.n_features,
        "word_ngrams": [1, 2],
        "char_ngrams": [3, 5],
        "max_chars": args.max_chars,
        "confidence_threshold": 0.0,
        "safety_bump": 0,
        "dataset": args.dataset.name,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "validation_kind": validation_kind,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "training_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    (args.output_dir / "fast_router.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
