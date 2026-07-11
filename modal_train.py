from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from routeur.metrics import SCORING_WEIGHTS

APP_NAME = "routeur-trainer"
VOLUME_NAME = "routeur-artifacts"
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "HF_HOME": str(ARTIFACTS_DIR / "hf-cache"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .pip_install(
        "datasets>=2.20",
        "numpy>=1.26",
        "scikit-learn>=1.5",
        "sentencepiece>=0.2",
        "torch>=2.4",
        "transformers>=4.44",
        "zstandard>=0.22",
    )
    .add_local_python_source("routeur")
)


def _optimize_temperature(level_logits: Any, level_labels: list[int]) -> float:
    """Find the temperature that minimizes validation NLL (proxy for ECE).

    Uses a simple grid search so no extra dependencies are required inside the
    Modal training image.
    """
    import numpy as np

    level_logits = np.asarray(level_logits)
    labels = np.asarray(level_labels) - 1
    one_hot = np.eye(5)[labels]
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.linspace(0.3, 3.0, 28):
        logits = level_logits / temperature
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        nll = -float(np.mean(np.sum(one_hot * log_probs, axis=-1)))
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def _calibrated_levels(
    probabilities: Any,
    threshold: float,
    bump: int,
    risk_preds: list[int] | None = None,
) -> list[int]:
    """Apply the same confidence bump used in production by TransformerRouter.

    In production the bump is only applied when the predicted risk is high,
    so the calibration loop must mirror that behaviour.
    """
    import numpy as np

    high_risk_index = RISKS.index("high")
    raw = np.asarray(probabilities)
    raw_pred = raw.argmax(axis=1)
    confidence = raw.max(axis=1)
    predictions: list[int] = []
    for index, (level, conf) in enumerate(zip(raw_pred, confidence, strict=True)):
        raw_level = int(level) + 1
        effective_threshold = min(float(threshold), 0.55) if raw_level <= 2 else float(threshold)
        high_risk = risk_preds is None or risk_preds[index] == high_risk_index
        routed = min(4, int(level) + int(bump)) if high_risk and float(conf) < effective_threshold else int(level)
        predictions.append(routed + 1)
    return predictions


def _calibrate_policy(
    y_true: list[int],
    probabilities: Any,
    costs: dict[int, float],
    risk_preds: list[int] | None = None,
) -> tuple[float, int, dict[str, float]]:
    import numpy as np

    from routeur.metrics import routing_metrics

    best: tuple[float, int, float, dict[str, float]] | None = None
    for threshold in np.arange(0.0, 0.81, 0.05):
        for bump in (0, 1):
            metrics = routing_metrics(
                y_true,
                _calibrated_levels(probabilities, float(threshold), bump, risk_preds),
                level_costs=costs,
            )
            objective = (
                -SCORING_WEIGHTS["severe_underroute_rate"] * metrics["severe_underroute_rate"]
                + -SCORING_WEIGHTS["underroute_rate"] * metrics["underroute_rate"]
                + -SCORING_WEIGHTS["overroute_rate"] * metrics["overroute_rate"]
                + -SCORING_WEIGHTS["cost_ratio_vs_oracle"] * abs(metrics["cost_ratio_vs_oracle"] - 1.0)
            )
            if best is None or objective < best[0] or (objective == best[0] and metrics["savings_vs_always_level_5"] > best[3]["savings_vs_always_level_5"]):
                best = (objective, bump, float(threshold), metrics)
    assert best is not None
    return best[2], best[1], best[3]


def _train_impl(
    *,
    dataset_volume_path: str,
    base_model: str,
    run_name: str,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    validation_ratio: float,
    confidence_threshold: float | None,
    safety_bump: int | None,
    seed: int,
    class_weight_power: float,
    gradient_accumulation_steps: int,
    amp_mode: str,
    truncation_strategy: str,
    patience: int = 3,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from datasets import Dataset, load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

    from routeur.capabilities import CAPABILITIES, RISKS, infer_capabilities, normalize_capabilities
    from routeur.io import read_jsonl, write_json
    from routeur.labels import DEFAULT_LEVEL_COSTS_USD
    from routeur.metrics import routing_metrics
    from routeur.modeling import MultiTaskRouterModel
    from routeur.schema import RouterExample
    from routeur.tasks import TASKS, normalize_task
    from routeur.tokenization import encode_head_tail_batch

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_path = ARTIFACTS_DIR / dataset_volume_path.lstrip("/")
    rows = [RouterExample.from_json(row).to_json() for row in read_jsonl(dataset_path)]
    if len(rows) < 100:
        raise ValueError("Need at least 100 examples for a useful multitask train/validation split.")
    task_to_id = {task: idx for idx, task in enumerate(TASKS)}
    gold_rows = [
        row
        for row in rows
        if (row.get("metadata") or {}).get("grader")
        and float((row.get("metadata") or {}).get("grader_confidence", 0.0)) >= 0.65
    ]
    if len(gold_rows) >= 300:
        composite = [f"{row['level']}:{row['task']}" for row in gold_rows]
        composite_counts = Counter(composite)
        candidate_strata = [
            label if composite_counts[label] >= 5 else f"{row['level']}:other"
            for row, label in zip(gold_rows, composite, strict=True)
        ]
        candidate_counts = Counter(candidate_strata)
        validation_size = max(1, int(round(len(gold_rows) * validation_ratio)))
        gold_strata = (
            candidate_strata
            if min(candidate_counts.values()) >= 2 and len(candidate_counts) <= validation_size
            else [str(row["level"]) for row in gold_rows]
        )
        _gold_train, val_rows = train_test_split(
            gold_rows,
            test_size=validation_ratio,
            random_state=seed,
            stratify=gold_strata,
        )
        val_prompts = {str(row["prompt"]).strip().lower() for row in val_rows}
        train_rows = [row for row in rows if str(row["prompt"]).strip().lower() not in val_prompts]
        validation_kind = "held_out_teacher_gold"
    else:
        train_rows, val_rows = train_test_split(
            rows,
            test_size=validation_ratio,
            random_state=seed,
            stratify=[row["level"] for row in rows],
        )
        validation_kind = "mixed_fallback"
    preference_rows = [
        row for row in train_rows
        if len((row.get("metadata") or {}).get("preference_winner", [])) == len(TASKS)
        and len((row.get("metadata") or {}).get("preference_loser", [])) == len(TASKS)
    ]
    if len(preference_rows) >= 200:
        _preference_train, preference_val_rows = train_test_split(
            preference_rows,
            test_size=0.10,
            random_state=seed,
        )
        preference_val_prompts = {str(row["prompt"]).strip().lower() for row in preference_val_rows}
        train_rows = [
            row for row in train_rows
            if str(row["prompt"]).strip().lower() not in preference_val_prompts
        ]
    else:
        preference_val_rows = []

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    input_prefix = (
        "Instruct: Classify this user prompt by difficulty, task, risk, and required model capabilities.\nQuery: "
        if "multilingual-e5" in base_model.lower() and "instruct" in base_model.lower()
        else ""
    )
    truncation_strategy = str(truncation_strategy).strip().lower()
    if truncation_strategy not in {"right", "head_tail"}:
        raise ValueError("truncation_strategy must be right or head_tail")

    def supervision(row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata") or {}
        source = str(row.get("source", ""))
        confidence = float(metadata.get("grader_confidence", 0.0))
        is_gold = bool(metadata.get("grader"))
        if is_gold:
            sample_weight = 0.75 + 0.25 * max(0.0, min(1.0, confidence))
        elif source.startswith("hf:agentlans/"):
            sample_weight = 0.60
        elif source.startswith("hf:SupraLabs/"):
            sample_weight = 0.70
        elif source.startswith("hf:anasnassar/"):
            sample_weight = 0.65
        elif source.startswith("hf:somukandula/") or source.startswith("hf:routellm/"):
            sample_weight = 0.55
        elif source.startswith("hf:lmarena-ai/arena-human-preference"):
            sample_weight = 0.45
        elif source.startswith("synthetic_"):
            sample_weight = 0.40
        else:
            sample_weight = 0.30
        task = normalize_task(row.get("task"))
        secondary = [normalize_task(value) for value in metadata.get("secondary_tasks", [])]
        task_targets = [0.0] * len(TASKS)
        task_targets[task_to_id[task]] = 1.0
        for value in secondary:
            if value in task_to_id and value != task:
                task_targets[task_to_id[value]] = 0.35
        annotated_capabilities = normalize_capabilities(metadata.get("required_capabilities"))
        capabilities = annotated_capabilities or infer_capabilities(str(row["prompt"]), task)
        capability_targets = [1.0 if capability in capabilities else 0.0 for capability in CAPABILITIES]
        risk = str(metadata.get("risk", "")).lower()
        if risk not in RISKS:
            risk = "high" if task == "safety" or int(row["level"]) == 5 else ("low" if int(row["level"]) <= 2 else "medium")
        winner_features = metadata.get("preference_winner", [])
        loser_features = metadata.get("preference_loser", [])
        has_preference = len(winner_features) == len(TASKS) and len(loser_features) == len(TASKS)
        return {
            "sample_weight": sample_weight,
            "task_targets": task_targets,
            "capability_targets": capability_targets,
            "capability_weight": 1.0 if annotated_capabilities else 0.25,
            "risk_label": RISKS.index(risk),
            "risk_weight": 1.0 if metadata.get("risk") in RISKS else 0.25,
            "preference_winner": [float(value) for value in winner_features] if has_preference else [0.0] * len(TASKS),
            "preference_loser": [float(value) for value in loser_features] if has_preference else [0.0] * len(TASKS),
            "preference_weight": 1.0 if has_preference else 0.0,
        }

    def tokenize(items: list[dict[str, Any]]) -> Dataset:
        prompts = [str(row["prompt"]) for row in items]
        if truncation_strategy == "head_tail":
            encoded = encode_head_tail_batch(
                tokenizer,
                prompts,
                max_length=max_length,
                input_prefix=input_prefix,
                padding=False,
            )
        else:
            encoded = tokenizer(
                [input_prefix + prompt for prompt in prompts],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
        encoded["level_labels"] = [int(row["level"]) - 1 for row in items]
        encoded["task_labels"] = [task_to_id[normalize_task(row.get("task"))] for row in items]
        labels = [supervision(row) for row in items]
        encoded["sample_weights"] = [row["sample_weight"] for row in labels]
        encoded["task_targets"] = [row["task_targets"] for row in labels]
        encoded["capability_targets"] = [row["capability_targets"] for row in labels]
        encoded["capability_weights"] = [row["capability_weight"] for row in labels]
        encoded["risk_labels"] = [row["risk_label"] for row in labels]
        encoded["risk_weights"] = [row["risk_weight"] for row in labels]
        encoded["preference_winner"] = [row["preference_winner"] for row in labels]
        encoded["preference_loser"] = [row["preference_loser"] for row in labels]
        encoded["preference_weights"] = [row["preference_weight"] for row in labels]
        return Dataset.from_dict(encoded)

    train_ds = tokenize(train_rows)
    val_ds = tokenize(val_rows)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels_level = torch.tensor([item.pop("level_labels") for item in features], dtype=torch.long)
        labels_task = torch.tensor([item.pop("task_labels") for item in features], dtype=torch.long)
        sample_weights = torch.tensor([item.pop("sample_weights") for item in features], dtype=torch.float)
        task_targets = torch.tensor([item.pop("task_targets") for item in features], dtype=torch.float)
        capability_targets = torch.tensor([item.pop("capability_targets") for item in features], dtype=torch.float)
        capability_weights = torch.tensor([item.pop("capability_weights") for item in features], dtype=torch.float)
        risk_labels = torch.tensor([item.pop("risk_labels") for item in features], dtype=torch.long)
        risk_weights = torch.tensor([item.pop("risk_weights") for item in features], dtype=torch.float)
        preference_winner = torch.tensor([item.pop("preference_winner") for item in features], dtype=torch.float)
        preference_loser = torch.tensor([item.pop("preference_loser") for item in features], dtype=torch.float)
        preference_weights = torch.tensor([item.pop("preference_weights") for item in features], dtype=torch.float)
        batch = collator(features)
        batch["level_labels"] = labels_level
        batch["task_labels"] = labels_task
        batch["sample_weights"] = sample_weights
        batch["task_targets"] = task_targets
        batch["capability_targets"] = capability_targets
        batch["capability_weights"] = capability_weights
        batch["risk_labels"] = risk_labels
        batch["risk_weights"] = risk_weights
        batch["preference_winner"] = preference_winner
        batch["preference_loser"] = preference_loser
        batch["preference_weights"] = preference_weights
        return batch

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
    preference_val_ds = tokenize(preference_val_rows) if preference_val_rows else None
    preference_val_loader = (
        DataLoader(preference_val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
        if preference_val_ds is not None
        else None
    )
    model = MultiTaskRouterModel.from_base_model(
        base_model,
        num_tasks=len(TASKS),
        num_capabilities=len(CAPABILITIES),
        num_risks=len(RISKS),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Some Hub checkpoints (notably mDeBERTa variants) are stored in FP16.
    # Keep FP32 master weights so GradScaler can safely unscale FP16 gradients.
    model.float().to(device)
    output_dir = ARTIFACTS_DIR / "runs" / run_name
    model_dir = output_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    status_path = output_dir / "status.json"
    started_at = time.time()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    completed_epochs = max(1, int(round(epochs)))

    def publish(event: dict[str, Any]) -> None:
        event = {"run_name": run_name, "gpu": gpu_name, **event}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        status_path.write_text(json.dumps(event, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Commit once per epoch so a local dashboard can follow the remote job.
        volume.commit()

    publish({"status": "running", "epoch": 0, "epochs": completed_epochs})
    level_counts = torch.bincount(torch.tensor([int(row["level"]) - 1 for row in train_rows]), minlength=5).float()
    task_counts = torch.bincount(torch.tensor([task_to_id[normalize_task(row.get("task"))] for row in train_rows]), minlength=len(TASKS)).float()

    def make_class_weights(counts: torch.Tensor) -> torch.Tensor:
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = (counts[present].sum() / counts[present]).pow(class_weight_power)
        weights[present] = weights[present] / weights[present].mean()
        return weights.to(device)

    level_weights = make_class_weights(level_counts)
    task_weights = make_class_weights(task_counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = max(1, int(steps_per_epoch * epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps)
    requested_amp_mode = str(amp_mode).strip().lower()
    if requested_amp_mode not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("amp_mode must be auto, bf16, fp16, or fp32")
    if device.type != "cuda" or requested_amp_mode == "fp32":
        use_amp = False
        autocast_dtype = torch.float32
    elif requested_amp_mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but is not supported by this GPU")
        use_amp = True
        autocast_dtype = torch.bfloat16
    elif requested_amp_mode == "fp16":
        use_amp = True
        autocast_dtype = torch.float16
    else:
        use_amp = True
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and autocast_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = max(1, int(patience))
    epochs_without_improvement = 0
    label_keys = {
        "level_labels",
        "task_labels",
        "sample_weights",
        "task_targets",
        "capability_targets",
        "capability_weights",
        "risk_labels",
        "risk_weights",
        "preference_winner",
        "preference_loser",
        "preference_weights",
    }

    def forward_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return model(**{key: value for key, value in batch.items() if key not in label_keys})

    def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample_weights = batch["sample_weights"]
        level_ce = torch.nn.functional.cross_entropy(
            outputs["level_logits"], batch["level_labels"], weight=level_weights,
            reduction="none", label_smoothing=0.03,
        )
        level_values = torch.arange(1, 6, device=device, dtype=outputs["level_logits"].dtype)
        expected_level = (torch.softmax(outputs["level_logits"], dim=-1) * level_values).sum(dim=-1)
        ordinal_loss = torch.nn.functional.smooth_l1_loss(
            expected_level, batch["level_labels"].to(expected_level.dtype) + 1.0, reduction="none"
        )
        level_loss = weighted_mean(level_ce + 0.20 * ordinal_loss, sample_weights)

        task_ce = torch.nn.functional.cross_entropy(
            outputs["task_logits"], batch["task_labels"], weight=task_weights,
            reduction="none", label_smoothing=0.03,
        )
        task_multilabel = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["task_logits"], batch["task_targets"], reduction="none"
        ).mean(dim=-1)
        task_loss = weighted_mean(task_ce + 0.20 * task_multilabel, sample_weights)

        risk_per_row = torch.nn.functional.cross_entropy(
            outputs["risk_logits"], batch["risk_labels"], reduction="none"
        )
        risk_loss = weighted_mean(risk_per_row, sample_weights * batch["risk_weights"])
        capability_per_row = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["capability_logits"], batch["capability_targets"], reduction="none"
        ).mean(dim=-1)
        capability_loss = weighted_mean(
            capability_per_row, sample_weights * batch["capability_weights"]
        )
        preference_delta = batch["preference_winner"] - batch["preference_loser"]
        preference_margin = 5.0 * (outputs["task_logits"] * preference_delta).sum(dim=-1)
        preference_per_row = torch.nn.functional.softplus(-preference_margin)
        preference_loss = weighted_mean(preference_per_row, batch["preference_weights"])
        total = (
            0.47 * level_loss
            + 0.21 * task_loss
            + 0.08 * risk_loss
            + 0.12 * capability_loss
            + 0.12 * preference_loss
        )
        return total, {
            "level": level_loss,
            "task": task_loss,
            "risk": risk_loss,
            "capability": capability_loss,
            "preference": preference_loss,
        }

    def evaluate_preferences() -> float:
        if preference_val_loader is None:
            return 0.0
        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in preference_val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = forward_batch(batch)
                delta = batch["preference_winner"] - batch["preference_loser"]
                margin = (outputs["task_logits"] * delta).sum(dim=-1)
                selected = batch["preference_weights"] > 0
                correct += int((margin[selected] > 0).sum().item())
                count += int(selected.sum().item())
        return correct / max(1, count)

    def evaluate_external_router() -> dict[str, Any]:
        try:
            external = load_dataset("somukandula/prompt-router-dataset", split="test")
        except Exception as exc:
            return {"accuracy": 0.0, "count": 0, "error": str(exc)}
        prompts = [str(row["prompt"]) for row in external]
        labels = [str(row["label"]) for row in external]
        predictions: list[str] = []
        model.eval()
        vision_index = CAPABILITIES.index("vision")
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size * 2):
                prompt_batch = prompts[start : start + batch_size * 2]
                if truncation_strategy == "head_tail":
                    encoded = encode_head_tail_batch(
                        tokenizer,
                        prompt_batch,
                        max_length=max_length,
                        input_prefix=input_prefix,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    encoded = tokenizer(
                        [input_prefix + prompt for prompt in prompt_batch],
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=True,
                    )
                outputs = model(**{key: value.to(device) for key, value in encoded.items()})
                levels = outputs["level_logits"].argmax(dim=-1).cpu().tolist()
                tasks = outputs["task_logits"].argmax(dim=-1).cpu().tolist()
                vision = (torch.sigmoid(outputs["capability_logits"][:, vision_index]) >= 0.5).cpu().tolist()
                for level, task_index, needs_vision in zip(levels, tasks, vision, strict=True):
                    task = TASKS[int(task_index)]
                    if needs_vision or task == "vision":
                        predictions.append("vision_model")
                    elif task in {"frontend", "backend", "coding", "data"}:
                        predictions.append("code_model")
                    elif int(level) + 1 <= 2:
                        predictions.append("cheap_small_text")
                    else:
                        predictions.append("strong_general")
        confusion = Counter(zip(labels, predictions, strict=True))
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "count": len(labels),
            "confusion": {f"{expected}->{predicted}": count for (expected, predicted), count in confusion.items()},
        }

    def evaluate() -> dict[str, Any]:
        model.eval()
        evaluation_started = time.time()
        level_true: list[int] = []
        level_pred: list[int] = []
        task_true: list[int] = []
        task_pred: list[int] = []
        level_logits_list: list[Any] = []
        level_probs: list[Any] = []
        risk_true: list[int] = []
        risk_pred: list[int] = []
        capability_true: list[list[int]] = []
        capability_pred: list[list[int]] = []
        loss_total = 0.0
        batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = forward_batch(batch)
                loss, _components = compute_loss(outputs, batch)
                loss_total += float(loss.item())
                batches += 1
                level_logits_list.extend(outputs["level_logits"].cpu().tolist())
                level_true.extend((batch["level_labels"] + 1).cpu().tolist())
                # Temperature is optimised after the loop; use T=1.0 for raw
                # predictions and store logits so calibration can apply it.
                level_pred.extend((outputs["level_logits"].argmax(dim=-1) + 1).cpu().tolist())
                task_true.extend(batch["task_labels"].cpu().tolist())
                task_pred.extend(outputs["task_logits"].argmax(dim=-1).cpu().tolist())
                level_probs.extend(level_prob.cpu().tolist())
                risk_pred.extend(outputs["risk_logits"].argmax(dim=-1).cpu().tolist())
                gold_risk = batch["risk_weights"] >= 0.75
                if gold_risk.any():
                    risk_true.extend(batch["risk_labels"][gold_risk].cpu().tolist())
                gold_capability = batch["capability_weights"] >= 0.75
                if gold_capability.any():
                    capability_true.extend(batch["capability_targets"][gold_capability].int().cpu().tolist())
                    capability_pred.extend((torch.sigmoid(outputs["capability_logits"][gold_capability]) >= 0.5).int().cpu().tolist())
        level_f1_values = f1_score(level_true, level_pred, labels=[1, 2, 3, 4, 5], average=None, zero_division=0)
        task_f1_values = f1_score(
            task_true, task_pred, labels=list(range(len(TASKS))), average=None, zero_division=0
        )
        # Optimize temperature on validation logits to match production calibration.
        optimal_temperature = _optimize_temperature(level_logits_list, level_true)
        calibrated_logits = np.asarray(level_logits_list) / optimal_temperature
        probability_array = np.exp(calibrated_logits) / np.exp(calibrated_logits).sum(axis=-1, keepdims=True)
        confidence_values = probability_array.max(axis=1)
        correctness = (probability_array.argmax(axis=1) + 1 == np.asarray(level_true)).astype(float)
        calibration_error = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            selected = (confidence_values >= lower) & (confidence_values < lower + 0.1)
            if selected.any():
                calibration_error += float(selected.mean()) * abs(
                    float(correctness[selected].mean()) - float(confidence_values[selected].mean())
                )
        one_hot_levels = np.eye(5, dtype=float)[np.asarray(level_true) - 1]
        brier_score = float(np.mean(np.square(probability_array - one_hot_levels).sum(axis=1)))
        return {
            "level_macro_f1": float(f1_score(level_true, level_pred, average="macro")),
            "level_f1_by_class": {str(index + 1): float(value) for index, value in enumerate(level_f1_values)},
            "task_accuracy": float(accuracy_score(task_true, task_pred)),
            "task_macro_f1": float(f1_score(task_true, task_pred, average="macro", zero_division=0)),
            "task_f1_by_class": {task: float(task_f1_values[index]) for index, task in enumerate(TASKS)},
            "risk_accuracy": float(accuracy_score(risk_true, risk_pred)) if risk_true else 0.0,
            "capability_micro_f1": float(f1_score(capability_true, capability_pred, average="micro", zero_division=0)) if capability_true else 0.0,
            "level_ece": calibration_error,
            "level_brier_score": brier_score,
            "val_loss": loss_total / max(1, batches),
            "inference_examples_per_second": len(level_true) / max(1e-6, time.time() - evaluation_started),
            "level_true": level_true,
            "level_pred": level_pred,
            "task_true": task_true,
            "task_pred": task_pred,
            "probabilities": probability_array,
            "risk_pred": risk_pred,
        }

    for epoch in range(completed_epochs):
        model.train()
        train_loss_total = 0.0
        train_steps = 0
        progress_every = max(1, len(train_loader) // 10)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            try:
                autocast_context = torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype)
            except (AttributeError, TypeError):
                autocast_context = torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype)
            with autocast_context:
                outputs = forward_batch(batch)
                loss, _loss_components = compute_loss(outputs, batch)
            scaler.scale(loss / gradient_accumulation_steps).backward()
            should_step = (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss_total += float(loss.detach().float().cpu().item())
            train_steps += 1
            if (step + 1) % progress_every == 0 and step + 1 < len(train_loader):
                publish({
                    "status": "batch",
                    "epoch": epoch + 1,
                    "epochs": completed_epochs,
                    "batch": step + 1,
                    "batches": len(train_loader),
                    "train_loss": train_loss_total / max(1, train_steps),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "elapsed_seconds": time.time() - started_at,
                })
        evaluation = evaluate()
        preference_accuracy = evaluate_preferences()
        costs = DEFAULT_LEVEL_COSTS_USD
        # Calibrate predictions before scoring so the selected checkpoint
        # reflects the actual production routing behaviour.
        calibrated_threshold, calibrated_bump, calibrated_business = _calibrate_policy(
            evaluation["level_true"], evaluation["probabilities"], costs,
            risk_preds=evaluation.get("risk_pred"),
        )
        epoch_business = calibrated_business
        # Select the checkpoint with the same quality/risk priorities used by
        # the cross-encoder sweep, rather than raw accuracy alone.
        score = (
            SCORING_WEIGHTS["level_macro_f1"] * evaluation["level_macro_f1"]
            + SCORING_WEIGHTS["task_macro_f1"] * evaluation["task_macro_f1"]
            + SCORING_WEIGHTS["capability_micro_f1"] * evaluation["capability_micro_f1"]
            + SCORING_WEIGHTS["risk_accuracy"] * evaluation["risk_accuracy"]
            + SCORING_WEIGHTS["preference_accuracy"] * preference_accuracy
            + SCORING_WEIGHTS["severe_underroute_rate"] * epoch_business["severe_underroute_rate"]
            + SCORING_WEIGHTS["underroute_rate"] * epoch_business["underroute_rate"]
            + SCORING_WEIGHTS["overroute_rate"] * epoch_business["overroute_rate"]
            + SCORING_WEIGHTS["level_ece"] * evaluation["level_ece"]
        )
        event = {
            "status": "epoch",
            "epoch": epoch + 1,
            "epochs": completed_epochs,
            "train_loss": train_loss_total / max(1, train_steps),
            "val_loss": evaluation["val_loss"],
            "level_macro_f1": evaluation["level_macro_f1"],
            "task_accuracy": evaluation["task_accuracy"],
            "task_macro_f1": evaluation["task_macro_f1"],
            "risk_accuracy": evaluation["risk_accuracy"],
            "capability_micro_f1": evaluation["capability_micro_f1"],
            "level_ece": evaluation["level_ece"],
            "level_brier_score": evaluation["level_brier_score"],
            "inference_examples_per_second": evaluation["inference_examples_per_second"],
            "level_f1_by_class": evaluation["level_f1_by_class"],
            "task_f1_by_class": evaluation["task_f1_by_class"],
            "preference_accuracy": preference_accuracy,
            "best_score": max(best_score, score),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": time.time() - started_at,
            "business": epoch_business,
        }
        print(json.dumps(event, sort_keys=True), flush=True)
        publish(event)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            publish({"status": "early_stopping", "epoch": epoch + 1, "epochs": completed_epochs, "best_score": best_score})
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    evaluation = evaluate()
    preference_accuracy = evaluate_preferences()
    external_router = evaluate_external_router()
    level_f1 = evaluation["level_macro_f1"]
    task_accuracy = evaluation["task_accuracy"]
    task_f1 = evaluation["task_macro_f1"]
    val_loss = evaluation["val_loss"]
    y_true = evaluation["level_true"]
    probabilities = evaluation["probabilities"]
    costs = DEFAULT_LEVEL_COSTS_USD
    raw_business_metrics = routing_metrics(y_true, evaluation["level_pred"], level_costs=costs)
    calibrated_threshold, calibrated_bump, business_metrics = _calibrate_policy(
        y_true, probabilities, costs, risk_preds=evaluation.get("risk_pred")
    )
    if confidence_threshold is not None:
        calibrated_threshold = confidence_threshold
    if safety_bump is not None:
        calibrated_bump = safety_bump
    business_metrics = routing_metrics(
        y_true,
        _calibrated_levels(probabilities, calibrated_threshold, calibrated_bump, risk_preds=evaluation.get("risk_pred")),
        level_costs=costs,
    )
    model.save_pretrained(model_dir, base_model=base_model)
    tokenizer.save_pretrained(model_dir)
    write_json(
        model_dir / "router_config.json",
        {
            "confidence_threshold": float(calibrated_threshold),
            "safety_bump": int(calibrated_bump),
            "max_length": max_length,
            "input_prefix": input_prefix,
            "truncation_strategy": truncation_strategy,
            "levels": [1, 2, 3, 4, 5],
            "tasks": list(TASKS),
            "capabilities": list(CAPABILITIES),
            "risks": list(RISKS),
            "task_accuracy": task_accuracy,
            "level_macro_f1": level_f1,
            "risk_accuracy": evaluation["risk_accuracy"],
            "capability_micro_f1": evaluation["capability_micro_f1"],
            "level_ece": evaluation["level_ece"],
            "level_brier_score": evaluation["level_brier_score"],
            "inference_examples_per_second": evaluation["inference_examples_per_second"],
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "wall_time_seconds": time.time() - started_at,
            "level_f1_by_class": evaluation["level_f1_by_class"],
            "task_f1_by_class": evaluation["task_f1_by_class"],
            "preference_accuracy": preference_accuracy,
            "external_router_accuracy": external_router["accuracy"],
        },
    )
    train_summary = {
        "run_name": run_name,
        "base_model": base_model,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "validation_kind": validation_kind,
        "gold_examples": len(gold_rows),
        "epochs": completed_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "input_prefix": input_prefix,
        "truncation_strategy": truncation_strategy,
        "confidence_threshold": float(calibrated_threshold),
        "safety_bump": int(calibrated_bump),
        "class_weight_power": class_weight_power,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "amp_mode": requested_amp_mode,
        "amp_dtype": str(autocast_dtype),
        "level_macro_f1": level_f1,
        "task_accuracy": task_accuracy,
        "task_macro_f1": task_f1,
        "risk_accuracy": evaluation["risk_accuracy"],
        "capability_micro_f1": evaluation["capability_micro_f1"],
        "level_ece": evaluation["level_ece"],
        "level_brier_score": evaluation["level_brier_score"],
        "inference_examples_per_second": evaluation["inference_examples_per_second"],
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "wall_time_seconds": time.time() - started_at,
        "level_f1_by_class": evaluation["level_f1_by_class"],
        "task_f1_by_class": evaluation["task_f1_by_class"],
        "preference_accuracy": preference_accuracy,
        "preference_validation_examples": len(preference_val_rows),
        "external_router_benchmark": external_router,
        "validation_loss": val_loss,
        "gpu": gpu_name,
        "progress_path": f"/runs/{run_name}/progress.jsonl",
        "metrics": business_metrics,
        "raw_metrics": raw_business_metrics,
        "model_path": f"/runs/{run_name}/model",
        "tasks": list(TASKS),
        "capabilities": list(CAPABILITIES),
        "risks": list(RISKS),
    }
    write_json(output_dir / "metrics.json", train_summary)
    publish({
        "status": "completed",
        "epoch": completed_epochs,
        "epochs": completed_epochs,
        "val_loss": val_loss,
        "level_macro_f1": level_f1,
        "task_accuracy": task_accuracy,
        "task_macro_f1": task_f1,
        "risk_accuracy": evaluation["risk_accuracy"],
        "capability_micro_f1": evaluation["capability_micro_f1"],
        "inference_examples_per_second": evaluation["inference_examples_per_second"],
        "preference_accuracy": preference_accuracy,
        "external_router_accuracy": external_router["accuracy"],
        "elapsed_seconds": time.time() - started_at,
        "business": business_metrics,
        "metrics": train_summary,
    })
    volume.commit()
    return train_summary


@app.function(image=image, volumes={ARTIFACTS_DIR: volume}, timeout=60 * 60 * 2)
def upload_dataset(volume_path: str, content: str) -> str:
    output = ARTIFACTS_DIR / volume_path.lstrip("/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    volume.commit()
    return str(output)


@app.function(image=image, volumes={ARTIFACTS_DIR: volume}, timeout=60 * 60 * 4)
def prepare_hf_dataset(volume_path: str, max_per_source: int = 10000, seed: int = 42) -> dict[str, Any]:
    from routeur.hf_data import build_hf_examples
    from routeur.io import write_jsonl
    from routeur.synthetic import generate_specialty_examples

    rows = build_hf_examples(max_per_source=max_per_source, seed=seed)
    rows.extend(generate_specialty_examples(max(1000, max_per_source // 4), seed=seed))
    output = ARTIFACTS_DIR / volume_path.lstrip("/")
    write_jsonl(output, rows)
    volume.commit()
    return {"path": str(output), "rows": len(rows)}


@app.function(image=image, volumes={ARTIFACTS_DIR: volume}, gpu=["L4", "A10", "T4"], timeout=60 * 60 * 8)
def train_economical(**kwargs: Any) -> dict[str, Any]:
    return _train_impl(**kwargs)


@app.function(image=image, volumes={ARTIFACTS_DIR: volume}, gpu=["L40S", "A100-40GB", "H100"], timeout=60 * 60 * 8)
def train_balanced(**kwargs: Any) -> dict[str, Any]:
    return _train_impl(**kwargs)


@app.function(image=image, volumes={ARTIFACTS_DIR: volume}, gpu="H100", timeout=60 * 60 * 8)
def train_h100(**kwargs: Any) -> dict[str, Any]:
    return _train_impl(**kwargs)


@app.function(image=image, volumes={ARTIFACTS_DIR: volume}, gpu=["H100:8", "H200:8", "B200:8"], timeout=60 * 60 * 8)
def train_max(**kwargs: Any) -> dict[str, Any]:
    return _train_impl(**kwargs)


@app.local_entrypoint()
def train(
    dataset: str,
    profile: str = "economical",
    base_model: str = "xlm-roberta-base",
    run_name: str | None = None,
    epochs: float = 3.0,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    max_length: int = 512,
    validation_ratio: float = 0.15,
    confidence_threshold: float | None = None,
    safety_bump: int | None = None,
    seed: int = 42,
    class_weight_power: float = 0.35,
    gradient_accumulation_steps: int = 1,
    amp_mode: str = "auto",
    truncation_strategy: str = "right",
    max_inline_upload_mb: int = 200,
    patience: int = 3,
) -> None:
    run = run_name or f"router-{int(time.time())}"
    dataset_path = Path(dataset)
    volume_dataset_path = f"/datasets/{dataset_path.name}"
    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        if size_mb > max_inline_upload_mb:
            raise ValueError(f"{dataset} is {size_mb:.1f} MB; upload it to the Modal volume first")
        upload_dataset.remote(volume_dataset_path, dataset_path.read_text(encoding="utf-8"))
    else:
        volume_dataset_path = dataset
    kwargs = {
        "dataset_volume_path": volume_dataset_path,
        "base_model": base_model,
        "run_name": run,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "validation_ratio": validation_ratio,
        "confidence_threshold": confidence_threshold,
        "safety_bump": safety_bump,
        "seed": seed,
        "class_weight_power": class_weight_power,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "amp_mode": amp_mode,
        "truncation_strategy": truncation_strategy,
        "patience": patience,
    }
    if profile == "economical":
        result = train_economical.remote(**kwargs)
    elif profile == "balanced":
        result = train_balanced.remote(**kwargs)
    elif profile == "h100":
        result = train_h100.remote(**kwargs)
    elif profile == "max":
        result = train_max.remote(**kwargs)
    else:
        raise ValueError("profile must be economical, balanced, h100, or max")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
