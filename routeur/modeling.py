from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_CAPABILITY_LOSS_WEIGHT = 0.15
DEFAULT_RISK_LOSS_WEIGHT = 0.10


class MultiTaskRouterModel(nn.Module):
    """Shared encoder for difficulty, task, risk and capability requirements."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_tasks: int,
        num_capabilities: int = 0,
        num_risks: int = 0,
        pooling: str = "mean",
        dropout: float = 0.1,
        capability_loss_weight: float = DEFAULT_CAPABILITY_LOSS_WEIGHT,
        risk_loss_weight: float = DEFAULT_RISK_LOSS_WEIGHT,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(getattr(encoder.config, "hidden_size", getattr(encoder.config, "dim", 768)))
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.pooler = (
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            if pooling == "hybrid"
            else None
        )
        self.level_head = nn.Linear(hidden_size, 5)
        self.task_head = nn.Linear(hidden_size, num_tasks)
        self.capability_head = nn.Linear(hidden_size, num_capabilities) if num_capabilities else None
        self.risk_head = nn.Linear(hidden_size, num_risks) if num_risks else None
        self.num_tasks = num_tasks
        self.num_capabilities = num_capabilities
        self.num_risks = num_risks
        self.capability_loss_weight = float(capability_loss_weight)
        self.risk_loss_weight = float(risk_loss_weight)

    def forward(
        self,
        *,
        level_labels: torch.Tensor | None = None,
        task_labels: torch.Tensor | None = None,
        capability_labels: torch.Tensor | None = None,
        risk_labels: torch.Tensor | None = None,
        **inputs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs.get("attention_mask")
        first = hidden[:, 0]
        if mask is None:
            mean = first
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = self.pooler(torch.cat((first, mean), dim=-1)) if self.pooler is not None else mean
        pooled = self.dropout(pooled)
        level_logits = self.level_head(pooled)
        task_logits = self.task_head(pooled)
        result: dict[str, torch.Tensor] = {"level_logits": level_logits, "task_logits": task_logits}
        if self.capability_head is not None:
            result["capability_logits"] = self.capability_head(pooled)
        if self.risk_head is not None:
            result["risk_logits"] = self.risk_head(pooled)
        if level_labels is not None and task_labels is not None:
            level_loss = nn.functional.cross_entropy(level_logits, level_labels)
            task_loss = nn.functional.cross_entropy(task_logits, task_labels)
            loss = 0.72 * level_loss + 0.28 * task_loss
            if self.capability_head is not None and capability_labels is not None:
                capability_loss = nn.functional.binary_cross_entropy_with_logits(
                    result["capability_logits"], capability_labels.float()
                )
                loss = loss + self.capability_loss_weight * capability_loss
            if self.risk_head is not None and risk_labels is not None:
                risk_loss = nn.functional.cross_entropy(result["risk_logits"], risk_labels)
                loss = loss + self.risk_loss_weight * risk_loss
            result["loss"] = loss
        return result

    @classmethod
    def from_base_model(
        cls,
        base_model: str,
        *,
        num_tasks: int,
        num_capabilities: int = 0,
        num_risks: int = 0,
        pooling: str = "hybrid",
        capability_loss_weight: float = DEFAULT_CAPABILITY_LOSS_WEIGHT,
        risk_loss_weight: float = DEFAULT_RISK_LOSS_WEIGHT,
    ) -> "MultiTaskRouterModel":
        from transformers import AutoModel

        return cls(
            AutoModel.from_pretrained(base_model),
            num_tasks=num_tasks,
            num_capabilities=num_capabilities,
            num_risks=num_risks,
            pooling=pooling,
            capability_loss_weight=capability_loss_weight,
            risk_loss_weight=risk_loss_weight,
        )

    def save_pretrained(self, output_dir: str | Path, *, base_model: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output / "encoder")
        heads = {"level_head": self.level_head.state_dict(), "task_head": self.task_head.state_dict()}
        if self.capability_head is not None:
            heads["capability_head"] = self.capability_head.state_dict()
        if self.risk_head is not None:
            heads["risk_head"] = self.risk_head.state_dict()
        if self.pooler is not None:
            heads["pooler"] = self.pooler.state_dict()
        torch.save(heads, output / "heads.pt")
        (output / "router_model.json").write_text(
            json.dumps(
                {
                    "architecture": f"multitask_{self.pooling}_pool",
                    "base_model": base_model,
                    "num_tasks": self.num_tasks,
                    "num_capabilities": self.num_capabilities,
                    "num_risks": self.num_risks,
                    "pooling": self.pooling,
                    "num_levels": 5,
                    "capability_loss_weight": self.capability_loss_weight,
                    "risk_loss_weight": self.risk_loss_weight,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "MultiTaskRouterModel":
        from transformers import AutoModel

        directory = Path(model_dir)
        config = json.loads((directory / "router_model.json").read_text(encoding="utf-8"))
        encoder = AutoModel.from_pretrained(directory / "encoder")
        model = cls(
            encoder,
            num_tasks=int(config["num_tasks"]),
            num_capabilities=int(config.get("num_capabilities", 0)),
            num_risks=int(config.get("num_risks", 0)),
            pooling=str(config.get("pooling", "mean")),
            capability_loss_weight=float(config.get("capability_loss_weight", DEFAULT_CAPABILITY_LOSS_WEIGHT)),
            risk_loss_weight=float(config.get("risk_loss_weight", DEFAULT_RISK_LOSS_WEIGHT)),
        )
        state = torch.load(directory / "heads.pt", map_location="cpu", weights_only=True)
        model.level_head.load_state_dict(state["level_head"])
        model.task_head.load_state_dict(state["task_head"])
        if model.capability_head is not None and "capability_head" in state:
            model.capability_head.load_state_dict(state["capability_head"])
        if model.risk_head is not None and "risk_head" in state:
            model.risk_head.load_state_dict(state["risk_head"])
        if model.pooler is not None and "pooler" in state:
            model.pooler.load_state_dict(state["pooler"])
        return model
