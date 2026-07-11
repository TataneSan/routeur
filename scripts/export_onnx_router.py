#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from routeur.modeling import MultiTaskRouterModel


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: MultiTaskRouterModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return (
            outputs["level_logits"],
            outputs["task_logits"],
            outputs["risk_logits"],
            outputs["capability_logits"],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and dynamically quantize a router for CPU ONNX inference.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    source_config = json.loads((args.model_dir / "router_config.json").read_text(encoding="utf-8"))
    model_metadata = json.loads((args.model_dir / "router_model.json").read_text(encoding="utf-8"))
    max_length = int(args.max_length or source_config.get("max_length", 128))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = MultiTaskRouterModel.from_pretrained(args.model_dir).eval()
    wrapper = ExportWrapper(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = tokenizer(
        "Classify this production routing request",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    if "token_type_ids" not in sample:
        sample["token_type_ids"] = torch.zeros_like(sample["input_ids"])
    float_path = args.output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        float_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["level_logits", "task_logits", "risk_logits", "capability_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "level_logits": {0: "batch"},
            "task_logits": {0: "batch"},
            "risk_logits": {0: "batch"},
            "capability_logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantized_path = args.output_dir / "model.int8.onnx"
    quantize_dynamic(
        float_path,
        quantized_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    tokenizer.save_pretrained(args.output_dir)
    config = {
        "architecture": "quantized_onnx_multitask_transformer_v1",
        "model_file": quantized_path.name,
        "levels": [1, 2, 3, 4, 5],
        "tasks": source_config["tasks"],
        "risks": source_config["risks"],
        "capabilities": source_config["capabilities"],
        "max_length": max_length,
        "input_prefix": source_config.get("input_prefix", ""),
        "temperature": source_config.get("temperature", 1.0),
        "confidence_threshold": source_config.get("confidence_threshold", 0.0),
        "safety_bump": source_config.get("safety_bump", 0),
        "safety_guard_mode": "off",
        "base_model": model_metadata.get("base_model"),
        "quantization": "dynamic_int8_qint8_per_channel_reduce_range",
    }
    (args.output_dir / "onnx_router.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.model_dir / "router_model.json", args.output_dir / "router_model.json")
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
