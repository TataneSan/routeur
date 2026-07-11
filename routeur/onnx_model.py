from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .fast_model import FastPredictions, _sigmoid, _softmax


class OnnxModelArtifact:
    """Tokenizer plus optimized ONNX multi-task classifier artifact."""

    def __init__(self, model_dir: str | Path) -> None:
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("Install routeur[onnx] to use the ONNX router") from exc

        directory = Path(model_dir)
        config_path = directory / "onnx_router.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing ONNX router config: {config_path}")
        self.config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        self.directory = directory
        self.tasks = tuple(str(value) for value in self.config["tasks"])
        self.risks = tuple(str(value) for value in self.config["risks"])
        self.capabilities = tuple(str(value) for value in self.config["capabilities"])
        self.levels = tuple(int(value) for value in self.config.get("levels", [1, 2, 3, 4, 5]))
        self.max_length = int(self.config.get("max_length", 128))
        self.input_prefix = str(self.config.get("input_prefix", ""))
        self.temperature = float(self.config.get("temperature", 1.0))
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = int(os.getenv("ROUTEUR_ONNX_THREADS", "1"))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        model_file = directory / str(self.config.get("model_file", "model.int8.onnx"))
        self.session = ort.InferenceSession(
            str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict(self, prompts: list[str]) -> FastPredictions:
        encoded = self.tokenizer(
            [self.input_prefix + prompt for prompt in prompts],
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {
            name: np.asarray(value, dtype=np.int64)
            for name, value in encoded.items()
            if name in self.input_names
        }
        level_logits, task_logits, risk_logits, capability_logits = self.session.run(None, inputs)
        return FastPredictions(
            level_probabilities=_softmax(level_logits / max(self.temperature, 1e-6)),
            task_probabilities=_softmax(task_logits),
            risk_probabilities=_softmax(risk_logits),
            capability_probabilities=_sigmoid(capability_logits),
        )
