from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def prepare_prompt(prompt: str, *, max_chars: int) -> str:
    """Bound feature extraction cost while retaining both instruction ends."""
    text = str(prompt)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n[...TRUNCATED...]\n" + text[-tail:]


def build_vectorizer(*, n_features: int, word_ngrams: tuple[int, int], char_ngrams: tuple[int, int]):
    """Build the stateless feature extractor shared by training and serving."""
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.pipeline import FeatureUnion
    except ImportError as exc:  # pragma: no cover - exercised by optional dependency users
        raise RuntimeError("Install routeur[fast] to use the sub-10ms router") from exc

    return FeatureUnion(
        [
            (
                "word",
                HashingVectorizer(
                    n_features=n_features,
                    alternate_sign=False,
                    analyzer="word",
                    ngram_range=word_ngrams,
                    norm="l2",
                    lowercase=True,
                ),
            ),
            (
                "char",
                HashingVectorizer(
                    n_features=n_features,
                    alternate_sign=False,
                    analyzer="char_wb",
                    ngram_range=char_ngrams,
                    norm="l2",
                    lowercase=True,
                ),
            ),
        ]
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True).clip(min=1e-12)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


@dataclass(frozen=True)
class FastPredictions:
    level_probabilities: np.ndarray
    task_probabilities: np.ndarray
    risk_probabilities: np.ndarray
    capability_probabilities: np.ndarray


class FastModelArtifact:
    """Portable linear multi-task model with deterministic hashed features.

    The artifact uses JSON and NPZ only. It deliberately avoids pickle/joblib,
    making models inspectable and safe to load from a public release.
    """

    def __init__(self, model_dir: str | Path) -> None:
        directory = Path(model_dir)
        config_path = directory / "fast_router.json"
        weights_path = directory / "fast_router.npz"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Fast router artifact is incomplete: {directory}")
        self.directory = directory
        self.config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        self.tasks = tuple(str(value) for value in self.config["tasks"])
        self.risks = tuple(str(value) for value in self.config["risks"])
        self.capabilities = tuple(str(value) for value in self.config["capabilities"])
        self.levels = tuple(int(value) for value in self.config.get("levels", [1, 2, 3, 4, 5]))
        self.max_chars = int(self.config.get("max_chars", 4096))
        self.vectorizer = build_vectorizer(
            n_features=int(self.config["n_features"]),
            word_ngrams=tuple(self.config.get("word_ngrams", [1, 2])),
            char_ngrams=tuple(self.config.get("char_ngrams", [3, 5])),
        )
        weights = np.load(weights_path, allow_pickle=False)
        self.level_coef = weights["level_coef"]
        self.level_intercept = weights["level_intercept"]
        self.task_coef = weights["task_coef"]
        self.task_intercept = weights["task_intercept"]
        self.risk_coef = weights["risk_coef"]
        self.risk_intercept = weights["risk_intercept"]
        self.capability_coef = weights["capability_coef"]
        self.capability_intercept = weights["capability_intercept"]
        # One sparse matrix multiplication is materially faster than four for
        # single-prompt serving. Keep slices so each head remains explicit.
        sizes = (
            len(self.level_intercept),
            len(self.task_intercept),
            len(self.risk_intercept),
            len(self.capability_intercept),
        )
        boundaries = np.cumsum((0, *sizes))
        self._level_slice = slice(int(boundaries[0]), int(boundaries[1]))
        self._task_slice = slice(int(boundaries[1]), int(boundaries[2]))
        self._risk_slice = slice(int(boundaries[2]), int(boundaries[3]))
        self._capability_slice = slice(int(boundaries[3]), int(boundaries[4]))
        self._all_coef = np.concatenate(
            (self.level_coef, self.task_coef, self.risk_coef, self.capability_coef), axis=0
        )
        self._all_intercept = np.concatenate(
            (self.level_intercept, self.task_intercept, self.risk_intercept, self.capability_intercept)
        )

    def predict(self, prompts: list[str]) -> FastPredictions:
        features = self.vectorizer.transform(
            [prepare_prompt(prompt, max_chars=self.max_chars) for prompt in prompts]
        )
        # scipy's CSR @ dense path scans the entire dense coefficient matrix,
        # which is unexpectedly costly for a single short prompt. Indexing only
        # the active hashed n-grams makes latency proportional to prompt length.
        logits = np.empty((features.shape[0], self._all_coef.shape[0]), dtype=np.float32)
        for row_index in range(features.shape[0]):
            start, end = features.indptr[row_index : row_index + 2]
            indices = features.indices[start:end]
            values = features.data[start:end].astype(np.float32, copy=False)
            logits[row_index] = self._all_coef[:, indices] @ values + self._all_intercept
        level_logits = logits[:, self._level_slice]
        task_logits = logits[:, self._task_slice]
        risk_logits = logits[:, self._risk_slice]
        capability_logits = logits[:, self._capability_slice]
        return FastPredictions(
            level_probabilities=_softmax(level_logits),
            task_probabilities=_softmax(task_logits),
            risk_probabilities=_softmax(risk_logits),
            capability_probabilities=_sigmoid(capability_logits),
        )
