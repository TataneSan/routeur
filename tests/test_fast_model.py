from __future__ import annotations

import json

import numpy as np

from routeur.capabilities import CAPABILITIES, RISKS
from routeur.fast_model import FastModelArtifact, prepare_prompt
from routeur.router import FastRouter, load_router
from routeur.tasks import TASKS


def write_artifact(path) -> None:
    path.mkdir()
    n_features = 16
    total_features = n_features * 2
    config = {
        "architecture": "hashed_word_char_linear_multitask_v1",
        "levels": [1, 2, 3, 4, 5],
        "tasks": list(TASKS),
        "risks": list(RISKS),
        "capabilities": list(CAPABILITIES),
        "n_features": n_features,
        "word_ngrams": [1, 2],
        "char_ngrams": [3, 5],
        "max_chars": 128,
    }
    (path / "fast_router.json").write_text(json.dumps(config), encoding="utf-8")
    np.savez_compressed(
        path / "fast_router.npz",
        level_coef=np.zeros((5, total_features), dtype=np.float32),
        level_intercept=np.asarray([5, 0, 0, 0, 0], dtype=np.float32),
        task_coef=np.zeros((len(TASKS), total_features), dtype=np.float32),
        task_intercept=np.asarray([5, *([0] * (len(TASKS) - 1))], dtype=np.float32),
        risk_coef=np.zeros((len(RISKS), total_features), dtype=np.float32),
        risk_intercept=np.asarray([5, 0, 0], dtype=np.float32),
        capability_coef=np.zeros((len(CAPABILITIES), total_features), dtype=np.float32),
        capability_intercept=np.full(len(CAPABILITIES), -5, dtype=np.float32),
    )


def test_prepare_prompt_keeps_both_ends():
    value = prepare_prompt("abcdefghij", max_chars=6)
    assert value.startswith("abc")
    assert value.endswith("hij")
    assert "TRUNCATED" in value


def test_fast_artifact_predicts_all_heads(tmp_path):
    model_dir = tmp_path / "model"
    write_artifact(model_dir)
    predictions = FastModelArtifact(model_dir).predict(["hello"])
    assert predictions.level_probabilities.shape == (1, 5)
    assert predictions.task_probabilities.shape == (1, len(TASKS))
    assert predictions.risk_probabilities.shape == (1, len(RISKS))
    assert predictions.capability_probabilities.shape == (1, len(CAPABILITIES))


def test_load_router_detects_fast_artifact_and_keeps_safety_guard(tmp_path):
    model_dir = tmp_path / "model"
    write_artifact(model_dir)
    router = load_router(model_dir)
    assert isinstance(router, FastRouter)
    normal = router.route("Say hello")
    assert normal.raw_level == 1
    safety = router.route("Review this production security vulnerability and exploit risk")
    assert safety.level == 5
    assert safety.task == "safety"
    assert safety.reason == "safety_guard"
