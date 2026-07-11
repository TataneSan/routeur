from __future__ import annotations

from typing import Any

from .capabilities import RISKS


def calibrated_levels(
    probabilities: Any,
    threshold: float,
    bump: int,
    risk_preds: list[int] | None = None,
) -> list[int]:
    """Apply the production confidence bump only to high-risk predictions."""
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
        routed = (
            min(4, int(level) + int(bump))
            if high_risk and float(conf) < effective_threshold
            else int(level)
        )
        predictions.append(routed + 1)
    return predictions
