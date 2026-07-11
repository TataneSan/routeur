from routeur.metrics import SCORING_WEIGHTS, routing_metrics


def test_routing_metrics_tracks_underroute_and_savings():
    metrics = routing_metrics([1, 3, 5], [1, 2, 4])

    assert metrics["accuracy"] == 1 / 3
    assert metrics["underroute_rate"] == 2 / 3
    assert metrics["severe_underroute_rate"] == 0
    assert metrics["savings_vs_always_level_5"] > 0


def test_scoring_weights_cover_key_metrics():
    assert "level_macro_f1" in SCORING_WEIGHTS
    assert "severe_underroute_rate" in SCORING_WEIGHTS
    assert "underroute_rate" in SCORING_WEIGHTS
    assert "overroute_rate" in SCORING_WEIGHTS
    assert "level_ece" in SCORING_WEIGHTS
    assert SCORING_WEIGHTS["severe_underroute_rate"] < 0
    assert SCORING_WEIGHTS["level_macro_f1"] > 0


def test_modal_calibration_only_bumps_high_risk_predictions():
    import numpy as np

    from routeur.calibration import calibrated_levels

    probabilities = np.asarray([[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]])
    # RISKS is low=0, medium=1, high=2. Only the second prediction is bumped.
    assert calibrated_levels(probabilities, threshold=0.8, bump=1, risk_preds=[0, 2]) == [3, 4]
