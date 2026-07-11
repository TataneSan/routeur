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
