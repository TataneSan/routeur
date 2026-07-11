from routeur.labels import label_trace_row


def test_label_trace_row_chooses_cheapest_passing_level():
    row = {
        "prompt": "debug code",
        "results": [
            {"level": 1, "quality": 0.2, "cost_usd": 0.01},
            {"level": 2, "quality": 0.8, "cost_usd": 0.02},
            {"level": 3, "quality": 0.9, "cost_usd": 0.03},
        ],
    }

    labelled = label_trace_row(row, min_quality=0.75)

    assert labelled["level"] == 2


def test_label_trace_row_falls_back_to_best_quality():
    row = {
        "prompt": "hard task",
        "results": [
            {"level": 1, "quality": 0.2},
            {"level": 5, "quality": 0.7},
            {"level": 4, "quality": 0.65},
        ],
    }

    labelled = label_trace_row(row, min_quality=0.9)

    assert labelled["level"] == 5
