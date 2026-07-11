from scripts.build_new_datasets import mmr_level


def test_mmr_level_maps_observed_quality_gap():
    assert mmr_level(9, 9) == 1
    assert mmr_level(7, 8) == 2
    assert mmr_level(5, 8) == 3
    assert mmr_level(3, 8) == 4
    assert mmr_level(2, 10) == 5
