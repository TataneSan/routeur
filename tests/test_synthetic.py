from routeur.synthetic import generate_seed_examples


def test_generate_seed_examples_covers_all_levels():
    rows = generate_seed_examples(25, seed=1)
    assert {row["level"] for row in rows} == {1, 2, 3, 4, 5}
