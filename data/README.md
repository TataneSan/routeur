# Data directory

Only tiny examples are versioned. Larger training files and external benchmark
derivatives are generated locally and ignored by Git.

Build the v4 mixture with:

```bash
python scripts/build_new_datasets.py \
  --base-dataset data/router_train_v3_gold_pairwise.jsonl \
  --output-dataset data/router_train_v4_mmr.jsonl
```

See `DATASETS.md` for provenance, licensing, label mappings, and evaluation rules.
