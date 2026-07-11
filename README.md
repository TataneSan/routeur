# Routeur: sub-10ms multi-model prompt routing

Routeur is a production-oriented prompt router that selects a concrete LLM from a
live catalog instead of returning only an abstract tier. It predicts difficulty,
task, risk, and required capabilities, then ranks callable models using public
Arena/LMArena priors, capability coverage, and relative cost.

The repository ships three routing paths:

- `OnnxRouter`: the promoted semantic path, available as a 5 MB economy model
  and a 12 MB accuracy model. Both are INT8 CPU artifacts below the 10 ms p95 SLO.
- `FastRouter`: a portable 52 MB linear multi-task baseline with word and
  character n-grams.
- `TransformerRouter`: a higher-capacity multilingual encoder used for offline
  experiments and accuracy comparisons when a 100+ ms CPU budget is acceptable.

The `level` field from 1 to 5 remains as an internal compatibility signal. The
primary decision is `model`, with ranked fallbacks in `model_candidates`.

## Measured results

The semantic students were trained on 49,984 unique prompts. The validation split
contains only teacher-labelled examples whose normalized prompts are excluded from
training. It is a validation set used for model selection, not an untouched test set.

| Metric | Linear baseline | Economy ONNX | Accuracy ONNX |
|---|---:|---:|---:|
| Exact level accuracy | 0.4839 | 0.4915 | **0.5123** |
| Adjacent accuracy | 0.9192 | 0.9268 | **0.9338** |
| Severe under-routing | 0.0278 | 0.0158 | **0.0132** |
| Task accuracy | 0.5950 | 0.5830 | **0.6183** |
| Savings vs always tier 5 | 0.8898 | **0.8907** | 0.8810 |
| Cost ratio vs labelled oracle | - | **1.0357** | 1.1276 |
| Artifact size | 52 MB | **5 MB** | 12 MB |

In 1,000-request uncached, single-prompt runs, economy/accuracy p95 latency was
2.84/4.76 ms on a local Xeon E3-1271 v3 and 1.61/3.62 ms on a Hetzner Xeon
E5-1650 v3. Measurements were sequential with one ONNX Runtime thread.

The enforced service objective is p95 below 10 ms. The former XLM-R path measured
112.3 ms median on the same machine. `models/profiles.json` is the machine-readable
profile comparison.

Exact, under-route, and over-route rates partition all decisions. Requiring both
under-routing and over-routing below 0.1% therefore requires at least 99.8% exact
accuracy. The current subjective teacher labels do not support that claim: even
high-confidence rows contain boundary disagreement. The repository reports the
measured ceiling instead of tuning on validation labels or manufacturing a score.

The full architecture, training procedure, cost analysis, and limitations are in
[`paper/routeur_paper.tex`](paper/routeur_paper.tex) and the compiled
[`output/pdf/routeur_paper.pdf`](output/pdf/routeur_paper.pdf).

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[onnx,dev]'

routeur route \
  --model-dir models/semantic-router-v1 \
  --prompt "Debug this Python production timeout"
```

Example output fields:

```json
{
  "model": "openai/gpt-5-4-mini-high",
  "model_candidates": ["anthropic/claude-opus-4-7", "..."],
  "level": 4,
  "raw_level": 4,
  "task": "coding",
  "risk": "medium",
  "required_capabilities": ["coding", "reasoning"],
  "confidence": 0.62,
  "reason": "onnx_model"
}
```

Use `models/semantic-router-tiny-v1` for the lowest cost and smallest artifact.
The default relies on the learned risk, safety, and capability heads. Applications
can enable `--safety-guard-mode model_confirmed` or `lexical`; both deliberately
trade higher cost and over-routing for conservative escalation.

## Verify latency

The benchmark generates unique requests so the result cannot be satisfied by the
router's LRU cache:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_latency.py \
  --model-dir models/semantic-router-v1 \
  --iterations 1000 \
  --max-p95-ms 10
```

The command exits with status 2 if p95 misses the target.

## Rebuild the recent data mixture

Raw third-party datasets are not copied into this Git repository. The reproducible
builder downloads their official releases, creates derived training rows, and keeps
external benchmarks separate from training:

```bash
pip install -e '.[fast,train]'

python scripts/build_new_datasets.py \
  --base-dataset data/router_train_v3_gold_pairwise.jsonl \
  --output-dataset data/router_train_v4_mmr.jsonl \
  --max-mmr-rows 15000
```

The v4 addition uses observed quality gaps from
[`JiaqiXue/mmr-routing-20k`](https://huggingface.co/datasets/JiaqiXue/mmr-routing-20k),
which contains multi-turn WildChat conversations scored independently for several
weak and strong LLMs. The builder also creates untouched evaluation sets from
[`Amorph/TwinRouterBench`](https://huggingface.co/datasets/Amorph/TwinRouterBench)
and
[`Lance1573/CodeRouterBench`](https://huggingface.co/datasets/Lance1573/CodeRouterBench).

See [DATASETS.md](DATASETS.md) for source roles, licenses, mapping rules, and
leakage controls.

## Train the fast model

```bash
python scripts/train_fast_router.py \
  --dataset data/router_train_v4_mmr.jsonl \
  --output-dir models/fast-router-v3 \
  --n-features 262144 \
  --alpha 3e-5 \
  --max-iter 30
```

The model has four supervised heads:

- five-way difficulty;
- eleven-way task specialty;
- low/medium/high risk;
- nine independent capability requirements.

Serving uses one sparse feature pass and one active-column matrix multiplication.
Only n-grams present in the prompt are scored, avoiding a dense scan of the full
coefficient matrix. Inputs are head-tail bounded at 4,096 characters so very long
prompts cannot violate the latency budget through unbounded tokenization work.

Artifacts contain only JSON and compressed NumPy arrays. No pickle or arbitrary
code deserialization is required.

## Transformer experiments on Modal

The existing GPU pipeline remains available for accuracy research:

```bash
pip install -e '.[train,modal]'

modal run modal_train.py \
  --dataset data/router_train_v4_mmr.jsonl \
  --profile h100 \
  --base-model google/bert_uncased_L-4_H-256_A-4 \
  --run-name semantic-router-v1 \
  --epochs 6 \
  --batch-size 128 \
  --max-length 128
```

Available profiles are `economical` (`L4` -> `A10` -> `T4`), `balanced`, `h100`,
and `max`. Training artifacts are stored in the Modal volume `routeur-artifacts`
under `/runs/<run-name>/`.

After downloading the selected Modal checkpoint, export and quantize it with:

```bash
python scripts/export_onnx_router.py \
  --model-dir artifacts/semantic-router-v1/model \
  --output-dir models/semantic-router-v1
```

Larger E5 and six-layer MiniLM experiments were rejected because they missed the
CPU latency objective or failed to improve the measured Pareto frontier.

## Data and trace labels

A direct training row uses:

```json
{"prompt":"Debug this Python traceback", "level":4, "task":"coding", "source":"production"}
```

Production evaluations can be converted into cost-aware labels:

```json
{
  "prompt": "Review this production patch",
  "results": [
    {"level": 1, "quality": 0.20, "cost_usd": 0.00002},
    {"level": 5, "quality": 0.95, "cost_usd": 0.00450}
  ]
}
```

```bash
routeur label \
  --input data/example_traces.jsonl \
  --output /tmp/router_train.jsonl \
  --min-quality 0.90
```

The labeler selects the cheapest tier meeting the quality threshold, or the
highest-quality result when none passes.

## Model catalog

`configs/models_lmarena.json` contains the generated 450-model policy snapshot.
The ranker combines task probabilities, required capabilities, public benchmark
coverage, tier fit, relative cost, and provider-diverse fallbacks. Non-callable
Arena entries can appear as reference candidates but are not selected as the
primary model while an eligible callable model exists.

The fast path uses cached per-task shortlists to avoid scanning all 450 models on
every request. This reduced policy overhead while preserving the strongest
specialists and provider diversity.

## Evaluation metrics

`routeur eval` reports:

- exact and adjacent level accuracy;
- mean absolute error;
- under-routing and severe under-routing;
- over-routing;
- routed cost versus the oracle and always-level-5 baselines;
- task accuracy and concrete model counts.

Exact level accuracy is not sufficient on its own. A useful router must also
control severe under-routing, preserve quality, and avoid spending the latency or
cost savings it was built to create.

## Tests

```bash
pytest -q
python -m compileall routeur scripts modal_train.py
```

## License

The Routeur source code is released under the MIT License. Upstream datasets retain
their own licenses and terms; they are downloaded directly from their publishers.
