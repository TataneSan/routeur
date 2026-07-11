# Routeur: sub-10ms multi-model prompt routing

Routeur is a production-oriented prompt router that selects a concrete LLM from a
live catalog instead of returning only an abstract tier. It predicts difficulty,
task, risk, and required capabilities, then ranks callable models using public
Arena/LMArena priors, capability coverage, and relative cost.

The repository ships two routing paths:

- `FastRouter`: a portable 52 MB linear multi-task model with word and character
  n-grams. It is the default production path and runs below 10 ms on an old CPU.
- `TransformerRouter`: a higher-capacity multilingual encoder used for offline
  experiments and accuracy comparisons when a 100+ ms CPU budget is acceptable.

The `level` field from 1 to 5 remains as an internal compatibility signal. The
primary decision is `model`, with ranked fallbacks in `model_candidates`.

## Measured results

The promoted `models/fast-router-v3` artifact was trained on 49,984 unique prompts.
Its validation split is held out exclusively from high-confidence teacher-labeled
examples; exact normalized prompts are removed from training.

| Metric | Fast v2 | Fast v3 | Change |
|---|---:|---:|---:|
| Gold exact level accuracy | 0.4751 | **0.4839** | +0.0088 |
| Gold adjacent accuracy | 0.9091 | **0.9192** | +0.0101 |
| Severe under-routing | 0.0372 | **0.0278** | -0.0095 |
| Task accuracy | 0.5703 | **0.5950** | +0.0246 |
| Risk accuracy | 0.7199 | **0.7356** | +0.0158 |
| Savings vs always tier 5 | 0.8804 | **0.8898** | +0.0093 |
| MMR unseen-turn accuracy | 0.1287 | **0.3280** | +0.1993 |
| MMR unseen-turn task accuracy | 0.5673 | **0.7497** | +0.1823 |

On an Intel Xeon E3-1271 v3, 1,000 uncached single-prompt decisions measured:

- median: **1.85 ms**
- p95: **2.97 ms**
- p99: **3.77 ms**
- maximum: 11.10 ms

The same 1,000-request protocol on the Hetzner Xeon E5-1650 v3 server measured
**1.73 ms median** and **1.89 ms p95**.

The enforced service objective is p95 below 10 ms. The former XLM-R path measured
112.3 ms median on the same machine. See `models/fast-router-v3/evaluation.json`
for the machine-readable report and limitations.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[fast,dev]'

routeur route \
  --model-dir models/fast-router-v3 \
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
  "reason": "fast_model"
}
```

The explicit lexical safety guard overrides learned predictions for high-stakes
security prompts and routes them to level 5.

## Verify latency

The benchmark generates unique requests so the result cannot be satisfied by the
router's LRU cache:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_latency.py \
  --model-dir models/fast-router-v3 \
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
  --base-model intfloat/multilingual-e5-large-instruct \
  --run-name router-e5-experiment \
  --epochs 4 \
  --batch-size 16 \
  --gradient-accumulation-steps 2
```

Available profiles are `economical` (`L4` -> `A10` -> `T4`), `balanced`, `h100`,
and `max`. Training artifacts are stored in the Modal volume `routeur-artifacts`
under `/runs/<run-name>/`.

The Transformer path is intentionally not promoted for latency-sensitive serving:
the best E5 experiment reached 0.5785 exact level accuracy on the teacher holdout,
but its encoder is hundreds of millions of parameters and does not meet the CPU
10 ms objective.

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
