# Dataset provenance and evaluation policy

Routeur separates training data from external benchmarks. Dataset builders preserve
the upstream source name in every row and use exact normalized-prompt deduplication.

## Objective-oracle sources added in v6

### DARS repeated-generation routing data

- Source: `AIGNLAI/DARS`.
- Role: primary objective-oracle training and external test data.
- Coverage: 90,000 train and 41,328 test generations over GPQA, MATH-500,
  and DROP, with six candidate models, prompt rewrites, repeated decodes,
  automatic scores, parse status, and observed cost.
- Mapping: `scripts/build_objective_oracle.py` aggregates every observation for
  a query-model pair. It selects the cheapest model whose one-sided lower
  confidence bound clears the quality threshold. If none clears it, the model
  with the strongest lower bound is selected as an explicit fallback.
- Stability: 200 bootstrap resamples estimate whether the route level is
  reproducible. The strict corpus retains only labels reproduced at least 99.8%
  of the time; all other rows require adjudication and are not gold labels.
- Leakage control: all rewrites and decodes sharing a `query_id` stay in the same
  partition. The upstream test split never enters training.

### R2-Bench model and token-budget traces

- Source: `JiaqiXue/R2-Bench` (MIT).
- Role: secondary objective-trace source and future joint model/budget training.
- Coverage: approximately 4.35 million published evaluations spanning 30,968
  queries, ten LLMs, and sixteen output-token budgets, with judge correctness
  scores and actual token counts.
- Raw size is approximately 25 GB, so the repository does not redistribute it.

## Sources added in v4

### MMR multi-turn routing data

- Source: `JiaqiXue/mmr-routing-20k`
- Upstream license: Apache-2.0; conversations originate from WildChat and remain
  subject to its upstream terms.
- Role: training plus a disjoint 3,000-row holdout.
- Signal: observed judge scores for Qwen3-0.6B and
  Qwen3-30B-A3B-Instruct-2507.
- Mapping: the five route levels are derived from weak-model quality, strong-model
  quality, and their gap. The mapping is implemented and tested in
  `scripts/build_new_datasets.py`.
- Context: later user turns include the visible conversation prefix.

The released v4 training mixture samples 15,000 MMR turns and caps any one level at
45% of that addition. It therefore improves real multi-turn coverage without
overwhelming higher-confidence teacher supervision.

### TwinRouterBench

- Source: `Amorph/TwinRouterBench`
- License: Apache-2.0.
- Role: external evaluation only.
- Coverage: 970 agentic routing decisions from SWE-bench, BFCL, mtRAG, QMSum, and
  PinchBench.
- Important limitation: the publisher marks parts of the release as weak-label
  degradation-search data rather than strict ground truth. Results are diagnostic,
  not a headline quality claim.

### CodeRouterBench OOD176

- Source: `Lance1573/CodeRouterBench`
- License: MIT.
- Role: external evaluation only.
- Coverage: 176 out-of-distribution coding and agent tasks evaluated against eight
  backend models.
- Mapping: choose the cheapest backend that actually resolves each task, then map
  its published `low`, `mid`, `high`, or `premium` tier to route levels 2-5. If no
  backend succeeds, the oracle is level 5.

## Existing mixture

The v3 base mixture contains teacher annotations and weak supervision drawn from:

- `agentlans/prompt-difficulty-model-ratings`;
- `HuggingFaceH4/no_robots`;
- `OpenAssistant/oasst1`;
- `allenai/WildChat-1M`;
- `SupraLabs/Prompt-Routing-Dataset`;
- `somukandula/prompt-router-dataset`;
- `anasnassar/llm-query-complexity-benchmark`;
- `routellm/gpt4_dataset`;
- `lmarena-ai/arena-human-preference-55k`;
- targeted synthetic specialty examples.

Before redistributing any derived dataset, review every upstream license and term.
This repository intentionally publishes the reproducible builders rather than raw
third-party data.

## Leakage controls

1. Teacher-gold validation rows are selected before training.
2. Every exact normalized validation prompt is removed from the full training mix.
3. Rows sharing an objective-oracle `query_id`, including paraphrases, are split
   as one group so a rewritten test question cannot leak into training.
4. MMR holdout prompts are excluded from both the base and MMR training rows.
5. TwinRouterBench and CodeRouterBench OOD176 never enter training.
6. Rewrites or near-duplicates can still exist across unrelated public corpora. For high-stakes
   benchmarking, add semantic deduplication and source-grouped splits.
