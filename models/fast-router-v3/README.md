# Fast Router v3 model card

Fast Router v3 is a portable linear multi-task classifier for prompt routing. It
uses deterministic hashed word and character n-grams and stores weights as compressed
NumPy arrays.

## Intended use

- Select a difficulty tier and concrete LLM under a 10 ms CPU routing budget.
- Predict task, risk, and required capabilities for downstream policy ranking.
- Act as a fast production alternative to a large Transformer classifier.

## Training data

49,984 unique prompts: the 35,012-row v3 mixture plus 15,000 sampled multi-turn MMR
rows, followed by exact-prompt deduplication. See the repository `DATASETS.md`.

## Held-out teacher metrics

- exact level accuracy: 0.4839
- adjacent accuracy: 0.9192
- macro-F1: 0.4739
- severe under-routing: 0.0278
- task accuracy: 0.5950
- task macro-F1: 0.5924
- risk accuracy: 0.7356
- savings versus always level 5: 0.8898

## Latency

On an Intel Xeon E3-1271 v3 with one BLAS/OpenMP thread, 1,000 unique requests:

- p50: 1.85 ms
- p95: 2.97 ms
- p99: 3.77 ms

On the Hetzner Xeon E5-1650 v3 server, the same protocol measured 1.73 ms p50,
1.89 ms p95, and 2.05 ms p99.

## Limitations

- Exact five-way difficulty remains noisy; adjacent accuracy is much stronger.
- Public benchmark labels use different tier definitions and cannot be compared as
  if they were one universal oracle.
- N-gram models are less robust to paraphrase and novel domains than large semantic
  encoders.
- A lexical safety guard is retained because learned safety predictions alone are
  not sufficient for high-stakes routing.
- The model catalog and its quality/cost priors must be refreshed as providers and
  model versions change.
