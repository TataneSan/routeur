# Semantic Router v1

The accuracy-optimized production profile. An 11.3M-parameter, four-layer BERT
student trained on Modal and dynamically quantized to INT8 for ONNX Runtime.

- teacher validation exact accuracy: 51.23%
- adjacent accuracy: 93.38%
- severe under-routing: 1.32%
- task accuracy: 62.46%
- savings versus always using level 5: 88.20%
- cost ratio versus the oracle: 1.117
- local CPU p95 latency: 4.76 ms
- Hetzner CPU p95 latency: 3.62 ms
- artifact size: approximately 12 MB including tokenizer

The default profile uses learned risk, safety, and capability heads. Applications
that prefer conservative escalation can pass `--safety-guard-mode model_confirmed`
or `--safety-guard-mode lexical`, accepting increased cost and over-routing.
