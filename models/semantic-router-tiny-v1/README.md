# Semantic Router Tiny v1

The cost-optimized production profile. A 4.4M-parameter, two-layer BERT student
trained on Modal and dynamically quantized to INT8 for ONNX Runtime.

- teacher validation exact accuracy: 49.02%
- adjacent accuracy: 92.62%
- severe under-routing: 1.64%
- savings versus always using level 5: 89.14%
- cost ratio versus the oracle: 1.029
- local CPU p95 latency: 2.84 ms
- Hetzner CPU p95 latency: 1.61 ms
- artifact size: approximately 5 MB including tokenizer

This model strictly dominates the earlier hashed linear router on the measured
accuracy, severe under-routing, cost ratio, artifact size, and latency metrics.
