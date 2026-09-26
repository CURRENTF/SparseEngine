# Llama-3.1-8B-Instruct TurboQuant: upstream vLLM comparison

Status: complete, 2026-09-26. Upstream vLLM `0.20.2` (`v0.20.2` tag commit `bc150f50299199599673614f80d12a196f377655`) ran on 6 × RTX 4090 48 GiB, with six independent TP1 workers (DP6). The BF16 model and frozen LongBench prompts match the [SparseEngine run](../turboquant_llama31_longbench_20260926/RESULTS.md), whose recorded Git commit is `7b32be0fa67fc06d0df8bfabab5a2700f8825591`. Both runs cover all 3750 LongBench v1 and 503 LongBench v2 samples; all model calls succeeded, with every source sample ID present exactly once.

| Benchmark | SparseEngine TurboQuant 4-bit | vLLM `turboquant_4bit_nc` | vLLM − SparseEngine |
| --- | ---: | ---: | ---: |
| LongBench v1 six-category average | 50.34 | 50.48 | +0.14 |
| LongBench v2 accuracy (%) | 31.01 | 29.03 | −1.98 pp |
| LongBench v2 parse failures / 503 | 23 | 24 | +1 |

| v1 category | SparseEngine | vLLM |
| --- | ---: | ---: |
| SDQA | 43.53 | 43.26 |
| MDQA | 46.42 | 47.15 |
| SUM | 29.06 | 29.06 |
| FewShot | 68.81 | 68.80 |
| Syn | 54.62 | 54.48 |
| Code | 59.61 | 60.16 |

The vLLM v2 result has 146 correct answers from 503 samples. Its domain, length, and difficulty scores are in [v2 aggregate metrics](v2_aggregate_metrics.json); [v1 aggregate metrics](v1_aggregate_metrics.json) contains all task and category scores. The SparseEngine aggregates are in its linked result package.

This comparison uses vLLM's upstream `turboquant_4bit_nc` implementation: 4-bit Lloyd-Max keys with Hadamard rotation and norm correction, plus 4-bit uniformly quantized values. SparseEngine's page-based adaptation uses a seeded orthogonal rotation and Gaussian codebook. vLLM's implementation omits QJL; neither result should be labeled the TurboQuant paper's original 3.5-bit implementation. The v2 decoding temperature is 0.1, so its outcome also includes sampling variation across engines.

Runtime settings and the launch pattern are in [runtime.json](runtime.json). LongBench v1 used official per-task generation limits, temperature 0, top-p 1, top-k 1. LongBench v2 used official pre-chat middle truncation at 120000 tokens, at most 128 generated tokens, temperature 0.1, top-p 1, top-k 0, and seed `42 + eval_index`. Prefix caching was disabled. All six v1 and six v2 shards exited 0; the merged outputs and scorer reports passed coverage and status checks. Raw outputs, per-sample scores, manifests, and logs remain on the evaluation machine under `turboquant-upstream-vllm-20260926/`.
