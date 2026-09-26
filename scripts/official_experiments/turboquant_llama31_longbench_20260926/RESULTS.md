# Llama-3.1-8B-Instruct SparseEngine TurboQuant 4-bit LongBench

Status: complete. The campaign ran on 2026-09-26 on 8 × RTX 4090 48 GiB (eight independent TP1 workers, DP8). All eight v2 shard exits succeeded; the merged output contains each of the 503 source IDs exactly once. LongBench v1 contains all 3750 samples across the 16 official tasks, with no model failures.

| Benchmark | Samples | Score | Parse failures | Model failures |
| --- | ---: | ---: | ---: | ---: |
| LongBench v1 six-category average | 3750 | 50.34 | 0 | 0 |
| LongBench v2 accuracy (%) | 503 | 31.01 | 23 | 0 |

| v1 category | Score |
| --- | ---: |
| SDQA | 43.53 |
| MDQA | 46.42 |
| SUM | 29.06 |
| FewShot | 68.81 |
| Syn | 54.62 |
| Code | 59.61 |

| v2 domain | Samples | Accuracy (%) |
| --- | ---: | ---: |
| Code Repository Understanding | 50 | 38.00 |
| Long In-context Learning | 81 | 27.16 |
| Long Structured Data Understanding | 33 | 39.39 |
| Long-dialogue History Understanding | 39 | 12.82 |
| Multi-Document QA | 125 | 27.20 |
| Single-Document QA | 175 | 36.00 |

| v2 length | Samples | Accuracy (%) |
| --- | ---: | ---: |
| short | 180 | 35.56 |
| medium | 215 | 28.37 |
| long | 108 | 28.70 |

| v2 difficulty | Samples | Accuracy (%) |
| --- | ---: | ---: |
| easy | 192 | 34.90 |
| hard | 311 | 28.62 |

SparseEngine Git commit: `7b32be0fa67fc06d0df8bfabab5a2700f8825591`. Base model weights: BF16 Llama-3.1-8B-Instruct. KV cache: TurboQuant 4-bit, page size 32, rotation seed 0. Exact engine settings are in [runtime.json](runtime.json).

This is SparseEngine's page-based TurboQuant adaptation using a seeded rotation and Gaussian codebook. It does not implement the paper's QJL residual correction or mixed outlier bit budgets.

LongBench v1: all 3750 samples, official per-task generation lengths, temperature 0, top-p 1, top-k 1, seed 42. LongBench v2: all 503 samples, official pre-chat middle truncation at 120000 tokens, maximum 128 generated tokens, temperature 0.1, top-p 1, top-k 0, seed 42. Both use a 131072-token model limit, batch size 1 per GPU, and prefix caching disabled.

One short v1 sample and one 120K-token v2 sample completed before the campaign. Validated aggregate sources: [v1](v1_aggregate_metrics.json) and [v2](v2_aggregate_metrics.json). Raw logs and per-sample outputs are under `/root/autodl-fs/outputs/Sparse-vLLM/turboquant-llama31-longbench-20260926/` on the evaluation machine.
