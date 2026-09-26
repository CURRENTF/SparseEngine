# Llama-3.1-8B-Instruct OnlyTerp TurboQuant vLLM LongBench

Status: complete. This measures the `turboquant_4bit_nc` KV cache in the
`OnlyTerp/turboquant` vLLM implementation at Git commit
`1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306` (vLLM 0.20.2). It is a
separate implementation from [SparseEngine's page-based adaptation](../turboquant_llama31_longbench_20260926/RESULTS.md), and is not a run of the TurboQuant paper's original repository.

| Benchmark | Samples | Score | Parse failures | Model failures |
| --- | ---: | ---: | ---: | ---: |
| LongBench v1 six-category average | 3750 | 50.48 | — | 0 |
| LongBench v2 accuracy (%) | 503 | 29.03 | 24 | 0 |

| v1 category | Score |
| --- | ---: |
| SDQA | 43.26 |
| MDQA | 47.15 |
| SUM | 29.06 |
| FewShot | 68.80 |
| Syn | 54.48 |
| Code | 60.16 |

| v2 domain | Samples | Accuracy (%) |
| --- | ---: | ---: |
| Code Repository Understanding | 50 | 30.00 |
| Long In-context Learning | 81 | 22.22 |
| Long Structured Data Understanding | 33 | 27.27 |
| Long-dialogue History Understanding | 39 | 12.82 |
| Multi-Document QA | 125 | 33.60 |
| Single-Document QA | 175 | 32.57 |

| v2 length | Samples | Accuracy (%) |
| --- | ---: | ---: |
| short | 180 | 33.89 |
| medium | 215 | 26.51 |
| long | 108 | 25.93 |

| v2 difficulty | Samples | Accuracy (%) |
| --- | ---: | ---: |
| easy | 192 | 30.21 |
| hard | 311 | 28.30 |

Device: 6 × NVIDIA GeForce RTX 4090 48 GiB, six independent TP1 workers
(DP6). Model: BF16 Llama-3.1-8B-Instruct. Each worker used
`kv_cache_dtype=turboquant_4bit_nc`, `max_model_len=131072`,
`max_num_seqs=1`, `max_num_batched_tokens=4096`, chunked prefill on,
`gpu_memory_utilization=0.5`, eager execution, and prefix caching off.
Generation used seed `42 + original sample index`, EOS enabled, top-p 1;
v1 used temperature 0/top-k 1 and official per-task output lengths;
v2 used temperature 0.1/top-k 0 and 128 output tokens. All six shard exits
were 0 for each benchmark, and merged inputs/outputs covered every frozen ID.

Scores: [v1 aggregate](v1_aggregate_metrics.json),
[v2 aggregate](v2_aggregate_metrics.json). Raw manifests, logs, merged inputs,
and predictions are under
`/root/autodl-fs/outputs/Sparse-vLLM/turboquant-upstream-vllm-20260926/`
on the evaluation machine. The evaluation runner lives outside Git under
`/root/autodl-tmp/scripts/tmp/run_tq_vllm_baseline.py`; its exact per-shard
commands are preserved in the raw manifests.
