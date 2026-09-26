# Llama-3.1-8B-Instruct HF StreamingLLM, aligned cache budget

Status: complete. All 3750 LongBench v1 and 503 LongBench v2 frozen inputs
generated successfully. This run aligns the HF sink/recent retention counts
to the historical SparseEngine StreamingLLM settings.

| Benchmark | Samples | Score | Parse failures | Model failures |
| --- | ---: | ---: | ---: | ---: |
| LongBench v1 six-category average | 3750 | 46.14 | — | 0 |
| LongBench v2 accuracy (%) | 503 | 29.62 | 19 | 0 |

| v1 category | Score |
| --- | ---: |
| SDQA | 33.81 |
| MDQA | 39.62 |
| SUM | 25.48 |
| FewShot | 67.07 |
| Syn | 51.75 |
| Code | 59.10 |

| v2 domain | Samples | Accuracy (%) |
| --- | ---: | ---: |
| Code Repository Understanding | 50 | 32.00 |
| Long In-context Learning | 81 | 24.69 |
| Long Structured Data Understanding | 33 | 27.27 |
| Long-dialogue History Understanding | 39 | 12.82 |
| Multi-Document QA | 125 | 31.20 |
| Single-Document QA | 175 | 34.29 |

| v2 length | Samples | Accuracy (%) |
| --- | ---: | ---: |
| short | 180 | 35.00 |
| medium | 215 | 27.91 |
| long | 108 | 24.07 |

| v2 difficulty | Samples | Accuracy (%) |
| --- | ---: | ---: |
| easy | 192 | 32.29 |
| hard | 311 | 27.97 |

Device: 3 × NVIDIA H100 80GB HBM3, three independent TP1 workers (DP3).
Weights: BF16 Llama-3.1-8B-Instruct. HF method: KVCache-Factory at Git commit
`68cd9551a63fedf362ddb0edb008badd04c24996`, unmodified, with
Transformers 4.44.2 and PyTorch 2.8.0+cu128. The baseline runner checkout has
no Git commit; its exact commands and runner identity are in the raw manifests.

| Benchmark | sink | recent | total KV tokens | generation |
| --- | ---: | ---: | ---: | --- |
| v1 | 4 | 2044 | 2048 | temperature 0, top-p 1, top-k 1, official per-task lengths |
| v2 | 512 | 4096 | 4608 | temperature 0.1, top-p 1, top-k 0, 128 tokens |

Both used the baseline's frozen tokenized prompts, a 131072-token model
limit, seed `42 + original sample index`, EOS enabled,
MLP prefill chunks of 8192 tokens, and single-request generation. KVCache-Factory
compresses at the prefill boundary. The maximum post-prefill cache lengths
are 2048 + 512 = 2560 for v1 and 4608 + 128 = 4736 for v2. Both are below
SparseEngine's decode compaction triggers of 4096 and 9216, respectively, so
decode compaction would not occur in these evaluations. The
[earlier HF condition](../llama31_hf_streamingllm_longbench_20260926/RESULTS.md)
used a different cache allocation and 8 × RTX 4090; score changes across the
two conditions cannot be attributed to cache allocation alone.

Scores: [v1 aggregate](v1_aggregate_metrics.json),
[v2 aggregate](v2_aggregate_metrics.json). The three shard exits for each
benchmark were 0; merged outputs had exactly one successful row per frozen
sample ID. Raw manifests, outputs, and logs are under
`/data1/haojitai/outputs/SparseEngine-Baselines/streamingllm-aligned-20260926/`.
