# Llama-3.1-8B-Instruct HF StreamingLLM LongBench

Status: complete. The eight-card campaign ran on 2026-09-26. All 3750 v1 and 503 v2 model generations succeeded. The v1 scorer initially lacked its official metric dependencies; it completed after installing `jieba`, `fuzzywuzzy`, and `rouge`, without rerunning or changing model outputs.

| Benchmark | Samples | Score | Parse failures | Model failures |
| --- | ---: | ---: | ---: | ---: |
| LongBench v1 six-category average | 3750 | 46.12 | 0 | 0 |
| LongBench v2 accuracy (%) | 503 | 29.82 | 22 | 0 |

| v1 category | Score |
| --- | ---: |
| SDQA | 33.77 |
| MDQA | 39.90 |
| SUM | 25.58 |
| FewShot | 66.87 |
| Syn | 51.75 |
| Code | 58.88 |

| v2 domain | Samples | Accuracy (%) |
| --- | ---: | ---: |
| Code Repository Understanding | 50 | 36.00 |
| Long In-context Learning | 81 | 25.93 |
| Long Structured Data Understanding | 33 | 30.30 |
| Long-dialogue History Understanding | 39 | 10.26 |
| Multi-Document QA | 125 | 31.20 |
| Single-Document QA | 175 | 33.14 |

| v2 length | Samples | Accuracy (%) |
| --- | ---: | ---: |
| short | 180 | 32.78 |
| medium | 215 | 28.84 |
| long | 108 | 26.85 |

| v2 difficulty | Samples | Accuracy (%) |
| --- | ---: | ---: |
| easy | 192 | 32.81 |
| hard | 311 | 27.97 |

| Condition | Value |
| --- | --- |
| Device | 8 × NVIDIA GeForce RTX 4090 48 GiB; TP1 per GPU, DP8 |
| Weights | Llama-3.1-8B-Instruct BF16 |
| HF implementation | KVCache-Factory pinned at `68cd9551a63fedf362ddb0edb008badd04c24996` |
| LongBench v1 | 3750 frozen inputs; sink 8 + recent 2040 = 2048 KV tokens; temperature 0, top-p 1, top-k 1 |
| LongBench v2 | 503 frozen inputs; sink 8 + recent 4096 = 4104 KV tokens; temperature 0.1, top-p 1, top-k 0 |
| Shared launch | Eight token-balanced single-request HF shards, seed 42 plus original evaluation row index, max model length 131072, MLP prefill chunk 8192, EOS enabled |

The pinned implementation compresses the KV cache once at the prompt boundary to the first sink tokens and last recent tokens. Its decode path appends generated tokens without recurring eviction. These scores therefore identify this HF implementation rather than continuous-eviction StreamingLLM.

The baseline runner checkout has no Git commit. Its run manifests contain the runner and settings hashes; the pinned KVCache-Factory Git commit above identifies the method implementation. Aggregate source files: [v1](v1_aggregate_metrics.json) and [v2](v2_aggregate_metrics.json). Raw manifests, outputs, logs, and score files are under `/root/autodl-fs/outputs/SparseEngine-Baselines/llama31-streamingllm-hf-20260926/` on the evaluation machine.
