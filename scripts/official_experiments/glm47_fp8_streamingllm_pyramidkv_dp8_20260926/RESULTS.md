# GLM-4.7-Flash-FP8 LongBench results

2026-09-26; 8 × RTX 4090 48 GiB. Each benchmark used eight independent TP1/EP1/DP1 workers (DP8). Model weights: FP8; MLA latent cache: BF16. Base Git commit: `7b32be0fa67fc06d0df8bfabab5a2700f8825591`.

| Method | LongBench v1 category mean | v1 samples | LongBench v2 accuracy | v2 correct / total | v2 parse failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| StreamingLLM | 48.18 | 3750 | 32.41 | 163 / 503 | 2 |
| PyramidKV (MLA) | 49.91 | 3750 | 31.61 | 159 / 503 | 0 |

## LongBench v1 category scores

| Category | StreamingLLM | PyramidKV |
| --- | ---: | ---: |
| SDQA | 34.68 | 39.14 |
| MDQA | 41.67 | 46.37 |
| SUM | 26.40 | 26.22 |
| FewShot | 69.20 | 70.09 |
| Syn | 53.00 | 53.00 |
| Code | 64.12 | 64.64 |

## LongBench v2 domain

| Group | Samples | StreamingLLM | PyramidKV |
| --- | ---: | ---: | ---: |
| Code Repository Understanding | 50 | 32.00 | 36.00 |
| Long In-context Learning | 81 | 34.57 | 34.57 |
| Long Structured Data Understanding | 33 | 30.30 | 24.24 |
| Long-dialogue History Understanding | 39 | 23.08 | 25.64 |
| Multi-Document QA | 125 | 32.00 | 26.40 |
| Single-Document QA | 175 | 34.29 | 35.43 |

## LongBench v2 official length

| Group | Samples | StreamingLLM | PyramidKV |
| --- | ---: | ---: | ---: |
| long | 108 | 26.85 | 25.93 |
| medium | 215 | 28.37 | 28.84 |
| short | 180 | 40.56 | 38.33 |

## LongBench v2 difficulty

| Group | Samples | StreamingLLM | PyramidKV |
| --- | ---: | ---: | ---: |
| easy | 192 | 33.33 | 31.25 |
| hard | 311 | 31.83 | 31.83 |

## Launch protocol

- Shared engine settings: `max_model_len=131072`, `engine_prefill_chunk_size=4096`, `max_num_batched_tokens=32768`, `max_decoding_seqs=16`, `gpu_memory_utilization=0.9`, CUDA Graph and prefix caching disabled. Exact method runtime arguments are in [results.json](results.json).
- LongBench v1: `benchmark/long_bench/pred.py --model GLM-4.7-Flash-FP8 --model_path <model> --sparse_method <method> --ws 8 --batch_size 4 --max_model_len 131072 --hyper_param <runtime.json> --output_root <v1_full>`; full 3750 samples.
- LongBench v2: eight shards of `benchmark/long_bench_v2/pred.py --model-path <model> --tokenizer-path <tokenizer> --sparse-method <method> --data-path <shard/data.json> --prepared-samples <shard/prepared_samples.json> --hyper-param-json <runtime.json> --all-samples --overflow-policy official-middle --truncate-max-tokens 120000 --preprocess-workers 1 --max-model-len 131072 --max-new-tokens 128 --batch-size 1 --temperature 0.1 --top-p 1 --top-k 0 --seed 42 --output-dir <v2_shardN>`. Merged results validate all 503 original IDs.
- StreamingLLM: sink 8, recent 4096, decode keep 0. PyramidKV: sink 64, recent 512, decode keep 4096, layer ratio 0.60 → 0.01, observation window 32, probability scoring, long prefill offload threshold 16384.

PyramidKV MLA cache support was added on top of the base commit in `snapkv.py`, `mla_latent.py`, and `method_registry.py`. The base commit alone does not reproduce that method result. Short and long prompt MLA smoke tests passed before the full run.

Raw result roots: `/root/autodl-fs/outputs/Sparse-vLLM/glm47-fp8-streamingllm-dp8-20260926/` and `/root/autodl-fs/outputs/Sparse-vLLM/glm47-fp8-pyramidkv-mla-dp8-20260926/`. Score sources: each root’s `v1_full/result.json` and `combined/aggregate_metrics.json`. The StreamingLLM v2 launcher reported exit code 2 after all shards completed because its script was updated during execution; all final shard artifacts and 503 IDs were validated before merging.
