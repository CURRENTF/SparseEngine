# Qwen3-4B-Thinking-2507 AIME 2024: C24 and Vortex QuEST

Device: one NVIDIA H100 80GB HBM3, GPU2 (`GPU-566a0ffb-6f93-413f-cdf1-bab7f7acd20e`). Each method received the same 60 requests in one generation call: 30 AIME 2024 train problems with two samples per problem. Normalized dataset SHA-256: `bfe60715bc6fd32f1f6438c9a479cd5b6f4daaaa6aa6255b375b81a7ea84c136`. SparseEngine commit: `7b32be0fa67fc06d0df8bfabab5a2700f8825591`; Vortex commit: `ab9ac68c7b9b81f6ba17752741f9e9d92444bf56`.

Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, 40,960 output-token cap, and 41,984 context length. Concurrency cap 24; prefix cache disabled; CUDA Graph active. Native QuEST uses sink 16, recent 64, selected 2048, chunk 16, and skips the first two layers. Vortex uses 128 selected pages of 16 tokens plus one first and four last pages, and also skips the first two layers. See [setting](setting.requested.json) and [compact results](results.json).

| Method | Correct / 60 | Trial 0 / 30 | Trial 1 / 30 | At least one correct / 30 | Generation time (s) | vs Vanilla |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 49 | 26 | 23 | 26 | 1125.0 | 1.00× |
| OmniKV | 48 | 25 | 23 | 25 | 747.1 | 1.51× |
| SparseEngine QuEST | 38 | 19 | 19 | 21 | 1269.9 | 0.89× |
| Vortex QuEST | 44 | 22 | 22 | 23 | 1262.3 | 0.89× |

Speedup divides Vanilla generation time by method generation time, without output-token normalization. Timing excludes startup, prompt preparation, and scoring. All methods generated and scored 60 requests successfully. SparseEngine QuEST logged three scheduler recompute preemptions; Vanilla and OmniKV logged zero. Vortex logged three KV-pool retraction events and completed all requests. Different output-token counts are preserved in `results.json`.

Vortex's GQA decode used FlashInfer paged attention and a Triton-compiled QuEST indexer. Because `vortex_max_topk_val` was unset, the indexer's `topk_output` used the `flashinfer/default` CUB-sort CUDA implementation rather than the `k_128` specialization. The performance effect of this dispatch choice was not isolated.

## Launch and artifacts

The launcher `scripts/tmp/run_aime4b_official_sampling_2try_c24_gpu2_20260925.sh` invoked `scripts/official_experiments/aime/run.py` once per native method with the copied setting and `--methods vanilla`, `--methods omnikv`, or `--methods quest`, `--gpus 2 --gpu-wait-timeout 21600 --execute`. Vortex used `scripts/official_experiments/aime/run_vortex_quest.py` with the same 60 prompt token sequences, `--concurrency 24 --max-model-len 41984 --max-new-tokens 40960 --mem-fraction-static 0.9 --seed 42`. MathBench scored all outputs with `math-verify==0.9.0`.

Raw outputs, per-sample scores, manifests, status, and logs: `/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_official_sampling_2try_c24_gpu2_20260925`.
