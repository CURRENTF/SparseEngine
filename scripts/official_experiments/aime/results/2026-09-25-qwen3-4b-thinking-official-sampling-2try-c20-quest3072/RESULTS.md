# Qwen3-4B-Thinking-2507 AIME 2024: C20 QuEST budget 3072

Device: one NVIDIA H100 80GB HBM3, GPU2 (`GPU-566a0ffb-6f93-413f-cdf1-bab7f7acd20e`). SparseEngine commit: `7b32be0fa67fc06d0df8bfabab5a2700f8825591`. The same 30 AIME 2024 train problems were sampled twice in one 60-request generation call; normalized dataset SHA-256: `bfe60715bc6fd32f1f6438c9a479cd5b6f4daaaa6aa6255b375b81a7ea84c136`.

QuEST budget: sink 16 + recent 64 + selected 2992 = **3072 total tokens** per sparse layer; chunk size 16; the first two layers skip sparse selection. C20 has batch/decode/residency caps of 20 and CUDA Graph capture sizes `[1, 2, 4, 8, 16, 20]`. Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, 40,960 output-token cap, and 41,984 context length. See [setting](setting.requested.json) and [compact results](results.json).

| C20 method | Correct / 60 | Trials / 30 | At least one correct / 30 | Generation time (s) | vs C20 Vanilla | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla reference | 51 | 24 / 27 | 27 | 1135.1 | 1.00× | 1,183,274 |
| QuEST 2128 reference | 40 | 19 / 21 | 23 | 907.2 | 1.25× | 1,414,757 |
| QuEST 3072 | 46 | 23 / 23 | 24 | 879.6 | 1.29× | 1,278,597 |

The two reference rows come from the [matched C20 result](../2026-09-25-qwen3-4b-thinking-official-sampling-2try-c20-vortex-quest/RESULTS.md) on the same GPU and dataset. Speedup divides Vanilla generation time by method generation time; output tokens are shown separately and are not used to normalize the timing. QuEST 3072 generated and scored all 60 requests successfully, with zero logged scheduler recompute preemptions. The score difference is an observed result from two samples per problem.

## Launch and artifacts

The launcher `scripts/tmp/run_aime4b_official_sampling_2try_c20_quest3072_gpu2_20260925.sh` invoked `scripts/official_experiments/aime/run.py` with `--methods quest --gpus 2 --gpu-wait-timeout 21600 --execute`, the copied setting, the Qwen3-4B-Thinking-2507 checkpoint, and the pinned 30-question dataset. MathBench scored the output with `math-verify==0.9.0`.

Raw outputs, per-sample scores, manifest, status, and logs: `/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_official_sampling_2try_c20_quest3072_gpu2_20260925`.
