# Qwen3-4B-Thinking-2507 AIME 2024: C20 and Vortex QuEST

Device: one NVIDIA H100 80GB HBM3, GPU2 (`GPU-566a0ffb-6f93-413f-cdf1-bab7f7acd20e`). Each method received the same 60 requests in one generation call: 30 AIME 2024 train problems with two samples per problem. Normalized dataset SHA-256: `bfe60715bc6fd32f1f6438c9a479cd5b6f4daaaa6aa6255b375b81a7ea84c136`.

Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, 40,960 new-token cap, and 41,984 context length. Prefix caching was disabled. CUDA Graph was active with capture sizes `[1, 2, 4, 8, 16, 20]`; all four methods had concurrency cap 20. SparseEngine commit: `7b32be0fa67fc06d0df8bfabab5a2700f8825591`. Vortex commit: `ab9ac68c7b9b81f6ba17752741f9e9d92444bf56`. See the [SparseEngine setting](../../setting.qwen3-4b-thinking-2507-official-sampling-c20-2try.json), [Vortex setting](vortex_quest.setting.json), and [compact results](results.json).

| Method | Correct / 60 | Trial 0 / 30 | Trial 1 / 30 | At least one correct / 30 | Generation time (s) | vs Vanilla |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 51 | 24 | 27 | 27 | 1135.1 | 1.00× |
| OmniKV | 48 | 25 | 23 | 26 | 804.2 | 1.41× |
| SparseEngine QuEST | 40 | 19 | 21 | 23 | 907.2 | 1.25× |
| Vortex QuEST | 45 | 24 | 21 | 25 | 1231.1 | 0.92× |

Speedup is Vanilla generation time divided by method generation time, without output-token normalization. Timing excludes model startup, prompt preparation, and scoring. SparseEngine synchronizes GPU call boundaries; Vortex times the blocking `Engine.generate()` call. All 60 requests generated and received a score in each method. Native results have 60 `success` scoring statuses each; Vortex has 59 `success` and one `parse_failed` (request 46), which counts as incorrect under the unchanged metric. The three SparseEngine logs recorded zero recompute preemptions; Vortex response metadata recorded zero retractions and six requests hitting the length cap.

Both QuEST configurations allow up to 133 pages of 16 tokens per KV head. Vortex pins the first page and last four pages, then selects 128 middle pages per KV head. SparseEngine shares one selected page set across KV heads and does not pin those first/recent pages. The workload and attention capacity match, while the selected pages can differ.

Vortex's GQA decode used FlashInfer paged attention and a Triton-compiled QuEST indexer. Because `vortex_max_topk_val` was unset, the indexer's `topk_output` used the `flashinfer/default` CUB-sort CUDA implementation rather than the `k_128` specialization. The performance effect of this dispatch choice was not isolated.

## Launch and artifacts

The launcher `scripts/tmp/run_aime4b_official_sampling_2try_c20_gpu2_20260925.sh` ran `scripts/official_experiments/aime/run.py` once for each native method with `--setting scripts/official_experiments/aime/setting.qwen3-4b-thinking-2507-official-sampling-c20-2try.json`, `--model /data2/pretrain_models/Qwen3-4B-Thinking-2507`, `--gpus 2 --gpu-wait-timeout 21600 --execute`, and `--methods` set to `vanilla`, `omnikv`, or `quest`. Vortex used `scripts/official_experiments/aime/run_vortex_quest.py` with the same 60 prompt token sequences, `--concurrency 20 --max-model-len 41984 --max-new-tokens 40960 --mem-fraction-static 0.9 --seed 42`. MathBench scored all methods with `math-verify==0.9.0`.

Raw outputs, manifests, status, and logs: `/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_official_sampling_2try_c20_gpu2_20260925`.
