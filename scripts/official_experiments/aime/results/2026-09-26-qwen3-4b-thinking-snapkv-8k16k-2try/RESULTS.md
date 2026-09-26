# Qwen3-4B-Thinking-2507 AIME 2024: SnapKV 8K and 16K, two tries

Device: NVIDIA H100 80GB HBM3, physical GPU 7; TP1/EP1/DP1. Execution used
clean Git commit `7b32be0fa67fc06d0df8bfabab5a2700f8825591`.

The 30 AIME 2024 `train` questions were each submitted twice in one 60-request
call per setting. All three completed settings used temperature 0.6, top-p 0.95, top-k 20,
min-p 0, seed 42, 40,960 output-token cap, 41,984 context limit, CUDA Graph,
and no prefix caching. Source dataset SHA-256:
`118487dcc92e1801ae88c56696e695baed421160e6594a9ad76949004d205404`;
normalized 60-request dataset SHA-256:
`bfe60715bc6fd32f1f6438c9a479cd5b6f4daaaa6aa6255b375b81a7ea84c136`.

| SnapKV total retained tokens per sparse layer | Sink + recent + selected | Batch/decode/resident cap | Correct / 60 | Trial 0 / 30 | Trial 1 / 30 | At least one correct / 30 | Both correct / 30 | Complete generation time (s) | Direct time ratio vs historical Vanilla | Generated text tokens | Recompute preemptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 16 + 64 + 8,112 | 64 | 49 (81.67%) | 26 | 23 | 26 | 23 | 1,454.8 | 0.75× | 1,259,189 | 6 |
| 8,192 | 16 + 64 + 8,112 | 48 | 46 (76.67%) | 23 | 23 | 24 | 22 | 820.0 | 1.33× | 1,273,454 | 0 |
| 16,384 | 16 + 64 + 16,304 | 30 | 51 (85.00%) | 24 | 27 | 27 | 24 | 1,039.4 | 1.05× | 1,203,132 | 0 |

All three per-sample files for each setting contain 60 distinct IDs with
`success` status. The score is mean single-answer accuracy across 60 responses,
not pass@2. Generation time covers the complete 60-request `model()` call and
does not normalize for generated-token count. Concurrency differs across rows,
so score or elapsed-time differences do not isolate the effect of retention
budget. The 16K C64 attempt was stopped before scoring
at the user's request and is excluded.

The direct time ratio is the historical Vanilla 60-request generation time,
1,087.8496 s, divided by each row's generation time. That Vanilla result used
an H100 on physical GPU 3, concurrency cap 32, and Git commit `a07c3e30`;
see the [earlier two-try result](../2026-09-25-qwen3-4b-thinking-official-sampling-2try-c32c64/RESULTS.md).
Different commits, concurrency caps, and generated-token counts limit this to
a descriptive end-to-end reference, not a matched throughput comparison.
The historical Vanilla submitted 60 requests but admitted at most 32 decodes
at once and had no recompute preemptions. The 8K C64 SnapKV run reached 60 active
decodes and preempted six requests after 7,169 or 8,193 completion tokens.
SnapKV decode eviction was enabled, with a physical-length trigger of
8,192 + 1,024 = 9,216 tokens; those requests were preempted before reaching
that trigger. The C64 run later entered the decode-eviction query-scoring path.
The 8K C48 run reached 48 active decodes and had no recompute preemptions.

The launch used `conda run --no-capture-output -p
/data2/haojitai/conda_envs/sparse-vllm-cu130-py312 python -u` with
`PYTHONPATH=<clean-worktree>:<clean-worktree>/src`, `CUDA_VISIBLE_DEVICES=7`,
and `scripts/official_experiments/aime/run.py`. Common arguments were
`--model /data2/pretrain_models/Qwen3-4B-Thinking-2507`,
`--data <run-root>/aime2024.json`, `--gpus 7 --methods snapkv
--gpu-wait-timeout 21600 --execute`. The 8K invocation used
`--setting <run-root>/setting.8k.json --output <run-root>/full_8k`;
the 8K C48 invocation used `--setting <run-root>/setting.8k-c48.json --output
<run-root>/full_8k_c48`; the 16K invocation used
`--setting <run-root>/setting.16k-c30.json --output <run-root>/full_16k_c30`.

Compact table data: [results.json](results.json). Raw settings, manifests,
logs, token outputs, and per-sample scores are under
`/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_snapkv_8k16k_2try_20260926`.
