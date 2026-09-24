# Qwen3-4B-Thinking-2507 AIME 2024: official sampling, two tries

Device: NVIDIA H100 80GB HBM3, physical GPUs 3, 4 and 5; TP1/EP1/DP1.
Git commit at execution: `a07c3e30be63646137cc6aeeff131adc987afe5d`.
The eight methods used the same [configuration](setting.requested.json) and
normalized dataset SHA-256
`bfe60715bc6fd32f1f6438c9a479cd5b6f4daaaa6aa6255b375b81a7ea84c136`.
The source AIME 2024 `train` export SHA-256 was
`118487dcc92e1801ae88c56696e695baed421160e6594a9ad76949004d205404`.

Each of the 30 problems was repeated twice with distinct request IDs. All 60
requests were submitted in one `model()` call per method; the engine applied
the C32 or C64 limit below. Sampling used temperature 0.6, `top_p=0.95`,
`top_k=20`, `min_p=0`, seed 42, a 40960-token output cap and
`max_model_len=41984`. CUDA Graph was active and prefix caching was disabled.
Every method produced 60 successful raw, parsed and scored samples.

## Quality

| Method | Correct / 60 | Trial 0 / 30 | Trial 1 / 30 | At least one correct / 30 | Both correct / 30 |
|---|---:|---:|---:|---:|---:|
| Vanilla | 49 (81.67%) | 24 | 25 | 25 | 24 |
| StreamingLLM | 11 (18.33%) | 7 | 4 | 8 | 3 |
| SnapKV | 35 (58.33%) | 18 | 17 | 18 | 17 |
| H2O | 33 (55.00%) | 16 | 17 | 18 | 15 |
| PyramidKV | 32 (53.33%) | 16 | 16 | 20 | 12 |
| OmniKV | 50 (83.33%) | 27 | 23 | 28 | 22 |
| QuEST | 44 (73.33%) | 21 | 23 | 24 | 20 |
| RKV | 31 (51.67%) | 16 | 15 | 18 | 13 |

The 60-sample score is the mean single-answer accuracy, not pass@2. The last
two columns group the two responses to each problem without changing that
score's denominator.

## End-to-end generation

| Method | GPU | Limit | 60-request generation time (s) | Direct time ratio vs Vanilla | Generated text tokens | Scheduler recompute preemptions |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 3 | 32 | 1087.8 | 1.00× | 1,187,197 | 0 |
| StreamingLLM | 3 | 64 | 334.6 | 3.25× | 800,900 | 0 |
| SnapKV | 3 | 64 | 676.8 | 1.61× | 1,459,300 | 0 |
| H2O | 3 | 64 | 743.3 | 1.46× | 1,634,361 | 0 |
| PyramidKV | 3 | 64 | 677.8 | 1.61× | 1,394,311 | 0 |
| OmniKV | 3 | 32 | 1895.5 | 0.57× | 1,321,198 | 7 |
| QuEST | 4 | 32 | 1288.5 | 0.84× | 1,357,859 | 4 |
| RKV | 5 | 64 | 867.5 | 1.25× | 1,716,632 | 0 |

Time is the synchronized wall time of the complete 60-request `model()` call,
including admission, prefill, decode, scheduling, scoring used during decode,
and any eviction or recomputation. It excludes model startup, prompt assembly
and answer evaluation. The ratio is Vanilla time divided by method time; it
does not normalize for output length. Generated text tokens are the MathBench
post-generation re-tokenization diagnostic, not an exact KV-slot count.
Preemptions count `recompute_replay_start` events; each has a matching
`recompute_replay_complete`. They do not count method-owned KV eviction or
deferred prompt admission.

## Launch and artifacts

The original GPU3 launcher
`scripts/tmp/run_aime4b_official_sampling_2try_c32c64_gpu3_20260925.sh`
ran Vanilla through PyramidKV. After PyramidKV completed, its orchestrator
was intentionally stopped so the remaining methods could run concurrently;
its `failed:137` wrapper exit is this queue split, not a method failure.
The isolated launcher
`scripts/tmp/run_aime4b_official_sampling_2try_parallel_method_20260925.sh`
ran `omnikv` on GPU3, `quest` on GPU4 and `rkv` on GPU5.

Both launchers invoked `scripts/official_experiments/aime/run.py` through
`conda run --no-capture-output -p /data2/haojitai/conda_envs/sparse-vllm-cu130-py312 python -u`
with `--model /data2/pretrain_models/Qwen3-4B-Thinking-2507`,
`--setting <run-root>/setting.requested.json`,
`--data <run-root>/aime2024.json`, `--output <run-root>/full`,
`--gpu-wait-timeout 21600` and `--execute`. The initial invocation used
`--gpus 3 --methods vanilla streamingllm snapkv h2o pyramidkv omnikv quest rkv`;
each isolated invocation used its assigned `--gpus` and single `--methods`
value. The exact per-method commands are in the raw manifests.

The [compact results](results.json) include each method's artifact and
manifest path. Raw outputs are under
`/data1/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_official_sampling_2try_c32c64_gpu3_20260925`
and
`/data1/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_official_sampling_2try_c32c64_parallel_gpu345_20260925`.
