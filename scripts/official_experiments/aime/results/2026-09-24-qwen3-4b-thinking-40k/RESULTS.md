# Qwen3-4B-Thinking-2507 AIME 2024, 40K output

Device: NVIDIA H100 80GB HBM3, physical GPU 3; TP1/EP1/DP1. Git HEAD at
execution: `a07c3e30be63646137cc6aeeff131adc987afe5d`.

Launch configuration: [`setting.requested.json`](setting.requested.json), copied
from run A. The shared recipe was revised after this run.
All 30 AIME 2024 `train` problems were submitted as one batch, with sequence
limits of 32, `max_new_tokens=40960`, `max_model_len=41984`, seed 42,
temperature 0.7, `top_p=1`, `top_k=0`, prefix caching off and CUDA Graph active.
The normalized dataset SHA-256 was
`118487dcc92e1801ae88c56696e695baed421160e6594a9ad76949004d205404`.
Each segment used the same command with its own output directory and method
subset: A `vanilla`; B `streamingllm snapkv h2o pyramidkv omnikv quest rkv`;
C `pyramidkv`. The B attempt at PyramidKV was stopped and is excluded from the
result table.

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD:$PWD/src" \
  conda run --no-capture-output -p "$ENV_PREFIX" python -u \
  scripts/official_experiments/aime/run.py \
  --setting "$RUN_ROOT/setting.requested.json" --model "$MODEL_PATH" \
  --data "$RUN_ROOT/aime2024.json" --output "$RUN_ROOT/full" \
  --gpus 3 --methods "${METHODS[@]}" --execute
```

| Method | Run | AIME 2024 pass@1 | 30-question generation time (s; lower is better) | Speedup vs Vanilla (higher is better) |
|---|:---:|---:|---:|---:|
| Vanilla | A | 25/30 (83.33%) | 583.9 | 1.00× |
| StreamingLLM | B | 5/30 (16.67%) | 259.2 | 2.25× |
| SnapKV | B | 13/30 (43.33%) | 405.1 | 1.44× |
| H2O | B | 16/30 (53.33%) | 425.1 | 1.37× |
| PyramidKV, corrected allocation | C | 3/30 (10.00%) | 437.5 | 1.33× |
| OmniKV | B | 26/30 (86.67%) | 360.3 | 1.62× |
| QuEST | B | 18/30 (60.00%) | 413.5 | 1.41× |
| RKV | B | 15/30 (50.00%) | 540.4 | 1.08× |

All eight included methods have 30 unique raw outputs, parsed outputs and
per-sample results; every sample has `success` status, and each aggregate
matches its per-sample scores. Each included run reached 30 active decode
sequences. The initial PyramidKV attempt in run B was
stopped because its old layer-slot distribution admitted only 12 of 30
concurrent sequences. Run C used the corrected distribution: the final layer
had 168,478 slots, with no preemption or runtime errors. Its 3/30 score is retained as
observed; the cause of the low score has not been established.

Source artifacts: run A
`/data1/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_40k_gpu3_20260924`, run B
`/data1/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_40k_gpu3_20260924_resume1`,
run C
`/data1/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_40k_gpu3_20260924_pyramid_fixed1`.
Each reported row comes from that run's `full/<method>/benchmark/math_bench/pred/aime/*/result.json`
and `perf_rank0.json`. The comparison uses the wall time of each complete
30-question `model()` generation call, with CUDA synchronization at its
boundaries; it includes prefill, decode, scheduling and eviction. Model loading,
prompt preparation and scoring are outside this interval. Outputs ended at
their natural lengths. The recorded Git HEAD is a base revision; local
method and configuration changes used for this run have not been committed.
