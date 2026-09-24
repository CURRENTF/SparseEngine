# Qwen3-4B-Thinking-2507 AIME 2024: decode observation rerun

Device: NVIDIA H100 80GB HBM3, physical GPU 3; TP1/EP1/DP1. Git HEAD at
execution: `a07c3e30be63646137cc6aeeff131adc987afe5d`. The decode
observation implementation was uncommitted at execution, so this HEAD alone
does not reconstruct the runtime.

Launch configuration: [`setting.requested.json`](setting.requested.json).
The run used the 30 pinned AIME 2024 `train` problems, dataset SHA-256
`118487dcc92e1801ae88c56696e695baed421160e6594a9ad76949004d205404`,
batch size 30, `max_new_tokens=40960`, `max_model_len=41984`, seed 42,
temperature 0.7, `top_p=1`, `top_k=0`, CUDA Graph on, and prefix caching off.
SnapKV used sink 16, recent 64, selected 4096, observation window 16, and
decode eviction interval 1024. PyramidKV used sink 16, recent 64, selected
7987 / 205 / 4096 at first / last / mean layer, observation window 32, and
decode eviction interval 1024. Both use the new decode query observation path.

```bash
RUN_ROOT=/data1/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_40k_gpu3_20260925_querywindow_aligned
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD:$PWD/src" TOKENIZERS_PARALLELISM=false \
  /home/haojitai/miniconda3/bin/conda run --no-capture-output \
  -p /data2/haojitai/conda_envs/sparse-vllm-cu130-py312 python -u \
  scripts/official_experiments/aime/run.py \
  --setting "$RUN_ROOT/setting.requested.json" \
  --model /data2/pretrain_models/Qwen3-4B-Thinking-2507 \
  --data "$RUN_ROOT/aime2024.json" --output "$RUN_ROOT/full" \
  --gpus 3 --methods snapkv pyramidkv --execute
```

| Method | AIME 2024 pass@1 | 30-question generation time (s; lower is better) | Direct time ratio vs prior Vanilla (higher is better) |
|---|---:|---:|---:|
| SnapKV, query window 16 | 19/30 (63.33%) | 437.6 | 1.33× |
| PyramidKV, query window 32 | 15/30 (50.00%) | 481.0 | 1.21× |

Both methods have 30 unique raw outputs, parsed outputs, and per-sample
results, all with `success` status. Each reached 30 active decode sequences;
the recorded decode Graph was active with seven captured sizes. The generation
time measures the complete 30-question `model()` call, including prefill,
decode, scheduling, and eviction. Model loading, prompt preparation, and score
evaluation are outside this interval.

The time-ratio reference is the 583.9-second Vanilla run in the
[2026-09-24 result package](../2026-09-24-qwen3-4b-thinking-40k/RESULTS.md).
Vanilla was not rerun with this code revision; the ratios compare complete
generation calls across runs and are not a matched-code speedup measurement.

Raw artifacts: `/data1/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_40k_gpu3_20260925_querywindow_aligned`.
The canonical aggregate is `full/final_summary.json`; each method's
`full/<method>/benchmark/math_bench/pred/aime/*/result.json` and
`perf_rank0.json` supply the score and generation time. The run status is in
`status.tsv` and `full/manifest.json`.
