# KVzip reconstruction score tile optimization

Measured results and numerical limits: [RESULTS.md](RESULTS.md).

KVzip reconstruction now launches final-score tiles only over the source chunk.
The probability denominator visits tiles intersecting the protected prefix,
source chunk, and causal replay, counting shared boundary tiles once. Absolute
key positions, GQA reduction, FP32 normalization, and global threshold selection
are preserved. A dense denominator grid is retained when it has fewer tiles.

Baseline revision: `5d96738429c16bbd2f2ba29f392b3fd74f084f02`.
The optimized implementation is in `src/sparseengine/kernels/triton/prefill_score.py`.

## Protocol

- NVIDIA H100 80GB HBM3; driver 580.65.06; Python 3.12.13;
  Torch 2.11.0+cu130; Triton 3.6.0; Nsight Systems 2025.3.2.
- Qwen3-4B-Instruct-2507, BF16, TP/EP/DP = 1, token-layout KV cache.
- Fixed synthetic traces: 32,768 input tokens, 128 greedy output tokens,
  concurrency 1 and 4, seed 42, no length jitter or prefix caching.
- Context length 32,640, protected prefix 64, question suffix 128;
  KVzip budget 4,096, reconstruction chunk 2,000, prefill chunk 8,192.
- CUDA Graph enabled, batch buckets `[1, 2, 4]`, async enabled;
  maximum model length 40,832, GPU memory utilization 0.80.
- Unprofiled: one warmup and three measured iterations per variant and batch.
  Report pooled request TTFT/TPOT means and total output tokens divided by
  total complete workload time. TPOT is `(finish-first)/127` at engine token
  publication; no added per-step synchronization.
- Profiled: one warmup and one measured iteration, matching unprofiled trace 0.
  NVTX-inclusive GPU kernel sums explain attribution; nested scopes are not
  additive and profiler times are not serving latency measurements.

Token-layout KVzip uses the union of per-head selections. Its 4,096 budget is
not a 4,096-token physical cache limit per layer. This experiment preserves
that behavior and does not establish downstream task quality.

## Reproduce

Activate the environment, set `MODEL_PATH`, `OUTPUT_ROOT`, `BASELINE_TREE`, and
`GPU` to an idle device. Keep outputs on a volume with sufficient capacity.
Create the baseline checkout and copy only the diagnostic harness:

```bash
git worktree add --detach "$BASELINE_TREE" 5d96738429c16bbd2f2ba29f392b3fd74f084f02
mkdir -p "$BASELINE_TREE/scripts/official_experiments"
cp -r scripts/official_experiments/snapkv_kvzip_nsight "$BASELINE_TREE/scripts/official_experiments/"
export MODEL_PATH GPU
for batch in 1 4; do
  RUN_ROOT="$OUTPUT_ROOT/baseline" bash "$BASELINE_TREE/scripts/official_experiments/snapkv_kvzip_nsight/run.sh" kvzip baseline "$batch"
  RUN_ROOT="$OUTPUT_ROOT/optimized-final" bash scripts/official_experiments/snapkv_kvzip_nsight/run.sh kvzip baseline "$batch"
  RUN_ROOT="$OUTPUT_ROOT/optimized-final" bash scripts/official_experiments/snapkv_kvzip_nsight/run.sh kvzip profile "$batch"
done
```

For a fresh baseline profile, run the same harness in the baseline checkout
with mode `profile`. Pass that output root as `--original-profile-root` below.
The reported baseline profiles reuse the matched original Nsight diagnosis.

```bash
PYTHONPATH=src:. python scripts/official_experiments/kvzip_score_tiles/analyze.py \
  "$OUTPUT_ROOT" --original-profile-root "$BASELINE_PROFILE_ROOT" \
  --output "$OUTPUT_ROOT/summary.json"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src:. python scripts/official_experiments/kvzip_score_tiles/bench.py \
  --baseline-source "$BASELINE_TREE/src/sparseengine/kernels/triton/prefill_score.py" \
  --output "$OUTPUT_ROOT/microbench.json"
SPARSEENGINE_TEST_CUDA=1 CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src:. python -m pytest -q -rs \
  tests/test_kvzip_method.py tests/test_prefill_score_dynamic_lengths.py \
  tests/test_prefill_score_kernel.py tests/test_prefill_score_per_head.py
```

To exercise the NCCL threshold/union check, expose two idle devices through
`TEST_GPUS` and run:

```bash
CUDA_VISIBLE_DEVICES="$TEST_GPUS" PYTHONPATH=src:. python -m pytest -q \
  'tests/test_kvzip_method.py::test_tp_threshold_and_union_match_global_head_population[nccl]'
```

The microbenchmark alternates baseline/optimized timing order over seven
rounds, 20 calls per round, after five warmup calls. It includes resets and
workspace access, and checks both versions against a per-head FP64 oracle.

`verify.py` accepts the canonical probe arguments plus
`--verification-output <file>` and optional `--baseline-source <source>`.
Run both variants with identical arguments and compare the JSON: selected
positions are hashed per layer and generated token IDs are retained. This
diagnostic adds synchronization and must not be used for latency results.
Optional `--score-output-dir <directory>` also saves per-request scores and
selected indices for investigating threshold-boundary differences.

Hardware counters remain unavailable (`ERR_NVGPUCTRPERM`). CUDA/NVTX timelines
and unprofiled measurements support the reported improvements; they do not
establish achieved bandwidth, occupancy, or a hardware compute/memory limit.
