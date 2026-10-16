# SnapKV and KVzip Nsight diagnosis

This package profiles the existing request benchmark. It does not change sparse
algorithms or introduce a timing implementation. See [RESULTS.md](RESULTS.md)
for the measured configuration and conclusions.

## Reproduce

Activate a compatible SparseEngine environment first. Set `MODEL_PATH` to
Qwen3-4B-Instruct-2507, `RUN_ROOT` to a new directory on persistent storage,
and `GPU` to an idle physical GPU. The launcher refuses occupied GPUs and
existing case directories. Large traces remain under `RUN_ROOT`.

```bash
for bs in 1 4; do
  for method in snapkv kvzip; do
    for mode in baseline profile; do
      bash scripts/official_experiments/snapkv_kvzip_nsight/run.sh "$method" "$mode" "$bs"
    done
  done
  for mode in baseline profile; do
    SCORE_MODE=logits bash scripts/official_experiments/snapkv_kvzip_nsight/run.sh snapkv "$mode" "$bs"
  done
done
python scripts/official_experiments/snapkv_kvzip_nsight/analyze.py \
  "$RUN_ROOT" --output "$RUN_ROOT/analysis.json"
```

`run.sh` uses `benchmark/efficiency/bench_probe.py` for three unprofiled measured
workloads after one warmup. The default SnapKV arm explicitly uses probability
scores; the additional logits arm uses the repository's default score mode.
Both use the declared 64-token observation window. KVzip uses its native
reconstruction scorer. Budgets have different semantics; this is cost diagnosis,
not equal-retention or equal-quality evaluation.

The profiling arm runs one matched measured workload after warmup. `capture.py`
adds NVTX ranges around existing profiler categories, KVzip reconstruction,
scoring, selection and compaction. It starts CUDA profiling after the probe's
warmup. It adds no per-step device synchronization. Its outer capture also
includes teardown: analyze the `model_run_prefill` and `model_run_decode` ranges,
not full-capture CUDA API totals. The NVTX wrapper replaces optional CPU timers
only for the profiling arm. KVzip retained layer lengths are recorded in
`kvzip_retention.jsonl` in execution order, including warmup.

## General Nsight workflow

1. Fix model, trace, context/output lengths, concurrency, retention semantics,
   precision, providers, topology and Graph policy. Check idle devices. Run an
   unprofiled baseline and validate completion, trace digests and Graph counters.
2. Use Nsight Systems to collect CUDA, NVTX and OS runtime activity after warmup:

   ```bash
   nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
     --cuda-graph-trace=node --capture-range=cudaProfilerApi \
     --capture-range-end=stop -o "$RUN_ROOT/timeline" python scripts/official_experiments/snapkv_kvzip_nsight/capture.py <probe arguments>
   ```

   The launcher provides the full working command. `cudaProfilerStart/Stop`
   requires a matching wrapper such as this package's `capture.py`.
3. Separate original prefill, reconstruction, scoring, selection/compaction,
   decode, sampling and host waits. Use launch correlation/NVTX projection to
   associate kernels with CPU scopes. Rank total contribution and launch count;
   distinguish GPU execution from CPU enqueue/wait time. Nested NVTX scopes are
   inclusive and must not be added together. CUDA Graph node tracing is useful
   for attribution but perturbs replay latency.
4. Select a few dominant kernels for Nsight Compute, on the same shapes. With
   performance-counter access, an example targeted command is:

   ```bash
   ncu --profile-from-start off \
     --kernel-name 'regex:_prefill_score_(partial_lse|final)_kernel' \
     --launch-count 2 --section SpeedOfLight --section LaunchStats \
     --section Occupancy --section MemoryWorkloadAnalysis \
     --section SchedulerStats --section WarpStateStats \
     -o "$RUN_ROOT/score" python scripts/official_experiments/snapkv_kvzip_nsight/capture.py <probe arguments>
   ```

   Confirm section names with `ncu --list-sections`. Inspect measured compute
   and DRAM/L2 throughput, grid/waves, occupancy, registers/spills and warp stalls
   together. Low occupancy alone is not a bottleneck diagnosis. Replay and cache
   control perturb execution; NCU wall time is not serving performance.
5. Form a bounded, source-backed optimization hypothesis, preserve correctness
   with an independent oracle, and rerun the original unprofiled workload. A
   shorter kernel is useful only in proportion to its end-to-end contribution.

Hardware-counter permission is independent of CUDA timeline access. This run
captured Nsight Systems CUDA/NVTX timelines successfully, but `ncu` returned
`ERR_NVGPUCTRPERM`. No achieved bandwidth, occupancy, roofline or SM-utilization
claim is made from these artifacts. Sampled `nvidia-smi` activity is not a
replacement for those measurements.

References: [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
