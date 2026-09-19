# Kernel Benchmark Protocol

Apply this protocol to every performance claim.

## Before Launch

1. Inspect every GPU's utilization, memory, and compute-process ownership.
2. Select an idle permitted device. If none is idle, wait or report instead of
   sharing a busy device.
3. Inspect Git status and preserve unrelated tracked and untracked work.
4. Record the kernel callable boundary and the end-to-end workload that
   produced the target shapes.
5. Establish correctness before collecting performance samples.

## Record the Case

Save a manifest containing:

- repository path, Git SHA, branch, and dirty status
- exact command and interpreter
- Torch, Triton, TileLang, CUDA, driver, and relevant kernel-package versions
- GPU name, compute capability, selected device, clocks or power constraints
  when controlled
- input shapes, dtypes, strides, seed, and data-generation method
- provider, launch configuration, graph mode, TP/EP topology, and cache state
- warmup count, timed repetitions, timing method, and synchronization boundary
- output paths and explicit run status

Do not overwrite previous raw results. Use a new run directory or immutable
case identifier.

## Microbenchmark Semantics

- Time the serving-relevant wrapper, not only an internal launch, unless the
  result is explicitly labeled kernel-only.
- Include required output reset, workspace preparation, conversion, or
  materialization. Report optional components separately when decomposition is
  useful.
- Exclude compilation from steady-state latency after recording cold compile
  time separately.
- Warm every specialization and synchronize before and after the timed region.
- Use CUDA events or another GPU-aware timer correctly; never time asynchronous
  launches with host wall time alone.
- Use identical inputs, streams, graph mode, and synchronization for baseline
  and candidate. Interleave them when long runs may drift.
- Save raw samples and report at least sample count and median. Add tail or
  dispersion statistics when they affect the decision.
- State whether caches are intentionally warm or cold. Do not mix regimes.
- Repeat suspicious gains and reject results affected by competing processes,
  throttling, compilation, or changing clocks.

## End-to-End Semantics

Match model/checkpoint, request trace, prompt and output lengths, batch and
concurrency, TP/EP, cache state, graph/provider settings, decoding parameters,
and metric window. Keep TTFT, TPOT/inter-token latency, request latency,
throughput, and kernel time distinct. Label partial serving runs separately
from completed end-to-end results.

## Artifact Minimum

Persist:

```text
run_manifest.json
raw_samples.jsonl
summary.json
stdout.log
stderr.log
```

Add profiler traces and `.ncu-rep` files when collected. Use structured status
and error fields; do not treat a non-empty output or a successful process start
as benchmark success.
