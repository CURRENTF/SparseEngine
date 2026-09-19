---
name: optimize-sparseengine-kernel
description: Optimize and integrate Sparse-Engine GPU kernels across Triton, TileLang, CUDA/CuTe, and external SGL kernels. Use when Codex needs to identify an LLM inference hotspot, choose a kernel implementation path, write or fuse a kernel, tune an existing kernel, build correctness or microbenchmark coverage, analyze Nsight Compute results, integrate a provider, or validate kernel and end-to-end performance.
---

# Optimize Sparse-Engine Kernel

Use one evidence loop from the serving workload to the kernel and back:

```text
matched LLM workload
  -> hotspot evidence
  -> implementation choice
  -> correctness oracle
  -> stable microbenchmark
  -> targeted tuning
  -> Nsight Compute
  -> operator/provider integration
  -> matched end-to-end validation
```

Do not assume that a kernel rewrite is useful before locating its contribution
to the requested workload. If the user already specifies a kernel, proceed but
state whether end-to-end hotspot evidence exists.

Sparse-Engine performance regressions often come from runtime shape variability,
host synchronization, control-plane work, metadata movement, or provider
routing rather than from the arithmetic kernel itself. Before rewriting a
kernel, check for CUDA Graph recapture, JIT compile-key growth, request-length
specialization, `.item()` or CPU-GPU round trips, prefix-cache RPCs, scheduler
scans, decode-reservation loops, sparse selection/top-k overhead, offload host
copies, LRU work, workspace allocation, and provider dispatch overhead.

## Load the Relevant Guidance

Read [benchmark-protocol.md](references/benchmark-protocol.md) before running
any performance experiment.

Read exactly one implementation guide first, then load another only when a
measured comparison requires it:

- Triton: [triton.md](references/triton.md)
- TileLang: [tilelang.md](references/tilelang.md)
- CUDA, CuTe, CUTLASS, or an external compiled kernel:
  [cuda-cute.md](references/cuda-cute.md)

Read [nsight-playbook.md](references/nsight-playbook.md) before collecting or
interpreting Nsight Compute data. Read
[operator-integration.md](references/operator-integration.md) before changing
provider selection, dependencies, workspaces, layouts, model call sites, or
production dispatch.

Use [reference-sources.md](references/reference-sources.md) only when an
upstream example is needed. Pin the exact source revision and inspect its
license before adapting code.

When available, use companion skills for their focused expertise:

- `llm-torch-profiler-analysis` for SGLang/vLLM/TRT-LLM trace analysis.
- `kernel-triton-writing` for Triton implementation details.
- `add-jit-kernel` for JIT CUDA integration without a large C++ project.
- `add-sgl-kernel` for CUTLASS or complex AOT integration.
- `kernel-cute-writing` for CuTe-specific implementation.
- `perf-nsight-compute-analysis` for deep Nsight Compute interpretation.
- `debug-cuda-crash` for illegal access, misalignment, or graph-capture faults.

Do not block when a companion skill is unavailable; follow this skill's local
references and report the missing capability.

## Execute the Workflow

### 1. Freeze the Scope

Inspect Git status before editing and preserve unrelated tracked and untracked
work. Define the operator, phase, workload, hardware, dtype, shapes, layouts,
parallel topology, CUDA Graph mode, and comparison baseline. Check all devices
and select an idle permitted GPU before starting a GPU task.

### 2. Locate the Cost

Profile a representative end-to-end workload. Separate prefill, decode,
sampling, communication, host overhead, graph replay, and compilation. Rank
hotspots by total contribution rather than kernel latency alone. Record fusion
and overlap opportunities, but treat them as hypotheses until measured.

When the workload uses sparse methods, account for scoring, selection,
eviction, slot metadata, prefix-cache refresh, and host-offload transfer time
as first-class costs. Do not preselect "scheduler", "CPU bottleneck", or
"kernel bottleneck" as the explanation before measuring the relevant boundary.

### 3. Choose the Implementation Path

Prefer the smallest path that can express the required computation:

- For unchanged standard semantics, first reuse a mature upstream public
  provider and its maintained dispatcher. Do not create a repository-owned
  replacement merely because the available local GPU favors one measured
  shape.
- Write or extend repository-owned kernels primarily for sparse or otherwise
  non-standard semantics that upstream implementations cannot express.
- Keep or improve Triton for portable repository fallbacks, correctness
  baselines, and existing repository-owned semantic kernels.
- Use TileLang when explicit tiling, pipelining, shared-memory layouts,
  tensor-core scheduling, TMA, or warp specialization materially helps.
- Prefer JIT CUDA when custom CUDA is needed without CUTLASS or a large AOT
  project.
- Use compiled SGL/CUTLASS/CuTe integration only when the required primitives,
  layouts, or performance cannot be reached cleanly with a JIT path.

Do not replace a mature upstream provider solely because another DSL looks
promising. Distinguish algorithmic improvements from device/shape tuning. An improvement
in computation, data movement, or parallelism may become a default across its
actual compatible domain after representative correctness and performance
checks. Explain why it generalizes; do not gate it on measured batch, length,
TP, or device-name lists. Use exact reproducible profiles for tuning-dependent
winners or specialized dispatch schedules. Keep performance claims scoped to
measurements and preserve actual hardware, dtype, layout, and compiler limits.

### 4. Establish Correctness

Create an independent Torch or mathematically direct oracle. Specify input and
output shapes, dtypes, strides, aliases, mutation, padding, empty cases,
numerical tolerances, and reduction semantics. Cover real model shapes,
boundary shapes, non-contiguous inputs when supported, repeated execution, and
CUDA Graph capture/replay when claimed.

Reject a candidate immediately when correctness fails. Never tune against a
known-wrong implementation.

### 5. Build the Microbenchmark

Benchmark the actual callable boundary used by serving, including required
workspace initialization, output reset, synchronization, or materialization.
Separate compile and cold-start cost from steady-state latency. Use shapes
derived from the workload, fixed inputs and seeds, sufficient warmup, raw
samples, and identical conditions for baseline and candidate.

### 6. Tune with Bounded Hypotheses

Change one explained dimension or run a declared finite matrix. Record every
candidate, including failures. Typical dimensions include tile shape, program
grid, threads or warps, pipeline stages, split count, vector width,
shared-memory layout, swizzle, async copy/TMA, fusion boundary, register use,
and workspace layout.

Tune offline. Do not benchmark or search configurations in the serving hot
path. Bind the selected configuration deterministically from validated shape
and device facts.

### 7. Profile Finalists

Run Nsight Compute only after correctness and stable timing identify a small
set of finalists. Compare the same shape and call boundary. Use measured
occupancy, achieved bandwidth, tensor-core utilization, register or local
memory use, shared-memory behavior, scheduler issue rate, and warp stalls to
support the next hypothesis. Do not infer a bottleneck from occupancy alone.

### 8. Integrate through Operators

Keep kernels under the repository-owned kernel packages and keep dependency
checks, support predicates, workspace ownership, launch selection, and
fallback policy under `src/sparseengine/operators/`. Resolve and bind a provider
before the forward hot path. Route unsupported configurations before launch;
never catch a runtime kernel failure and silently switch providers.

Keep atomic correctness eligibility separate from default upstream-first
portfolio policy and exact local profile overlays. Follow
[operator-integration.md](references/operator-integration.md) and treat
`$review-operator-organization` as authoritative for production selection.

### 9. Return to End-to-End Measurement

Repeat the original workload with the same model, request trace, concurrency,
context/output lengths, TP/EP topology, cache state, graph mode, and metric
window. Report kernel latency improvement separately from end-to-end latency,
throughput, TTFT, or TPOT. A faster microbenchmark is not an end-to-end win.
Stage logs, component microbenchmarks, and Nsight timelines are diagnostic
evidence; they do not establish serving throughput unless the matched workload
is rerun with the same request and timing contract.

## Acceptance Gates

Require all applicable gates before calling the work complete:

1. Independent correctness equivalence passes.
2. Provider selection and rejection paths are tested.
3. Repository-owned kernel claims and local profile overrides exercise their
   claimed devices, dtypes, shapes, and graph modes on hardware. External
   atomic providers separately record upstream-declared support and the local
   adapter coverage that was actually exercised.
4. Microbenchmark raw samples and summary are saved.
5. Nsight evidence exists for hardware-level bottleneck claims.
6. Matched end-to-end validation supports serving-level claims.
7. Commands, Git state, environment, selected provider, and artifacts are
   recorded.

Mark unrun validation explicitly. Keep a compatible baseline for contracts the
new implementation cannot execute. Algorithmic default promotion does not need
an exhaustive device/shape/TP matrix: validation records describe observed
coverage, not an enablement whitelist. Tuning-dependent profile overrides remain
limited to their recorded domain. Do not claim unmeasured performance wins.
