# Triton Kernel Guide

Use this guide for repository-owned Triton kernels and wrappers under
`src/sparseengine/kernels/triton/`.

## Inspect Before Editing

Trace the serving call through its wrapper, launch grid, kernel, output use,
and synchronization boundary. Identify compile-time constants, supported
layouts, mutation, workspace, CUDA Graph constraints, and the actual model
shape distribution. Compare against existing neighboring kernels before
creating a new family.

Use the external `kernel-triton-writing` skill when it is available. Keep this
guide authoritative for Sparse-Engine-specific integration and validation.

## Establish the Baseline

- Use an independent Torch oracle, not another kernel with the same reduction
  or indexing strategy.
- Cover masked tails, empty inputs, odd lengths, large production shapes,
  non-contiguous inputs when supported, aliases, and repeated launches.
- Validate the wrapper as well as the JIT function. A correct kernel with an
  incorrect grid, stride, or output contract is still wrong.
- Warm every required specialization before CUDA Graph capture and prove replay
  stability when graph support is claimed.

## Tune the Relevant Dimensions

Measure a bounded matrix chosen from the kernel structure:

- program decomposition and grid mapping
- block sizes and vector width
- `num_warps` and `num_stages`
- coalescing and redundant loads
- masks and compile-time specialization
- reduction order and accumulation dtype
- fusion versus intermediate tensor traffic
- register pressure, spills, and occupancy
- launch count and graph behavior

Avoid using private Triton internals for steady-state dispatch. Separate
offline tuning from production launch: tune once for a declared key, then bind
a plain deterministic launcher or static configuration. Never invoke autotune
search inside a request hot path.

## Interpret Results

Check whether the wrapper includes extra casts, padding, allocation, output
reset, or synchronization hidden by a kernel-only timer. Preserve numerical
semantics when changing reduction order or precision. Attribute a win to
fusion only when eliminated launches or memory traffic appear in the matched
measurement.
