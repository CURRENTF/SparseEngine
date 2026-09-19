# CUDA, CuTe, CUTLASS, and SGL Kernel Guide

Use a compiled or JIT CUDA path only when its required primitives, layout
control, or measured performance justify the added integration cost.

## Choose the Smallest Integration

- Prefer JIT CUDA when the kernel does not require CUTLASS or a large C++
  project. Use `add-jit-kernel` when available.
- Use CuTe when explicit NVIDIA tensor layouts and architecture-specific
  primitives are central. Use `kernel-cute-writing` when available.
- Use SGL/CUTLASS AOT integration for complex compiled projects or packaged
  kernels. Use `add-sgl-kernel` when available.
- Keep a verified Triton or Torch implementation as the correctness baseline
  and, where supported, the portable provider.

## Control the Contract

Declare supported compute capabilities, CUDA/toolchain versions, dtypes,
alignments, layouts, workspace, streams, graph behavior, and mutation. Keep
build and package availability checks inside the provider and import compiled
extensions lazily.

Do not expose CUTLASS packing, reordered projections, descriptor formats, or
workspace details to model classes. The selected provider owns physical
layouts and preparation. Reject unsupported configurations during resolution
or preparation; do not catch a launch failure and switch implementations.

## Validate

Compare the actual wrapper against an independent oracle over production and
boundary shapes. Exercise the minimum declared dependency/toolchain and every
claimed architecture on real hardware. Check sanitizer or crash diagnostics
for indexing and lifetime changes, and validate CUDA Graph capture/replay when
advertised.

Measure compile/startup separately from steady-state execution. Include any
descriptor construction, packing, workspace clearing, or output conversion
that remains in the service call boundary.
