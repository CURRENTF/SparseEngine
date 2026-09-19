# TileLang Kernel Guide

Use this guide for kernels under `src/sparseengine/kernels/tilelang/` and their
lazy adapters under `src/sparseengine/operators/`.

## Establish the Contract

Start from a direct Torch oracle and, when useful, a validated Triton provider.
Record the exact TileLang, TVM-FFI, Torch, CUDA, driver, and GPU architecture
versions. Validate the installed versions used by the production dependency
contract; do not rely only on a newer developer checkout.

Keep source files stable and present while compiling. TileLang JIT may inspect
Python source, so do not benchmark code stored only in a transient shell body
or a file that another worktree operation can remove. Warm every shape needed
by CUDA Graph before capture.

## Tune Systematically

Classify the candidate as launch-, memory-, latency-, or compute-sensitive,
then explore a bounded subset of:

1. Tile shapes along token, head, and hidden dimensions.
2. Block, thread, warp, or warpgroup mapping.
3. Pipeline stage count and the shared-memory cost of each stage.
4. Shared-memory layout, padding, bank conflicts, and swizzle.
5. Async copy or TMA eligibility, alignment, and producer/consumer overlap.
6. Tensor-core tile compatibility and achieved utilization.
7. Fragment size, live ranges, register pressure, spills, and occupancy.
8. Split-KV or split-reduction parallelism and combine-kernel overhead.
9. Fusion benefit versus added score reset, workspace, or atomic traffic.
10. Launch shape and CUDA Graph replay behavior.

Measure after every change or declared matrix. Keep raw results for losing
configurations so the same search is not repeated. Preserve the best verified
version rather than the last attempted version.

## Integrate Safely

Keep the TileLang module limited to kernel definitions. Put device and package
checks, shape support, lazy compilation, output/workspace ownership, static
launch-config selection, and routing under the operator provider. Import
TileLang lazily so an unselected provider does not initialize the compiler.

Select split counts or other tuned values from offline-calibrated device and
shape buckets. Never benchmark in `forward()`. Route unsupported dtype, layout,
capacity, architecture, or dependency versions to another provider before
launch, and surface kernel compilation or execution failures.

Validate padded heads, indirect slots, invalid rows, score/output capacity,
atomic reductions, caller-owned workspaces, no-score paths, and CUDA Graph
capture when the kernel supports them.
