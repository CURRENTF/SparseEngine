# Kernel Reference Sources

Use upstream code as design evidence and few-shot material, not as an implicit
dependency or an unquestioned performance baseline. Before adapting code,
record the exact commit, source path, license, local changes, supported
hardware, and benchmark protocol.

## TileLang

- [Tile-AI/TileLang](https://github.com/tile-ai/tilelang): inspect official
  examples and documentation for the installed version. The DeepSeek MLA
  examples are especially relevant to tiled attention, split-KV, layouts,
  pipelining, shared-memory swizzle, and warp specialization.
- [DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels): use as a
  kernel portfolio and testing reference. Treat each kernel's hardware,
  dependency, and benchmark assumptions as local to its pinned revision.

## Serving Integration

- [SGLang](https://github.com/sgl-project/sglang): inspect current in-tree
  Triton and TileLang serving kernels, provider/backend dispatch, graph
  constraints, tests, and benchmarks. Search the pinned checkout rather than
  relying on remembered paths because the tree changes frequently.
- Inspect SGLang's JIT kernel path before introducing a standalone C++ project.
  Inspect the separate SGL kernel package only when CUTLASS, CuTe, AOT build,
  or packaged binary integration is required.

## Reuse Rules

1. Prefer a repository-owned or installed-version example over `main`.
2. Compare tensor semantics, layouts, dtypes, scaling, masking, and mutation
   before comparing implementation shape.
3. Port the smallest relevant mechanism rather than copying an entire module.
4. Retain required license and provenance files and describe local changes.
5. Rebuild correctness and benchmark baselines under Sparse-Engine's actual
   serving contract.
