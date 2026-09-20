# Architecture

SparseEngine has one native runtime under `src/sparseengine/`: a sparse-first
inference engine with its own scheduler,
  model runner, cache managers, sparse controller, model definitions, and
  Triton kernels.

## SparseEngine Flow

The SparseEngine inference path is:

1. `sparseengine.LLM(model, **kwargs)`
2. `LLMEngine` validates kwargs against the canonical `Config` fields.
3. `src/sparseengine/config.py` validates engine config and method compatibility.
4. `CacheManager.create()` in `engine/cache_manager/base.py` selects a cache
   manager, and `engine/sparse_methods/factory.py` selects the logical method
   runtime.
5. `Scheduler`, `ModelRunner`, the method-agnostic `SparseController`, its
   selected `SparseMethodRuntime`, kernels, and the selected cache manager
   execute prefill and decode.

`src/sparseengine/layers/attention.py` is intentionally generic. It writes K/V,
asks the sparse controller for the logical read view, and lets the cache manager
customize decode-time views through hooks such as `build_decode_view(...)`.

## Method Ownership

New SparseEngine sparse methods should follow the cache-manager-first design and
the [sparse method runtime architecture](sparse-method-runtime.md):

- Persistent physical cache state and prefix-coupled metadata belong in
  `src/sparseengine/engine/cache_manager/`.
- Per-step and cross-layer score/selection orchestration belongs in a
  `src/sparseengine/engine/sparse_methods/` runtime.
- `src/sparseengine/engine/sparse_controller.py` remains the stable generic
  facade and must not grow method-name hot-path branches.
- `src/sparseengine/layers/attention.py` should only call generic hooks or shared
  kernels.
- Public runtime arguments should use canonical names documented in
  [runtime-parameter-semantics.md](../configuration/runtime-parameter-semantics.md).

When adding a first-class method, use the repo-local `$add-sparse-method` skill.
It encodes the expected file placement, cache-manager hooks, and validation
workflow.

## Scheduling Ownership

Prefill scheduling is part of the method contract. Default policies live in
`src/sparseengine/method_registry.py`, `Config` resolves and validates the policy,
and `src/sparseengine/engine/scheduler.py` implements the scheduling behavior.

The engine currently supports:

- `all_chunked`: all prefill requests are chunked and batched through the normal
  scheduler limits, with at most `engine_prefill_chunk_size` tokens per sequence.
- `long_bs1full_short_batch`: after any supported prefix is attached, requests
  with at most `long_prefill_offload_threshold` residual tokens use atomic full
  prefill and may batch together. Larger residuals run alone as chunked RawKV
  offload prefill, with chunks capped by `engine_prefill_chunk_size`.

DeltaKV and PyramidKV implement the long branch through
`prefill_execution_mode()`. The threshold defaults to `65536` tokens (64K).
When `engine_prefill_chunk_size` is omitted, `Config` defaults it to the threshold; an
explicit chunk size must satisfy `0 < engine_prefill_chunk_size <=
long_prefill_offload_threshold`. `Config` raises `max_num_batched_tokens` to the
threshold when necessary so a boundary-sized full prefill remains atomic.
PyramidKV applies the boundary to the residual after chain-prefix attachment.
DeltaKV does not expose prefix caching and fails fast if an attached-prefix
prefill reaches its cache manager because its compressed row state has no
prefix-residency contract.

Do not encode a method's prefill policy in benchmark scripts or one-off config
defaults. Add the method-to-policy mapping to the registry and update
`tests/test_prefill_schedule_policy.py`.

## Important Files

- `src/sparseengine/configs/groups.py` and `runtime.py`: canonical runtime fields,
  defaults, and validation. Public and internal names are identical.
- `src/sparseengine/config.py`: compatibility import facade for `Config`.
- `src/sparseengine/method_registry.py`: supported method names and prefill policy
  defaults.
- `src/sparseengine/engine/cache_manager/base.py`: shared cache-manager contracts,
  construction routing, and hooks.
- `src/sparseengine/engine/sparse_controller.py`: method-agnostic engine facade.
- `src/sparseengine/engine/sparse_methods/`: method runtime interface, factory,
  per-step score/selection orchestration, and mechanism-specific runtimes.
- `src/sparseengine/engine/scheduler.py`: prefill/decode scheduling and admission.
- `src/sparseengine/layers/attention.py`: generic K/V storage and attention
  compute path.
- `benchmark/`: LongBench, MathBench, SCBench, NIAH, and multimodal benchmark
  entrypoints.
- `benchmark/model_adapters/sparseengine.py`: shared native generation adapter for
  text benchmarks.
