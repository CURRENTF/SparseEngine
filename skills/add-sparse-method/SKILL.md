---
name: add-sparse-method
description: Add or refactor a first-class Sparse-Engine sparse method alongside vanilla, SnapKV, OmniKV, QuEST, and DeltaKV. Use when Codex needs to introduce a new `sparse_method`, move method logic out of `attention.py` or `utils/`, add method-specific cache metadata or decode-time view building, wire config and registration, and preserve the repo's cache-manager-first architecture.
---

# Add Sparse Method

Implement new Sparse-Engine methods as explicit runtime methods, not as ad-hoc helpers. Keep `attention.py` generic, let `cache_manager` own method state, and let `SparseController` keep scheduling and cross-layer coordination responsibilities.

Performance is part of method support. Do not treat a method as supported just
because it produces tokens: decode throughput must be reasonably competitive
with the closest existing sparse method on the same GPU, model, context length,
batch size, and graph/eager setting. A Python-heavy debug path, bs=1-only path,
or quadratic per-token decode scoring path is acceptable only as an explicitly
marked prototype or blocked state, not as a completed implementation.

Correctness must be exercised end to end. Unit tests, import checks, and a
single-token smoke run are not enough for a new method: run at least one
benchmark or regression-quality task that loads the model through the public
runtime method, generates outputs, records per-run status/artifacts, and can be
compared against a baseline or existing sparse method.

## Start Here

Read [references/file-map.md](references/file-map.md) before changing code.

Read [references/quest-pattern.md](references/quest-pattern.md) when the new method:
- stores persistent metadata
- depends on decode-time `q`
- needs a custom `build_decode_view(...)`
- is being added as a full peer to `omnikv`, `snapkv`, `quest`, or `deltakv`

## Placement Rules

Follow this placement order.

1. Put the method's core runtime logic in `src/sparseengine/engine/cache_manager/<method>.py` when the method owns any persistent state.
2. Put persistent page, chunk, token, or compressed metadata in the cache manager, not in `utils/`.
3. Use `CacheManager.on_kv_stored(...)` when metadata must be updated after KV is written.
4. Use `CacheManager.build_decode_view(...)` when the method needs the current layer's decode-time `q`.
5. Keep `src/sparseengine/layers/attention.py` method-agnostic. It may call generic hooks, but should not grow method-specific branches unless adding a new reusable hook.
6. Put cross-layer observation, attention-score collection, or scheduler-facing sparse orchestration in `src/sparseengine/engine/sparse_controller.py`.
7. Put hidden-state capture, activation steering, and per-sequence steering state in `src/sparseengine/engine/activation_controller.py`, with `SparseController` owning the lifecycle and model files calling only a generic hook.
8. Use `src/sparseengine/utils/` only for truly generic helpers shared by multiple methods. Do not place an entire method implementation there.
9. Add custom kernels under `src/sparseengine/kernels/triton/` or another explicit runtime module, then call them through the method's cache manager or shared decode path.

## Decision Rules

Use these rules before editing code.

- For paper/official-method support, first compare the requested method against the paper and official implementation at the level of algorithm semantics: retention score formula, candidate set, trigger schedule, fixed sink/recent/observation tokens, normalization, sampling/eval protocol, and required metadata. If strict alignment can be achieved with localized cache-manager/controller changes that fit the existing Sparse-VLLM architecture, implement the strict alignment directly instead of adding an approximation. If strict alignment requires broad framework changes such as new attention contracts, scheduler semantics, cache layout ownership, CUDA graph capture invariants, or custom kernels beyond the method-owned path, stop and ask the user whether to pursue the architectural change or accept a documented approximation.
- Do not silently ship an approximate paper method when the strict method is locally adaptable. If an approximation is kept for speed or safety, make it opt-in or clearly named, document which paper semantics it relaxes, and benchmark its quality/performance separately from the strict method.
- If the method only changes logical token selection and has no persistent state, reuse the existing cache manager if possible and prefer `SparseController`.
- If the method keeps method-specific metadata across steps, create a dedicated cache manager file.
- If the method needs the current `q`, do not try to force the logic into `prepare_forward()` or `get_read_view()` alone. Implement a decode-time hook through `build_decode_view(...)`.
- If the method changes physical KV allocation or page ownership, keep that logic in the cache manager and register it as a first-class method.
- If the method is meant to become a repo-supported algorithm, do not hide it behind a one-off helper in `attention.py`.
- If the method adds decode-time scoring, estimate and bound its per-step work before benchmarking. Avoid per-layer Python loops, CPU syncs, or dense O(context^2) scoring on the decode hot path unless the run is explicitly a prototype and reported as too slow.
- If an initial benchmark shows obviously poor decode speed, optimize the hot path or report the method as not performance-ready. Do not claim first-class support from a smoke run that is much slower than vanilla, SnapKV, QuEST, or the nearest comparable method.
- If a throughput result uses a very short output length, treat it as a smoke/debug number only. Short generations can be dominated by warmup, first-token, scheduler, graph-capture, and measurement overhead, so they are not valid decode-speed evidence.
- If the method only works at `batch_size=1`, treat that as a limitation to document or a blocker to fix. First-class method support should exercise both `bs=1` and at least one `bs>1` case unless the user explicitly scopes the method to single-request decoding.

## Editing Workflow

1. Add canonical config knobs to `src/sparseengine/configs/groups.py` or `runtime.py`; use the same names in public and internal code.
2. Register the method in `src/sparseengine/engine/cache_manager/base.py` and `src/sparseengine/engine/cache_manager/__init__.py` if needed.
3. Create or update `src/sparseengine/engine/cache_manager/<method>.py`.
4. Touch `src/sparseengine/engine/sparse_controller.py` only for controller responsibilities.
5. Touch `src/sparseengine/layers/attention.py` only to use generic hooks or shared kernels.
6. Update README and benchmark examples after the method runs.
7. Compile touched Python files with `python -m py_compile`.
8. Run at least one correctness-oriented benchmark or regression task that exercises the new public method end to end and saves result artifacts.
9. Run at least one throughput benchmark.
10. Cover multiple batch sizes: at minimum `bs=1` and one `bs>1` benchmark/smoke. If `bs>1` fails, keep the failure explicit and do not call the method first-class complete.
11. Compare decode throughput against `vanilla` and the closest existing sparse method using the same model, GPU, prompt length, batch size, output length, and `decode_graph` setting.
12. If the method is slower than expected, profile the method-owned hooks first, then optimize or document the implementation as prototype/blocked instead of declaring support complete.
13. For performance claims, use enough generated tokens to reach a stable decode region. Prefer `output_len >= 128` for quick checks and `output_len >= 256` for meaningful comparisons; use the benchmark's full-admission/stable-window metrics when available.

## Architecture Guardrails

Do not do these things.

- Do not put a new method's main logic in `src/sparseengine/utils/`.
- Do not add method-specific decode branches directly inside `Attention.forward()` when a cache-manager hook can express the same behavior.
- Do not make `SparseController` own persistent cache metadata that belongs to one method.
- Do not make `SparseController` own method-specific activation steering logic; delegate to `activation_controller.py`.
- Do not bake one method's paper constants, model allowlists, prompt markers, delimiters, or artifact filenames into this skill. Put them in method-specific code, README docs, or a method-specific reference file, and make missing required artifacts fail fast.
- Do not couple README claims to behavior that the code does not implement.
- Do not claim support from unit tests alone; run a benchmark/regression path that actually constructs the engine, generates outputs, and records status/artifacts for the new method.
- Do not leave `bs>1` untested. If multi-request decode/admission is unsupported or failing, report it as a limitation or blocker.
- Do not hide bad decode performance behind tiny smoke tests, eager-only runs, or unmatched benchmark settings.
- Do not report short-output decode throughput as a final performance result.
- Do not leave unbounded or routinely-triggered O(context^2) decode work in a first-class method without a fast kernel/vectorized path and an explicit performance result.

## Validation

Compile first.

```bash
python -m py_compile \
  src/sparseengine/config.py \
  src/sparseengine/engine/cache_manager/base.py \
  src/sparseengine/engine/cache_manager/__init__.py \
  src/sparseengine/engine/cache_manager/<method>.py \
  src/sparseengine/engine/sparse_controller.py \
  src/sparseengine/layers/attention.py
```

Benchmark after correctness is established.

```bash
python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <MODEL_PATH> \
  --methods <method> \
  --lengths 128000 \
  --batch_sizes 1,8 \
  --output_len 256 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"tensor_parallel_size":1}'
```

Use the same command shape to run a baseline and a comparable sparse method:

```bash
python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <MODEL_PATH> \
  --methods vanilla,<nearest_existing_method>,<method> \
  --lengths <REALISTIC_LENGTH> \
  --batch_sizes 1,<REALISTIC_BATCH_GT_1> \
  --output_len 256 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"tensor_parallel_size":1,"decode_graph":true}'
```

Report decode throughput, prefill throughput, memory accounting, graph status,
number of decoded tokens, batch sizes tested, correctness/regression artifact
paths, and whether the benchmark used a performance backend. If the method only
passes short-output smoke tests, only passes `bs=1`, or is clearly slow, say so
directly and keep correctness or performance work open.

When the user asks to add a new method, follow this skill first and only then invent method-specific details.
