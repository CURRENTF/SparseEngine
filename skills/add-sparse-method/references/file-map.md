# File Map

Use this map to decide which repo files must change when adding a new Sparse-Engine method.

## Always Review

- `src/sparseengine/configs/groups.py` and `src/sparseengine/configs/runtime.py`
- `src/sparseengine/engine/cache_manager/base.py`
- `src/sparseengine/layers/attention.py`
- `src/sparseengine/engine/sparse_controller.py`
- `README.md`

## Add a First-Class Method

Touch these files when the method becomes a supported `sparse_method`.

- `src/sparseengine/configs/groups.py` or `src/sparseengine/configs/runtime.py`
  Add config fields, validation, and defaults.
- `src/sparseengine/engine/cache_manager/<method>.py`
  Put method state, metadata, cache layout, and decode-time hooks here.
- `src/sparseengine/engine/cache_manager/base.py`
  Register `CacheManager.create(...)` routing and add generic hooks only if the existing hooks are insufficient.
- `src/sparseengine/engine/cache_manager/__init__.py`
  Export the new cache manager when appropriate.
- `src/sparseengine/engine/activation_controller.py`
  Put method-specific hidden-state steering or activation capture here, with
  `SparseController` owning the lifecycle and model files calling only a generic
  hook.
- `README.md`
  Document method semantics, knobs, and benchmark examples.

## Touch `SparseController` Only for Controller Work

Edit `src/sparseengine/engine/sparse_controller.py` when the method:

- reuses observed attention scores
- needs cross-layer propagation
- shares dynamic logical views with other layers
- changes scheduler-facing sparse state

Do not move method-owned cache metadata here.

If the method needs hidden-state steering, keep the steering math and per-seq
activation state in `activation_controller.py`; `SparseController` should only
create the controller, call its prepare/post hooks, and expose a generic model
hook.

## Touch `attention.py` Only for Generic Hooks

Edit `src/sparseengine/layers/attention.py` when you need to:

- call a new generic cache-manager hook
- wire a new shared kernel path
- keep the store-view, read-view, and decode-view call sequence consistent

Do not bury a full method implementation in `attention.py`.

## Add Kernel Code Only When Needed

Touch `src/sparseengine/kernels/triton/` or another explicit kernel module when:

- the existing decode or prefill kernels are the bottleneck
- the method requires a new layout-aware fused operator
- the method cannot be expressed as view selection plus existing kernels

## Minimum Validation Set

After editing code:

1. Compile touched Python files with `python -m py_compile`.
2. Run one small correctness-oriented benchmark or regression task that actually
   loads the public method, generates outputs, and saves status/artifacts.
3. Run one throughput benchmark.
4. Cover multiple batch sizes: at minimum `bs=1` and one `bs>1` run. Treat an
   untested or failing `bs>1` path as a documented limitation or blocker.
5. Compare against at least one existing method on the same machine.
6. Treat poor decode speed as an implementation blocker, not just a benchmark note.
   Use matched settings against `vanilla` and the closest existing sparse method;
   do not use eager-only, bs=1, or tiny-context smoke results as evidence that the
   method is performance-ready.
7. Do not use very short generations as decode-speed evidence. Runs with small
   `output_len` are useful for debug only; use `output_len >= 128` for quick
   performance checks and prefer `output_len >= 256` for meaningful comparisons.
