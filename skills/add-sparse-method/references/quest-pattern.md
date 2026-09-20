# QuEST Pattern

Use QuEST as the reference design for adding a method that is query-aware at decode time and owns persistent metadata.

## Structural Lessons

QuEST is treated as a full method, not a helper.

- It lives in `src/sparseengine/engine/cache_manager/methods/quest.py`.
- It is registered through `CacheManager.create(...)`.
- It keeps page metadata in the cache manager.
- It uses `build_decode_view(...)` because decode-time selection depends on the current `q`.
- It does not store the method's main logic in `utils/`.

## Why `build_decode_view(...)` Exists

`prepare_forward()` and `SparseController.get_read_view()` happen before the current layer's decode-time `q` is available.

If a method needs the current `q`, it cannot be fully resolved during controller preparation alone. In that case:

1. keep the method's persistent state in the cache manager
2. let `attention.py` stay generic
3. call `cache_manager.build_decode_view(...)` right before decode kernels

This is the right pattern for methods like QuEST.

## Why QuEST Does Not Belong in `utils/`

QuEST owns:

- method-specific cache allocation policy
- page slot bookkeeping
- page min/max metadata
- decode-time view construction

Those are runtime method responsibilities, not generic utilities. Use `utils/` only for helpers that are reusable across methods.

## Prefill Semantics

QuEST prefill stays dense in the attention sense, but it still updates page metadata after KV writes through `on_kv_stored(...)`.

That means QuEST can affect prefill throughput even when sparse page selection only happens at decode time.

## Performance Lessons

The first QuEST bottleneck was metadata maintenance, not decode attention itself.

Useful optimizations:

- batch metadata updates instead of looping page-by-page in Python
- keep `attention.py` unchanged and optimize method-owned hooks first
- use profiler tags to split `quest_update_metadata`, `quest_build_decode_view`, `quest_score_pages`, and `quest_pack_slots`

## Official Kernel Caveat

The official QuEST repo is useful as a kernel reference, but its Python wrapper and controller path are still strongly shaped around batch size 1 in some places.

Do not assume the official wrapper can be dropped into SparseEngine's batched engine unchanged. Check the wrapper and batch assumptions before planning an integration.
