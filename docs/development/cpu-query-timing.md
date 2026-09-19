# Host-side cache and scheduler timing

Set `SPARSEENGINE_CPU_TIMING_INTERVAL_S=30` before starting the engine to emit
periodic `cpu_timing` JSON records. The default `0` leaves decorated functions
unwrapped. The value must be finite and nonnegative.

The probe covers scheduling, postprocessing, decode reservation, prefill
reservation, prompt admission budgets, prefix lookup/capacity queries, cache
eviction, forward-end cache updates and sequence cleanup in the standard cache
path. It does not change scheduling or insert device synchronization.

Each process/thread reports its own window, rank, and named stages:

- `calls`, `cpu_ms`: call count and total **thread CPU time**;
- `wall_ms`, `max_wall_ms`: elapsed total and longest call;
- `errors`: calls that raised; the original exception still propagates.

Timings are inclusive: parent and child categories overlap and must not be
summed. CPU time excludes sleeping but may include CUDA driver busy polling;
wall time can include device waits, descheduling and nested calls. Neither
field is GPU kernel execution time or evidence that a CPU bottleneck is proven.

Reports are emitted after the outermost instrumented call returns once the
interval has elapsed, not by a background timer. Idle or stuck threads do not
emit periodic heartbeats, and a final incomplete window is not flushed. No
additional cache-capacity queries are made merely to populate a report.

The prefix index reports `prefix_cache_freeable_scans` for initial full-index
construction and `prefix_cache_freeable_snapshot_builds` for immutable capacity
snapshots. Subsequent membership changes update affected ancestor paths rather
than rebuilding the tree view. Snapshot copying and resident-slot weighting can
still be linear in the number of freeable blocks. Mutation work is included in
its owning stage (for example sequence release), not necessarily the capacity
query stage; compare both when evaluating this optimization.
