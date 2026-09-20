# Troubleshooting

## Chain eviction with free KV slots

Physical free slots can include space reserved for already admitted requests.
`chain_admission` logs a JSON object at WARNING when admission needs reclamation
or fails for lack of capacity; use `LOG_LEVEL=DEBUG` for normal admission plans.
`pressure` distinguishes `kv_slots` from `resident_rows` (both may apply).
The same event includes cache free-slot budgets, separate chain/decode
reservations, the incoming request's required slots/rows, and the resulting
deficits. Per-layer arrays follow `kv_layer_indices`; do not sum layers as if
they were independent token capacities. Counts are slots, not bytes.

`outcome=planned` does not prove that the plan executed. `victim_chain_ids`
identifies planned permanent evictions; `demote_chain_ids` identifies chains
whose CPU snapshots preserve resumability. `chain_evicted` records actual
removal from the chain index, not completion of GPU payload release.
`chain_demoted` records release of GPU residency while retaining the CPU copy.
Correlate events by chain/sequence ID and keep server errors alongside them;
chain eviction alone is not evidence of CUDA OOM.

## `SamplingParams` Does Not Allow Greedy Decoding

`SamplingParams.temperature` must be `> 1e-10`. Use a tiny temperature such
as `1e-5` for almost-greedy decoding.

## `Insufficient KV cache slots to admit prompt`

The engine cannot allocate enough KV slots for the prompt or prompt chunk.
Increase `gpu_memory_utilization`, reduce `max_model_len` or batch size, or
reduce the keep-token budgets.

## TensorRT-LLM DeepGEMM compilation cache

CUDA workers use separate TensorRT-LLM DeepGEMM cache directories to avoid
concurrent cold-compilation writes. The cache root is selected in this order:
`SPARSEENGINE_TRTLLM_DG_CACHE_ROOT`, `TRTLLM_DG_CACHE_DIR`, then
`${XDG_CACHE_HOME:-~/.cache}/sparseengine/trtllm-deepgemm`. Both explicit variables
name a root; each worker exclusively leases a child directory, printed at startup.
Set the root to a writable data disk when the home filesystem is space-limited.

Concurrent workers, including separate servers with the same rank, hold separate
file locks for their entire process lifetime. After exit, a later worker can
reuse a released directory. Installed package/toolchain metadata and DeepGEMM
headers separate cache namespaces. For custom toolchains or modified external
headers not covered by package metadata, select a new cache root per build.
The filesystem must support advisory file locks across participating hosts.
Sequential engines in the same process reuse its directory because
the upstream compiler retains the first path; changing the root requires a new
process. Set cache variables before starting the engine or using this external
kernel directly. Direct kernel users must also avoid initializing its compiler
before the engine configures its cache.

Directories are retained after exit for diagnostics; remove unused worker
directories and lock files only after all processes using the root stop. Never
remove an active lock file. Old UUID directories are not migrated automatically.
Other FlashInfer and Triton caches
are unaffected. This isolates cache writes; compiler and kernel failures still
propagate.
