# Troubleshooting

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
