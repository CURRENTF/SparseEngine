# Startup prefill memory profiling

Before model construction, automatic prefill limits reuse the activation-headroom
heuristic: `floor(total_memory * (1 - gpu_memory_utilization) /
(intermediate_size_per_tp_rank * activation_dtype_bytes * 16))`. The smallest
total memory among participating ranks determines the shared estimate. The
ordinary automatic ceiling is the former 65,536-token batch default; an explicit
chunk, decode concurrency or atomic full-prefill threshold may require a higher
ceiling, but must still fit the estimate. Both automatic limits are equal for
ordinary chunked scheduling. For the long/full policy, the automatic chunk is
bounded by the unchanged offload threshold and a fully automatic budget is the
larger of that threshold and the decode concurrency limit. CUDA device properties
supply total memory without live memory queries on every device in the parent.
The resolved integers are fixed before model/workspace allocation, prefix-cache
fingerprinting and the profile below. Explicit batch budgets skip this heuristic.
This does not search for the largest feasible workload or measure performance.

Startup measures one prefill step after the small compilation warmup. The batch
uses the maximum request count allowed by `max_num_seqs_in_batch`, resident row
capacity and the token budget. New tokens are distributed evenly across requests,
filling `max_num_batched_tokens` when the per-request chunk and context limits
permit it.

Only the first request has synthetic history: its history plus current chunk
reaches `max_model_len - 1`, leaving room for the sampled token. All other
requests start with no history. For a 50,000-token context limit, 32 requests and
a 4,096-token budget, each chunk has 128 tokens. The first request has 49,871
history tokens, and the batch has 53,967 visible KV tokens in total. If the first
chunk already covers the context limit, the same step runs without history.

The history probe uses the ordinary StandardCacheManager allocation and view
paths, a dense SparseController, and the existing model forward and bound
attention providers. A startup-only cache manager seeds zero-valued history
without evaluating the preceding tokens. Same-shape layers share physical KV
storage: each layer overwrites the current chunk while the synthetic history
remains zero. MLA retains its latent/RoPE storage layout; explicit KV supports
both uniform and heterogeneous layer shapes.

History storage is allocated before the memory profile begins and remains live
until the profile finishes, so it does not count as transient model memory.
Afterward all temporary sequences and cache bindings are released and the
original runtime is restored, including on forward failure. This single prefill
peak feeds the existing per-rank KV capacity decision. Logs report the batch,
new-token count, maximum context and chunk lengths, synthetic history length,
total visible KV tokens and measured peak.

This is a representative allocation probe, not a worst-case workload search or
a numerical/throughput benchmark. It does not reconstruct sparse-method history,
long-history compression/score state, recurrent history, or simultaneous long
requests. Unmeasured costs rely on the memory left outside
`gpu_memory_utilization`; no extra analytical reserve, iterative sizing, or
method-specific seeding is introduced. Existing workspace limits still apply,
so the probe can expose a maximum-context MLA workspace failure during startup.
