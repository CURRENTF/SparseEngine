# Runtime Parameters

Common runtime parameters below can be passed to `LLM(model, **kwargs)` or `Config`. JSON configs and benchmark manifests use the same names. Defaults are initial configuration values; `None` means automatic resolution or unset.

## Model and Execution

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | str | Required | Model path. |
| `max_model_len` | int / None | `None` | Maximum context length. Automatic values are limited by model and runtime capacity; explicit values cannot exceed the model limit. |
| `gpu_memory_utilization` | float | `0.9` | Fraction of GPU memory available to the engine. |
| `tensor_parallel_size` | int | `1` | Attention tensor parallel size. |
| `decode_graph` | bool | `True` | Enable decode CUDA Graphs. Set to `False` for eager execution or methods without graph support. |
| `decode_graph_capture_sizes` | `auto` / list[int] | `auto` | Decode batch sizes to capture. Automatic planning captures every size up to the graph budget, then favors small batches and spreads the remaining sizes through `max_decoding_seqs`. An explicit list must include `max_decoding_seqs`. |
| `decode_graph_startup_capture_limit` | int / None | `None` (32) | Maximum number of startup decode graphs. Automatic capture plans use up to this many graphs. |

Shared and routed MoE branches run concurrently by default during DP=1 decode
in GLM-4.7-Flash, Qwen3.5/3.6 MoE, and Gemma 4 MoE (its dense MLP branch),
including supported FP8 configurations. Prefill remains serial; existing fused
shared-expert paths remain fused. Qwen3 MoE and MiniMax M2 have no shared branch
and are unaffected.

This optimization is automatic and independent of `async_scheduling`. Supported
decode CUDA Graphs capture the branch dependencies. Request scheduling and each
model's gating, normalization, and reduction order are preserved.

## Scheduling and Token Budgets

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_num_batched_tokens` | int / `auto` | `auto` | Token budget per scheduling step, estimated at startup from model metadata, TP and device memory. |
| `max_num_seqs_in_batch` | int | `32` | Maximum requests per batch. |
| `max_decoding_seqs` | int / None | `None` | Maximum decode batch size; defaults to `max_num_seqs_in_batch` and sets the final graph capture size. |
| `engine_prefill_chunk_size` | int / `auto` / None | `auto` | Per-request prefill token limit per step; `auto` (or `None`) follows the final batch token budget, subject to method constraints. |
| `long_prefill_offload_threshold` | int | `65536` | Long-request threshold in tokens for the policy that prefills long requests in full and batches short requests. |
| `mla_prefill_history_chunk_size` | int | `16384` | Maximum historical KV tokens processed at once during MLA prefill. Smaller values reduce history workspace memory. |
| `decode_reservation_tokens` | int | `1024` | Maximum token window reserved for subsequent decode steps. Must be positive; this is not the total output limit. |

Both parameters resolve to integers at startup and remain fixed during execution.
For ordinary chunked scheduling, both automatic values use the same budget.
An explicit batch token budget determines the automatic chunk. An explicit chunk
is preserved and must fit the automatic batch budget, or startup fails. Both
values may still be set independently as explicit integers.

For `long_bs1full_short_batch`, the automatic budget must fit a complete
`long_prefill_offload_threshold`, and the automatic chunk cannot exceed that
threshold. With both parameters automatic, both normally equal the threshold;
a larger decode concurrency limit can require a larger batch budget. An
insufficient estimate fails without changing the offload threshold. Reduce the
threshold or set the batch token budget explicitly. Automatic sizing requires
activation headroom, so `gpu_memory_utilization=1.0` requires an explicit batch
token budget.

Automatic sizing is a conservative estimate, not an optimal-throughput or OOM
guarantee. Startup logs and worker information expose the resolved integers;
fix and record both values for reproducible experiments. Engine defaults do not
replace parameters explicitly supplied by benchmark runners.

## Asynchronous Execution

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `async_scheduling` | bool / None | `None` | Automatically enable asynchronous scheduling for compatible CUDA configurations. `False` selects synchronous execution; explicit `True` rejects incompatible configurations. |
| `async_max_inflight` | int | `2` | Maximum submitted, uncollected batches during asynchronous scheduling. Minimum: `2`. |

Asynchronous scheduling supports DP=1 text-only models without recurrent
attention, with radix or no prefix cache and no prefix offload. Other
configurations retain synchronous scheduling in automatic mode. Sparse decode
and independent sparse prefill use their existing lifecycle hooks; existing
method, storage, and CUDA Graph compatibility rules still apply. Both eager and
supported decode Graph execution can use asynchronous scheduling.

GPU computation and KV updates remain ordered. Methods requiring committed CPU
token history wait at that dependency boundary; enabling asynchronous scheduling
does not remove required waits or change selection and compression semantics.
Output publication remains ordered. EOS/cancellation can leave bounded already
submitted work; its outputs are discarded and storage retires after its users
complete. Capacity preemption drains outstanding work before synchronous
recompute recovery. This option does not enable mixed prefill/decode batches.

## Sparse Methods and Shared Budgets

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sparse_method` | str | `""` | Cache/decode method. An empty string or `vanilla` selects full attention. See [Sparse Methods](../features/sparse-methods.md) for available methods. |
| `sink_keep_tokens` | int | `64` | Number of initial tokens to retain. |
| `recent_keep_tokens` | int | `512` | Number of recent tokens to retain. |
| `decode_keep_tokens` | int | `4096` | Token budget for sparse selection; interpretation depends on the method. |
| `full_attention_layers` | str / list[int] | `"auto"` | Full-attention layers: automatic profile, comma-separated string, or index list. Unregistered OmniKV / DeltaKV models require calibration or an explicit list. |
| `sparse_prefill_score_mode` | str / None | `None` | Scoring mode: automatic, `probability`, or `logits`. SnapKV/PyramidKV use the same mode for prefill and decode observation windows. Logits scoring is limited to SnapKV, PyramidKV, and H2O and requires float32 scores. |

Token-count budgets must be nonnegative integers, not ratios. QuEST derives its total selection budget from `sink_keep_tokens + decode_keep_tokens + recent_keep_tokens`; `quest_token_budget` cannot be set directly.

## Prefill Sparsity

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_sparse_method` | str / None | `None` | Independent prefill acceleration: `h2o_prefill`, `flashprefill_v2`, or `omnikv_prefill`; `""` disables it. When omitted, only H2O enables `h2o_prefill` by default. |
| `omnikv_prefill_full_attention_layers` | str / list[int] / None | `"auto"` | Full-attention layers for OmniKV prefill, independent of decode settings. |
| `omnikv_prefill_keep_tokens` | int | `4096` | Historical token selection budget for OmniKV prefill. |
| `omnikv_prefill_sink_keep_tokens` | int | `8` | Initial tokens retained during OmniKV prefill. |
| `omnikv_prefill_recent_keep_tokens` | int | `128` | Recent tokens retained during OmniKV prefill. |
| `flashprefill_v2_abs_threshold` | float / None | `None` | FlashPrefill V2 sparsity threshold in `[0, 1]`. Enabling the method requires an explicit value calibrated for the model. |

`flashprefill_v2` requires explicit KV models; `omnikv_prefill` does not support radix prefix reuse. See [Sparse Methods](../features/sparse-methods.md) for compatible combinations and [FlashPrefill V2](../features/flashprefill-v2.md) for tuning.

## SnapKV and PyramidKV

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `observation_window_size` | int | `32` | Positive number of final prefill queries used for prompt scoring and maximum recent decode queries used at each eviction. Decode history resets after compaction. |
| `snapkv_decode_eviction` | bool | `False` | Enable periodic decode eviction for SnapKV. PyramidKV always enables it. |
| `decode_eviction_interval` | int | `1024` | Positive physical KV growth above the per-layer retention budget before SnapKV/PyramidKV evict. The score is computed after decode Graph replay, only when this boundary is reached. |

## KVzip

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `kvzip_token_budget` | int | `4096` | Total prompt tokens retained after reconstruction; positive. Generated tokens append without further eviction. |
| `kvzip_score_chunk_size` | int | `2048` | Original-context tokens reconstructed per forward; positive. |
| `kvzip_prev_postfix_size` | int | `64` | Maximum preceding context tokens included in each replay; nonnegative. |

See [KVzip constraints](../features/sparse-methods.md#kvzip) for reconstruction
headroom and the distinction from original per-head KVzip.

## H2O

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `h2o_prefill_budget` | int | `8192` | KV budget after intermediate prefill chunk compression. Must be at least the decode budget. |
| `h2o_decode_budget` | int | `4096` | KV budget after final prompt compression. Must be positive. |
| `h2o_recent_ratio` | float | `0.5` | Fraction of the budget assigned to recent tokens, in `(0, 1)`. |
| `h2o_prefill_score_window` | int | `128` | Query window for prefill scoring. `0` uses the whole current chunk; probability mode accepts `[0, 128]`. |
| `h2o_decode_eviction` | bool | `False` | Enable ongoing decode scoring and eviction. Forces probability scoring when enabled. |
| `h2o_decode_eviction_interval` | int | `128` | Decode eviction interval. Capacity pressure may trigger earlier eviction. |
| `h2o_decode_score_fusion` | bool | `True` | Reuse attention scores during decode eviction to reduce separate scoring overhead. |

By default, compression happens during prefill without ongoing decode eviction. MLA decode eviction uses approximate scores and is not fully aligned with original H2O.

## DeltaKV

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `deltakv_checkpoint_path` | str / None | `None` | Compatible compressor checkpoint path, required for reportable inference. |
| `deltakv_neighbor_count` | int | `4` | Number of reference neighbors. |
| `deltakv_center_ratio` | float | `0.1` | Reference center ratio. |
| `deltakv_latent_dim` | int | `128` | Compressor latent dimension; must match the checkpoint. |
| `deltakv_latent_quant_bits` | int | `4` | Latent-state quantization bits: `0`, `2`, or `4`; `0` disables quantization. |
| `deltakv_latent_quant_group_size` | int | `0` | Latent-state quantization group size. Automatically set to `32` with the default 4-bit configuration. |

See [DeltaKV](../features/deltakv.md) for model and checkpoint configuration.

## RoPE and Context Limits

RoPE settings come from the model checkpoint's `rope_parameters`, with support for the legacy `rope_scaling` field.

| Model path | Supported settings |
| --- | --- |
| Standard one-dimensional MHA/GQA | `default`, `linear`, `yarn`, and `llama3`; `dynamic` and `longrope` are unsupported. |
| DeltaKV | Only `default` and `llama3`. |
| Qwen3.5 MRoPE, GLM-4.7-Flash MLA, Gemma4 per-layer RoPE | Use their own RoPE configurations; standard one-dimensional scaling parameters are not accepted. |

YaRN's effective context length is `original_max_position_embeddings × factor`; explicit context limits cannot exceed it.

## Benchmark adapter

Text benchmarks share `benchmark/model_adapters/sparseengine.py`. It accepts the
same public parameter names, constructs the native engine, and exposes a small
generation callable for LongBench, MathBench, NIAH, and RULER core. SCBench uses
its native `sparseengine` attention type. There is no `--backend hf` option.

## Decode reservation window

`decode_reservation_tokens` is a positive integer (default: `1024`) shared by
all cache methods and prefix modes. It bounds the next decode-step window,
not total output length or a sparse eviction interval. The first output token
comes from prefill; subsequent decode steps reserve up to this window or the
remaining output limit. Methods account for page rounding, eviction peaks,
and compressed pools in their own resource units.

The scheduler renews active decode windows before admitting more prefill work.
Renewal failure uses existing preemption/recompute recovery. A sole request may
receive a smaller window if at least one step fits. A window boundary never
forces a queue rotation or changes scoring/eviction. Small windows can increase
preemption under load; larger windows can delay new requests. Chain admission
still checks suffix-prefill and CPU restore capacity separately.

## R-KV retention

`rkv` uses a global retained token set across KV layers and attention TP ranks.
Its total retention budget is `sink_keep_tokens + decode_keep_tokens +
recent_keep_tokens`; for this method these fields contribute only to the total,
not separate protected regions. Exactly `rkv_observation_tokens` trailing tokens
are protected, with no fixed sink. This window also determines the number of
recent decode queries used for scoring; prompt queries are excluded.

`rkv_compression_interval` sets decode-buffer boundaries and must be at least
the observation window. Compression also requires physical length >= total
budget + interval. A long prompt therefore remains resident until a decode
boundary, rather than being compressed during prefill.

`rkv_kernel_size` is the positive odd importance-pooling width (default 7).
`rkv_alpha` weights importance in `alpha * importance - (1-alpha) * redundancy`.
`rkv_score_chunk_mb` bounds scoring tiles (default 512 MiB); startup validates
that it can score one request at `max_model_len`, including materialized keys
and queries. The retention budget does not bound the full prompt before its
first decode compression. Increase this workspace for long contexts if the
startup check reports insufficient bytes; the cache allocator
reserves this workspace separately. Increase it if even one scoring tile cannot
fit. There is no automatic Full-KV fallback on scoring failure.

Legacy RKV approximation overrides `rkv_redundancy_window` (nonzero),
`rkv_max_redundancy_tokens` (non-null), `rkv_similarity_threshold` (other than 0.5),
and `rkv_recent_similar_keep` (other than 1) are rejected. Existing RKV results
must be rerun before comparison because the token-selection semantics changed.
