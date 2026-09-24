# Core Sparse Methods

SparseEngine is built around a cache-manager-first sparse runtime. The engine
supports physical eviction, logical masking, and hybrid compression without
forcing `attention.py` to own method-specific state.

## Supported Methods

Set `sparse_method` to one of the following method names.

| Method | Family | Description | Main Runtime Knobs |
| --- | --- | --- | --- |
| `palu` | Low-rank KV | [Activation-whitened grouped K/V compression](palu.md), with fused decode reconstruction. | `palu_checkpoint_path` |
| `vanilla` | Dense baseline | Full attention baseline. Use it to verify correctness and measure the non-sparse engine path. | Common engine knobs only. |
| `streamingllm` | Physical eviction | StreamingLLM-style fixed sink plus recent-window cache. Tokens outside the retained prefix/tail policy are physically evicted from the active KV cache. | `sink_keep_tokens`, `recent_keep_tokens` |
| `attention-sink` | Physical eviction | Alias-style attention-sink policy with the same sink-token and recent-window retention model. It is useful for comparing sink-window behavior against other physical eviction methods. | `sink_keep_tokens`, `recent_keep_tokens` |
| `snapkv` | Physical eviction | SnapKV-style token selection uses an end-of-prompt observation window to keep important prompt KV. Decode eviction is optional (`snapkv_decode_eviction=false` by default). When enabled, each sparse layer scores its last `observation_window_size` decode queries and compacts after its physical row reaches the layer budget plus `decode_eviction_interval`. Scoring runs after the decode Graph replay, only at eviction boundaries. | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `observation_window_size`, `snapkv_decode_eviction`, `decode_eviction_interval`, `sparse_prefill_score_mode` |
| `kvzip` | Physical eviction | Repository token-shared KVzip reconstruction scoring, followed by one global prompt-token compaction. | `kvzip_token_budget`, `kvzip_score_chunk_size`, `kvzip_prev_postfix_size` |
| `h2o` | Physical eviction | Intermediate prefill chunks can be compacted to `h2o_prefill_budget`; the final prompt is compacted to `h2o_decode_budget`. Decode is score-free by default and grows with generated tokens. Optional `h2o_decode_eviction` accumulates decode probabilities and periodically evicts physical KV. | `h2o_decode_eviction`, `h2o_decode_budget`, `h2o_decode_eviction_interval`, `h2o_prefill_budget`, `h2o_recent_ratio`, `h2o_prefill_score_window`, `sparse_prefill_score_mode` |
| `pyramidkv` | Physical eviction | PyramidKV-style layer-dependent KV retention. Decode eviction is always enabled. Each sparse layer scores its last `observation_window_size` decode queries and compacts after its physical row reaches that layer's budget plus `decode_eviction_interval` (default 1024). Scoring runs after the decode Graph replay, only at eviction boundaries. | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `observation_window_size`, `decode_eviction_interval`, `sparse_prefill_score_mode` |
| `omnikv` | Logical masking with optional offload | Cross-layer token selection; optionally keep sparse-layer history in pinned CPU memory and fetch the exact selected KV for decode. | `full_attention_layers`, `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `enable_omnikv_offload` |
| `quest` | Query-aware page selection | QuEST selects token pages from persistent min/max page summaries. Prefill stays dense. Explicit-KV models score in key coordinates; GLM-4.7-Flash scores the fused MLA latent/RoPE cache with the matching absorbed decode query while keeping the compute payload latent. | `quest_chunk_size`, `quest_skip_layers`, `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens` |
| `deltakv` | Hybrid compression | Slim compressor-backed DeltaKV runtime. Legacy `deltakv-less-memory*` names normalize here for older configs, but real benchmark runs still require a matching compressor checkpoint. | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `full_layer_kv_quant_bits` |

SparseEngine uses `sparse_method` unchanged in public commands, `LLM(...)`, the
runtime config, and internal consumers.



## KVzip

Use `LLM(model, sparse_method="kvzip", kvzip_token_budget=4096)` to retain
at most 4096 prompt tokens after context reconstruction. Shorter prompts retain
all KV. The first output token comes from the original dense prefill; subsequent
decode attends to retained prompt KV and all generated KV.

This is the repository's `kvzip_global` token-shared variant, not the original
paper's non-uniform per-head eviction. Scores and retained positions are shared
across layers, heads and attention TP ranks. No additional checkpoint is needed.
`kvzip_token_budget` is the complete prompt budget; `sink_keep_tokens`,
`recent_keep_tokens` and `decode_keep_tokens` do not add protected regions.

Supported models are Qwen2, Qwen3 and Llama with explicit KV storage. Chunked
prefill, batched requests, tensor parallelism, eager decode and decode CUDA Graph
use the same physical cache. Prefix caching (radix or chain), offload, sparse
prefill overrides, recurrent/MLA/shared-KV models and MoE are rejected.

Reconstruction uses teacher-forced chunks of `kvzip_score_chunk_size` tokens
(default 2048), with up to `kvzip_prev_postfix_size` preceding tokens (default 64).
The replay instruction, preceding tokens and chunk must fit
`engine_prefill_chunk_size`; the full prompt plus replay must fit
`max_model_len`. Allow this extra capacity when selecting prompt lengths.
Reconstruction adds prefill work; compression alone does not establish a speedup.


## OmniKV KV offload

OmniKV offload keeps full-attention KV on GPU and sparse-layer history in CPU
memory, fetching the exact selected KV for attention and reusing GPU-cached KV.
Enable it when GPU memory is limited. When concurrency exceeds GPU-resident KV
capacity, it can substantially reduce TTFT by reducing request waiting. Transfers
can lower decode TPS, especially when all requests already fit on GPU.

| Parameter | Description |
|---|---|
| `enable_omnikv_offload` | Default `False`. Set to `True` with `sparse_method="omnikv"` to enable offload. |
| `omnikv_offload_cache_tokens` | GPU cache capacity in tokens per request per sparse layer. Default `None` sizes it automatically; `0` disables LRU caching. A positive value must cover the selected-token budget. Larger caches use more GPU memory to reduce transfers. |

Prefill acceleration is selected separately with `prefill_sparse_method`.
SparseEngine currently supports `h2o_prefill` for intermediate-chunk KV
compaction, `flashprefill_v2` for sparse prefill attention computation, and
`omnikv_prefill` for chunked prefill with cross-layer history selection. They
are alternatives on one axis and can each be combined with a compatible
cache/decode method. See
[runtime parameter semantics](../configuration/runtime-parameter-semantics.md#prefill-sparsity)
for the H2O prefill/decode combination matrix and the omitted-versus-empty
compatibility rule.

`omnikv_prefill` can pair with vanilla (`sparse_method=""`) or OmniKV decode.
It retains full KV storage and changes only prefill attention reads. Set
`omnikv_prefill_full_attention_layers="auto"` (the default) to use a prefill-specific
profile when registered, or reuse the model's OmniKV decode profile, filtered to
global KV layers. An unregistered model fails
explicitly and needs calibration first. An explicit list can override it and
must include the first global KV layer. Sliding-window layers are excluded
and always retain their normal attention. These layers and the following budgets are
independent of the decode configuration:

| Parameter | Meaning |
|---|---|
| `omnikv_prefill_keep_tokens` | Selected historical tokens, excluding sink, recent and current chunk; default 4096. |
| `omnikv_prefill_sink_keep_tokens` | Always retained leading tokens; default 8. |
| `omnikv_prefill_recent_keep_tokens` | Always retained history immediately before the current chunk; default 128. |

The entire current chunk remains visible with causal masking. The method uses
full-chunk raw-QK scoring in float32: explicit KV averages over queries then
takes the head maximum; MLA takes the maximum over both queries and heads.
`sparse_prefill_score_mode` does not change it. `engine_prefill_chunk_size` affects both quality and saved
history-attention work. One full-prompt chunk has no historical work to prune.
Sparse prefill can affect quality even with vanilla decode.

Explicit KV, MLA latent storage, and Gemma 4 global attention (including shared-KV
consumer layers) are supported. Gemma 4 sliding-window layers neither score nor
consume OmniKV prefill selections. Contiguous explicit KV and MLA can use OmniKV
CPU KV backing with `enable_omnikv_offload=true`, including independent prefill
and decode full-attention layer lists. Heterogeneous Gemma 4 KV storage retains
its existing offload restriction.

With `enable_prefix_caching=true`, `prefix_cache_mode="auto"` selects `chain`;
explicit `chain` is also accepted. Send the complete logical context and returned
`chain_id` to continue the same completed turn. Chain reuse preserves the actual
computed KV; it does not recompute earlier turns under a new chunk partition.
Chunk-wide query selection can change a shared prefix's KV, so `radix` is rejected.
Keep `enable_prefix_cache_offload=false`: idle chain snapshots do not support
this shared slot layout. `enable_omnikv_offload` remains available independently.

> [!NOTE]
> The two score-free decode contracts have different paper provenance. The
> [SnapKV paper](https://arxiv.org/abs/2404.14469) selects prompt KV from an
> observation window at the end of the prompt; adding decode-time rescoring and
> eviction would be a SparseEngine extension. The
> [H2O paper](https://arxiv.org/abs/2306.14048) instead defines dynamic retention
> over successive decode steps. SparseEngine's intermediate-chunk H2O compaction
> is its own prefill extension. Final-prompt compaction instead belongs to the
> decode contract because it creates the shorter cache used during generation,
> even though the mutation executes at the final-prefill boundary. Optional
> online score updates move toward the original H2O algorithm; periodic batched
> eviction and bounded prefill observation remain system/algorithm variants.

SnapKV defaults `sparse_prefill_score_mode` to `logits`, while PyramidKV defaults
to `probability`. H2O (including standalone `h2o_prefill`) defaults to
`sparse_prefill_score_mode="logits"` and `h2o_prefill_score_window=128`.
Explicit score-mode and window settings override these defaults, except that
`h2o_decode_eviction=True` forces `sparse_prefill_score_mode="probability"`.

The H2O default is an approximation using a bounded query window. To select
full-chunk normalized attention-mass scoring, explicitly set
`sparse_prefill_score_mode="probability"` and `h2o_prefill_score_window=0`.
In that mode, every KV layer independently sums normalized attention
probabilities over the full current query chunk and accumulates attention mass
across prefill chunks. H2O probability mode emits a performance warning because
it needs additional QK scoring even when attention LSE is reused. Both modes
retain independent scores for every H2O KV layer.

`h2o_decode_eviction` defaults to `False`. Enable it with `sparse_method="h2o"`
to accumulate normalized attention mass at every decode step and physically
retain heavy hitters plus recent tokens. Eviction returns each triggered row
to `h2o_decode_budget` at `budget + h2o_decode_eviction_interval`; memory pressure
can trigger earlier eviction of over-budget active rows. The switch forces
probability scoring even if `logits` was requested, with a warning. It preserves
`h2o_prefill_score_window`: probability mode accepts 0 through 128, and a nonzero
window is allowed. MLA latent models use an explicit approximation: apply
`softmax(scale * RAW_QK_REDUCED)` to head-max decode logits, then accumulate and
evict. This is not equivalent to normalizing each head before reduction and is
not fully aligned with original H2O; enabling it emits a once-per-process warning.
MLA prefill scoring and the default score-free decode behavior are unchanged.

## Prefill Scheduling Policies

Prefill scheduling is method-specific and registry-owned. The source of truth
is `src/sparseengine/method_registry.py`; benchmark scripts and user configs
should not redefine method semantics.

| Policy | Runtime Semantics | Current Default Methods |
| --- | --- | --- |
| `all_chunked` | Every prefill request is capped by `engine_prefill_chunk_size` and normal scheduler batch limits; `long_prefill_offload_threshold` is ignored. | `vanilla`, `streamingllm`, `attention-sink`, `snapkv`, `h2o`, `quest`, `omnikv` |
| `long_bs1full_short_batch` | After supported prefix attachment, residuals at or below `long_prefill_offload_threshold` use atomic full prefill and may batch. Larger residuals are isolated and use RawKV offload chunks capped by `engine_prefill_chunk_size`. | `pyramidkv` and DeltaKV-family methods |

DeltaKV-family methods and PyramidKV keep `long_bs1full_short_batch` as the only
public policy. The threshold defaults to `65536` tokens (64K). If
`engine_prefill_chunk_size` is omitted, it defaults to that threshold; explicit
values must be positive and no larger than the threshold. `Config` raises
`max_num_batched_tokens` to fit one threshold-sized full prefill when necessary.
With full-layer KIVI enabled, DeltaKV keeps a small resident raw tail pool for
decode and a separate `max_model_len`-sized prefill staging buffer. Batched
short prefills share that staging buffer through disjoint per-request ranges;
the resident raw-tail slot count is not a prefill batch limit.
PyramidKV classifies the residual after chain-prefix attachment. DeltaKV does
not support prefix caching and rejects attached-prefix prefill before mutating
compressed or quantized row metadata.

## Prefix cache modes

`enable_prefix_caching=true` supports two deliberately separate layouts.
`prefix_cache_mode=auto` chooses radix for vanilla/OmniKV/QuEST and a linear
chain for SnapKV/H2O/PyramidKV/R-KV/SkipKV. Selecting `omnikv_prefill` changes
vanilla/OmniKV to chain mode. `radix` and `chain` can be
requested explicitly, but incompatible method/mode pairs fail fast.
GLM-4.7-Flash latent QuEST supports radix prefix caching (including CPU offload)
with decode CUDA Graph. Prefix blocks must match `quest_chunk_size`. Page selection
uses each TP rank's local query heads, so TP and single-rank results need not
be equivalent. Offload restores latent/RoPE cache and page summaries together;
the existing attention TP=1 or TP=2 offload restriction still applies.
Existing vanilla/OmniKV radix trees can be physically compacted with the
SnapKV- or KVzip-scored maintenance API described in
[Prefix cache pruning](prefix-cache-pruning.md). QuEST uses whole-page scoring
and selection, while Vanilla and OmniKV use token-granular selection.

The chain layout keeps one owner `seq_id` across turns and never branches.
Callers send the complete logical context plus the returned `chain_id`; only
the suffix after the verified processed boundary is forwarded. Method KV and
metadata remain in the cache manager. Idle chains are reclaimed by strict
LRU, while active writers are pinned. Rank 0 keeps the processed logical token
IDs in compact 32-bit storage so text continuations preserve the resident BPE
tokenization. This CPU history is bounded by
`max_model_len * max_num_seqs_in_gpu` and reclaimed with the chain.

### Chain CPU offload

Set `enable_prefix_cache_offload=true` with `enable_prefix_caching=true`,
`prefix_cache_mode="chain"` (or `auto`), and an explicit
`prefix_cache_host_size_gb`. This uses the existing offload configuration for
StreamingLLM, SnapKV, H2O, PyramidKV, R-KV, and SkipKV on their supported model
paths with TP=1 or TP=2. Contiguous explicit KV and MLA latent/RoPE storage are
supported; models with recurrent/linear layers are rejected at initialization.
Pinned host memory, device streams, and the installed SGL cache-transfer APIs
are required.

Every completed turn asynchronously copies the entire retained cache and method
state to CPU. The GPU copy remains available for a quick continuation. Under
GPU capacity pressure, an idle chain with a completed CPU snapshot can release
its GPU slots and row without losing its ID. A CPU-only continuation restores
the full snapshot before processing the suffix. A new writer first waits for
any previous copy to finish and invalidates that CPU snapshot; the next turn
completion refreshes it, including changes to old tokens and scores.

The host size limits snapshot tensor bytes **per rank**. Under host pressure,
LRU CPU copies are reclaimed; chains whose GPU copy still exists retain their
ID, while CPU-only victims become `chain_gone`. A single snapshot larger than
the budget fails explicitly at turn completion. Logical token history and
Python metadata are separate from the snapshot tensor budget; offload extends
the driver's bounded token-history allowance by one token per four configured
host bytes. CUDA Graph execution uses the same restored storage; transfers and
allocation happen outside capture/replay.

`Config` resolves `None`, empty string, and `auto` to the registry default. An
explicit policy that does not match the method default fails fast so experiments
do not silently change scheduler semantics. Treat any policy override as an
explicit ablation and document it with the benchmark result.

## Runtime Ownership

- Persistent physical cache state and prefix-coupled metadata belong in
  `src/sparseengine/engine/cache_manager/`.
- Per-step/per-layer logical state, score orchestration, cross-layer selection,
  and mutation triggers belong in a `SparseMethodRuntime` under
  `src/sparseengine/engine/sparse_methods/`.
- `src/sparseengine/engine/sparse_controller.py` is the stable method-agnostic
  facade and must not contain method-name hot-path branches.
- `src/sparseengine/layers/attention.py` should stay generic and call shared
  hooks.
- New first-class methods must register their default prefill policy in
  `src/sparseengine/method_registry.py` and cover it in
  `tests/test_prefill_schedule_policy.py`.

See the [sparse method runtime architecture](../design/sparse-method-runtime.md)
for the complete interface, ownership, prefix-cache, CUDA Graph, and extension
contract.

## Query-Aware Knobs

`quest` runtime knobs:

- `quest_chunk_size`: QuEST page/chunk size in tokens
- `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens`: QuEST derives
  its decode-time token budget once during config construction by summing these
  three values
- `quest_skip_layers`: keep the first N layers dense during decode

`quest_token_budget` is no longer a runtime input. Passing it fails fast; remove
it and configure the three common keep-token fields instead.
