# Independent KV-head storage and selection

`head_slots` is a physical layout shared by native KVzip, a chain-boundary
SnapKV variant, and AdaKV. It is independent of the attention implementation.
`kv_cache_layout="auto"` resolves to `head_slots` for AdaKV and `token` otherwise;
existing methods retain their token-layout behavior.

## Runtime contract

| Method | Selection and trigger | Budget | Prefix lifecycle |
| --- | --- | --- | --- |
| KVzip | Reconstruction-query maximum; global threshold across layers/KV heads/context positions, before question prefill | `kvzip_token_budget`; see [KVzip runtime](kvzip-runtime.md) | Independent requests; no prefix cache |
| SnapKV with head slots | Last observed query window; query mean for probability mode, query maximum for logits; maximum within each GQA group; selection at chain turn completion | Optional `[KV layer][KV head]` total budgets, including sink/recent; `None` disables compression | Exclusive GPU-resident chains |
| AdaKV | Same query-window probability scorer and GQA maximum; final-prefill and chain-turn compaction | Per-layer total `KV_heads * (sink + decode_keep + recent)` with per-head alpha safeguards | Independent requests or exclusive GPU-resident chains |

AdaKV follows NVIDIA/KVPress AdaKVPress revision
`71640b4f9061054a7630c5049bb9ee659a01523c`'s safeguard/global-selection objective,
extended to ragged candidates and protected sink/recent entries. Each head
retains at least `min(length, max(1, floor(alpha * average_budget)))` entries or
its protected entries, whichever is larger. Stable ties use head then candidate
order. This is not a claim of exact FFY0/AdaKV algorithm or benchmark parity.
The head-slot SnapKV variant also has a different trigger from ordinary
prompt-compacting SnapKV; selecting the layout is explicit.

The allocator is the single CPU ownership ledger. The cache manager owns
physical slots, original positions, per-head lengths, observed query rings,
compaction, chain residency and graph metadata. Method runtimes own selection
and final-prefill triggers. All layers' selections are validated before any
mapping is published or discarded slot reused. Compaction changes mappings and
returns discarded slots without copying retained historical K/V.

This integration carries the head-related work from `experiment/head-nonuniform`
at base `6541591ea0efd1df7ccca3e11708bb5d46e78371`, including its uncommitted
head implementation. It adapts namespace, observation-window naming, KV-layer
ownership and reservation interfaces to the current runtime. Unrelated
benchmark/paper/skill changes are excluded.

## Storage, scheduling and compatibility

Each layer stores K/V in `[physical_head_slots, head_dim]`. Index and original
position tables have shape `[resident_rows, KV_heads, max_model_len]`; lengths
are `[resident_rows, KV_heads]`. A slot has one exclusive head/token owner.
Logical append tokens and physical head slots are different units: each new
token costs `KV_heads` slots in every layer. Chain promises and decode windows
share this unit and are counted once. Admission budgets reserve the complete
uncompressed prefill peak. Failed plans must leave ownership unchanged.

Supported model contracts are homogeneous explicit BF16 causal RoPE attention,
Llama/Qwen2/Qwen3, TP=EP=DP=1. Sliding windows, position/logit biases, quantized
models/KV, shared or heterogeneous KV, recurrent layers, radix cache, head-cache
offload, and decode-time eviction are rejected. Native KVzip prefix caching
remains unsupported; importing the chain implementation does not implement
shared radix pruning.

Query-ring allocations and writes belong only to enabled SnapKV/AdaKV
compression. KVzip keeps its existing reconstruction scorer and does not
allocate or maintain this query ring. Token-layout methods allocate none of the
head-slot storage or metadata.

## Compute, graph and cost review

Attention reuses SGL FA3 on SM90 and FlashInfer FA2/tensor-core decode on SM120
through a virtual MQA adapter: `(request, KV head)` is a virtual sequence with
its original GQA group. The adapter packs Q and restores output order; it reads
history through independent indices rather than gathering historical K/V.
The vLLM/Tangram ragged-layout reference is revision
`6fa551fc8f6edcc118a2a39b3554ee1520e91edd`; its serving runtime and static page-group
allocation are not imported. The SGL/FlashInfer adapters retain their existing
prepared-provider and page-size-one contracts.

The existing tiled `prefill_score_fwd` supports token-reduced, query-head and
KV-head outputs. Head slots extend addressing/lengths behind constexpr flags;
KVzip reconstruction retains separate constexpr scoring semantics. The source
query-ring write kernel is reused. No new dependencies or attention kernels
are introduced.

For L layers, R rows, H KV heads, Q query heads, head dimension D, context T,
window W and append batch C: metadata costs `L*R*H*(8*T+4)` bytes, BF16 observed
queries cost `2*L*R*W*Q*D`, append metadata costs `8*L*C*H`. Per-event scoring is
`O(L*Q*W*T*D)` with tiled LSE and `4*H*T` score output; fixed selection sorts
per head, while AdaKV additionally sorts the layer's H*T population. Selection
indices for all layers stay live until validation. The existing conservative
auxiliary envelope includes these buffers and AdaKV's `64*H*T` sorting/mask
scratch, and is deducted before KV-pool allocation. CPU conversion of selected
indices occurs only at compaction boundaries. Decode appends and updates
metadata; no query-window scoring/sorting occurs per generated token.

Memory accounting classifies storage-owned tensors as KV, counts aliases once,
and derives live KV bytes from the sum of retained head lengths across layers.
The dense baseline counts complete token slots (`physical_head_slots // H`).
Compaction releases pool capacity for reuse; it does not shrink the allocated
pool, so allocated-byte savings and retained-byte reduction are distinct.

SGL decode uses fixed-address, batch-only graph inputs. Query-ring writes
consume padded-row masks and dynamic positions; pruning and CPU allocation
stay outside capture. FlashInfer head decode owns one public graph wrapper per
captured batch. Startup plans FA2 split-KV for the full context capacity once;
captured device packing updates each layer's indices, indptr and last-page
lengths. Shorter heads mask empty splits, including padded requests. Plan,
wrapper and workspace identities remain fixed; replay does not replan or
synchronize each layer. Keep upstream graph splitting enabled: the no-split
FA2 path omits the valid-CTA mask for its padded grid. Workspace initialization
occurs only at startup. The capacity split envelope can add empty-split/merge
work for short heads, so compare matched request workloads before making an
efficiency claim. Graph eligibility is a provider contract; actual replay,
numerics and performance require independent validation artifacts.

## Validation entrypoints

```bash
python -m pytest -q tests/test_head_slot_allocator.py tests/test_adakv_head_slots.py
SPARSEENGINE_TEST_CUDA=1 python -m pytest -q \
  tests/test_head_slot_attention_cuda.py tests/test_prefill_score_per_head.py \
  tests/test_kvzip_method.py
python scripts/debug/validate_head_slots.py --model "$MODEL" --output "$OUT/fixed" --eager
python scripts/debug/validate_head_slots.py --model "$MODEL" --output "$OUT/adakv" \
  --budget-policy adakv --head-budget 28 --eager
python scripts/debug/validate_head_slots.py --model "$MODEL" --output "$OUT/adakv-oracle" \
  --budget-policy adakv --head-budget 28 --eager --check-materialized-attention
```

The model validator checks two interleaved chains over three turns, real scorer
and selection oracles, physical ownership and final release. Use its
`--materialized-reference --selection-reference "$OUT/fixed"` modes to compare
independent CPU-ledger addressing with ordinary BF16 attention. Omit `--eager`
only on a graph-compatible provider and require actual replay evidence.
`--check-materialized-attention` checks production and independently materialized
attention against an FP32 position-mask oracle on identical Q/K/V at every real
model call. The BF16 reference keeps the virtual batch and tensor-core arithmetic
but independently packs queries, gathers CPU-ledger history and builds dense
addresses. Each materialized prefill call receives a fresh planning scope for
its ordinary page table, allowing the same upstream arithmetic as production.
The saved summary identifies both bindings. Separate BF16 networks can amplify
reduction-rounding differences;
the full-logit comparison retains its stricter tolerance and must report any
failure separately from per-operation oracle checks and greedy-token agreement.
Correctness smoke and source-branch measurements do not establish quality or
performance parity. Matched performance comparisons use the canonical
[efficiency runbook](../en/benchmarking/efficiency.md).
