# KVzip runtime contract

Native `sparse_method="kvzip"` follows SNU KVzip revision
`5d8472975d286e3da6b52178af39d18904330c0d` (`model/wrapper.py`,
`model/template.py`, `attention/score.py`). Scores and thresholding retain the
original `[layer, KV_head, context_token]` domain. Storage is explicit:
`kv_cache_layout="token"` shares each layer's head-selection union;
`kv_cache_layout="head_slots"` retains independent per-head KV pairs.
The legacy prefix-maintenance policy `kvzip_global` has separate semantics.

## Algorithm and request boundary

Requests declare `context_length` (exclusive end of system prefix plus context)
and `context_prefix_length` (exclusive end of protected system prefix).
The prompt requires a nonempty question suffix. The scheduler stops at the
context boundary; compression precedes question prefill and the first sampled
token. Missing boundaries fail for compressible requests. Short contexts remain
dense. Boundaries survive TP IPC and recomputation.

Teacher-force the original instructions and model assistant suffix. The first
instruction is `\n\nRepeat the previous context exactly.`; later instructions
are `\n\nRepeat the part of the previous context exactly, starting with `,
followed by the preceding eight tokens. Default source chunks contain 2000
tokens and exclude the protected prefix. Replay model attention sees the full
original cache. The independent scorer normalizes over the protected prefix,
the current original chunk and causally available replay KV. Only probabilities
for the current original chunk contribute scores. Reduce by maximum over replay
queries and the query heads belonging to each KV head. Keep KV heads separate.
FP32 LSE and accumulation differ from SNU's model-dtype probability arithmetic;
algorithmic masks and reductions align, without a bitwise-parity claim.

`kvzip_token_budget` is the average unprotected context-token allowance across
layers and KV heads. Sort the global score population and use rank
`layers * global_KV_heads * budget - 1`, retaining scores strictly greater than
the threshold. TP gathers head shards to the leader and broadcasts the scalar
threshold. Ties can reduce retained KV pairs. Restore chronological order and
protect the complete system prefix, question and generated KV. No additional
sink/recent protection or decode eviction is applied.

For token storage, take the union of original head selections within each
layer, then TP MAX the position mask. All heads see this union, so physical
context-token retention can exceed the nominal budget and original per-head
attention is not preserved. Collapsing scores before global thresholding would
change the algorithm and is not used. For head-slot storage, keep each head's
selected addresses and length; discarded KV pairs return to that layer's pool.
The prefix is protected in every head, including heads with no selected context.

## Owners, compatibility and cost/reuse review

`KVzipRuntime` owns replay orchestration and global selection.
`KVzipReconstructionMixin` shares source history, replay validation and cleanup.
`KVzipCacheManager` uses SnapKV token pools and layer compaction.
`KVzipHeadSlotCacheManager` uses independent per-layer head-slot pools and a CPU
ownership ledger, with device-resident typed address/length views. Sequence owns
generic context boundaries; Scheduler caps chunks generically; the auxiliary
hook runs unsampled forwards. Attention and Scheduler remain method agnostic.

Token storage supports explicit independent Qwen2/Qwen3/Llama KV layers and
attention TP. Head-slot storage is currently restricted to TP=EP=DP=1,
unquantized BF16 homogeneous explicit KV. H100/SM90 uses SGL FA3 with
head dimension 128 or 256. RTX Pro 6000/SM120 uses FlashInfer FA2 prefill
(head dimension 64, 128 or 256) and its public tensor-core decode wrapper. Unsupported providers fail resolution explicitly.
Prefix reuse/offload, recurrent/MLA/shared-KV/MoE, sparse-prefill overrides and
sliding/position-biased attention are rejected. Head-slot graph metadata has
stable writes and lengths plus an empty padding row. SGL FA3 and FlashInfer
head-slot decode support batch-only CUDA Graphs. FlashInfer plans the full
context-capacity split envelope once, then packs independent layer/head page
metadata on device inside capture. No per-layer host planner or replay-time
recapture is required. The plan and workspace remain fixed across head-length
changes; shorter contexts mask empty splits. See the
[head-slot design](head-nonuniform-design.md) for cost and lifecycle details.

Let `L` include prefix, `C` be source chunk, `R` replay length, `A` layers,
`Hq` query heads, `Hkv` KV heads and `D` head dimension. Context prefill and
`ceil((L-prefix)/C)` replay forwards remain. Replay attention costs
`O(A Hq R L D)`; the tiled scorer costs `O(A Hq R (L+R) D)` without materializing
`[Hq,R,L]`. Scores cost `4 A Hkv L` bytes; global sort costs
`O(A Hkv L log(A Hkv L))`. Compaction changes address tables without copying KV.
Replay scratch reserves at most `engine_prefill_chunk_size` logical tokens per
layer, multiplied by `Hkv` for head-slot pools. Business admission and decode
cannot consume that reserve; replay may consume its own reserve and releases
it even when a forward fails. Metadata, packed queries/outputs and scorer/sort
workspace are deducted before allocating physical KV. Startup profiling counts
head-slot payload and metadata separately.

Reuse `prefill_score_fwd`/`PrefillScoreWorkspace` for masked probability maxima.
For head-slot scoring, pack Q into virtual MQA requests while keeping the same
original-head identity. Reuse SGL FA3 prefill/decode through the same virtual MQA
view; only Q and address/length metadata change layout. The SM120 adapter
reuses FlashInfer paged wrappers with a new eager plan for each layer or a fixed
capacity graph plan. Its tensor-core
decode path handles GQA groups unsupported by the CUDA-core wrapper. Upstream ordinary LSE
includes all original keys and cannot replace the reconstruction denominator.
The imported head-slot publish/store/query-reorder kernels adapt addressing;
no new dependencies or attention kernel clones are introduced. No efficiency
claim is made without a matched benchmark.

## Validation boundaries

Independent tests cover global threshold/ties, shared unions and true head
selections, real Gloo/NCCL TP thresholding, replay rollback, startup capacity and
resource release. CUDA score oracles cover FP16/BF16, masked tails, multiple
query/head tiles and virtual MQA identity. Head-slot tests cover strided KV
writes, ragged original-position attention, graph padding and cache ownership.
Attention tests select SM90 SGL FA3 or SM120 FlashInfer and skip other devices.
They change head addresses/lengths between layers within one provider scope;
graph cases additionally check dynamic lengths, padding, capacity boundaries,
workspace reuse and the absence of runtime replanning. Kernel/scorer tests
do not establish real-model head-slot quality or engine graph parity. Private
run artifacts record actual executions and results; bounded RULER smoke is not
a full quality benchmark, and no performance claim follows from it.
