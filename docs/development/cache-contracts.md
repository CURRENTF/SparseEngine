# Cache integration contracts

This document defines two protocols, not a common cache state machine. The
runtime implementation remains in the existing engine components. The test
harness lives exclusively under `tests/cache_contracts`; production must never
import it. Structural validation at `CacheManager.create` catches missing hooks;
only behavioral tests establish conformance.

## Ownership and measurement

A Chain is one exclusive mutable continuation. `ChainCacheIndex` owns its logical
identity, history boundary, ACTIVE/IDLE state and admission promise. A request ID
continues to own its method's physical rows while the chain is retained. The
manager owns those rows and any score/query/sentence sidecars. A CPU snapshot is
an additional immutable representation, not another owner of the GPU row. An
IDLE CPU-only chain must have a completed, valid snapshot and no GPU row.

A radix block is an immutable shared payload owned by `RadixPrefixIndex`. A row
borrows its storage only while holding a block reference. Private suffix and
recompute storage belongs to the request. Matching logical tokens does not
transfer ownership: duplicate materialization retains the original block and
leaves the duplicate physical allocation private. Logical block width, physical
resident-token count, pages and bytes are different units.

Use the existing allocator counters as the physical ledger and the existing
Chain admission/DecodeReservations records as future promises. These are not
interchangeable and must not be mirrored in another runtime ledger. Account an
independent layer pool separately. Standard and OmniKV share a physical slot
allocator: a repeated layer-axis requirement describes one pool and must not be
summed. QuEST reserves whole physical pages, including a partially filled page.
Staging, recurrent state, host snapshots and graph padding each retain their
existing explicit owner; none may disappear behind a scalar `min(free_slots)`.

The observed high-water mark includes allocation before compression, raw-prefill
staging, full layers, delayed eviction and restore. The final sampled output has
not yet acquired a KV entry. Async submitted-but-uncollected outputs have already
consumed their step's physical allocation. Collection is not another allocation.

## Chain functions and call boundaries

| Function and owner | Caller, input and result | Permitted mutation and failure residue |
| --- | --- | --- |
| `LLMEngine.admit_request(prompt, sampling_params)` | Request entry normalizes input and selects the configured protocol; returns admission metadata and request ID. | Validate limits, multimodal compatibility and routing before scheduler visibility. Failure must not leave a runnable request. This is not permission to modify another Chain. |
| `ChainCacheCoordinator.plan_admission(chain_id=..., seq_id=..., token_ids=...) -> ChainAdmissionPlan` | Driver asks before enqueue/apply. The plan identifies the exact owner, processed prefix, reclaim candidates and slot/row promises. | No index ownership change or physical allocation/free. Polling may retire an already submitted offload event; bounded diagnostics may change. A rejected plan leaves the Chain records unchanged. |
| `chain_capacity_deficits(*, suffix_tokens, generation_tokens, existing_slots_by_layer, outstanding_reserved_slots_by_layer, outstanding_reserved_rows, needs_resident_row) -> (required_slots, required_rows, deficits, row_deficit)` | Coordinator computes one candidate's additional physical peak in KV-layer order. `generation_tokens` includes the final output without KV. | Read-only CPU metadata; no allocation, eviction, score reset or promise acquisition. Include restore when no row is resident. Negative or truncated dimensions are not a valid answer. A method may overestimate but cannot underestimate an observed execution peak. |
| `ChainCacheCoordinator.validate_admission_plan(plan, input_token_count=..., input_prefix_digest=...)` / `ModelRunner.chain_validate_admission_plan(...)` | Each TP rank revalidates the driver's exact logical plan against its rank-local capacity before mutation. | No allocation. Compare logical facts and capacity, not physical slot IDs. Disagreement rejects the plan on every rank. A plan is not a permanent capacity certificate across intervening mutations. |
| `ModelRunner.chain_apply_admission(plan)` / `RuntimeState.chain_apply_admission(plan)` | After TP agreement, before execution preparation. Returns the applied record summary. | Runner rejects mutation of owners with unretired async tickets. Coordinator applies explicit IDLE reclaim choices and transfers logical ownership; RuntimeState frees victim payloads and restores a resumed CPU-only owner. A failed restore retains its completed host source and is not executable. Failure after rank divergence is a worker failure, not permission to continue on successful ranks. |
| `chain_has_residency(seq_id) -> bool` | Cleanup and reclaim query whether any physical row exists. | Read-only; partial preparation counts as residency. This predicate must never gate release of nonphysical reservations or sidecars. |
| `chain_physical_residency(seq_id) -> tuple[int, ...]` / `chain_physical_kv_len(layer_idx, seq_id) -> int` | Finish/snapshot/runtime obtain occupied lengths, not retained logical history length. | Read-only CPU row lengths; missing required layers are errors, not zero-length success. |
| `RuntimeState.prepare_step(seqs, is_prefill)` / manager preparation | Scheduler-selected requests only; returns execution inputs and publishes method-specific row/slot metadata. | Allocate actual step storage under existing promises. A prepared step may own rows even if execution has not started. Method runtime may choose different staging and compression layouts. A failed partial preparation is cleaned only after the runner establishes the relevant stream/event safety boundary. |
| `RuntimeState.on_forward_end(seqs, is_prefill)` | After successful method execution hooks, before final completion is exposed. | Clear a Chain prefill promise only at the final residual chunk; it is not released after an intermediate chunk. Final physical lengths come from the manager, not prompt size. |
| `on_chain_turn_finished(seq_id, processed_token_count) -> None` | RuntimeState first checks the ACTIVE owner; the runner has retired affected work. | Finalize method state only. Do not publish IDLE, choose a victim or alter history. Failure invalidates the turn and triggers request cleanup; it must not expose a half-finalized resumable Chain. |
| `RuntimeState.chain_finish(chain_id, seq_id, digest, count)` | Completion path after execution retirement. Returns retained logical/physical metadata. | Validate owner before method finalization, release the decode promise, publish the processed boundary through the coordinator, then snapshot if enabled. Failure in method finalization, boundary publication or snapshot submission invalidates the turn. A stale owner changes nothing. |
| `RuntimeState.chain_reclaim_idle(chain_id, expected_seq_id, demote)` | Scheduler/control path, only for IDLE exact owner. | Demotion requires completed host backing before GPU release; eviction drops backing and logical ownership. An ACTIVE owner is never a pressure victim. |
| `RuntimeState.chain_invalidate(chain_id, expected_seq_id=...)` | Cancel, explicit discard or failed-turn cleanup. | Check expected owner; remove the logical record/host backing, then release all request payloads and promises even when no GPU row exists. A repeated allocator release is safe; repeating a logical invalidate may still return the existing not-found/tombstone error. These are deliberately different idempotence contracts. |
| `free_seq(seq_id) -> None` | RuntimeState only after execution is safe to release. | Release all present rows and method sidecars, including partially prepared layers. Repetition is a no-op. Do not mutate the Chain index or another owner. Metadata/allocator corruption or an unquiesced device failure is fatal, not a recoverable capacity rejection. |

`allocate_chain_restore(seq_id, lengths_by_layer) -> dict[layer, slot_view]` is a
manager-owned transaction, not an offload allocator callback. It rejects an
existing row, wrong layer count, negative/noninteger length, missing row or
insufficient physical capacity before any layer is claimed. Unexpected failure
while allocating later layers cleans every earlier layer with the method's
ordinary `free_seq`. Return borrowed occupied slot views; the offload controller
freezes the IDs before asynchronous use. This boundary leaves no partially
restored owner on an ordinary allocation failure. No KV copying or host-account
mutation occurs here.

`chain_token_slots(layer, seq_id)` returns the occupied slot view.
`chain_storage_tensors(layer)` returns the storage pair for that method's payload,
including distinct latent/rope widths. `snapshot_chain_method_state(seq_id)`
returns auxiliary tensors and metadata without changing the live method;
`restore_chain_method_state(seq_id, state)` rebinds that state to the newly
allocated rows. They do not assume all methods have the same auxiliary state.

`ChainOffloadController.save(seq_id, state)` freezes source IDs and method state,
charges actual host tensors, and retains source keepalive until D2H completes.
`wait(seq_id)` completes that snapshot; `invalidate(seq_id)` waits before making
an old snapshot unusable for a rewritten live row; `drop(seq_id)` waits before
freeing host backing. `restore(seq_id)` requires a valid snapshot, delegates
physical allocation to `allocate_chain_restore`, transfers the whole payload and
method state, and establishes completion before reuse. On transfer failure the
controller must quiesce its transfer stream before rolling back device rows. A
failed synchronization is a fatal device error; restarting service on the same
allocator is not a supported recovery.

## Radix functions and call boundaries

| Function and owner | Caller, input and result | Permitted mutation and failure residue |
| --- | --- | --- |
| `refresh_prefix_cache_hit(seq) -> None` | Entry/scheduler refreshes a new prompt's speculative hit metadata. | Lookup/memo/access statistics only; no row, reference, promotion or prefilled-token advance. Retain at least the work needed to produce logits. A lookup result can become stale before attachment. |
| `_attach_prefix_cache_if_needed(seq) -> None` | Manager prefill preparation, after capacity scheduling and before attention. | Revalidate chain, block residency, payload geometry and row capacity; acquire refs; promote missing payloads; then publish row aliases. Repetition after successful attach is a no-op. Failure removes this request's new refs/aliases/row but never frees a successfully submitted promotion's destination. That destination remains index-owned and operation-tracked. |
| `_allocate(seq_id, size) -> slots` / paged or layer-local allocator | Manager preparation under the scheduler's promise. | Reject invalid size, row overflow and capacity before committing ownership. A synchronous metadata-write failure restores the old row length/pointer and drops only a newly claimed row. Eligible IDLE index blocks evicted while looking for capacity need not be resurrected. Reserved prefill slots remain owned by their enclosing reservation on failure. |
| `_record_prefix_materialization(seq, token_ids, slots) -> None` | Executed-token recording; async submission uses a deferred record list. | Validate token/slot lengths, freeze IDs and form complete/partial candidates. Candidates remain request-private until publication. Never publish KV for a discarded/unretired async result. |
| `publish_pending_prefix_blocks(seqs) -> None` | Successful forward completion, or async collection after its completion event. | Transfer complete new blocks to the index and hold request refs. Preserve an existing block's payload on duplicate logical ID. Earlier committed blocks may survive a later insertion failure; cleanup frees only uncommitted private storage and drops request refs. The marking hook must make the row's shared ranges agree with every inserted payload. |
| `_release_prefix_request(seq_id) -> list[PrefixCacheBlock]` | Standard/QuEST request cleanup, after private-storage reclamation. | Release borrowed/materialized refs and discard pending/runtime/lookup metadata exactly once. Does not free physical shared storage or submit transfers. The returned newly unreferenced candidates belong to the index. |
| `free_seq(seq_id) -> None` | Runtime terminal/preemption cleanup after execution safety. | Free private complement ranges/pages, release refs, zero/return row, then schedule optional write-through. A host-pressure/submission failure after detach must leave the request fully detached. Other readers and index-owned storage survive. |
| `reset_prefix_cache() -> None` | Explicit reset/warmup boundary after drain. | Reject live rows/references. Retire transfers before returning host/device index storage. It is not a shortcut for request cancellation. |

For mixed KV/recurrent models, `PrefixCacheCoordinator`, not the manager-local
radix attach routine, owns the reference transaction. Its KV participant supplies
`build_prefix_kv_payload(seq, block_start, block_end) -> payload`,
`validate_prefix_kv_attach(seq) -> row_preexisted`,
`attach_prefix_kv_payloads(seq, payloads) -> None`, and
`rollback_prefix_kv_attach(seq, payloads, row_preexisted=...) -> None`. The rollback
preserves a preexisting empty row and removes only the attachments made by that
transaction. `mark_materialized_prefix_kv_payload` transfers the request's KV
ownership to a block; `free_prefix_kv_payload` frees index-owned KV, never a live
reader's private row. The coordinator separately acquires/releases recurrent
state and owns the joint block reference. It must not make both KV and recurrent
participants own the same reference release.

The KV attach participant validates the entire batch before claiming a new row,
then copies the packed aliases and publishes row metadata. A synchronous write
failure clears the new aliases and returns only a newly claimed row. QuEST
validates packed token/page correspondence once for the batch. The single-payload
case uses the same batch transaction with a one-element list.

Mixed offload keeps its existing `allocate_prefix_kv_payloads_device`,
`free_prefix_kv_payload_device`, `prefix_kv_payload_nbytes` and recurrent transport
participants. Their units are physical payloads/bytes; they are not a second
Chain interface. Unsupported recurrent Chain snapshots remain rejected.

Radix H2D submission returning `None` can be a successful zero-resident-token
promotion, not a failure. Attachment records successful submission separately
from the optional operation handle. A failed attach after H2D submission leaves
the destination tracked by the index until completion; returning it to the free
stack would allow the next request to race the DMA. D2H source storage similarly
cannot be reused before its transfer finishes. Layer readiness waits remain
outside graph capture and do not imply whole-operation retirement.

## Scheduling and execution are part of the contract

`prompt_admission_budgets(waiting_seqs, chunk_size)` and
`prompt_admission_costs(seq)` describe named physical constraints.
`on_prompt_admitted(seq, costs)` records method-owned admission state only after
the scheduler selects that request. A free row is a separate constraint from
free tokens. A lookup hit does not turn host-only blocks into free GPU capacity.

`DecodeReservations.acquire_many(seqs, allow_short=..., prefill_reserve=...,
budgets=...)` returns the first request that cannot acquire a window, or `None`.
Earlier successful acquisitions intentionally remain owned by their requests;
this is not an all-or-nothing batch API. `release(seq_id)` is idempotent and never
frees physical storage. Outstanding costs are derived from live physical
residency and `completion + pending` progress, not a second aggregate counter.
Window size must be a positive integer. Method costs must be monotone upper
bounds over a horizon and use only budgets actually exposed by that method.

`AsyncExecution.submit` owns each ticket's inputs, deferred prefix records and
host/device keepalive. `collect(ticket, discarded=...)` waits for that result,
excludes discarded owners from publication, and retires/recycles its buffers.
`assert_releasable(seq_ids)` permits unrelated tickets to remain in flight, but
rejects any target with an unretired result even if its event is complete. It does
not synchronize or poll GPU events. Runner free/finish/Chain mutation boundaries
call it; scheduler cancellation/preemption must retire affected tickets first.

Graph preparation reserves only real request rows/pages and publishes stable
buffers. Padded lanes must not consume request allocations or publish prefix
records. Captured execution cannot perform lookup, host allocation, eviction,
H2D/D2H submission or ownership release. The runner rejects release while stream
capture is active. Graph host-input buffers remain ticket-owned until retirement;
changing the next batch cannot overwrite a previous ticket's pinned input.

TP preserves the same logical owner, processed boundary and plan on all ranks;
rank-local physical IDs may differ. Validate before applying. Capacity rejection
on one rank cannot be accepted on another. Async feedback and retirement preserve
rank order. Do not introduce a per-token collective, global GPU synchronization
or a conformance callback into normal decode. A fatal launch/collective/device
failure requires worker termination/quarantine rather than pretending local
Python rollback can undo already submitted work on other ranks.

Cancellation, preemption and execution failure use the existing driver/runner
control boundaries. `RuntimeState._free_seq_payload` releases the decode promise,
manager storage, coordinator refs and recurrent state, and always removes runtime
residency. A recoverable post-detach offload error cannot strand the remaining
owners; errors still propagate. Method-specific reset/recompute policy remains
inside the existing scheduler/runtime and must not be approximated by setting a
cache counter to its initial value.

## Executable acceptance and migration

`tests/cache_contracts/cases.py` explicitly binds all six regular Chain methods,
plus vanilla and OmniKV with `omnikv_prefill`, and the three radix methods. The
coverage test compares those choices to the production mode registry, so adding
an advertised mode without an actual fixture is a failure, not a silent skip.
Storage-family observers are test-only and closed: independent per-layer slots,
shared slots, and QuEST pages. A genuinely new layout adds an independently
reviewed observer and mutation tests; it does not register a production callback.

The oracle enumerates the actual allocated pool and checks the disjoint union of
free IDs and unique physical owners. It separately checks row ownership, exact
shared references, page/token geometry, host storage bytes and Chain logical
ownership. It never calls a production capacity-summary hook to discover used
storage. Compensating counter errors, missing slots and free/live aliases are
intentional negative tests. Cost monotonicity is supplemented by real allocator
execution; it is not itself a proof of physical sufficiency.

Run the zero-skip CPU gate with:

```bash
python scripts/validation/validate_cache_contracts.py --tier cpu
python scripts/validation/validate_cache_contracts.py --tier regression
python scripts/validation/validate_cache_contracts.py --tier cuda
```

The CPU gate exercises all advertised configurations with actual manager classes,
exclusive/shared ownership, allocation boundaries, partial restore failure,
duplicate materialization, publication/attach failure, repeated cleanup, host
failure after detach and async release interleavings. Offload CPU tests replace
transport only, not allocators or method state. The CUDA tier exercises real
transport and the existing async CUDA suite. Missing hardware, missing target
files, zero collected tests and skipped advertised contract cases are not passes.
Reports retain failures/skips and the exact command.

Rollout is deliberately staged without parallel production ledgers. First merge
the structural contracts and complete configuration matrix; unsupported/missing
hooks must fail at construction. Then migrate physical restore/cleanup and run
the independent allocator/ownership gate plus existing capacity regressions.
Next validate method selection/compression and full request entry, cancellation,
preemption, mixed recurrent models, graph/non-graph and TP on the project's real
model/hardware integration workloads. CPU fixtures cannot certify those device
execution paths. Finally compare CPU preparation and end-to-end serving metrics
against the pinned baseline on the same machine, batch sizes and workloads.

Delete old paths only after their replacement gate passes: direct offload access
to row deques is replaced by `allocate_chain_restore`; duplicated Standard/QuEST
reference cleanup is replaced by `_release_prefix_request`; missing-row release
errors are replaced by idempotent partial cleanup. Do not delete method-specific
compression algorithms, numerical tests, async/graph tests or capacity formulas
merely because a shared protocol exists. Remove redundant tests only after the
new gate covers the same fault with an independent observation. Do not declare a
full-device or performance migration complete from CPU fixture results alone.

## CPU cost constraints

Factory validation runs once and installs no wrappers. Pool scans, tensor reads,
sets of physical IDs and mutation checking are exclusively test code. New
all-layer preflight is at Chain restore, not every decode token. Allocation
transaction guards use CPU metadata already touched by allocation. Terminal
release scans existing in-flight tickets only on control paths and never
synchronizes the device. Decode reservation costing, method selection, graph
metadata kernels, normal forward hooks and TP collectives retain their existing
hot call structure. These design constraints do not substitute for a benchmark
on the serving workload; performance regression is a release gate, not a claim
inferred from the absence of a new framework.

Prefix attachment deduplicates in-flight transfers by object identity, in first
encounter order. For B matched blocks and U distinct transfers this takes O(B + U)
expected CPU work per request, including merging into the current step. Each
transfer is waited once per layer even when several blocks or requests share it.
QuEST validates resident page geometry in one packed comparison before mutation
and reuses the packed tensors for row publication. Device result reads and row
copy submissions do not grow with the number of matched pages.

Use the same driver and fixtures in both worktrees for CPU comparisons:

```bash
SPARSEENGINE_PLATFORM=cpu CUDA_VISIBLE_DEVICES='' \
  python scripts/validation/benchmark_cache_contracts.py \
  --attach-blocks 16 128 1024 --attach-iterations 20 \
  --iterations 2000 --samples 7 --output "$RUN_DIR/baseline.json"
# Run the candidate with identical arguments and CPU affinity, adding:
# --baseline "$RUN_DIR/baseline.json" --output "$RUN_DIR/candidate.json"
```

The optional attach cases include one shared transfer and one transfer per block.
Timing includes real CPU attachment, row publication and request cleanup, with
transport stubbed and ownership checks outside the measured region. JSON records
raw samples, command and environment. These cases do not measure
GPU synchronization, DMA or serving latency.

Replay references are keyed by stable block ID for both radix managers and mixed
coordinators. Each request acquires a reference once, even across repeated chunks;
releasing it visits only that request's references. Attached root chains use their
logical block index for membership. Mixed recurrent-byte totals are cached against
the index's insert/remove epochs and reset identity.

Offload polling visits pending transfers, not retained host snapshots. Each
direction submits to one ordered stream, so a pending head stops event polling.
Prefix transfer queues use constant-time head removal; chain transfers also allow
constant-time removal by targeted wait/drop. Empty prefix blocks participate in
the same root-to-leaf residency transaction as their nonempty relatives.

Chain admission aggregates promises over ACTIVE owners only. The index updates
that view on creation, resume, finish, invalidation and reset; a failed restore
removes the abandoned ACTIVE entry. Victim ordering is computed only when a
physical or host capacity deficit requires reclamation. It preserves the existing
LRU/ACTIVE-snapshot priority and deterministic tie breaking.

Under capacity pressure, radix publication reserves space once for all distinct
new blocks in the request's pending batch, after protecting existing parents and
duplicates. Device demotion counts resident children once and updates counts as
leaves leave the device. Recurrent-byte pressure uses actual payload sizes in one
weighted eviction pass, including zero-size leaves needed to reach their parents.
Chain victim selection heapifies candidates once and orders only the consumed
prefix; classifying backed victims preserves each partition's original order.

Async prefix recording freezes slot IDs once on the producer stream; retirement
consumes the owned snapshot without another copy. QuEST validates completed pages
and returned index pages in batches, retaining CPU page IDs for ownership while
returning request-private page IDs directly on the device. Hit-capacity snapshots
include physical slot weights and index identity, so repeated admission queries
reuse them until the physical-capacity epoch changes. Chain routing snapshots are
immutable and rebuilt only on logical lifecycle changes, rather than each decode
step's dispatcher refresh.

Standard/OmniKV return an evicted or demoted batch of device slot IDs with one
allocator copy, after validating all payload lengths and the aggregate capacity.
Empty compacted blocks release their metadata without returning physical slots.
Offload host-index construction preserves `None` as a full-block offset range:
full batches expand block IDs on the device; mixed compacted batches pack native
CPU arrays and transfer once, preserving block order and explicit retained offsets.

Lookup memos stay available while their request object is live; weak ownership
removes them when cancelled waiting requests are collected, including requests
that never acquired a KV row. TP batch lookups retain the last completed batch's
ID entries until all fragments of the next lookup batch finish, then retain only
the entries queried in that batch. This avoids LRU thrashing when queues exceed
the individual-lookup cache capacity or when requests are cancelled. RPC splits
carry first/last markers; singleton fallbacks preserve those markers. Failed
fragments discard the partial batch memo, retaining the previous completed pass.
Individual lookups outside a batch still use a bounded ID cache. Explicit request
release removes all memo references. Token IDs and
path membership are still checked before a memoized hit is reused. Stable radix
hashes pack each block in one native operation, preserving signed-int64
little-endian wire bytes. ACTIVE chains whose prefill promises have been cleared
contribute zero to admission reservations without reading their physical layers.
