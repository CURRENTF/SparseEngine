# Batch-Only Decode CUDA Graph Review

Read this reference when a review touches an operator reachable from captured
decode, decode graph input preparation, provider graph state, or sparse
mixed-length decode.

## Scope and Vocabulary

Batch-only is the only maintained decode CUDA Graph shape policy. Do not add or
preserve bucketed-only graph implementations, context-bucket routing, parallel
provider families, or configuration surfaces merely to keep a second graph
architecture alive. Eager may remain as an independent correctness or
unsupported-graph path; it must not leak context-dependent dispatch into
captured decode.

Use these terms consistently:

- **batch-only graph**: graph identity depends on batch capacity but not actual
  per-step context lengths;
- **strict batch-only**: one forward graph per batch and sampling topology for
  a fixed model/method, with `dense` or `unified` as its sole decode path;
- **context capacity**: a capture-time storage and launch upper bound, not a
  replay-time graph bucket;
- **static launch plan**: capture-time tile, warp, stage, split envelope, grid
  envelope, compiled variant, and workspace capacity;
- **replay-before metadata**: dynamic state prepared outside the captured graph
  before replay, also called graph-out preparation;
- **graph-in preparation**: fixed device work captured before operator forward;
- **stable graph state**: typed inputs, provider state, workspaces, wrappers,
  and keepalive owners whose addresses and capacities remain fixed.

Reading `context_lens` or exposing `plan()` does not by itself violate
batch-only. The violation is allowing actual context to change graph identity,
captured topology, static launch plan, workspace shape, tensor addresses, or
provider binding.

## Operator and Provider Adaptation

- Define graph identity from batch capacity and sampling topology for a fixed
  model/method and capture-time tensor/layout contract. Use `dense` for dense
  decode and `unified` for sparse decode. Actual `context_lens` must not enter
  graph keys or cause runtime capture.
- Resolve model/hardware tuning tables and compile-time choices before capture.
  A table selected for a fixed model architecture and hardware combination is
  valid static configuration. Tile, warp, stage, compiled variant, grid
  envelope, and workspace shape are not replay metadata.
- Flag replay-time host thresholds that switch kernel chains, launch variants,
  split envelopes, or workspaces. Replace them with a fixed envelope plus
  device-side effective scheduling, bind another batch-only provider, or reject
  the unsupported contract during resolution/preparation.
- Dynamic lengths may drive device masking, effective split/range metadata, or
  an explicit replay-before provider plan when those updates write only stable
  graph state and leave the captured launch contract unchanged.
- Do not split decode batches or capture separate short/long graphs. Represent
  per-row dense/sparse selection with dynamic lengths and method-owned metadata
  inside the unified path. Seal the startup plan; length changes must not JIT,
  reselect a provider or variant, grow workspace, or recapture. This does not
  change method-specific prefill scheduling or cache transformation policies.
- Require `supports(spec, caps)` and preparation to validate dtype, shape,
  layout, capacity, padding, workspace, and batch-only compatibility before
  forward. Do not treat a few fixed-shape experiments as production support.
- When a standard upstream provider already exposes a graph-stable lifecycle,
  adapt that lifecycle instead of cloning its kernel. Use a repository-owned
  fixed-grid provider for missing SparseEngine semantics, portable fallback, or
  an algorithmic improvement supported by representative validation, or an exact
  measured tuning override. Choose defaults deliberately through the portfolio.
- Fail unsupported capacity or layout before cache mutation. Once bound, do not
  switch provider, allocate a larger workspace, or fall back after execution
  begins.

## Unified Inputs and Participant Lifecycle

The unified input mechanism standardizes public replay inputs and update order;
it does not combine every tensor into one allocation or expose provider and
sparse-algorithm internals to the graph runner.

### Common input contract

Keep shared replay inputs in typed, fixed-address runner-owned state. At minimum
distinguish token ids, positions, context lengths, request indices, KV
write-slot mappings, and valid-row state. Every registered slot declares:

- shape, dtype, and device;
- batch axis and capacity;
- padding policy;
- semantic/value source and copy policy;
- stable-address requirement.

Prefer explicit `DecodeGraphInputs`-style fields. Flag an indefinitely growing
`dict[str, Tensor]`, an untyped memory blob, or a positional runner API carrying
method- and provider-private tensors.

### Ownership

For every field distinguish storage owner, semantic owner, and per-step value
producer:

- graph runner: common decode input storage, padding, capture/replay, and graph
  identity;
- cache manager: physical KV storage, page/slot metadata, and physical cache
  views;
- `SparseController`: logical sparse selection, cross-layer observation, and
  attention-score coordination;
- provider: static kernel plan, schedule buffers, private graph state,
  workspace, external wrapper, and physical weight/layout;
- model/attention layer: stable operator semantics only.

Do not move provider workspace into the common registry or physical cache
metadata into `SparseController`. The runner coordinates lifecycle and copy
order without taking ownership of private algorithms or layouts.

### Participant lifecycle

Use a typed lifecycle equivalent to:

```text
init_decode_graph_state(contract, inputs)
prepare_out_graph(step, state)
prepare_in_graph(state)
graph_keepalive_tensors(state)
```

- initialization allocates stable private buffers/workspaces, resolves the
  static plan, initializes wrappers/JIT once, and records capacity;
- graph-out preparation updates dynamic host metadata or executes a documented
  provider plan, writing only stable state;
- graph-in preparation contains fixed device work captured before forward;
- keepalive ownership prevents captured tensors, workspaces, wrappers, or
  outputs from being released or replaced.

Coordinate provider preparation once before each model replay, outside
per-layer attention forward. Model and attention code consume prepared state and
must not contain sparse-method branches, provider names, external-wrapper
access, or graph lifecycle calls.

### Padding

Pad real batches to their capture bucket with an explicit active-row contract.
Padding rows use safe token, position, slot, page, and score metadata. Prove
they cannot access or mutate a live request's KV cache, sparse score, or
controller state. Do not rely on an incidental sentinel that a kernel still
dereferences before masking.

## External Graph Wrappers

For FlashInfer paged decode and comparable external providers with a public
CUDA Graph lifecycle:

- Use the upstream graph-enabled wrapper instead of an ordinary eager wrapper,
  raw internal kernel, or repository reimplementation of its planner. Bind one
  wrapper to each captured batch/topology state that needs distinct storage.
- Provider state owns fixed-capacity page indptr, page indices, last-page
  lengths, integer/float workspaces, output owners, and the wrapper. The runner
  invokes the participant lifecycle but never reads wrapper-private fields or
  constructs provider-specific page metadata.
- Run context-dependent `plan()` or the documented fast-plan path during
  replay-before preparation. Planning may change contents, not wrapper/workspace
  identity, input/output addresses, launch contract, or captured `run()`
  topology.
- Captured forward calls only the already-bound wrapper `run()`. Flag planning
  in forward, wrapper recreation, real-length-driven `masked_select`, `cat`, or
  allocation, workspace replacement, and runtime backend switching.
- Reuse persistent host/GPU staging. If the public API requires D2H or host
  planning, keep the synchronization boundary explicit and report its p50/p95
  cost separately; it must not alter captured addresses.
- If the minimum supported upstream version has no public wrapper contract that
  satisfies these invariants, reject the provider for batch-only during binding.
  Do not reach through private APIs or silently fall back after replay starts.
- Validate constructor, plan/fast-plan, and run with a real installation at the
  declared minimum version. Mocks do not prove lifecycle compatibility.

## Required Review Evidence

For every claimed model/method/provider configuration require:

- one startup-captured graph per batch/sampling state and no actual
  context bucket in graph keys;
- repeated replay across representative, historical-threshold boundary,
  ragged, padded, and maximum-capacity contexts;
- unchanged graph count, `recapture_count == 0`, and stable registered input,
  workspace, output, and wrapper-owner addresses;
- no replay-time JIT, static plan/variant reselection, context-sized allocation,
  workspace growth, provider switch, or per-layer host planning;
- independent numerical comparison for output and every required score, LSE,
  cache mutation, or other side effect;
- padding and maximum-capacity memory-safety tests;
- real-model fixed and churn coverage, selected-provider/binding evidence, and
  matched performance results;
- isolated timing for CPU metadata preparation, H2D/D2H, provider planning,
  waits, and graph replay when replay-before work is nontrivial.

Test algorithm thresholds below, at, and above the boundary, including mixed
rows in one batch and length transitions within the same captured graph without
state loss or new capture. Do not use an algorithm or historical kernel-tuning
threshold to define a separate decode topology path.

## Finding Severity Additions

- P1: a claimed batch-only path performs runtime recapture, changes captured
  topology/variant from actual context, replaces captured addresses, switches
  provider after binding, or lets padding access/mutate live request state.
- P2: a changed provider rejects batch-only cleanly but leaves a required
  model/method without a production batch-only provider; replay preparation has
  avoidable per-layer or allocation overhead; or new code extends only the
  retired bucketed graph path without an explicit migration purpose.
- P3: terminology, binding-report, or ownership documentation is unclear while
  behavior remains correct and observable.

Use the main skill's P0-P3 definitions for all other findings.
