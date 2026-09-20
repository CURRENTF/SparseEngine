---
name: add-sparse-method
description: Add or refactor a first-class SparseEngine sparse method across configuration, capability registration, cache and controller ownership, scheduler admission, typed attention views, model and storage compatibility, CUDA Graph and prefix-cache lifecycles, operators, and reproducible validation. Use for a new canonical sparse_method or an integration whose sparse runtime semantics change; do not use for a kernel/provider optimization that leaves method semantics unchanged.
---

# Add Sparse Method

Treat a sparse method as a runtime capability contract, not as a file template. Start from the method's semantics, route each responsibility to its existing owner, and add only the narrow contracts the method actually needs.

Always read the
[official runtime architecture](../../../docs/en/design/sparse-method-runtime.md) and
[references/file-map.md](references/file-map.md) before editing. The official
document defines the stable runtime boundary; the file map distinguishes files
to inspect from files that usually need modification.

## 1. Classify The Change

Use this skill when the change introduces or materially changes a canonical `sparse_method`, including its state, selection, scheduling, storage, lifecycle, or compatibility semantics.

Do not create a new sparse method merely to add:

- an attention compute implementation with unchanged semantics; use the operator/provider and kernel workflow;
- a storage layout required by several methods; extend the storage/model layout contract;
- a model integration that only adapts an existing method;
- a config-only experiment or a private ablation with no first-class runtime contract.

If classification is ambiguous, write the semantic difference from the nearest existing method first. If there is no difference in runtime behavior, it is not a new method.

## 2. Declare The Capability Contract

Before editing code, complete the contract in [references/runtime-contract.md](references/runtime-contract.md). At minimum, decide:

- identity: public name, canonical internal name, aliases, defaults, and external assets;
- state semantics: logical views, physical eviction, paged metadata, compression, activation state, or combinations;
- selection semantics: when scores are produced, which queries/keys they describe, and whether state crosses layers or steps;
- storage compatibility: explicit KV, heterogeneous explicit KV, MLA latent storage, and actual-key materialization needs;
- scheduling and admission: prefill mode, batch compatibility, capacity budgets, reservation costs, and offload behavior;
- lifecycle support: eager execution, CUDA Graph, radix prefix cache, chain prefix cache, and restore/fork/free behavior;
- model and topology support: model types, TP/EP/DP assumptions, and provider requirements;
- implementation cost: phase-specific time complexity, peak memory, kernel reuse candidates, and hot-path data movement;
- validation evidence: correctness oracle, quality workload, performance workload, and unsupported combinations.

Do not fill unknowns with permissive defaults. Unsupported combinations must be represented explicitly and fail during validation or initialization, not in a later kernel.

### Design For Eager And CUDA Graph From The Start

Consider decode CUDA Graph support in the initial design, before choosing
metadata, cache-update kernels, and workspace layouts; do not treat it as an
afterthought once an eager-only implementation is complete. Prefer a shared
compute and state-transition path for eager execution and graph replay, with
differences confined to preparation, capture/replay, and justified launch plans.
Follow the [graph design contract](references/runtime-contract.md#eager-and-cuda-graph-design)
and [replay validation](references/validation.md#cuda-graph-replay-checks).

An initial eager-only scope must identify concrete blockers and the intended
adaptation, distinguishing deferred work or missing GPU evidence from a genuine
semantic/provider limitation. Do not advertise graph support until validated,
or change the algorithm merely to make capture possible.

### Required Cost And Reuse Review Before Implementation

Complete the cost and reuse review in
[references/runtime-contract.md](references/runtime-contract.md#cost-and-reuse-review)
before implementation. Architecture compliance alone is insufficient: account
for scoring, selection, metadata, mutation, and attention together, including
temporary storage and repeated work. Identify existing repository kernels and
upstream providers by symbol/path and explain each reuse, narrow extension, or
new-kernel decision. Explicitly check vLLM and SGLang (including `sgl-kernel`)
for reusable kernels; prefer direct reuse when the existing interface and
execution contract fit, and use their implementations as references for narrow
adaptations when direct reuse does not fit.

Choose an implementation that avoids unnecessary full-domain intermediates,
host synchronization, and repeated allocation or copying. Preserve the declared
algorithm: changing selection granularity, scoring, update frequency, or cache
lifecycle to make it faster requires explicit authorization for that semantic
change. Distinguish an intentionally slow correctness oracle from the serving
implementation. A required expensive algorithm is not grounds to silently
replace it with an approximation.

## 3. Route Responsibilities By Owner

Preserve these boundaries:

- Define canonical runtime fields in `configs/groups.py` or `configs/runtime.py`, use the same names in public and internal code, and reject unknown inputs at the engine boundary. Keep composed validation in the relevant config group, not the compatibility facade `config.py`.
- Register static method capabilities, aliases, schedule defaults, score contracts, model/topology compatibility, prefix-cache support, graph support, and assets in `method_registry.py`.
- Keep persistent cache-coupled method state, physical allocation, eviction/compaction, and attention-view materialization in `engine/cache_manager/`.
- Keep `engine/sparse_controller.py` as the stable method-agnostic facade. It
  constructs typed lifecycle requests/events and delegates; do not place method
  algorithms or method-name hot-path branches there.
- Keep per-step/per-layer logical state, score workspace orchestration,
  cross-layer observation/selection, and mutation triggers in an
  `engine/sparse_methods/` `SparseMethodRuntime`.
- Keep hidden-state capture and activation reuse in `engine/activation_controller.py`.
- Expose admission budgets, reservation costs, execution modes, and lifecycle state through `RuntimeState` and `MemoryOracle`; the scheduler consumes this generic contract.
- Pass typed selections, views, payloads, and writes through the cache/attention boundary. Keep `layers/attention.py` and model attention call sites method-agnostic.
- Let `ModelSpec`, `RuntimeLayout`, `ParallelTopology`, and storage protocols own physical cache layout compatibility. A sparse method does not own the model's KV representation.

Do not add method-name branches to `SparseController`, Attention, Scheduler, or
ModelRunner. If a method requires behavior not expressible by an existing
generic contract, add the smallest typed hook at the owning boundary. The
existing `SparseMethodRuntime` interface is the logical strategy boundary; do
not introduce a second broad strategy framework.

## 4. Choose A Runtime Mechanism, Not A Method Name

Read [references/method-families.md](references/method-families.md), then select reference implementations by shared mechanics: logical view, physical compaction, scored selection, query-aware paged selection, activation reuse, or compressed/multi-pool storage.

Select or add a runtime by shared logical mechanics. The current shallow runtime
families include pass-through, scored compaction, joint decode compaction, and
dynamic cross-layer selection. Reuse a runtime base only after verifying score
representation, trigger order, selection domain, mutation order, graph buffer
lifetime, and prefix behavior.

CacheManager inheritance is a separate physical-storage axis. Do not choose a
runtime parent because two methods share a CacheManager base, and do not choose
a CacheManager parent because two methods share runtime orchestration. If the
full lifecycle does not match, add a separate implementation and share only a
narrow helper.

For QuEST-like query-aware selection over explicit paged KV, also read [references/quest-pattern.md](references/quest-pattern.md). Do not apply that pattern to MLA latent or custom payload methods without proving the same view contract is valid.

## 5. Preserve Typed Data-Plane Contracts

Prefer the existing data types and protocols:

- `SparseSelection` for selection results;
- `AttentionViewMeta` for logical attention metadata;
- `PrefillComputeView` and `DecodeComputeView` for execution-time views;
- `ExplicitKVPayload` and `MlaLatentPayload` for storage-specific payloads;
- typed cache writes and `AttentionCacheStorage` for cache mutation and storage access.

Use `build_decode_view` when the method only changes the logical explicit-KV view. Use `build_decode_compute_view` when execution requires a different payload or materialization step. Do not pass method-specific tuples, hidden tensor side channels, or config objects through attention.

## 6. Route Kernels At The Semantic Boundary

- Put alternative attention computation behind `OpRegistry`, a typed `*OpSpec`, `DeviceCaps`, and an operator provider. Prepare and bind providers before capture when CUDA Graph is supported.
- Preserve unified full-attention ownership when a method changes only one
  phase. Prefill and decode may resolve different atomic providers, but their
  shared cache contract must be validated and both phases must remain under one
  prepared `FullAttentionProvider` lifecycle.
- Reuse mature upstream providers for unchanged standard attention, GEMM, MoE,
  normalization, and communication semantics across their declared support
  domains. A sparse method may add requirements to `OpSpec`, but it must not
  force a repository kernel when an upstream provider satisfies the same typed
  contract.
- Add repository-owned production kernels for the method's actual semantic
  gap, such as score production, custom cache access, selection, compaction,
  compression, reconstruction, or state mutation. Keep standard local kernels
  as portable fallbacks or exact-profile overrides.
- Keep local benchmark profiles separate from atomic correctness eligibility.
  A method-specific measurement may add an exact dispatch route; it must not
  narrow an upstream provider's standard support domain.
- Invoke method-internal selection, metadata, compaction, compression, or reconstruction kernels from the state owner, usually CacheManager or a `SparseMethodRuntime`, never from the controller facade merely because it is globally reachable.
- Keep kernel adapters thin and keep policy, allocation, and lifecycle decisions out of kernel modules.
- Do not silently fall back to another provider or dense behavior. Any fallback must be an explicit registered capability with tested semantics.

Read and apply `$review-operator-organization` whenever a sparse-method change
adds or changes provider selection, physical weight layout, or production
kernel dispatch.
Use `$optimize-sparseengine-kernel` when implementing or tuning a GPU kernel;
reusing an existing kernel does not by itself require a tuning project.

## 7. Integrate In Dependency Order

Implement the smallest vertical slice in this order:

1. Canonical public configuration and static registry contract.
2. Model/storage/topology validation.
3. State ownership, allocation, and lifecycle operations.
4. `SparseMethodRuntime` score/selection orchestration, runtime factory binding,
   and typed view construction.
5. Scheduler admission and prefill execution semantics.
6. Operator/provider or method-internal kernels, if required.
7. Prefix-cache, CUDA Graph, and offload integration for every advertised capability.
8. Documentation, evaluation configuration, and reproducibility artifacts.

Inspect all layers, but edit only the owners affected by the declared contract. Research-code scope matters: do not combine a method addition with an unrelated architecture rewrite.

## 8. Validate Claims, Not Just Imports

Read and follow [references/validation.md](references/validation.md). Validation must cover every capability advertised by the registry and every relevant storage/layout path.

Run cheap static and unit checks first, then targeted runtime correctness, lifecycle matrices, quality evaluation, and matched performance benchmarks. Fail fast on missing models, assets, datasets, unsupported layouts, or unavailable providers. Preserve raw outputs and per-sample status so experimental results remain auditable.
