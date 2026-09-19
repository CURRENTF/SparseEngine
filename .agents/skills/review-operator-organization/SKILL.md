---
name: review-operator-organization
description: Review Sparse-Engine operator architecture, provider selection, platform capability boundaries, kernel ownership, dependency compatibility, weight layouts, batch-only CUDA Graph adaptation, fallback semantics, and validation. Use for diffs touching src/sparseengine/operators, src/sparseengine/platforms, Triton or external kernels, model-to-operator call sites, quantized weight loading, CUDA Graph constraints, optional kernel dependencies, or backend removal and migration.
---

# Review Operator Organization

Review the complete selection path, not an isolated kernel:

```text
model semantic call
  -> OpSpec
  -> atomic capability filter(DeviceCaps)
  -> exact local profile overlay, when one matches
  -> default kernel portfolio policy
  -> selected atomic provider or prepared dispatch plan
  -> weight preparation / workspace
  -> local or external kernel
```

Use `docs/en/design/provider-selection.md` as the primary design context and
keep its `docs/zh/` mirror aligned, but judge the executable code and tests.

## Inspect the Change

Read all changed and directly coupled files in these areas:

- `src/sparseengine/operators/`: specs, providers, registries, and resolvers.
- `src/sparseengine/platforms/`: platform discovery and `DeviceCaps`.
- `src/sparseengine/kernels/triton/`: repository-owned Triton implementations.
- `src/sparseengine/kernels/tilelang/`: repository-owned TileLang implementations.
- `src/sparseengine/kernels/external/`: thin external-kernel adapters.
- Model and loader call sites that construct specs or prepare physical weights.
- Dependency declarations and installation documentation for external providers.
- Resolver, kernel-equivalence, integration, and model tests.

Trace at least one supported configuration and every changed rejection or
fallback path from model construction through execution.

## Enforce the Boundaries

### Model and Operator

- Keep models expressed in stable operator semantics. Flag imports of concrete
  providers, external kernel packages, architecture checks, or backend names.
- Keep implementation choices out of `OpSpec`; fields describe dtype, shape,
  quantization, topology, graph, workspace, and mutation requirements.
- Do not add model-level or global backend configuration such as
  `moe_backend`. Tests and microbenchmarks may force a provider through an
  internal interface.

### Platform and DeviceCaps

- Treat `DeviceCaps` as an immutable snapshot produced and cached by the active
  `Platform`, never as a second platform abstraction.
- Keep device discovery, streams, synchronization, memory, and communication
  operations in `Platform`.
- Keep only stable selection facts in `DeviceCaps`. Do not place provider
  factories or optional-library availability there.
- Flag direct `torch.cuda` or architecture probing in common model/operator
  code when the platform can supply the fact.
- Treat ROCm or NPU placeholders as unsupported until an implementation and
  tests exist; a TODO must not advertise support.

### Provider and Resolver

- Require `supports(spec, caps)` to cover every condition needed by execution:
  platform, architecture, dtype, quantization format, shape alignment,
  TP/EP topology, CUDA Graph behavior, workspace, toolchain, and external API
  availability.
- Keep atomic eligibility about correctness and compatibility. Local benchmark
  coverage, exact profiled devices or shapes, and performance confidence are
  not atomic support conditions unless the kernel is actually incorrect or
  unusable outside that domain.
- Separate atomic providers, default portfolio policy, exact profile overlays,
  and prepared dispatch plans. Flag a single free-form numeric priority that
  mixes these decisions or a profile-backed plan registered as if it were an
  atomic implementation.
- Distinguish an unsupported semantic contract, an absent optional dependency,
  and a broken or incompatible installed dependency. A normal candidate
  rejection may select another provider during resolution; a broken legal
  environment must remain an actionable startup failure rather than a silent
  performance fallback.
- Check optional dependencies lazily inside their provider. An unselected
  provider must not import or initialize a heavyweight dependency.
- Resolve once before weight preparation and bind the provider outside the
  forward hot path.
- Let one semantic provider own multi-phase lifecycles without forcing the
  phases to share an atomic kernel. For full attention, keep prefill and decode
  atomic registries independent, validate their shared cache contract, and bind
  both through one prepared full-attention owner. Prefill-only kernels must not
  implement a fake decode path.
- Require a deterministic portfolio policy among fully supported atomic
  providers and useful selection evidence and rejection reasons.
- Allow fallback only during resolution or preparation: reject the unsupported
  provider and select another provider for that operator. Once execution has
  begun, do not catch a kernel failure and silently switch implementations.
- Ensure one provider's build or JIT failure does not disable unrelated
  operators.

### Decode CUDA Graph

Batch-only is the only maintained decode CUDA Graph shape policy. For any
review touching captured decode, graph input preparation, provider graph state,
or sparse topology paths, read and enforce
[references/batch-only-decode-graph.md](references/batch-only-decode-graph.md).
That reference defines graph identity, static versus dynamic metadata, unified
input ownership, participant lifecycles, external wrappers, validation, and
finding severity. Eager may remain as a separate correctness path or for
operators that do not support graph capture; do not preserve a second bucketed
graph architecture.

### Kernel Portfolio

- Treat standard operations as upstream-first. Prefer a mature upstream public
  provider and its maintained dispatcher across the upstream-declared support
  domain when it satisfies the Sparse-Engine operator contract.
- Keep repository-owned standard Triton or other local implementations as
  portable production fallbacks, correctness baselines, or algorithmic defaults.
  Computation, data-movement, or parallelism improvements may enter the default
  portfolio across their real compatible domain with a generalization rationale
  and representative validation. Do not demand a measured shape/batch/TP/device
  whitelist. Use exact profiles for tuning-dependent winners and specialized
  schedules; a local win alone does not establish a general algorithmic gain.
- Concentrate repository-owned production kernels on sparse or otherwise
  non-standard semantics that upstream providers cannot express, such as
  score production, custom cache layouts, state mutation, selection,
  compaction, compression, or reconstruction.
- Treat `repo_nonstandard` as a semantic classification, not a hardware-scope
  classification. Prefer portable-by-construction Triton or TileLang kernels
  for repository-owned nonstandard contracts.
- Derive a portable DSL provider's atomic support from its semantic and tensor
  contract, DSL/toolchain availability, hardware features actually used, and
  known compiler limitations. Do not add device-name or architecture
  whitelists solely because local hardware or validation coverage is limited.
- Permit optimistic binding on unvalidated devices when no incompatibility is
  known. A prepare, JIT, warmup, or first-execution failure must preserve the
  actionable error and must not trigger silent provider reselection. Once an
  incompatibility is confirmed, prefer a feature- or toolchain-based exclusion
  over a permanent device-model whitelist.
- Keep broad portable eligibility separate from evidence. Record only tested
  devices and shapes as validated, and make performance profiles and claims no
  broader than reproducible measurements. Do not describe an unmeasured device
  as performance-validated merely because the DSL kernel is eligible there.
- Let matched local performance profiles override the upstream default only
  for their exact recorded device, contract, shape, topology, graph mode, and
  runtime bucket. A profile may add a dispatch route; it must not narrow an
  atomic provider's correctness support domain.
- Do not reject an upstream atomic provider only because every supported
  device or shape was not benchmarked locally. Validate the adapter contract,
  package/API compatibility, and the boundary conditions Sparse-Engine adds,
  then record upstream-declared support separately from local evidence.
- Treat raw or internal upstream entry points as locally maintained adapter
  contracts. Do not inherit broad upstream engine validation or dispatch
  claims unless Sparse-Engine actually uses the corresponding public interface.
- Keep Torch or naive implementations as explicit correctness oracles, not
  automatic production candidates.
- Do not introduce Hub-downloaded or frozen kernels as implicit runtime
  dependencies.
- Keep kernel-specific layout conversions and workspaces behind the provider
  interface.

### Weights and Layouts

- Let checkpoint loading identify logical projections and scales.
- Let the selected provider own physical packing, ordering, shuffling,
  padding, and scale layout.
- Flag `[gate, up]` versus `[up, gate]` choices or provider-specific offsets in
  model classes.
- Resolve before finalizing weights, tag or otherwise preserve layout
  ownership, and prohibit provider switching without re-preparing weights.
- Flag repeated transpose, repack, or dequantization work in forward.

## Verify External API Compatibility

For every added external API call:

1. Find the declared minimum package version in `pyproject.toml` and docs.
2. Inspect the function signature and required artifacts at that exact minimum
   version, not only in the currently installed environment.
3. Verify every keyword, enum, dtype, layout, and return contract used by the
   provider.
4. Keep core Python/JIT-cache minimums compatible. Keep optional cubin
   packages out of required dependencies unless execution truly requires them.
5. Add a regression check when a parameter was introduced after an older
   supported release.

Treat a legal dependency environment that crashes on provider invocation as
P1 even when the developer environment has a newer working version.

## Require Validation

Match validation to the changed selection surface:

- Resolver tests: supported and rejected architectures, dtype/shape variants,
  dependency absent or too old, toolchain unavailable, graph mode, and TP/EP.
- Portfolio tests: prove that an exact profile hit creates only its recorded
  override, a profile miss returns to the upstream-first default, and sparse
  semantic requirements select a compatible non-standard provider without
  changing standard-operation policy.
- Failure tests: aggregate actionable rejection reasons and prove fallback is
  local to the affected operator.
- Kernel tests: compare against an independent Torch oracle over boundary
  lengths, non-contiguous layouts, state mutation, padding, real model shapes,
  and every claimed dtype.
- Hardware tests for repository-owned kernels: exercise available
  representative architectures and record the exact tested domain. Lack of
  access to another device is not by itself a reason to narrow a portable DSL
  provider's atomic eligibility. Exact local performance overrides must still
  be tested on every device, contract, shape, and runtime bucket they claim.
- External-provider tests: use upstream-declared device support for atomic
  eligibility, validate the public API and Sparse-Engine adapter on available
  representative hardware, and label the evidence honestly. Lack of local
  access to every upstream-supported device is not by itself a reason to
  shrink that provider to a local hardware whitelist.
- Integration tests: run the actual model path so provider binding, weight
  layout, prefill/decode routing, and generation are covered.
- Quality checks: use the model chat template and deterministic decoding;
  non-empty output alone is not a quality assertion.
- Reproducibility: record selected providers, capability summary, dependency
  versions, selection basis (`upstream_default`, `profile_override`,
  `semantic_fallback`, or dependency degradation), model, command, device,
  and test result.

Do not accept a performance result before correctness equivalence. Do not
accept mocked provider-selection tests as proof that the minimum external
package version can execute the real call.

## Classify Findings

- P0: wrong output, corrupt state/cache, or incompatible physical weight
  layout that invalidates inference.
- P1: a validated or explicitly guaranteed environment crashes, the resolver
  selects a provider despite a known contract incompatibility, runtime silently
  changes providers, or model semantics depend on a backend layout. Also use
  P1 when local profile evidence narrows a standard upstream atomic support
  domain or a broadly preferred repository kernel makes performance claims
  without evidence appropriate to that scope.
- P2: missing rejection coverage, incomplete observability, hot-path selection
  overhead, or a confirmed portability gap in an otherwise portable DSL
  provider. Do not classify a clear prepare/JIT failure on previously
  unvalidated hardware as P1 solely because eligibility was optimistic, unless
  the failure contradicts claimed validation, ignores a known incompatibility,
  corrupts state, or is silently masked.
- P3: naming or documentation clarity that does not change execution.

Report the broken boundary, the concrete supported configuration that exposes
it, and the smallest architectural correction.
