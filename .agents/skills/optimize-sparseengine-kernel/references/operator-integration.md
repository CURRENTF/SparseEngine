# SparseEngine Operator Integration

Read the sibling
[`review-operator-organization`](../../review-operator-organization/SKILL.md)
skill before changing production dispatch. Treat its architecture and
validation rules as authoritative.

## Keep Ownership Explicit

```text
model semantic call
  -> OpSpec
  -> atomic capability filter(DeviceCaps)
  -> exact profile overlay
  -> upstream-first portfolio policy
  -> selected atomic provider or prepared dispatch plan
  -> provider-owned preparation/workspace
  -> Triton, TileLang, JIT, or external kernel
```

- Keep repository-owned Triton kernels under `src/sparseengine/kernels/triton/`.
- Keep repository-owned TileLang kernels under
  `src/sparseengine/kernels/tilelang/`.
- Keep thin third-party adapters under `src/sparseengine/kernels/external/`.
- Keep support checks, dependency availability, static launch selection,
  workspaces, and provider binding under `src/sparseengine/operators/`.
- Keep models expressed in semantic operations rather than backend names,
  package imports, device probes, or physical weight layouts.
- Keep device discovery and stable capability facts under the platform layer.

## Separate Eligibility From Performance Policy

- Atomic `supports(spec, caps)` describes correctness and compatibility. Do
  not put local benchmark coverage or `requires profiled ...` conditions in an
  upstream atomic provider unless execution is actually invalid outside them.
- Prefer mature upstream public providers for standard operations across their
  declared support domains. Keep repository-owned standard kernels as
  portable fallbacks, correctness baselines, or algorithmic defaults. Reserve
  exact-profile overrides for tuning-dependent choices.
- Use repository-owned production kernels for sparse or otherwise non-standard
  semantics that upstream providers cannot express.
- Promote algorithmic improvements through the default portfolio when their
  computational rationale and representative validation support generalization.
  Keep real compatibility constraints; do not turn measured shapes, batches,
  TP counts, or device names into an enablement whitelist.
- Represent tuning-dependent matched performance data as a separate exact
  profile overlay that references eligible atomic providers and produces a
  prepared dispatch plan.
  A profile miss returns to the default portfolio; it does not make an atomic
  provider unsupported.
- Record whether selection came from an upstream default, profile override,
  semantic fallback, or dependency degradation. Do not use an undifferentiated
  numeric priority as a substitute for these decisions.

## Resolve Before Execution

Make `supports(spec, caps)` cover platform, architecture, dtype, quantization,
shape and alignment, layouts, topology, graph behavior, workspace, toolchain,
and external API availability. Import optional compilers and kernel packages
lazily. Bind the provider outside the forward hot path.

Distinguish an unsupported semantic contract, an absent optional dependency,
and a broken installed dependency. Allow fallback only by rejecting a normal
unsupported or unavailable candidate during resolution or preparation. Once
execution begins, surface compilation and launch failures. Do not make one
provider failure disable unrelated operators, and do not hide a broken legal
environment as a normal performance fallback.

Keep phase-specific atomic selection separate when the execution contracts
differ. Full attention uses independent prefill and decode registries under one
prepared `FullAttentionProvider`; the owner validates their shared cache
contract and binds/closes both phases together. A provider that implements only
prefill, such as FlexPrefill, belongs only to that phase portfolio.

## Validate the Complete Path

Add or update:

- independent kernel-equivalence tests
- resolver selection and rejection tests
- missing and minimum-version dependency tests
- boundary shape, dtype, stride, padding, and workspace tests
- CUDA Graph capture/replay tests where supported
- actual model-path integration coverage
- profile-hit and profile-miss tests that preserve broad atomic eligibility
- matched evidence for exact profile overrides; representative correctness and
  performance checks plus a generalization rationale for algorithmic defaults
- explicit tested coverage, without claiming unmeasured performance or requiring
  an exhaustive matrix before broad compatible default selection

Record the selected provider, selection basis, evidence source, and rejection
reasons in reproducible artifacts.
After implementation, use `$review-operator-organization` for the focused
architecture review and `$code-review` for the complete diff.
