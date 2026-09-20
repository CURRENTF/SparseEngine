---
name: code-review
description: Review SparseEngine diffs for correctness, sparse-runtime and operator architecture, scheduling semantics, reproducibility, public documentation hygiene, performance, and tests. Use when reviewing PRs, git diffs, sparse method integrations, operator/provider or kernel changes, platform capability handling, cache-manager or scheduler changes, benchmark/evaluation scripts, docs changes, or when the user asks for a code review; if no range is specified, diff the current branch against main.
---

# Code Review

## Workflow

### Step 1: Determine the Diff

Use the user's SHAs, patch, or PR when provided. Otherwise:

```bash
git fetch origin main --quiet
CURRENT_BRANCH=$(git branch --show-current)
MERGE_BASE=$(git merge-base origin/main HEAD)
git diff --stat "$MERGE_BASE..HEAD"
git diff "$MERGE_BASE..HEAD"
```

If `CURRENT_BRANCH` is `main`, ask which commits or files to review.

### Step 2: Load Review Standards

Read [sengine-review-standards.md](references/sengine-review-standards.md).

For sparse-method additions or refactors, also read [`$add-sparse-method`](../add-sparse-method/SKILL.md). For policy changes, inspect `src/sparseengine/method_registry.py`, `src/sparseengine/engine/scheduler.py`, and `tests/test_prefill_schedule_policy.py`.

For changes to operators, providers, platforms, kernels, quantized weight
layouts, external kernel dependencies, or model-to-operator call sites, read
and apply
[`$review-operator-organization`](../review-operator-organization/SKILL.md).

### Step 3: Review

Prioritize:

- inference correctness and tensor/cache invariants
- SparseEngine architecture boundaries
- platform abstraction boundaries
- operator/provider selection and kernel ownership, including upstream-first
  standard operations and repository ownership of non-standard sparse semantics
- multi-phase operator ownership: independently selected prefill/decode kernels
  must share one validated full-attention binding and lifecycle
- separation of atomic correctness eligibility, default portfolio policy,
  exact profile overlays, and prepared dispatch plans
- prefill policy, long/short split, and `long_bs1full_short_batch`
- OpenAI-compatible request lifecycle, streaming, cancellation, and sampling contracts
- research reproducibility and fail-fast behavior
- public docs hygiene: stable user-facing docs must not contain local experiment ledgers, internal development notes, private paths, GPU occupancy notes, `scripts/tmp` launchers, uncommitted-worktree details, or agent/Codex workflow instructions
- hot-path performance and tests, including CUDA Graph/JIT specialization,
  host synchronization, scheduler and prefix-cache RPC overhead, sparse
  scoring/selection, offload copies, LRU work, workspace allocation, and
  provider dispatch

### Step 4: Report

Lead with findings, highest severity first. Use absolute file paths and line numbers.

```text
[P1] Short title
File: /absolute/path/to/file.py:123
What: ...
Why: ...
How: ...
```

Severity:

- `P0`: invalidates inference, corrupts results, or hides experiment failure.
- `P1`: likely bug, architecture violation, scheduler/policy regression, or serious performance regression.
- `P2`: meaningful test, reproducibility, edge-case, or maintainability gap.
- `P3`: minor clarity, style, or docs issue.

End with:

- `Open Questions` only when they affect correctness or confidence.
- `Summary` in 1-3 sentences.
- `Validation Notes` with commands run and missing coverage.
- `Assessment`: `Ready to merge? Yes / No / With fixes`, plus concise reasoning.

## Rules

- Do not approve without reading relevant code and tests.
- Do not comment broadly on files outside the diff unless needed to explain the changed behavior.
- Flag local benchmark, exact profiled device, or profiled shape conditions in
  an upstream atomic provider's support predicate unless they are required for
  correctness. Flag free-form priority changes that mix atomic eligibility,
  upstream default policy, and exact performance routing.
- Do not treat benchmark results as trustworthy if method, policy, config, command, checkpoint, sample status, or outputs are missing.
- Require evidence for the claimed metric and measurement boundary. Do not
  present component microbenchmarks, CPU-only timings, isolated kernel medians,
  stage logs, or profiler timelines as matched serving TTFT, TPOT, or throughput.
  Serving comparisons require the metric definition (including numerator and
  denominator for rates), timing boundary, matched workload, model, hardware,
  topology, concurrency, graph mode, cache budget, and failure count. Component
  comparisons require relevant shapes, dtypes, callable and timing boundaries,
  environment, and failure count; mark inapplicable serving fields as N/A.
  For paper efficiency comparisons, apply
  [`paper-efficiency`](../paper-efficiency/SKILL.md): a validated continuous
  decode window can support decode-window throughput, but not serving TPOT or
  full-workload throughput.
- Do not assume the cause of a performance regression is the scheduler, CPU, or
  a specific kernel without evidence from the changed workload boundary.
- If no issues are found, say so clearly and mention residual test or benchmark risk.
