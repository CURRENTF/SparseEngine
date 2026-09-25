# Repo Skills

This repository includes repo-local Codex skills.

## Available skills

- `add-sparse-method`: Add or refactor a first-class SparseEngine sparse method following this repo's architecture. Use when Codex needs to introduce a new `sparse_method`, move method logic out of `attention.py` or `utils/`, add method-specific cache metadata or decode-time view building, and preserve the cache-manager-first design. File: `.agents/skills/add-sparse-method/SKILL.md`
- `code-review`: Review SparseEngine diffs for correctness, sparse-runtime and operator architecture, scheduling semantics, reproducibility, performance, and tests. Use when reviewing PRs, git diffs, sparse method integrations, operator/provider or kernel changes, cache-manager or scheduler changes, benchmark/evaluation scripts, OpenAI serving changes, or when the user asks for a code review. File: `.agents/skills/code-review/SKILL.md`
- `review-operator-organization`: Review operator/provider boundaries, device capability selection, kernel ownership, dependency compatibility, weight layouts, fallback semantics, and validation. Use for changes under `operators/`, `platforms/`, Triton kernels, external kernel integrations, or model-to-operator call sites. File: `.agents/skills/review-operator-organization/SKILL.md`
- `optimize-sparseengine-kernel`: Find, implement, tune, profile, and integrate SparseEngine GPU kernels across Triton, TileLang, CUDA/CuTe, and external SGL providers. Use for kernel hotspots, fusion, correctness baselines, microbenchmarks, Nsight Compute analysis, provider integration, or matched end-to-end performance validation. File: `.agents/skills/optimize-sparseengine-kernel/SKILL.md`
- `paper-efficiency`: Standardize paper efficiency comparisons and capacity sweeps using shared entrypoints, explicit defaults, and boundary-only decode timing. Use for paper benchmarks and deciding whether results can share a figure, not figure-only styling. File: `.agents/skills/paper-efficiency/SKILL.md`

## How to use

- In this repo, invoke the sparse-method skill as `$add-sparse-method`.
- In this repo, invoke the review skill as `$code-review`.
- Invoke focused operator reviews as `$review-operator-organization`;
  `$code-review` loads it automatically for relevant diffs.
- Invoke the end-to-end kernel workflow as `$optimize-sparseengine-kernel`; it
  loads only the selected DSL and profiling references.
- Invoke `$paper-efficiency` for paper-level engine efficiency comparisons.
- Keep method-specific runtime state in `src/sparseengine/engine/cache_manager/`.
- Keep `src/sparseengine/layers/attention.py` generic and hook new methods through shared cache-manager interfaces when possible.

# Task Running Rules

1. Before running a task, check whether each device is idle. Select an idle device when one is available. If all devices are busy, wait first; if the wait becomes too long, report the situation instead of starting the task on a busy device. Ignore the above requirements when the user indicates that the GPU can be shared with other processes.
2. Do not hardcode private paths (including local machine paths and remote paths) in test scripts; pass them via variables or arguments instead. Scripts located under `scripts/tmp/` are exempt from this restriction.
3. When using a conda environment, activate it or use `conda run`; invoking only its absolute `python` path does not expose environment-provided executables such as `ninja` to child processes.

# Standardized Efficiency & Performance Benchmark Suite

The canonical runbooks are:

- [English efficiency benchmark runbook](docs/en/benchmarking/efficiency.md)
- [简体中文效率基准运行手册](docs/zh/benchmarking/efficiency.md)

Follow the runbook's matched-trace, idle-GPU, artifact-validation, and metric-
interpretation rules. Do not treat sampled GPU activity as theoretical MFU/MBU;
use the documented Nsight diagnostic for kernel-timeline attribution.

## Benchmark Entrypoints and Shared Statistics

- For paper comparisons, use the defaults and acceptance contract in
  [.agents/skills/paper-efficiency/SKILL.md](.agents/skills/paper-efficiency/SKILL.md).
  Main decode results use continuous windows with boundary-only synchronization,
  preserving supported async/overlap execution. Step-synchronized runs are diagnostics.
- Use `scripts/benchmarks/run_efficiency_probe.sh` (idle-GPU checks and sweeps)
  or `benchmark/efficiency/bench_probe.py` (explicit engine/TP configuration)
  for request TTFT/TPOT and end-to-end throughput. Do not create another runner
  for a new model, method, or shape; extend the existing arguments if needed.
- All latency distributions and stage-throughput math belong in
  `benchmark/efficiency/metrics.py`. Existing runners import it. Reaggregate
  probe artifacts with
  `python3 benchmark/efficiency/metrics.py <RUN_DIR>/request_samples.jsonl`.
  This command prints JSON and does not require CUDA or overwrite old artifacts.
- Use `benchmark/microbench.py` for separately timed prefill/decode engine
  steps. CUDA step synchronization is opt-in via `--synchronize_step_timing`,
  for stage diagnostics only; without it, synchronized stage rates are null.
  Do not add per-step CUDA synchronization to request TTFT/TPOT measurements:
  timestamp token publication events and preserve the engine's execution rhythm.
  The old
  `scripts/benchmarks/bench_sparse_engine.py` command remains a compatibility
  wrapper; new callers should use the canonical microbench path.
- TTFT and TPOT distributions pool individual measured requests across
  iterations, reporting mean/P50/P95/P99. TPOT is `(finish-first)/(O-1)` for
  `O > 1`; never subtract prefill interference or scheduling waits. Declare
  the arrival/first/finish observation boundary; engine events are not HTTP
  client latency. Keep differing vLLM timing sources separate in comparisons.
- Stage throughput requires actual computed token counts and separately
  accumulated, synchronized, non-overlapping stage time, including scheduling,
  sampling, scoring, eviction/compaction and cleanup. Exclude prefix hits from
  computed prefill tokens and prefill-produced tokens from decode token work.
  A selected decode window must report its admission/warmup/truncation scope.
- Default probe first-token/decode event windows are diagnostic window rates,
  not execution-stage rates, even in fixed-batch mode. Explicit `--decode-only-steps`
  instead measures a validated continuous full-residency decode window with
  boundary-only synchronization. The fixed-batch adapter supports native per-rank
  boundaries, vLLM/Tangram async queues, and HiSparse QuEST TP1 overlap queues;
  validate each model/topology/version with a smoke before a paper sweep.
  The default probe's old `prefill_token_throughput_tps` and
  `decode_token_throughput_tps` fields are compatibility aliases only; prefer
  `first_token_window_throughput_tps` and `batch_decode_token_throughput_tps`.
  Microbench's legacy `ttft`/`itl` fields are batch observations/proxies, not
  request distributions. Never derive stage throughput from mean TTFT or TPOT.
- `output_token_throughput_tps` is all output tokens divided by complete
  measured workload time. A request metric contract change invalidates old
  aggregate comparisons; reaggregate available per-request artifacts or rerun.

# Kernel Provider Policy

1. Separate atomic correctness eligibility, the default portfolio, exact
   performance-profile overlays, and validation evidence. Do not use local
   benchmark coverage or performance confidence as an atomic support condition.
2. For standard operations, use a mature upstream public provider as the
   default across its declared compatible domain. Keep repository-owned Triton
   or other portable implementations as correctness baselines and fallbacks for
   unsupported contracts or absent optional dependencies; an installed but
   broken compatible dependency must fail explicitly. Algorithmic improvements
   may enter the default portfolio across their actual compatible domain when
   the computation, memory traffic, or parallelism change explains generalization
   and representative correctness/performance checks support the choice. Do not
   require a whitelist of measured shapes, batch sizes, TP counts, or GPU models.
   Retain real dtype, layout, hardware-feature, and known compiler constraints.
3. A matched local performance profile may override the upstream default only
   for its exact recorded device, contract, shape or runtime bucket, topology,
   graph mode, and toolchain. A profile miss returns to the upstream-first
   default (including adopted algorithmic improvements); it must not select
   Triton merely because that setting was unmeasured
   or because FlashInfer or SGL did not win every benchmarked setting.
   Reserve exact profiles for device/toolchain/shape tuning or specialized
   dispatch schedules. Validation artifacts describe tested coverage; they are
   not enablement whitelists. Performance claims still require measurements.
4. Do not narrow an upstream provider's correctness domain because local
   hardware or shape coverage is incomplete. Validate the adapter and the
   SparseEngine-specific boundary, and record upstream-declared support separately
   from locally validated correctness and performance evidence.
5. Resolve and prepare providers before execution. Unsupported candidates may
   be rejected during resolution, but a prepare, JIT, warmup, or execution
   failure must preserve the actionable error and must not silently reselect a
   different provider.


# Research Code Skill

You are writing research code, not production SaaS code.

Primary goals:
1. Make experiments reproducible.
2. Make results easy to verify.
3. Keep implementation minimal and readable.
4. Avoid hiding failures.

Rules:
- Prefer simple, explicit code over abstraction-heavy frameworks.
- Do not introduce new dependencies unless necessary. If necessary, explain why.
- Do not add broad fallback logic, silent exception handling, or auto-recovery paths unless explicitly requested.
- Do not mask errors with default values, random substitutes, empty outputs, or warning-only behavior.
- Fail fast with clear error messages when required files, configs, checkpoints, datasets, or API keys are missing.
- Keep changes scoped to the requested experiment or bug.
- Preserve existing experiment semantics unless the user explicitly asks to refactor.
- Add comments only for non-obvious research logic, tensor shapes, algorithmic choices, or paper-specific details.

# Research Code Reliability Rules

This is a research codebase. The priority is trustworthy experimental results.

1. Do not hide failures. Missing files, bad configs, failed API calls, parse errors, and metric errors must be explicit.
2. Do not add fallback behavior unless requested. Any fallback must be opt-in, logged, and reflected in final results.
3. Every evaluated sample must have an explicit status: success, invalid_input, model_failed, parse_failed, metric_failed, or skipped_by_policy.
4. Save raw outputs, parsed outputs, per-sample results, and aggregate metrics separately.
5. Do not change metric definitions or sample inclusion rules unless explicitly requested.
6. Bound all retries, loops, API calls, and parsing attempts.
7. Validate inputs at config, dataset, model-loading, parsing, and metric boundaries.
8. Save enough run information to reproduce the experiment: config, command, model, dataset split, prompt, decoding parameters, seed, and sample count.
9. Make the smallest correct change. Avoid unrelated refactors, new dependencies, and renamed interfaces.

# Test Design Rules

Tests in this LLM inference repository must protect an independent correctness,
safety, or reproducibility contract. A test is not useful merely because it can
run without a GPU.

1. Do not add tests that only restate ordinary defaults, exact tuning constants,
   bucket boundaries, model budgets, provider choices, supported-model lists,
   registry contents, benchmark manifests, or source/AST shape. These values are
   expected to change deliberately and their definitions are already the source
   of truth.
2. Do not copy a production table or condition into a test and assert that both
   copies match. Prefer one authoritative definition plus validation at the
   boundary that consumes it.
3. CPU tests are appropriate only when the behavior is owned by CPU code or the
   test supplies an independent oracle: scheduler/cache/allocator state
   transitions, lifecycle and resource accounting, failure propagation,
   serialization boundaries, or mathematically derived results. "Runs on CPU"
   alone is not a reason to add a test.
4. Prefer properties and invariants over literals: capacity is not exceeded,
   failed admission does not mutate state, ordering is deterministic, resources
   are released, and outputs match an independent reference implementation.
5. Provider or kernel routing mocks may test explicit failure handling, but they
   do not prove GPU correctness or performance. Numerical kernel behavior needs
   a CUDA test with an independent numerical oracle; performance choices need a
   reproducible matched benchmark, not a unit test that freezes the winner.
6. Do not generate exhaustive parameter cross-products from declarative
   registries unless each case exercises distinct behavior. Use representative
   cases for shared behavior and focused tests for real exceptions.
7. Every new test should name a realistic regression it catches and why existing
   coverage would not catch it. Retain tests for public contracts, trust
   boundaries, explicit failure states, and previously observed regressions.
8. When a profile, threshold, budget, or provider decision is intentionally
   retuned, update its authoritative config or benchmark artifact. Do not add or
   preserve a unit test whose only purpose is to prevent that intentional change.

# Git Rules

## Git Commit Messages Rules
1. **Specification**: Strictly follow the Conventional Commits specification.
2. **Format**: Use the format `<type>: <description>`.
3. **Allowed Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.
4. **Style**:
   - Write the description in English, using the imperative mood (e.g., "add" not "added").
   - Start the description with a lowercase letter.
   - Keep the entire line under 200 characters.

# Docs Rules

Keep official documentation focused on stable user-visible behavior and operational constraints; place internal implementation details, provider/kernel selection rationale, benchmark methodology and results, and transient engineering plans in development documentation unless users need them to use or troubleshoot the feature.
