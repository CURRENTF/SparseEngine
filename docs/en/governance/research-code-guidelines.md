# Research Code Guidelines

This repository is research code. The main goal is trustworthy and reproducible experimental results, not production-style abstraction.

## Sparse Method Contributions

When adding or refactoring a first-class Sparse-Engine sparse method, keep
method-specific runtime state in `src/sparseengine/engine/cache_manager/`, keep
`src/sparseengine/layers/attention.py` generic, and hook new behavior through
shared cache-manager interfaces when possible.

## Research Code Priorities

Primary goals:

1. Make experiments reproducible.
2. Make results easy to verify.
3. Keep implementation minimal and readable.
4. Avoid hiding failures.

Working rules:

- Prefer simple, explicit code over abstraction-heavy frameworks.
- Do not introduce new dependencies unless necessary. If necessary, document why.
- Do not add broad fallback logic, silent exception handling, or auto-recovery paths unless explicitly requested.
- Do not mask errors with default values, random substitutes, empty outputs, or warning-only behavior.
- Fail fast with clear error messages when required files, configs, checkpoints, datasets, or API keys are missing.
- Keep changes scoped to the requested experiment or bug.
- Preserve existing experiment semantics unless an explicit refactor is requested.
- Add comments only for non-obvious research logic, tensor shapes, algorithmic choices, or paper-specific details.

## Reliability Rules

The priority is trustworthy experimental results.

1. Do not hide failures. Missing files, bad configs, failed API calls, parse errors, and metric errors must be explicit.
2. Do not add fallback behavior unless requested. Any fallback must be opt-in, logged, and reflected in final results.
3. Every evaluated sample must have an explicit status: `success`, `invalid_input`, `model_failed`, `parse_failed`, `metric_failed`, or `skipped_by_policy`.
4. Save raw outputs, parsed outputs, per-sample results, and aggregate metrics separately.
5. Do not change metric definitions or sample inclusion rules unless explicitly requested.
6. Bound all retries, loops, API calls, and parsing attempts.
7. Validate inputs at config, dataset, model-loading, parsing, and metric boundaries.
8. Save enough run information to reproduce the experiment: config, command, model, dataset split, prompt, decoding parameters, seed, and sample count.
9. Make the smallest correct change. Avoid unrelated refactors, new dependencies, and renamed interfaces.

## Benchmark Artifacts

For benchmark scripts, the expected output layout should make failures and scoring auditable:

- `raw_outputs.jsonl`: raw model responses and generation metadata.
- `parsed_outputs.jsonl`: parsed predictions and parser status.
- `per_sample_results.jsonl`: sample id, input metadata, prediction, target, correctness, and explicit status.
- `aggregate_metrics.json`: final metrics computed from the per-sample results.
- `run_info.json`: command, environment, model path, dataset path, split, sample count, decoding parameters, seed, method parameters, and relevant environment variables.

If a fallback is explicitly enabled, the per-sample result and aggregate metadata must record which samples used it. Silent fallback results are not acceptable for this repo.
