# SparseEngine Review Standards

Use these checks for SparseEngine's Python/Triton research engine.

## Architecture

- Method-specific runtime state belongs in `src/sparseengine/engine/cache_manager/`.
- `src/sparseengine/layers/attention.py` should stay generic and use hooks such as `on_kv_stored(...)` and `build_decode_view(...)`.
- `SparseController` owns cross-layer observation, attention-score collection, and scheduler-facing coordination, not method-owned cache metadata.
- Public APIs, `Config`, and internal consumers should all use `sparse_method` without a field-alias translation layer.
- New first-class methods should update config, `method_registry.py`, cache-manager routing, exports, docs, and policy tests.
- DeltaKV-family runtime behavior should stay cache-manager-first. Review method state in `src/sparseengine/engine/cache_manager/`, not ad hoc branches in `attention.py`, `utils/`, or benchmark scripts.

Flag method logic hidden in `utils/`, method branches added directly to `attention.py`, or benchmark scripts redefining method semantics.

## Platform Abstraction

Platform-specific behavior should flow through `src/sparseengine/platforms/` and shared device/runtime interfaces.

For platform abstraction changes, inspect `src/sparseengine/platforms/` plus changed shared-runtime call sites such as config normalization, model runner setup, cache managers, sparse controller, CUDA graph runner, loaders, and profilers.

Treat direct hardware-specific calls in common runtime code as an architecture issue. Code outside `src/sparseengine/platforms/`, narrowly scoped hardware backends, guarded one-off probes, or tests should avoid explicit APIs such as `torch.cuda`, CUDA device strings, CUDA memory queries, CUDA synchronization, and backend-specific distributed names. Prefer `sparseengine.platforms.current_platform`, platform methods, cache-manager/model-runner `device` fields, and backend-neutral config aliases so CUDA, ROCm, CPU, or future devices follow the same control flow.

## Scheduling

Prefill policy is registry-owned in `src/sparseengine/method_registry.py`.

- `all_chunked`: all prefill requests are capped by `engine_prefill_chunk_size` and normal scheduler limits.
- `long_bs1full_short_batch`: long requests run one complete prefill with batch size 1; short requests remain chunked and batchable.

Review long/short split carefully. Use `long_bs1full_short_batch` only for methods that need complete long-prefill before sparse/cache transformation, such as PyramidKV and DeltaKV-family methods. Policy, bucket, admission, or decode-priority changes need focused coverage in `tests/test_prefill_schedule_policy.py`.

## Correctness and Cache Accounting

Check tensor shapes, dtypes, devices, strides, `slot_mapping`, `active_slots`, `req_indices`, `context_lens`, and `cu_seqlens_q` across prefill/decode.

Cache-manager capacity hooks must agree: `num_free_slots`, `reserved_prefill_slots(...)`, `prompt_admission_costs(...)`, `prompt_admission_budgets(...)`, and `prefill_step_free_slots(...)`. Partial prefill, preemption, temp slots, staging views, and decode scratch must not leak slots or drop sequences.

Prefer fail-fast errors with method/cache-manager name, prompt length, needed/free slots, and budgets.

## OpenAI-Compatible Serving

Serving changes must preserve SparseEngine engine semantics instead of adding server-only execution paths.

For OpenAI-compatible serving or client changes, inspect `src/sparseengine/entrypoints/openai/api_server.py`, `src/sparseengine/entrypoints/openai/client.py`, `src/sparseengine/sampling_params.py`, `src/sparseengine/layers/sampler.py`, and `tests/test_openai_api_server.py`.

- Request JSON should reject unknown fields and validate `model`, `n`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, and `logprobs` before entering the engine.
- `CompletionRequest` and chat request handling should map directly to `SamplingParams`; sampler changes need coverage for greedy, top-p, top-k, and logprob behavior.
- Request admission, cancellation, streaming disconnects, and non-streaming errors must call `abort_request(...)` or otherwise release owned KV slots. Finished requests should use the engine's normal free path.
- Streaming responses must emit valid SSE frames, terminate with `data: [DONE]`, and avoid exposing partial stop text or inconsistent logprobs.
- DeltaKV-family methods must stay disabled for OpenAI serving unless serving correctness and memory behavior are explicitly validated.
- Per-request logging should report prompt/completion/total tokens and bounded TPS metrics without per-token noise or unbounded background loops.

Expected focused coverage: `tests/test_openai_api_server.py` for request validation, lifecycle, streaming, cancellation, and logprobs; `tests/test_sampler.py` for sampling contract changes.

## Research Reliability

Evaluation and benchmark code must not hide failures.

- Every evaluated sample needs a status: `success`, `invalid_input`, `model_failed`, `parse_failed`, `metric_failed`, or `skipped_by_policy`.
- Save raw outputs, parsed outputs, per-sample results, and aggregate metrics separately.
- Save enough run metadata: config, command, model, split, prompt, decoding params, seed, sample count, method, prefill policy, chunk size, prompt length, batch size, and checkpoint path when relevant.
- Keep retries, loops, API calls, parsing, and benchmark runs bounded.
- Do not silently change metric definitions or sample inclusion.

## Public Documentation Hygiene

Public repo docs should read like stable user-facing project documentation, not
developer scratch notes or private experiment ledgers.

For docs changes, inspect `README.md`, `docs/`, `benchmark/*/README.md`, and
`scripts/README.md` when they are in scope. Flag public-facing Markdown that:

- Records chronological local runs, failed attempts, dated campaign notes, GPU
  occupancy, or "worktree had uncommitted changes" details.
- Embeds host-specific paths such as `/root/...`, `/home/<user>/...`,
  `/data2/<user>/...`, AutoDL-specific defaults, or private artifact/log paths
  instead of placeholders like `<MODEL_ROOT>`, `<DATA_ROOT>`, `<OUTPUT_ROOT>`,
  and `<CHECKPOINT_ROOT>`.
- Points users to `scripts/tmp`, personal launchers, private notes, Codex/agent
  workflows, or repo-local skills from public README/docs.
- Stores raw experiment result history in `docs/` instead of a stable runbook,
  reproducibility checklist, methodology summary, or cited artifact output.

It is acceptable for `AGENTS.md`, `.agents/`, and `skills/` to contain
agent-facing instructions. The issue is exposing those details through public
project docs.

## Performance and Validation

Hot paths should avoid unnecessary CPU/GPU sync, host transfers, broad logging, repeated allocation, large tensor copies, and Python per-token loops. Kernel changes must preserve masks, bounds, block sizes, GQA/MHA behavior, and score collection.

Expected validation:

- Python changes: `python -m py_compile` touched files.
- Scheduler/policy changes: `tests/test_prefill_schedule_policy.py`.
- Runtime parameter changes: `tests/test_runtime_param_normalization.py`.
- OpenAI serving or sampler changes: `tests/test_openai_api_server.py` and `tests/test_sampler.py`.
- Research fail-fast changes: `tests/test_research_fail_fast.py`.
- Sparse method changes: one correctness-oriented run before throughput benchmarks.

Before GPU work, check device idleness and use an idle device when available.
