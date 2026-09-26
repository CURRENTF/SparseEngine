# SparseVLLM Regression Tests

## Purpose

This document describes how to run the fixed SparseVLLM regression harness under
`benchmark/sparseengine_regression/`.

The harness is intended for reproducible method/model checks across:

- `quality`: LongBench v1 mini, LongBench v2, and the length-stratified RULER
  core set.
- `longbench_v2`: focused LongBench v2 quality only.
- `ruler`: focused RULER core quality only.
- `perf`: prefill/decode throughput and memory accounting.
- `stress`: fixed-length high-concurrency SparseVLLM admission/decode stress.
- `stress_v2`: synthetic serving-trace stress with shared-prefix and multi-turn
  workloads, variable prompt lengths, and prefix-cache hit validation for
  supported methods.
- `validate`: manifest and output-artifact validation.
- `agent_trace`: recorded MiniSWE HTTP trajectories against an already-running
  OpenAI-compatible vanilla server; see below.

The test plan is controlled by
`benchmark/sparseengine_regression/manifest.json`.

## Recorded agent performance regression

This layer uses an external trace manifest instead of the synthetic model/method
matrix. It does not start a GPU server or execute tools. Prepare an idle-GPU server
separately, using SparseEngine vanilla or upstream vLLM; retain its actual MiniSWE
`server_manifest.json`. Match cache state and client/network placement between runs.

The standard published corpus is the appendable 100-agent, 8,000-request trace
at [`JitaiHao/SWE-lite-trace100` main](https://huggingface.co/datasets/JitaiHao/SWE-lite-trace100).
Pin revision `6bce0bb91ba68948248fd44b914ef77664b6b4e6` for this version.
Always match the trace manifest hash when comparing runs.

Reuse existing recordings:

```bash
python -m benchmark.sparseengine_regression.agent_trace \
  --run-dir "$MINISWE_RUN" --expected-instances 100 \
  --legacy-server-requests "$SERVER_REQUESTS" --output "$TRACE_DIR"
```

Repeat the legacy option for recovery-run request directories. By default, import
selects the first 100 IDs in frozen `instances.txt`. Use `--selection longest_turns`
to rank trajectories by model request count, breaking ties by source order.
`total_prompt_tokens` and `max_prompt_tokens` rank by the recorded response usage.
No selection filters by outcome. Import joins exact response IDs. Missing/duplicate requests, unmatched call counts,
cross-server trajectories and negative gaps fail explicitly. Terminal agent
time/step limits remain in the corpus. Inputs/outputs are exact saved payloads;
legacy delays are **estimates** calculated as
`next_log_timestamp - next_request_elapsed - previous_log_timestamp`.
Unknown logging overhead prevents a precision guarantee. Tool/network/retry waits
are retained. Historical generation duration is used only to derive the gap, not
stored as a replay field or used as a performance target.

For new exact client-boundary recordings, add `--record-agent-trace --slice 0:100`
to `python -m benchmark.swe_bench_lite.run`, retaining its usual model, API,
manifest, dataset, Docker and generation arguments. The default `mini-extra` is
wrapped by `timed_mini.py`. Synchronous non-streaming HTTP calls are associated
with each instance and flushed to `agent_trace.jsonl`; incomplete/failed HTTP
traces cannot be exported as complete. Export without the legacy option. This
uses the existing MiniSWE/httpx environment. Headers/credentials are not saved;
prompts and tool outputs can be private, so keep trace corpora outside Git.

```bash
python benchmark/sparseengine_regression/run_suite.py --layer agent_trace \
  --agent_trace "$TRACE_DIR" --agent_api_base "$API_BASE" \
  --agent_server_manifest "$SERVER_MANIFEST" --agent_concurrency 16 \
  --agent_allow_estimated_timing \
  --output_root "$OUTPUT_ROOT" --run_id agent_baseline
# Repeat with a matching fresh server and new run_id, adding:
# --agent_baseline "$OUTPUT_ROOT/sparseengine_regression/agent_baseline/agent_trace.json"
# --agent_max_slowdown 1.10
```

Only legacy imports require the explicit estimated-timing flag. Workers replay
whole agents with rolling concurrency. The first turn has no delay; subsequent
turns wait the recorded gap **after the current response**, then send the original
next input. This is not replay of the original absolute arrival schedule. The
target alias is substituted; original prompts/tools/sampling settings are retained,
but stop conditions are removed and `ignore_eos=true` plus the recorded completion
count fixes decode work. Actual output counts must match. Generated text is not
executed or substituted into later prompts; this is not a solution-quality test.

`agent_trace.json` reports current HTTP latency P50/P95/P99 and output tokens /
whole replay wall time **including waits**, not pure GPU throughput or TTFT/TPOT.
Per-agent files retain raw current outputs and failures/skipped dependent turns.
Any failed request fails the gate. Matched results gate P95 latency and complete
replay duration against a supplied baseline (default maximum slowdown 1.10).
Trace hash, model, GPU UUIDs, backend, engine settings, concurrency and timeout must
match. Without a baseline this creates a baseline, not a regression-pass claim.
`--dry_run` validates the corpus/config without contacting the server.

For local replay without recorded tool/network waits, use
`--agent_think_time_scale 0`. To exercise incremental radix pruning, add
`--agent_prefix_prune_keep_ratio 0.2 --agent_prefix_prune_tokenizer "$MODEL_PATH"`.
The shared MiniSWE client accumulates previously unpruned tool-body tokens until
`--agent_prefix_prune_trigger_tokens` (default 8192), then keeps the selected
fraction across all pending ranges and checks physical reuse on the next turn.
A threshold of 1 restores per-turn pruning. Below-threshold tails remain cached.
Queued KVzip tasks and their reconstruction chunks can share model forwards;
per-task ranges and retention budgets remain independent.
Use a fresh server cache for each replay. Request latency excludes pruning;
whole replay elapsed time includes pruning and the scaled waits. These settings
are part of the comparison contract; this replay does not measure solution quality.

For diagnosis, start the server with `SPARSEENGINE_CPU_TIMING_INTERVAL_S=10`
and replay with `SPARSEENGINE_PREFIX_PRUNE_TIMING=1`. Server `cpu_timing`
records include async submit/collect, TP coordination, routing snapshots and
prefix pruning; `prefix_prune_timing` records queue/execution time and request
counts. Per-agent prune events add client RPC and tool-selection wall times.
CPU timings are inclusive host observations: do not sum nested stages or treat
them as GPU kernel time. These diagnostics add no device synchronization and
are disabled by default; compare performance with the same logging settings.

## Prerequisites

Configure these paths for the machine running the suite:

- Working directory: `<REPO_ROOT>`
- Conda env: `<CONDA_ENV>`
- Output root: `<OUTPUT_ROOT>`
- LongBench data: `<LONGBENCH_ROOT>`
- LongBench v2 JSON/JSONL export: `<LONGBENCH_V2_DATA>`
- Models:
  - `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
  - `<MODEL_ROOT>/Qwen3-4B-Instruct-2507`
  - `<MODEL_ROOT>/Llama-3.1-8B-Instruct`
- Compressor checkpoints:
  - `<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor`
  - `<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor`
  - `<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor`

Set the environment before running the suite:

```bash
cd <REPO_ROOT>

export SPARSEENGINE_OUTPUT_DIR=<OUTPUT_ROOT>
export SPARSEENGINE_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>
export SPARSEENGINE_LONGBENCH_V2_DATA=<LONGBENCH_V2_DATA>

export DELTAKV_MODEL_QWEN25_7B=<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M
export DELTAKV_MODEL_QWEN3_4B=<MODEL_ROOT>/Qwen3-4B-Instruct-2507
export DELTAKV_MODEL_LLAMA31_8B=<MODEL_ROOT>/Llama-3.1-8B-Instruct

export DELTAKV_COMPRESSOR_QWEN25_7B=<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor
export DELTAKV_COMPRESSOR_QWEN3_4B=<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor
export DELTAKV_COMPRESSOR_LLAMA31_8B=<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor

export PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}
```

Initialize the pinned official LongBench repository before the first v2 run:

```bash
git submodule update --init benchmark/long_bench_v2/upstream
```

The submodule supplies the official prompt and implementation provenance. The
official `THUDM/LongBench-v2` dataset is distributed separately; export its
train split to one local `.json` or `.jsonl` file and point
`SPARSEENGINE_LONGBENCH_V2_DATA` to that immutable input.

The manifest also contains `qwen25_32b`; omit it unless there is enough GPU
memory and the corresponding model/checkpoint environment variables are set.

## Quick Unit Tests

Run the unit tests that protect the regression harness, RULER generators and
grading, manifest policy, and OmniKV full-layer selector:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python -m pytest \
  tests/test_sparseengine_regression_grading.py \
  tests/test_omnikv_full_layer_selector.py \
  tests/test_ruler_tasks.py \
  tests/test_ruler_vt_regression.py \
  tests/test_longbench_v2.py \
  -q
```

Expected result for the current harness: all tests pass.

## Manifest Validation

Use `validate` before long GPU runs. It resolves runtime paths, writes the
resolved manifest, and creates empty required artifact files.

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer validate \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id validate_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

Use `--no-allow_skipped_policy` when missing model/checkpoint paths should fail
the run instead of being recorded as skipped.

## Common Run Commands

All commands write to:

```text
<OUTPUT_ROOT>/sparseengine_regression/<run_id>/
```

### Quality

By default, `--layer quality` runs LongBench v1 mini, LongBench v2, and RULER
core. Pass a subset such as `--quality_benchmarks longbench_v2` only when a
focused run is intentional.

LongBench-mini uses:

- tasks: `qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc`
- LongBench batch size: `100`
- SparseVLLM `max_num_seqs_in_batch`: `16`
- SparseVLLM `max_decoding_seqs`: `16`
- samples per task: `50`

Run OmniKV against vanilla baselines:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id omnikv_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For a full non-32B quality run:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,quest,deltakv,deltakv-less-memory \
  --run_id quality_3models_all_methods_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For TP decode CUDA graph v1 quality validation, keep LongBench data-worker
parallelism at its default and pass engine TP through the regression-suite
override:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_quality_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

This compares sparse methods against TP vanilla in the same run. A/B/C grades
are recorded; crashes or D grades fail the TP graph quality gate.

For the fast TP prefix-cache + decode-graph regression gate, keep the same
method coverage but use explicit small-sample overrides and a child-command
timeout. This still exercises LongBench quality, SCBench quality, and stress,
but it is a smoke/regression gate rather than the full 50-sample-per-task
quality suite:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --quality_benchmarks longbench \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --quality_tasks qasper,hotpotqa \
  --quality_batch_size 2 \
  --quality_samples_per_task 2 \
  --quality_min_required_samples 2 \
  --quality_sparseengine_max_num_seqs_in_batch 2 \
  --quality_sparseengine_max_decoding_seqs 2 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_quality_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

Then run the matching SCBench quality and prefix-hit stress layers:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer scbench \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --scbench_decode_cuda_graph \
  --scbench_tasks scbench_kv \
  --scbench_num_eval_examples 1 \
  --scbench_max_turns 2 \
  --scbench_max_seq_length 1024 \
  --scbench_batch_size 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_scbench_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>

conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer stress \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --require_prefix_cache_hit \
  --stress_length 256 \
  --stress_request_counts 2 \
  --stress_output_len 2 \
  --stress_max_num_seqs_in_batch 2 \
  --stress_max_decoding_seqs 2 \
  --stress_max_decode_steps_after_full 1 \
  --stress_admission_wave_size 1 \
  --stress_wave_decode_gap_steps 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_stress_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### LongBench V2 Quality

LongBench v2 is additive: it does not replace the existing LongBench v1 mini
gate. The source benchmark covers roughly 8K to 2M words. The canonical
regression profile uses 120 natural samples across post-chat-template token
buckets `32K-64K`, `64K-96K`, and `96K-127K` (40 samples per bucket), with
`max_model_len=131072`, so it
remains runnable on 128K-class models while extending beyond the v1 runner's
121K limit. Selection is deterministic for a dataset, tokenizer, and seed.

The runner uses the official zero-shot direct-answer prompt from the pinned
submodule. It does not copy the upstream API client and does not use upstream's
head/tail truncation: prompts outside the configured capacity are excluded
before deterministic selection, and an underfilled bucket fails. The gate also
requires exact vanilla/sparse sample alignment, an identical source-data hash,
and complete explicit sample statuses. A non-empty response that does not
contain the official answer pattern is retained as `parse_failed` and scored as
incorrect, matching the official evaluator; model/runtime execution failures
still invalidate the run.

This fixed 120-sample profile uses greedy decoding to reduce run-to-run noise. It
is a repository regression gate, not a reproduction of the official full
503-sample leaderboard protocol.

Run only the canonical v2 gate:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer longbench_v2 \
  --models qwen25_7b \
  --methods vanilla,omnikv \
  --command_timeout_s 7200 \
  --run_id longbench_v2_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For a model with a verified context window above 128K, extend both the runtime
limit and the token buckets explicitly; this is a separate profile, not a silent
change to the canonical gate:

```bash
python benchmark/sparseengine_regression/run_suite.py \
  --layer longbench_v2 \
  --models qwen25_7b \
  --methods vanilla,omnikv \
  --longbench_v2_max_model_len 262144 \
  --longbench_v2_token_buckets_json \
  '[{"name":"128k-192k","min_prompt_tokens":131072,"max_prompt_tokens":196607,"samples":4},{"name":"192k-255k","min_prompt_tokens":196608,"max_prompt_tokens":261888,"samples":4}]' \
  --output_root <OUTPUT_ROOT>
```

### RULER Core Quality

The fixed self-contained set runs `niah_single_1`, `niah_multikey_2`, `vt`,
`cwe`, and `fwe`, covering retrieval, multi-hop tracing, and two aggregation
contracts. It uses `16K,32K,64K,98K` target context lengths and grades every
task and context length independently against the exactly aligned vanilla
dataset from the same run. A failure in one task/length bucket therefore cannot
be hidden by an average. Every sample must reach at least 90% of its target
sequence length. Raw, parsed, per-sample, dataset, aggregate, and grade
artifacts are retained.
With 10 samples per task/length bucket, the default matrix runs 200 generations
per model/method pair.

The task contracts and prompts follow NVIDIA RULER. Deterministic synthetic
word pools replace optional `wonderwords` and large word assets, so this is a
repository regression set rather than an official leaderboard dataset.
Essay-based NIAH and `qa_1`/`qa_2` are excluded because they require downloaded
external corpora.

Run the RULER core set without also running LongBench-mini:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer ruler \
  --models qwen25_7b \
  --methods vanilla,omnikv \
  --ruler_tasks niah_single_1,niah_multikey_2,vt,cwe,fwe \
  --ruler_context_lengths 16384,32768,65536,98304 \
  --ruler_samples_per_length 10 \
  --command_timeout_s 7200 \
  --run_id ruler_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For supported methods, prefix-cache validation immediately replays each
deterministic batch in the same engine. The gate requires nonzero hit requests
and hit tokens and requires replay outputs and scores to equal the primary
pass exactly:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer ruler \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --ruler_tasks vt \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --ruler_context_lengths 16384,32768 \
  --ruler_samples_per_length 2 \
  --command_timeout_s 1800 \
  --run_id ruler_prefix_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

Inspect `ruler.json`, `grade_summary.json`, and each task/method directory's
`prefix_cache_summary.json`. This is a quality and cache-correctness gate, not
a prefix-cache performance benchmark.

### Performance

Performance uses:

- prompt lengths: `16000,64000`
- batch sizes: `1,4`
- output tokens: `256`
- decode CUDA graph requested where the method supports it

For sparse methods, the benchmark also runs vanilla for the same shape so the
suite can compute decode speedup.

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_perf_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For TP decode CUDA graph v1 performance validation:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_perf_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

Inspect `perf.jsonl` for `decode_graph_expected=true` and
`decode_graph_active=true`.

### Stress

Stress currently uses:

- prompt length: `16000`
- request count / batch size: `80`
- output tokens: `64`
- `max_num_seqs_in_batch=80`
- `max_decoding_seqs=80`
- max decode steps after full admission: `32`

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer stress \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_stress80_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Stress V2

`stress_v2` uses `scripts/benchmarks/bench_prefix_cache.py` as a regression
layer for serving-like traces. Unlike fixed `stress`, it runs seeded synthetic
requests with:

- workloads: `shared_prefix,multiturn`
- supported methods: `vanilla`, `omnikv`, `quest`
- `vanilla` cases: `baseline_full,prefix_full`
- `omnikv` case: `prefix_omnikv`
- `quest` case: `prefix_quest`
- sessions / turns: `8 / 4`
- shared-prefix requests: `8`
- output tokens: `64`
- max active requests: `8`
- variable multi-turn user lengths: `128..1024`
- variable session-prefix lengths: `1024..4096`
- variable shared suffix lengths: `512..4096`
- max prompt length: about `16.5k` tokens for multi-turn and `12.3k` for
  shared-prefix
- prefix-cache block size: `16`

The gate fails if a prefix-cache-enabled case does not observe cache hits or if
the realized prompt lengths do not vary. Unsupported methods are recorded as
`skipped_by_policy` because this layer specifically validates prefix-cache
serving behavior.

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer stress_v2 \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --run_id stress_v2_qwen3_serving_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Combined Layers

`nightly` runs quality and performance. It does not run stress.

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer nightly \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id nightly_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

`pre-refactor` runs quality, performance, and stress.

## Result Records

Keep this file as the stable regression runbook. Do not add chronological
experiment records or local result indexes here. If a repo-facing result claim
is needed, cite the original run artifact path directly.

## OmniKV Full-Layer Selection

OmniKV full layers are model-specific. Use
`python -m sparseengine.utils.select_omnikv_full_layers` before publishing a new model's
OmniKV or OmniKV-aligned DeltaKV regression numbers.

The selector runs an offline decode-attention coverage calibration on a
LongBench task, chooses `--num-full-layers` layers, and writes the selected
layer string to `selected_full_layers.json`. This is not an online runtime mode:
the selected string must be passed back as `full_attention_layers`.

Example for Qwen2.5-7B with six full layers:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python -m sparseengine.utils.select_omnikv_full_layers \
  --model-path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --longbench-root <LONGBENCH_ROOT> \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir <OUTPUT_ROOT>/omnikv_full_layer_calibration_$(date -u +%Y%m%d)/qwen25_7b_full6 \
  --num-full-layers 6 \
  --num-samples 32 \
  --topk 2048 \
  --random-decode-points-per-sample 8 \
  --num-sink-tokens 0 \
  --num-recent-tokens 32 \
  --prefill-chunk-size 512 \
  --torch-dtype bfloat16 \
  --device cuda
```

Key outputs:

- `selected_full_layers.json`: selected layer ids and
  `full_attention_layers` string for runtime configs.
- `per_sample_points.jsonl`: sampled decode points used for calibration.
- `pair_scores.npy` and `segment_scores.npy`: raw coverage matrices for audit.
- `run_info.json`: command, git state, model/data paths, and calibration
  settings.

To use the selected layers in an ad hoc Sparse-VLLM run, copy the
`full_attention_layers` value into `--hyper_params`:

```bash
PYTHONPATH=$PWD:$PWD/src python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <MODEL_DIR> \
  --methods omnikv \
  --lengths 131072 \
  --batch_sizes 4 \
  --output_len 128 \
  --hyper_params '{"sparse_method":"omnikv","full_attention_layers":"0,2,4,11,16,22","decode_keep_tokens":4096,"recent_keep_tokens":32,"sink_keep_tokens":0,"engine_prefill_chunk_size":512}'
```

For regression runs, update `methods.omnikv.model_configs` in
`benchmark/sparseengine_regression/manifest.json`. If a DeltaKV regression config
is intentionally aligned to OmniKV observation/full layers, update the matching
DeltaKV model config in the same manifest and record that alignment in the run
summary. The current manifest uses:

```text
qwen25_7b:  0,2,4,11,16,22
qwen3_4b:   0,1,3,9,13,16,21,28
llama31_8b: 0,2,7,13,16,26
```

Run `validate` and rerun OmniKV quality/perf/stress after changing these
layers.

## Outputs

Each run writes:

- `resolved_manifest.json`: manifest after environment-variable resolution.
- `grade_summary.json`: command records, grades, and final status.
- `metrics.json`: quality aggregate records.
- `perf.jsonl`: flattened performance rows.
- `memory.json`: memory grades derived from performance rows.
- `stress.json`: stress rows and stress grades.
- `stress_v2.json`: serving-trace stress rows and stress_v2 grades.
- `ruler.json`: RULER aggregate records by task, model, and method.
- `longbench_v2.json`: LongBench v2 aggregate records by model and method.
- `raw_outputs.jsonl`, `parsed_outputs.jsonl`, `sample_results.jsonl`: quality
  generation artifacts, when quality is run.
- Layer-specific logs:
  - `quality/<model>/<method>/run.log`
  - `ruler/<task>/<model>/<method>/run.log`, with dataset, aggregate, and optional
    prefix-cache replay artifacts in the same directory
  - `longbench_v2/<model>/<method>/run.log`, with selection, source hashes,
    per-sample results, and aggregate metrics in the same directory
  - `perf/<model>/<method>.log`
  - `stress/<model>/<method>.log`

Quick summary command:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("<OUTPUT_ROOT>/sparseengine_regression/<run_id>")
data = json.loads((root / "grade_summary.json").read_text())
print("status:", data["status"])
print("worst_required_grade:", data.get("worst_required_grade"))
for grade in data.get("grades", []):
    print(grade.get("task"), grade.get("model"), grade.get("method"), grade.get("context_length"), grade["name"], grade["grade"], grade["status"], grade["metrics"])
PY
```

## Regression Rubrics

The executable gate rules live in `benchmark/sparseengine_regression/grading.py`.
The stable human-facing rubric lives in
`benchmark/sparseengine_regression/rubrics.md`.

Update `benchmark/sparseengine_regression/rubrics.md` only when stable ABCD
rubric definitions change. Do not add dated campaign results, open blockers,
run IDs, or remote log paths to the rubric file.

## Troubleshooting

- Missing model or compressor paths:
  - Run `validate`.
  - Check `resolved_manifest.json`.
  - If a run should fail on missing paths, pass `--no-allow_skipped_policy`.
- Import errors:
  - Ensure `PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}`.
  - Use an environment with the dependencies from [Getting Started](../getting_started/README.md).
- Quality dataset errors:
  - Set `SPARSEENGINE_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>`.
- GPU memory failures:
  - Do not add fallback behavior inside the harness.
  - Record the exact run ID, model, method, layer, log path, and error in the
    issue note.
- A command exits early:
  - Inspect `<run_id>/grade_summary.json`; failed commands are recorded with
    `returncode`, `cmd`, and `log_path`.
  - Inspect the layer-specific log path from the command record.
