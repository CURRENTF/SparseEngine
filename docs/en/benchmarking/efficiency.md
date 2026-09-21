# Efficiency and Throughput Benchmark Suite

[简体中文](../../zh/benchmarking/efficiency.md) | English

This runbook owns measurement definitions, entrypoints, and support limits.
Keep experiment-specific parameters, sweeps, and replot instructions in
[scripts/official_experiments/](../../../scripts/official_experiments/).
New defaults do not retroactively change historical protocols.

## Entrypoints and Minimal Example

| Purpose | Entrypoint |
| --- | --- |
| Request TTFT/TPOT, E2E, fixed/churn comparisons | `benchmark/efficiency/bench_probe.py`; optional orchestration via `scripts/benchmarks/run_efficiency_probe.sh` |
| Continuous decode throughput for paper figures | The same probe with explicit `--decode-only-steps` |
| Step-synchronized stage diagnostics | `benchmark/microbench.py --synchronize_step_timing`, separate from main figures |
| Shared statistics and offline request aggregation | `benchmark/efficiency/metrics.py` |

Run from the repository root in an activated environment (use conda activate or
conda run). Check dependencies, model access, persistent output capacity, and
every participating GPU's idle state with `nvidia-smi`. Wrappers check idle
devices; the standalone Python CLI requires a manual check. Use a new run directory.

This request-mode smoke checks functionality, not paper performance:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:$PWD/src" python3 \
  benchmark/efficiency/bench_probe.py \
  --engine sparseengine --sparse-method vanilla --model-path "<MODEL_PATH>" \
  --tensor-parallel-size 1 --monitor-gpus 0 \
  --scenario fixed --prompt-lens 4096 --output-lens 32 --batch-sizes 1 \
  --num-warmups 0 --num-iters 1 --output-dir "<NEW_RUN_DIR>"
```

See `python3 benchmark/efficiency/bench_probe.py --help` for all arguments.
`--monitor-gpus` uses physical IDs aligned with `CUDA_VISIBLE_DEVICES`.
The probe defaults to `max_num_batched_tokens=65536` per scheduler.
The probe explicitly sets SparseEngine's `engine_prefill_chunk_size=8192`; override
it through `--hyper-params`. The scheduler token budget and prefill chunk size
are separate controls. vLLM has no equivalent independent chunk-size option in
these adapters; its chunking follows the available scheduler token budget.
Record this difference and per-replica/global budgets in cross-topology comparisons.
Match checkpoints, weight/KV dtypes, traces, GPUs/TP/DP/EP, budgets, Graph policy,
seed, lengths, jitter, warmups, and repetitions across systems. Report
sink/recent/selected/full-layer budgets separately: equal names do not establish
equal work or quality.

Continuous windows require explicit `--scenario fixed --prompt-length-jitter 0
--output-length-jitter 0 --decode-only-steps 256 --decode-only-warmup-steps 32`.
These are not ordinary request-mode defaults. For a capacity sweep and its smoke,
use the [sparse decode efficiency experiment entrypoint](../../../scripts/official_experiments/sparse_decode_efficiency/README.md#boundary-sync-rerun);
do not copy timing runners. Cover periodic scoring/eviction, extending the
window uniformly across comparisons when necessary.

Additional entrypoints:

- [Probe wrapper](../../../scripts/benchmarks/run_efficiency_probe.sh):
  `SYSTEMS MODEL_NAME_OR_PATH PHYSICAL_GPU_IDS`; consult the script for variables
  and model aliases. Aliases fix TP; arbitrary model paths default to TP2.
  Use the Python CLI for explicit topology. Sparse settings come from the
  [regression manifest](../../../benchmark/sparseengine_regression/manifest.json).
  OmniKV requires calibrated layers; set `BENCH_MANIFEST_MODEL_ID` or
  `OMNIKV_FULL_ATTENTION_LAYERS` explicitly for custom calibration.
  Single-layer configurations require an explicit ablation option.
- [Unified suite](../../../scripts/benchmarks/run_unified_efficiency_suite.sh):
  argument order is `GPUS SYSTEMS MODEL_NAME`; runs synthetic and LongBench.
  Set `SPARSEENGINE_LONGBENCH_DATA_DIR` and check final `suite_status.json`.
- [Nsight diagnostic](../../../scripts/benchmarks/run_efficiency_profile.sh):
  use after a standard run identifies a suspicious case;
  arguments are `SYSTEM MODEL_PATH GPUS`. The wrapper currently supports native
  vanilla/SnapKV and vLLM vanilla with default TP2. Requires `nsys` and
  performance-counter permissions; produces `timeline.nsys-rep`.

<a id="measurement-contract"></a>

## Metrics and Timing Contract

### Continuous Decode: Main Paper Results

Throughput = actually completed decode tokens / full coordinator wall-clock window.

- Finish prefill, wave admission, and full-batch warmup before the window.
  Keep the same request set; prefill, preemption, turnover, partial batches,
  Graph capture, unexpected eager execution, and incomplete windows invalidate it.
- Wait for relevant work on all ranks/devices and drain in-flight work only at
  the two edges. Preserve compatible async/overlap and algorithm-required
  synchronization; add no per-step CUDA sync or synchronous RPC. Count completed,
  not submitted, tokens over the same interval; validate Graph state per rank.
- Include scheduling, driver work, sampling, scoring, eviction, and communication.
  This is not isolated kernel time. Save per-request start/end context lengths,
  admission, warmup, and measured steps; wave admission can produce unequal
  starting contexts.
- Finish and verify every requested output. Do not shorten windows, lower a point's
  batch, or truncate to conceal failure. Input/output changes or truncation need
  explicit approval and matched reruns under the new protocol.
- Each workload currently rebuilds the engine. Discarded workloads prime persistent
  compilation caches; every measured workload also discards full-batch warmup steps.
  This is not reuse of a live warmed engine across workloads.
- Boundary synchronization perturbs request latency, so this mode does not report
  TTFT/TPOT. Measure request/E2E metrics separately for overall serving efficiency.
  Never mix step-sync and boundary-sync results in comparisons or aggregates.

If full-batch residency is too brief, distinguish early request completion from
actual KV capacity limits. With approval, try transferring input tokens to output
in a separate bounded pilot; do not shorten the warmup/window or disable async.
Record both length pairs, the reason, and measured context lengths: equal total
length does not imply equal decode work. Rerun and label an adjusted curve rather
than mixing points or claiming it establishes the original setting's capacity.

### Request Mode: TTFT/TPOT and E2E

The default synthetic probe uses deterministic random traces, refreshed per
iteration and matched by seed/case across systems. Default jitter varies lengths;
churn includes oversubscription and turnover. Continuous decode uses the fixed
trace recorded in its manifest; do not assume it matches the default probe trace.

Fixed request mode accepts `--enable-prefix-caching --shared-prompt
--prompt-length-jitter 0` for identical prompts within each workload. Each repeat
uses a fresh prefix, separate from warmup. All requests are submitted together;
the engine decides admission and reuse. Inspect per-request `num_cached_tokens`
instead of assuming that enabling caching eliminates all but one prefill.
This mode supports SparseEngine/vLLM request probes (vLLM DP1), without prefill
waves or continuous decode windows. TPOT and decode event windows remain distinct
from execution-stage timing.

| Metric | Definition and boundary |
| --- | --- |
| TTFT | Request arrival to first token |
| TPOT | `(finish - first) / (output tokens - 1)`; null for one token, includes scheduling waits and intervening prefill |
| `output_token_throughput_tps` | All output tokens / complete workload time, i.e. E2E |
| `first_token_window_throughput_tps` | Input tokens / first-token event window |
| `batch_decode_token_throughput_tps` | Subsequent output tokens / earliest first token to latest completion |

The last two are potentially overlapping event windows, not execution-stage rates.
Default `stage_metrics_status=not_measured`; old `prefill_token_throughput_tps`
and `decode_token_throughput_tps` are compatibility aliases.
`tpot_concurrency_proxy_tps` is concurrency × 1000 / mean TPOT, not observed throughput.

The `per_request_distribution_v3` contract pools measured requests across
iterations for mean/P50/P95/P99; the old mean of batch TTFT maxima is
`batch_max_ttft_ms_mean`. Native token observations occur at step return without
extra step synchronization (`sparseengine_step_token_publication_no_extra_sync_v1`).
vLLM legacy finished_time and V1 last_token_ts have distinct `timing_source`
values. These are engine events, not HTTP client latency; compare matching boundaries.

### Step Synchronization: Stage Diagnostics

`--synchronize_step_timing` divides actual computed tokens by accumulated complete
synchronized step time. It includes work inside steps but excludes intervening
driver work. Prefill excludes prefix hits; decode excludes prefill-produced tokens.
Logical input is not computed work. Without step sync or an explicit continuous
window, stage rates are null. Legacy `ttft/itl` are batch observations/proxies,
not request distributions.

For fixed full-batch diagnostics, add
`--require_full_decode_batch --decode_warmup_steps_after_full N`:
require complete outputs without preemption, exclude warmup and falling-batch tails,
and reject truncation. Record admission, warmup, and truncation policy.
GPU activity is sampled by `nvidia-smi`, not theoretical MFU/MBU, and cannot
attribute CPU or launch overhead.

<a id="support"></a>

## Support and Validation Status

The table describes adapter paths, not GPU validation of every model, topology,
or dependency version. Update it when capabilities change. Smoke-test timing
boundaries, complete outputs, actual Graph and async/overlap behavior for each
target combination before measuring performance; implementation alone is not evidence.

| Mode | Current scope and limits |
| --- | --- |
| Default request probe | SparseEngine / vLLM, fixed/churn, explicit TP; model capabilities still apply |
| Native continuous decode | Per-rank boundary synchronization implemented, not permanently TP1-only; current orchestration uses DP1, TP/EP depend on model/engine capabilities |
| vLLM / Tangram continuous decode | Async queue boundary draining implemented; smoke-test each external version/model; no wave admission |
| HiSparse QuEST continuous decode | TP1 overlap queue boundary draining implemented; not an MLA adapter, no wave admission |
| Vortex QuEST continuous decode | Shared SGLang completed-work collector; DP1 and EP1 or EP=TP, with TP-group boundary synchronization and per-rank Graph/work validation implemented; smoke each model/topology before claiming GPU validation; no wave admission |
| Synchronized vLLM stage diagnostic | Previously validated v0.26.0; set `VLLM_ENABLE_V1_MULTIPROCESSING=0`; DP1, EP1 or EP=TP, no prefix cache/wave; supplied chunk size must equal scheduler token budget |

Continuous orchestration currently requires fixed batch, zero jitter, seed 42,
Graphs on, and prefix caching off. Native admission waves are supported.
Report unsupported contracts or measurement failures; do not disable async,
fall back to step-sync, or reduce measured work to manufacture a successful run.

For vLLM-compatible forks, use `--engine-kwargs @config.json` and `--backend-label`.
Options cannot override entrypoint-controlled model, capacity, seed, prefix cache,
or timing. Set `--sparse-method` to the actual algorithm and record the exact
fork revision and resolved configuration.

<a id="artifacts"></a>

## Acceptance, Artifacts, and Troubleshooting

Inspect terminal state and raw evidence, not just printed throughput:

| Mode | Required checks |
| --- | --- |
| Default request probe | `run_status.json`, `run_manifest.json`, and `summary.json` report `success`; complete valid `raw_samples.jsonl` and `request_samples.jsonl` |
| Continuous decode | `run_status.json` reports `completed`; every `performance.jsonl` row reports `success`; validate `repetitions[].decode_window`, completed-step records and full outputs, not the request-mode file checklist |
| Unified suite | Additionally require successful `suite_status.json`, matched source-ID coverage and sample counts |

Request mode also saves `comparison_report.md`, `case_hardware/*.json`, and, when
applicable, `operator_runtime_stats.json` for provider bindings and actual paths.
Reaggregate with `python3 benchmark/efficiency/metrics.py <RUN_DIR>/request_samples.jsonl`:
no CUDA, JSON to stdout, no artifact overwrite; missing fields or failed samples
raise errors. Verify missing `timing_source` boundaries rather than guessing.
Batch-only aggregates cannot recover request quantiles.

Save commands, configuration, Git commit and dirty status, dependencies, model, trace
hash, GPU/topology, and failures. Separate raw outputs, repetitions, and aggregates.
Unless explicitly requested, do not copy/archive source, save worktree patches,
generate per-file source fingerprints, or require source-hash equality for
execution, resumption, or reuse. Untracked source alone must not block a run.
Keep configuration, input-data, model, and result validation. Experimenters decide
whether code changes require remeasurement.
A dirty flag does not make uncommitted source reconstructible from the commit alone.
Aggregate throughput as total tokens / total time and retain dispersion.
Maximum concurrency requires an integer boundary and max+1 capacity failure;
an arbitrary crash is not capacity evidence. Multi-method queues preserve
method-local smoke/measurement failures and continue
other methods, but retain a failed final status. GPU contention, lost resources,
storage errors, and user interruption stop the entire queue.
Changed contracts, hardware, or
Graph/backend policies require new baselines, preserving the old data.

Keep scripts, reusable configurations, and plotting code under
`scripts/official_experiments/<experiment>/`. Version per-repeat measurements,
replot JSON/CSV, resolved configurations, and raw-data checksums separately in the
project's Research-Vault data directory. Pass data locations explicitly to the
runner and plotter; repo code plus the data bundle must support replotting.
Keep large raw outputs/logs on persistent data storage, indexed by the data bundle
and Research-Vault records. Do not store the experiment in tmp or overwrite old runs.

| Symptom | Action |
| --- | --- |
| Busy GPU | Wait or choose another idle device; do not terminate other users' processes |
| Missing dependencies/model | Check the activated environment, imports, model config, and access |
| OOM | Preserve failure; do not silently lower batch/length in a fixed matrix; follow the declared capacity-search policy |
| Output collision | Choose a new directory, preserving historical runs |
| Window/Graph/async validation fails | Keep logs and diagnose the adapter; do not substitute another synchronization protocol |
| Hardware sampling or Nsight permissions fail | Check sample JSON, tools, and permissions; coarse activity is not a replacement for counters |
