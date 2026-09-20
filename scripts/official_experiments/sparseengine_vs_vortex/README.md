# SparseEngine vs Vortex

Paper comparison recipes, measurement adapters, recorded JSON data, and the
combined figure. `official_experiments` means experiments maintained by this
project, not endorsement by Vortex's authors or official algorithm parity.

## What lives here

- [`session/`](session/README.md): six entry points plus one campaign config for
  Tangram/HiSparse orchestration, paired LongBench v2 quality, raw validation and
  efficiency-plus-quality figures. Historical H2O/GLM debugging recipes are
  archived with their runs rather than maintained as additional entry points.
- `configs/cases.json`: portable command/config templates for Qwen3-4B and
  GLM-4.7-Flash, including common-concurrency QuEST, H2O/H2O-like, native H2O
  wave admission, and a native smoke case.
- `prepare_runs.py`, `run_queue.py`, `gpu_guard.py`: resolve paths and run bounded
  single-GPU queues with task-owned GPU guards. Preparation itself runs no inference.
- `vortex_stage.py`, `vortex_server.py`, `decode_observer.py`: baseline server
  lifecycle and scheduler timing observation. They do not implement sparse methods.
- `summarize.py`, `export_plot_data.py`: validate raw artifacts and export plot data.
- `plot.py`: Matplotlib + Seaborn, one combined figure, automatic layout, no title,
  2.5:1 aspect ratio, arithmetic means with sample-standard-deviation error bars.
- `capture_sources.py`: record Git versions/dirty status, model JSON
  identities, and GPU inventory without copying model weights.
- `data/decode.json`: the default pure-decode figure data; `data/e2e.json` keeps
  the earlier end-to-end output metric separate. `data/decode_summary.json`
  retains the 30 measured windows; `data/provenance.json` identifies measured code.

SparseEngine continues to use `benchmark/efficiency/bench_probe.py`. Vortex uses
its own canonical probe. All decode-window arithmetic is owned by SparseEngine's
`benchmark/efficiency/metrics.py`; the observer supplies events to that collector.
The old `benchmark/efficiency/plot_vortex_comparison.py` CLI delegates to `plot.py`.
Actual vFlow algorithms remain in the prepared external Vortex repository;
this directory adds neither vFlow operators nor algorithm lifecycle interfaces.

## Scope of the recorded comparison

H100, TP1, BF16, 32,768 input / 512 output tokens, seed 42, prefix caching off,
one warmup and three measured iterations. Common concurrency is matched within
each model/method pair. H2O uses native logits scoring with window 128, prefill
budget 8192, decode budget 4096, and eviction interval 128. The checked-in config
is authoritative for all remaining parameters, including Vortex's adaptation.

The pure-decode window begins only after full target residency and completed
prefill/admission, discards eight full decode steps, and measures 256 steps.
Only the boundaries synchronize CUDA. Timing includes scheduler/control work,
scoring and eviction within the window; it is not isolated kernel time. Each
window must have 256 Graph replays, no captures, no eager decode and no prefill.
Vortex overlap remains enabled; its probe prepares each measured trace separately.
The large native H2O cases use wave admission; admission time is excluded from
this window metric, but not from end-to-end workload time.

Vortex H2O-like lacks native H2O's prompt statistics, permanent exclusion and
physical compaction. QuEST head policies also differ. These results do not prove
equal quality, official algorithm parity, or that every method is faster in one
engine. Tested concurrency bounds are not exhaustive maximum-capacity searches.
Request latency in these runs is boundary-synchronization-perturbed; do not label
it ordinary request latency or confuse end-to-end output throughput with decode.

## Redraw without models or GPUs

Run from the repository root. Set `NATIVE_ENV` to the prepared Conda environment
(Matplotlib >= 3.7, Seaborn >= 0.13, and pandas), and `FIGURE_DIR` to a fresh
external output directory:

```bash
conda run -p "$NATIVE_ENV" python scripts/official_experiments/sparseengine_vs_vortex/plot.py \
  --output-dir "$FIGURE_DIR"
```

Outputs: `comparison.png`, `comparison.pdf`, `comparison.svg`, and
`plot_manifest.json` (data hashes, versions and dimensions). Pass
`--data scripts/official_experiments/sparseengine_vs_vortex/data/e2e.json` explicitly
to redraw the separate E2E measurement; never combine the two metrics.

## Prepare and run the recorded decode recipe

Prerequisites: this checkout's decode-window probe, two compatible prepared
Conda environments, model directories under `MODEL_ROOT`, and the prepared
Vortex fork under `VORTEX_REPO`. `VORTEX_OVERLAY` must contain the recorded
FlashInfer overlay's `flashinfer` and `nvidia_cutlass_dsl/dsl_packages` directories.
The configured `cubins` location is a runtime artifact cache, not an obligatory
pre-existing input; missing required runtime artifacts must still fail explicitly.
Use a Vortex checkout compatible with these interfaces and run a smoke before
measurement. Preparation does not compare source hashes with historical runs.

The recorded engines included source changes beyond their base commits.
`data/provenance.json` retains source/archive hashes; the referenced source
archives and large raw runs are external, not bundled here. A base SHA alone
does not reproduce the measured fork. The relocated launch scripts have CPU and
offline-artifact validation, not a new GPU rerun.

Set all paths and select currently idle physical devices. `RUN_ROOT` must not
exist. This command only writes a plan and configs; it does not reserve a GPU:

```bash
python3 scripts/official_experiments/sparseengine_vs_vortex/prepare_runs.py \
  --model-root "$MODEL_ROOT" --vortex-repo "$VORTEX_REPO" \
  --vortex-overlay "$VORTEX_OVERLAY" --native-env "$NATIVE_ENV" \
  --vortex-env "$VORTEX_ENV" --output-root "$RUN_ROOT" \
  --qwen-gpu "$QWEN_GPU" --glm-gpu "$GLM_GPU"
```

Use `--conda` when Conda is not on PATH and `--cuda-home` for the toolkit location.
`--models qwen` or `--models glm` prepares only that model. Port overrides are
available in `--help`. The exact paper concurrency is fixed: insufficient
capacity is an error, not permission to silently lower the batch size.

Before launching, record Git versions and model identity in a fresh external directory:

```bash
python3 scripts/official_experiments/sparseengine_vs_vortex/capture_sources.py \
  --vortex-repo "$VORTEX_REPO" --model-root "$MODEL_ROOT" \
  --output-dir "$RUN_ROOT/provenance_start"
```

`execution_plan.json` lists each queue's argv. Run each queue in its own durable
terminal/tmux session, retaining stdout/stderr in an external log. For example:

```bash
python3 scripts/official_experiments/sparseengine_vs_vortex/run_queue.py \
  --root "$RUN_ROOT/qwen" --gpu "$QWEN_GPU" --native-env "$NATIVE_ENV"
```

The guard refuses busy devices and retains its allocation between jobs. Foreign
contention stops only this task; it never kills another user's process. Use a
fresh root after failure. Each prepared queue stops and releases its guard after
draining; Ctrl-C requests cleanup of the queue's own children. Inspect
`guard.log`, each case's `run.log` / `exit.json`, and any `contention.json` on failure.

## Validate and export a completed full run

```bash
python3 scripts/official_experiments/sparseengine_vs_vortex/summarize.py \
  --root "$RUN_ROOT" --output "$RUN_ROOT/summary_rebuilt.json"
python3 scripts/official_experiments/sparseengine_vs_vortex/export_plot_data.py \
  --summary "$RUN_ROOT/summary_rebuilt.json" \
  --template scripts/official_experiments/sparseengine_vs_vortex/data/decode.json \
  --output "$RUN_ROOT/plot_data.json"
```

The validator requires complete per-request/probe artifacts, matched traces and
valid replay-only windows; bundled plot JSON is not a replacement for those raw
checks. A single-model run needs a matching case manifest passed via `--cases`
to the validator. Record relevant checkout changes when interpreting a rerun.
Recorded JSON source paths are relative to their original external run roots.
Keep raw logs, caches, model files and newly rendered figures outside Git.
