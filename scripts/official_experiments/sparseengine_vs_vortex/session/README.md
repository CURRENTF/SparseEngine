# Efficiency and paired quality campaign

Six scripts, one declarative `campaign.json`. Numerical settings and the final
arm matrix live in that config; there is no build/align/repair/finalize chain.
Date/retry suffixes in arm IDs identify existing results, not required retries.

| Script | Responsibility |
| --- | --- |
| `prepare_cohort.py` | Common untruncated medium cohort and exclusion audit |
| `build_campaign.py` | Select a checkout; generate all final quality and new 32K jobs once |
| `run_quality.py` | Longest-context smoke, then the complete paired cohort |
| `serve_quality.py` | Own an external server, readiness deadline, quality run and cleanup |
| `summarize_campaign.py` | Per-sample quality aggregation and raw 32K/128K stage validation |
| `plot_comparison.py` | Three efficiency figures with quality columns and provenance |

Common code stays shared: `../run_queue.py` and `../gpu_guard.py` own guarded
execution; `benchmark/long_bench_v2/` owns scoring; `benchmark/microbench.py`
owns stage timing. `../../sparse_decode_efficiency/` owns the 128K sweep and
validator; `../prepare_runs.py` owns the original boundary-window experiment.
Historical H2O fusion/MLA diagnostics and retry scripts remain only in their
run archives; reproducing those debugging stages uses archived code/configs.

## Prepare and generate

Run from the repository root with an activated native environment or Conda.
Models, datasets, cohorts, caches and outputs stay outside the checkout.

```bash
EXP=scripts/official_experiments/sparseengine_vs_vortex/session
conda run -p "$NATIVE_ENV" python "$EXP/prepare_cohort.py" \
  --model-root "$MODEL_ROOT" --data "$DATASET" --output "$PREPARED_DIR"
python3 "$EXP/build_campaign.py" --paths "$PATHS_JSON" \
  --root "$RUN_ROOT" --prepared "$PREPARED_DIR" --gpus "$GPU_IDS"
```

`PATHS_JSON` is a caller-owned JSON object of absolute paths. Required keys:
`model_root`, `dataset`, `conda`, `native_env`, `vllm_env`, `vortex_env`,
`tangram_env`, `hisparse_env`, `vortex_repo`, `vortex_overlay`, `cuda_home`,
`scratch_root`. Native/vLLM/Vortex use Conda; Tangram/HiSparse use prepared
venvs. Keep the scratch path short for Unix sockets. Baselines require the
compatible prepared forks/overlay, not arbitrary upstream installations; retain
their Git version/dirty status and environment identity with the run.

Generation launches no GPU work. Destinations must be fresh.
`--source "$REPO"` selects the checkout to execute directly; the default is
this repository. Git commit/dirty status and resolved configuration are recorded.
Control scripts are also executed from that checkout, without source copies or hash gates.
Outputs include `plan.json`, `campaign.json`, `queued_commands.json`,
individual configs and sealed `queues/<gpu>/jobs/`.

Inspect generated commands and GPU ownership, then launch the parent's guarded
runner in durable tmux sessions, once per selected idle GPU:

```bash
python3 scripts/official_experiments/sparseengine_vs_vortex/run_queue.py \
  --root "$RUN_ROOT/queues/$GPU" --gpu "$GPU" \
  --native-env "$NATIVE_ENV" --conda "$CONDA_EXE"
```

Independent-arm failures are recorded, not silently retried. Use a fresh run
for retries. Consolidation does not resolve MLA OOMs, missing full-residency
measurements or abnormal generated answers.

## Validate and redraw

```bash
conda run -p "$NATIVE_ENV" python "$EXP/summarize_campaign.py" \
  --root "$RUN_ROOT" --prepared "$PREPARED_DIR" \
  --efficiency128 "$CAPACITY_PLOT_DATA" --output "$SUMMARY_DIR"
conda run -p "$NATIVE_ENV" python "$EXP/plot_comparison.py" \
  --root "$RUN_ROOT" --summary-dir "$SUMMARY_DIR" \
  --old-data "$BOUNDARY_PLOT_DATA" --sm-data "$SM_COMPARISON_DATA" \
  --output "$FIGURE_DIR"
```

Summary/figure destinations must be fresh. The summarizer defaults to
`ROOT/prepared_v2` and `ROOT/efficiency128/plot_data.json` for historical runs.
`--watch` polls artifacts for at most 12 hours; it does not repair failures or
diagnose anomalies. Validation/redraw needs no GPU inference.

The preview retains the recorded matrix and common 198-sample medium subset,
not the full official medium split. Parse failures remain incorrect answers;
missing results are not zeroes. Keep 256-step boundary-only windows separate
from per-step-synchronized full-batch rates. HiSparse fixed-ratio quality is not
fixed-token-budget matched; GLM quality is TP1 while 128K efficiency is TP2/EP2.
MLA H2O is approximate; Tangram/HiSparse MLA support remains unavailable.
