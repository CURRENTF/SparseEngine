# Sparse decode efficiency comparison

Keep scripts and reusable configs in this package. Measurements, resolved run
configs, and plot inputs live in a separate Research-Vault project data directory.
Set `DECODE_DATA_ROOT` to that directory and pass it explicitly to the commands
below. Use a new subdirectory for each run/export; preserve historical results.
Generated figures and large raw outputs belong in the persistent run directory.
Run-specific configuration variants and plot selections live under
`$DECODE_DATA_ROOT/configs/`; pass their full paths with `--config` or
`--grid-config`. The current base template is `config.boundary-sync.json`.
The legacy `config.json`, `config.omnikv-total2048.json`, and
`profile_ablation.json` remain here for the existing step-sync and ablation runners.
One-off queue, point-refresh, and device-relocation wrappers are archived under
`$DECODE_DATA_ROOT/scripts/` with their experiment records. Invoke those scripts by
full path; reusable preparation, sweep, validation, and plotting remain here.

## README figures

The repository READMEs reuse three archived 128K-only figures, in this order:

| View | Archived figure | README asset under `docs/assets/` |
| --- | --- | --- |
| Largest measured batch | `decode_capacity_128k_lowbs_max_batch.png` | `sparse_engine_throughput.png` |
| Absolute throughput lines | `decode_capacity_128k_lowbs.png` | `sparse_decode_efficiency_lowbs.png` |
| Relative throughput lines | `decode_capacity_128k_lowbs_delta_vllm.png` | `sparse_decode_efficiency_relative_vllm.png` |

SVLLM is the SparseEngine label retained in these figures.

All use H100 80GB, 128K input / 2K output and `boundary_sync_v2`: 32 warmup
steps followed by a continuous 256-step full-residency decode window, with one
discarded and three measured workloads. Rates pool measured decode tokens and
window time; they are not end-to-end serving throughput. Bars select each
method's largest measured batch, which need not be its peak throughput or a
verified capacity maximum. Absolute lines display Qwen through BS4 and GLM
through BS8. Relative lines show `100 * (method throughput / vLLM Vanilla
throughput - 1)` at matching measured concurrency, with vLLM at 0%. Points without
a matching baseline are omitted, so Qwen extends through BS3 and GLM through BS6.

The GLM Vanilla and OmniKV BS6 points use memory utilization 0.92; preceding
points and the vLLM baseline use 0.90. This supplement does not establish a
capacity maximum at 0.92. Framework-specific selection and cache policies
remain distinct; these figures do not establish equal quality.

## Framework color families

The default `palettes/framework_families.json` keeps a fresh, modern palette.
Each framework has a distinct color family: SENGINE uses related teal shades,
vLLM blue, Tangram coral, Vortex lavender, and HiSparse warm gold. Methods
within SENGINE retain distinct markers and explicit framework/method labels.
Reuse this family assignment when adding methods; preserve archived palettes
when reproducing historical figures.

## Relative throughput lines

Use `--line-metric relative-vllm` to plot
`100 * (method throughput / vLLM Vanilla throughput - 1)` on continuous axes.
Zero is vLLM Vanilla, positive values are improvements, and negative values
are regressions. Pair only exact measured concurrency within the same panel
and input/output/timing protocol; never interpolate or extrapolate a missing
baseline. The derived `delta_points.json` and CSV retain numerator, denominator,
source artifacts and explicit exclusion reasons. Original throughput data remain
unchanged. Relative views use linear percentage y axes and no maximum-batch bars.

```bash
python scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py \
  --plot-data "$DECODE_DATA_ROOT/<validated-export>/plot_data.json" \
  --line-metric relative-vllm \
  --output-dir "$DECODE_OUTPUT_ROOT/delta" \
  --export-data-dir "$DECODE_DATA_ROOT/<new-delta-export>"
```

The broken-axis `--batch-bands` layout is **deprecated**. Retain archived figures
for historical reproduction only; new renderings display a deprecation banner
and emit `DEPRECATED.md`. Prefer continuous low-concurrency panels or the relative
throughput view. Absolute throughput panels and largest-observed-batch bars
remain supported.

## Low-batch lines and maximum-batch bars

Use the same validated portable export for both views; no measurement rerun is
needed. The presentation config selects 128K panels, limits the Qwen line panel
to BS4 and GLM to BS8, and adds model/topology captions below each panel.

```bash
python scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py \
  --plot-data "$DECODE_DATA_ROOT/<validated-export>/plot_data.json" \
  --presentation-config scripts/official_experiments/sparse_decode_efficiency/presentation.128k.json \
  --output-dir "$DECODE_OUTPUT_ROOT/plots" \
  --export-data-dir "$DECODE_DATA_ROOT/<new-export>"
```

The JSON retains every measured batch in the selected panels; only the line
display is restricted. A separate 1×2 bar figure takes each method's throughput
at its largest validated batch, not the highest throughput across batches.
Capacity verification status is retained in the exported data, without markers
or footnotes on the figure. Bars use the paper/whitegrid theme from
`sparseengine_vs_vortex/plot.py`, borderless fills, shared y limits, and direct
framework/method/throughput/batch labels above each bar, without a legend. This borrows visual styling only, retaining the recorded
pooled token/time rates rather than substituting a mean or error bars.
Bars start at zero; line panels start at their visible
minimum minus 20 tok/s. Both plots use the same method colors. Normal/log-y lines,
linear bars, PNG/PDF/SVG files, and selected bar data are exported. Replotting the
new `plot_data.json` preserves this layout without the presentation argument.

## Boundary-sync rerun

For a context-length scan with request metrics at each method's verified maximum,
use `run_context_capacity.py --config CONFIG --repo CHECKOUT --gpus GPU_IDS`.
The configuration extends the boundary-sync template with `model_id`, `input_lens`
and `lanes`. It runs the existing integer capacity search for each length, then
the canonical request probe at every successfully verified maximum. The output
root must be new; failed or incomplete capacity searches do not produce a maximum.
Request failures remain separate from the successful decode capacity evidence.
Configuration paths use the same environment-variable expansion as the sweep.
Request probes reuse each measured case's resolved token budget; omitting the
campaign override retains the sweep default. Tangram request probes use its
configured environment and fork arguments. HiSparse/Vortex currently have only
decode-only adapters and are rejected by this entrypoint before a campaign starts.
Request timeouts stop the affected process and continue to the remaining cases,
with a failed final queue status. Loss of the GPU reservation or external
contention aborts the campaign and prevents acceptance of the affected result.

To measure one conservative KV-slot estimate instead of searching, provide
`conservative_candidates_by_length`, keyed by input length and lane. Each entry
contains `concurrency`, a validated BS1 `reference_artifact`, and the slot-budget
calculation evidence. The reference replaces an extra smoke; the candidate still
must pass full measurement validation. Each lane runs one candidate, with no
binary search, max+1 probe, or automatic retry. Results report
`verified_concurrency`, `maximum_verified: false`, and `max_concurrency: null`.

Set `max_num_batched_tokens` and `engine_prefill_chunk_size` explicitly to override
the sweep's historical 8192-token defaults. `native_admission` maps method names
to `wave_size` and `decode_gap_steps`; `wave_size: 0` disables staged admission.
Without an override, the historical SnapKV wave size/gap of 1 remains unchanged.
H2O is available as `sengine-h2o`, with its explicit parameters in `methods.h2o`.
Zero jitter request traces now preserve the exact requested length at every batch
size (`random-varlen-v2`); old multi-request traces could vary by one token.

Use `config.boundary-sync.json` for new paper curves. It routes the existing
capacity sweep through `benchmark/efficiency/bench_probe.py`, which reuses the
microbench adapters. Historical configurations below retain step-sync semantics;
their measurements are not reusable in this rerun.

Protocol: 131072/2048 tokens, greedy/ignore-EOS, zero jitter, seed 42, Graph on,
prefix cache off, memory fraction 0.9. Drain async/overlap work only at the two
edges of a 256-step full-batch window after 32 full-batch warmup steps. Complete
all 2048 output tokens. Run one discarded workload and three measured workloads,
each in a fresh engine; aggregate total tokens / total time and retain per-run
dispersion. A discarded workload primes persistent compilation caches, not a
shared live engine. Native rank boundaries support TP/EP; external compatibility
remains model/version-specific and must pass smoke before the full sweep.

Budgets: SnapKV total 8192 with native wave size/gap 1; QuEST and OmniKV total
2048, including 64 sink + 512 recent, with OmniKV auto full layers. Tangram uses
4096-token prefill chunks with matching `max_num_batched_tokens`; other lanes
use 8192. The pinned Tangram path fails block compaction for the 8192-token
budget with 2048-token chunks on the Qwen FP8 configuration. HiSparse QuEST uses its pinned
96/8160 page-selection ratio plus 32 recent pages, approximately 2048 at 128K,
and grows during decode. Do not describe it as a fixed token budget.

If baseline admission leaves too few full-batch steps, the run fails rather
than truncating the window. An explicitly approved new config can set
`curve_protocols[model][lane]` with `input_len`, `output_len`, `reason` and
`label_suffix`, preserving input+output total span. The plot labels this exception;
the 32/256 decode warmup/window and complete-output requirement remain unchanged.
No length adjustment is enabled by default.

For a bounded length-transfer trial, pass `--probe-concurrency B --probe-only`
to the canonical sweep with a separate config/attempt. Small transfers do not
guarantee capacity; retain measured context lengths and never merge adjusted
points into the original curve. `run_lanes.sh` accepts additional sweep arguments
after its seven positional arguments and `--` (use an empty wait-status argument
when no queue dependency is needed).

For the complete two-model campaign, `run_boundary_sync.py` records the current
checkout path and Git version, resolves the config, and checks paths.
Set the environment variables below, then prepare a new run:

```bash
python scripts/official_experiments/sparse_decode_efficiency/run_boundary_sync.py \
  --run-root "$DECODE_OUTPUT_ROOT" \
  --data-root "$DECODE_DATA_ROOT/<new-run>"
```

Run `scripts/official_experiments/sparse_decode_efficiency/run_boundary_sync.py`
under tmux with `--run-root "$DECODE_OUTPUT_ROOT" --run`. It starts separate
idle-GPU queues for Qwen TP1 and GLM TP2/EP2, each with fresh per-lane smoke,
and exports plots only after both capacity sweeps succeed. Inspect `status.tsv`,
`<model>.run.log`, and each campaign's case logs for progress or failures.

Set the path variables listed below, plus `TANGRAM_ENV` and `HISPARSE_ENV` for
the pinned external virtual environments. Keep `DECODE_SCRATCH_ROOT` short
(under 60 bytes) for Unix-domain socket paths. Use a new `DECODE_OUTPUT_ROOT`.

```bash
python scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py \
  --config scripts/official_experiments/sparse_decode_efficiency/config.boundary-sync.json \
  --repo "$BENCHMARK_REPO" --model qwen3-30b-fp8 --gpus auto:1 \
  --lanes sengine-vanilla,vllm-vanilla,sengine-snapkv,sengine-quest,sengine-omnikv,tangram-snapkv,hisparse-quest \
  --smoke-only --attempt smoke1
```

The smoke is 4096/64 tokens, B2, four warmup steps and 16 measured steps,
one measured workload. It is not a paper point. Successful and failed smoke
records are saved as `smoke.json` in each lane's raw smoke run directory,
regardless of `--export-measurements-dir`; keep them with the experiment archive.
After validation, omit `--smoke-only` and use a new attempt name to run the full
capacity search. Add `--export-measurements-dir` pointing to a new measurements
directory under `$DECODE_DATA_ROOT`: each non-smoke measurement is exported
immediately, even if a later point fails.
Run GLM separately with `--model glm4.7-flash --gpus auto:2` and the five native/
vLLM lanes (default `--lanes`). External MLA combinations remain unsupported
metadata and are omitted from plots. Full sweeps should run in tmux.

After a classification-only repair, native SnapKV boundary-sync sweeps may use
`--reuse-cases-from PREVIOUS_ATTEMPT_ROOT` with a new attempt and selected checkout
manifest. A fresh smoke still runs. Reuse requires identical campaign, GPU,
command (except artifact paths), hyperparameters and model config.
Source-file equality is not checked; decide whether code changes require remeasurement.
Successful points are checked
against raw repetitions again. Capacity failures need explicit evidence, including
the exact skipped batch/path for a missing graph; unrelated failures stop the sweep.
Old artifacts remain unchanged, and the new attempt records reuse hashes.
For an explicitly requested device move, `--reuse-equivalent-gpus` compares GPU
model, total memory, compute capability, driver, MIG mode, power limit and maximum
clocks, and records both assignments and the comparison. A mismatch fails. This
permits mixed physical-device provenance on equivalent hardware, not a claim of
identical timing or measured capacity on the replacement device; original points
retain their actual source GPU and raw evidence.
Use `--hold-reservation-seconds N` (at most 86400) to retain the guarded GPU
allocation after completion or a non-contention failure. Explicitly release it
by terminating the owning sweep. This is visible occupancy, not an exclusive
device lock; foreign contention still invalidates the run. When holding, pass
the sweep's own `status.tsv` to the plot waiter, since its queue reaches a terminal
state before the wrapper process exits.
To avoid releasing the device between queues, `--handoff-reservation-pid PID`
temporarily recognizes an existing same-user guard. Pause only that old guard's
monitoring and stop its benchmark workers first; retain its CUDA allocation.
After the new guard reports ready, terminate/resume the old guard so it exits.
The new sweep waits for that exit before starting work. Handoff is bounded to
60 seconds and verifies process ownership and PID identity; never pass a foreign
process or leave the old guard suspended after the transfer.

Multi-lane sweeps isolate explicit benchmark/measurement failures: a failed
smoke skips that method; a failed formal case preserves prior points and stops
only that method's scan. Other methods continue under the same reservation.
`lane_failure.json`, failed `capacity.json`, and `queue_summary.json` preserve
failures; the final queue status and exit code remain failed/nonzero. Capacity
failures still refine the boundary. Infrastructure, artifact-write errors,
reservation loss/contention, and user interruption stop the whole queue.

For an already-running legacy fail-fast sweep, `continue_failed_queue.py --help`
describes a one-shot sidecar. It waits for an explicitly classified method error,
then starts only untouched methods with fresh smokes and the same benchmark
checkout/config. It requires an idle held reservation and uses verified Linux
pidfds for handoff; it never retries failed/started methods or mutates old source.
Include the sidecar's terminal status in plot dependencies. A failed curve still
requires explicit partial-figure policy; continuation does not fabricate a
complete capacity result.

Relocated orchestration loads and preflights statistics from the explicit
`--repo` before acquiring GPUs, not from the orchestration directory's parents.
For an explicitly scheduled next lane after a successful held queue, the sidecar
supports `--explicit-followup`. Missing boundary points can share one reservation
using `--probe-concurrency N --additional-probe-concurrency M --probe-only`.
`--complete-from-capacity PATH --completion-note TEXT` revalidates prior raw
points and assembles a completed curve only when the probes establish all required
powers and an integer maximum/max+1. It preserves source/protocol provenance;
unclassified failures, duplicate points and unresolved boundaries remain errors.
Use `--allow-failed-methods` with explicit follow-up to continue after a typed,
persisted lane-failure summary. The failed predecessor remains failed; contention,
resource loss, user interruption and missing/mismatched failure evidence still
block handoff. This does not retry failed methods.

Only after every supported sweep has a verified integer maximum and max+1
capacity failure, export to a **new external data directory**:

```bash
python scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py \
  --config "$DECODE_OUTPUT_ROOT/<model>/<attempt>/campaign.json" \
  --output-dir "$DECODE_OUTPUT_ROOT/plots" \
  --export-data-dir "$DECODE_DATA_ROOT/boundary-sync-v2"
```

The exporter checks every repetition against raw completed work, full outputs,
context progression, all-rank Graph counters, and its window clock. It saves
`plot_data.json`, `points.csv`, palette and source provenance into the data directory.
Source provenance records Git commit and dirty status, not per-file code hashes.
Legacy source fingerprints are omitted on export. Raw-data checksums remain;
the legacy per-repetition `source_sha256` field refers to measurement artifacts,
not source code. A dirty commit alone cannot reconstruct uncommitted changes.
`plot_data.json` includes per-repeat tokens/time/context windows and artifact
hashes. Retain large raw logs/token outputs in the persistent run root; preserve
their index in Research-Vault. Version the portable data bundle in Research-Vault
separately from the experiment code. Replot by passing that JSON to `--plot-data`;
no model, GPU or external artifact path is needed. Normal/log-y figures include
PNG/PDF/SVG and preserve the existing legend, inset, font and canvas settings.

If the user explicitly stops a sweep and requests observed points only, a plotting
config may declare `partial_curves[model][lane]` with a `capacity_artifact` relative
to `output_root` and a stop `reason`. Optional `theoretical_max_concurrency` is
metadata, never an invented measured point. Exports mark that curve `partial`,
leave the verified maximum/max+1 null, and retain its measured extent and failures.
The default complete-sweep validation is unchanged for all other curves.

Independent queues can use `plot_when_finished.sh CONFIG BENCHMARK_REPO EXPORT_DIR
PLOT_RUN_DIR STATUS_FILES...` under tmux. It waits for terminal queue statuses,
then invokes the same raw validator and plotter; failed or incomplete curves are
not silently accepted. Use a new plot-run/export directory for each attempt.

### 128K / 32K comparison grid

`$DECODE_DATA_ROOT/configs/config.boundary-sync.32k2k.json` retains the boundary-only protocol with a 32768
token prompt. HiSparse's history-page ratio is 96/2016 at this length, plus 32
recent pages; it is approximately 2048 tokens, not a fixed decode budget.
External implementations retain their declared algorithm differences: native
QuEST keeps the first two layers full and explicit sink tokens, while the pinned
HiSparse path has its own per-head page selection; native SnapKV pooling is 1,
Tangram's historical pooling is 7. The separate 32K `matched-external` config sets
Tangram pooling to 1 before formal measurement, matching native SnapKV without
changing either implementation. Equal headline budgets do not prove equal selection or quality.

The full-KV sweep uses the minimum startup KV slots over ranks/repetitions divided
by input+output as a probe hint for Vanilla/QuEST/OmniKV/vLLM. It still measures
the curve and verifies the integer boundary. SnapKV wave admission needs its own
peak/residency checks and is not estimated with this full-KV formula.

`$DECODE_DATA_ROOT/configs/config.grid-128k32k.json` selects exact attempt artifacts in row-major order:
128K Qwen/GLM above 32K Qwen/GLM. Point each source variable to its resolved run
config. This only validates/plots existing results; it does not launch inference.

```bash
export DECODE128_CONFIG="$DECODE128_ROOT/config.json"
export DECODE32_CONFIG="$DECODE32_ROOT/config.json"
export DECODE32_MATCHED_CONFIG="$DECODE32_ROOT/config.matched-external.json"
python scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py \
  --grid-config "$DECODE_DATA_ROOT/configs/config.grid-128k32k.json" \
  --output-dir "$FIGURE_DIR" --export-data-dir "$NEW_DATA_EXPORT_DIR"
```

Grid sources may explicitly permit partial curves or missing-data omissions with
reasons. Every accepted point still undergoes full raw validation; invalid JSON
and invalid measurements are errors, not missing data. Each panel requires at
least two formal curves with two points each, including a Vanilla baseline.
Unverified maximums remain null;
no interpolation, smoke results, theoretical endpoints or zero-filled failures
are exported. Partial curves can lack intermediate powers and are not described
as complete capacity sweeps. The grid uses independent panel protocols, shared
four-column union legend and low-concurrency insets, with no extra single-panel
figures. `--plot-data` replots the same grid entirely from the exported JSON.

## Historical step-sync results

This package preserves the experiment recipe. The external data directory retains
all 44 accepted historical points across 10 curves. Original raw outputs remain in
the persistent run archive; historical JSON/CSV uses run-root-relative artifact IDs.

The original export uses OmniKV selected-history budget 2048 (total 2624).
The `config.omnikv-total2048.json` variant aligns OmniKV with QuEST at total
2048: 1472 selected + 64 sink + 512 recent. It retains auto full-attention layers.
The updated 44-point comparison is in `$DECODE_DATA_ROOT/omnikv-total2048/`; the original
`$DECODE_DATA_ROOT/plot_data.json`, CSV, config and provenance are preserved unchanged.

`$DECODE_DATA_ROOT/sm-parallel/` is the context-independent SM-scaled MLA profile update.
Only GLM OmniKV is remeasured; the other nine curves are preserved exactly.
It includes portable plot data, actual per-rank launch plans and comparison
data. Both previous exports remain available unchanged.

`$DECODE_DATA_ROOT/tangram-hisparse/` extends that comparison with Tangram SnapKV and the
HiSparse QuEST PR-series on Qwen3-30B FP8. Their GLM MLA combinations are explicit
N/A entries. The original 44 points remain unchanged and were revalidated from
raw steps and complete outputs before export. The separate Triton MLA SM rule
does not change these original attention paths. See
[`../triton_mla_sm_schedule/`](../triton_mla_sm_schedule/) for the external stage
adapters, their timing/algorithm differences, tuning recipe and validation.

## Contents

- `sweep_decode_capacity.py`: idle-GPU guarded sweep and integer boundary search,
  calling the selected checkout's canonical `benchmark/microbench.py`.
- `decode_capacity_guard.py`: keeps the reservation between cases; foreign GPU
  contention aborts this queue without terminating foreign processes.
- `config.json`: measured model topologies, method parameters, and runtime path
  placeholders. Resolved configuration is saved as `campaign.json` in each attempt.
- `plot_decode_capacity.py`, `palettes/framework_families.json`: standalone Matplotlib /
  Seaborn plotting with native constrained layout and editable method colors.
- `$DECODE_DATA_ROOT/plot_data.json`: authoritative replot input, including capacity attempts,
  stage sums, all accepted points, and configuration; `$DECODE_DATA_ROOT/points.csv` is its flat
  export. `$DECODE_DATA_ROOT/environment.json` records the measured software/hardware environment.
- `$DECODE_DATA_ROOT/provenance.json`: original source and artifact identity; no private paths.

## Replot without models or GPUs

Run from the repository root using an environment with Matplotlib and Seaborn
(measured plotting versions: Matplotlib 3.11.1, Seaborn 0.13.2):

```bash
python scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py \
  --plot-data "$DECODE_DATA_ROOT/tangram-hisparse/plot_data.json" \
  --output-dir "$DECODE_OUTPUT_ROOT/replot"
```

This creates `decode_capacity_128k2k` and `decode_capacity_128k2k_logy` in PNG,
PDF, and SVG, plus per-model figures. Both use a base-2 concurrency axis; y is
linear or logarithmic respectively. Override colors with `--palette FILE.json`.
Figures omit titles and protocol annotations for use with an external caption.
Overview canvases are 8.75 × 3.5 inches; per-model canvases are 5 × 3.625 inches.
Canvas width/height and marker sizes are scaled by 1.25 from the previous layout;
line widths and other style parameters are unchanged.
Font sizes remain unchanged; the y label is `Throughput (tok/s)`.
Legends use fixed columns (4 in the overview,
2 per model), with explicit `Tangram (SnapKV)` and `HiSparse (QuEST)` labels,
not automatic width-based wrapping; constrained layout positions the artists.
Each panel includes a native inset of observed concurrency <= 6 points, using
linear inset axes in both overview variants and matching curve colors/markers.
The inset introduces no additional samples; its bounds and connectors mark the
visible zoom region. Unsupported combinations remain in the exported metadata
but have no curve or N/A annotation in the figures.
Export validation checks completeness and token/time arithmetic; it cannot
revalidate raw steps or token outputs that are not bundled in the portable export.

## Measurement contract

Input is 131072 tokens; every request completes 2048 output tokens. Qwen3-30B-A3B-
Instruct-2507-FP8 uses TP1/EP1; GLM-4.7-Flash BF16 uses TP2/EP2, **not DP2**.
Both use H100 80GB and memory utilization 0.9. The five lanes are native vanilla,
vLLM vanilla, SnapKV, QuEST, and OmniKV.

- SnapKV retains 8192 prompt tokens (64 sink + 512 recent + 7616 selected), with
  wave size 1 and decode gap 1 for admission.
- QuEST uses total budget 2048 (64 sink + 512 recent + 1472 selected).
- OmniKV uses auto full-attention layers and 2048 selected tokens in addition to
  64 sink + 512 recent tokens. Its budget convention differs from QuEST's.
- Pure decode throughput is actual computed decode tokens divided by accumulated
  synchronized full-batch engine-step time, including engine bookkeeping. Exclude
  mixed/prefill steps, admission, the first 32 full-batch decode steps, and tails
  after batch occupancy falls. This is not request TPOT or end-to-end throughput.
- Sweep powers of two, then binary-search the integer maximum and verify failure
  at maximum + 1. Preserve successful boundary-search points and failed attempts.
  Maxima are conditional on this configuration, not theoretical device limits.
- One workload per concurrency; no repeated-run confidence intervals or smoothing.

## Run a new sweep

GPU execution needs a compatible benchmark checkout, explicitly selected by
`--repo`. The repository's canonical microbench supports synchronized full-batch
stage adapters for both engines, and includes the SnapKV bootstrap capacity
repair. Set `BENCHMARK_REPO` to this checkout for new runs. For exact historical
source reproduction, use the archive identified in `$DECODE_DATA_ROOT/provenance.json`; the
recorded base commit alone is insufficient to reconstruct that source.

Set these environment variables to your actual absolute paths:

| Variable | Meaning |
| --- | --- |
| `BENCHMARK_REPO` | Compatible benchmark checkout |
| `CONDA_EXE` | Conda executable |
| `SPARSE_ENGINE_ENV`, `VLLM_ENV` | Native and vLLM conda environment prefixes |
| `QWEN3_MODEL`, `GLM47_MODEL` | Local model directories |
| `DECODE_OUTPUT_ROOT` | New persistent campaign output directory |
| `DECODE_SCRATCH_ROOT` | Scratch directory on a disk with sufficient space |

Alternatively, make an ignored `config.local.json` with literal absolute paths
and pass it instead. Never overwrite the original campaign. The example uses
automatic idle-device selection; explicit GPU indices are also accepted.

```bash
python scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py \
  --config scripts/official_experiments/sparse_decode_efficiency/config.json \
  --repo "$BENCHMARK_REPO" --model qwen3-30b-fp8 --gpus auto:1 --check-only

python scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py \
  --config scripts/official_experiments/sparse_decode_efficiency/config.json \
  --repo "$BENCHMARK_REPO" --model qwen3-30b-fp8 --gpus auto:1 --attempt run1

python scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py \
  --config scripts/official_experiments/sparse_decode_efficiency/config.json \
  --repo "$BENCHMARK_REPO" --model glm4.7-flash --gpus auto:2 --attempt run1
```

Use tmux for the long-running sweep commands. `--check-only` checks paths and
required adapter option presence without touching GPUs; it is not a CUDA
correctness test or a proof of measurement-contract equivalence. The sweep also
validates complete raw outputs and full-batch step sums for every accepted point.

After both models complete, use `plot_decode_capacity.py --config PATH` with an
attempt's resolved `campaign.json` to validate the new campaign's raw artifacts
and regenerate both plots. This requires exactly one completed sweep per
model/lane; incomplete or ambiguous campaigns fail explicitly.

## Rerun OmniKV with the aligned total budget

Use the same measured runtime source and environment as the original campaign;
do not combine new-runtime OmniKV measurements with old-runtime baselines.
Set the same path variables above, but choose a **new** `DECODE_OUTPUT_ROOT`.
Pass `config.omnikv-total2048.json` to the canonical sweep with
`--lanes sengine-omnikv --attempt total2048`, once per model. The existing sweep
runs a fresh smoke, preserves its GPU reservation, and searches through the exact
integer capacity boundary. `run_aligned_omnikv.sh BENCHMARK_REPO RESOLVED_CONFIG
OUTPUT_ROOT` runs these two queues serially under tmux; its config must already
contain absolute runtime paths and its output root must match the third argument.

After both queues finish, merge only the replacement curves and revalidate all
ten curves against raw data:

```bash
python scripts/official_experiments/sparse_decode_efficiency/merge_omnikv_rerun.py \
  --original-root "$ORIGINAL_CAMPAIGN_ROOT" \
  --rerun-root "$DECODE_OUTPUT_ROOT" \
  --output-dir "$DECODE_OUTPUT_ROOT/plots-combined"
```

The merger rejects additional config, model-config, or hyperparameter changes.
It does not check source-file equality. It retains every other method's original numerical
values, validates the rerun's boundary, and writes new PNG/PDF/SVG plus JSON/CSV.
It refuses an existing output directory; previous figures and measurements are
never overwritten.
The `portable/` subdirectory contains path-independent JSON/CSV. Artifact IDs
start with `original/` or `omnikv-total2048/`, identifying the original campaign
root or the rerun campaign root respectively. Replotting does not require either
raw root; full revalidation does.

## MLA split-profile ablation

### Context-independent SM-scaled profile rerun

`sm_parallel_nearest_v1` chooses the closest split count in `[4, 8, 16, 32]`
to `SM_count / (batch * head_tiles)`, with lower splits winning exact ties.
The target is one CTA per SM; context length and capacity do not select splits.
Other launch parameters and provider eligibility remain unchanged. This is a
hardware-scaled heuristic, not cross-GPU performance validation.

Use the original measured runtime with **only** the reviewed MLA runtime and
provider-binding patch. `update_mla_profile_results.py prepare --repo
"$BENCHMARK_REPO" --base-plot "$BASE_PLOT_DATA" --run-root "$PROFILE_RUN_ROOT"
--scratch-root "$DECODE_SCRATCH_ROOT"`
records Git commit/dirty status and config checksums, without copying source or
checking source-file equality. `BASE_PLOT_DATA` is the full-path, raw-validated aligned-budget
export, not its portable copy. Use a short absolute scratch path on the output
volume (at most 65 characters) for multiprocessing Unix sockets.
Set `CONDA_EXE` and `SPARSE_ENGINE_ENV`, then run
`run_sm_parallel_profile.sh "$BENCHMARK_REPO" "$PROFILE_RUN_ROOT"
"$VALIDATION_ROOT" auto:2` under tmux. `VALIDATION_ROOT` comes from the successful
`tilelang_mla_split_profiles/validate_rule.sh` correctness run on the patched
implementation. No additional split search or context benchmark is performed.

The canonical sweep reruns GLM OmniKV to its integer capacity boundary. Export
checks every run's runtime identity, hyperparameters and bound split plans,
revalidates all raw steps/outputs, and retains the other nine curves only after
checking they do not select the changed provider. New linear/log-y PNG/PDF/SVG,
JSON/CSV and actual generated launch plans are written to `PROFILE_RUN_ROOT/export`.
The old aligned-budget and original-budget results are never overwritten.

### Historical fixed-32 control

`profile_ablation.json` and `run_mla_profile_ablation.py` reproduce the GLM
TP2/EP2 BS4/BS5 ABBA control: the existing split table versus fixed 32 splits.
This is **not autotuning**. Other provider choices and method parameters remain
unchanged. The runner reserves an idle GPU pair across numerical checks, smokes,
and all formal cases, then uses the same canonical microbench and raw validator.

The temporary runtime switch is not a supported engine setting. Historical
reproduction requires the original measured source plus the archived
`$DECODE_DATA_ROOT/mla-split-profile-ablation/profile-bypass.patch`; apply it only to an
explicitly selected experimental checkout. The runner rejects an unpatched
checkout and checks the numerical oracle's actual split count before benchmarking.
Remove the patch again after the experiment. Do not apply it to unrelated or
newer runtime code without reviewing the changed contract.

Set `BENCHMARK_REPO`, `CONDA_EXE`, `SPARSE_ENGINE_ENV`, `GLM47_MODEL`,
`DECODE_SCRATCH_ROOT`, and a fresh `ABLATION_OUTPUT_ROOT`, then run under tmux:

```bash
python3 scripts/official_experiments/sparse_decode_efficiency/run_mla_profile_ablation.py \
  --config scripts/official_experiments/sparse_decode_efficiency/profile_ablation.json

python3 scripts/official_experiments/sparse_decode_efficiency/summarize_mla_profile_ablation.py \
  --run-root "$ABLATION_OUTPUT_ROOT" --output-dir "$ABLATION_OUTPUT_ROOT/export"
```

The exporter requires all eight formal runs, validates raw tokens and stage
sums, checks both ranks' launch plans, and compares operator metadata after
excluding split counts. Its portable JSON/CSV retain every repetition; two-run
means are descriptive, not confidence intervals. This focused ablation does not
replace the ten capacity curves or establish a universal split-count default.
# Vortex supplement

`prepare_vortex.py --help` prepares the missing QuEST curves for both input
lengths from a resolved boundary-sync config. It records native and Vortex
checkout versions, writes per-model JSON configs and `jobs.json`, and invokes the existing
capacity sweep with `--run-root`. Model paths, environment, overlay and output
directories are explicit arguments. Run the entry point in tmux from the selected checkout.

The `vortex` engine uses the shared SGLang completed-work window collector,
preserves overlap, and synchronizes the TP group only at the two boundaries.
Every rank must report the same request/context/token window and captured-Graph
counts. The measurement uses rank0 wall time with all ranks completing each edge.
GPU support is established by each job's smoke, not by accepting CLI arguments.

QuEST budgets are 4 sink + 32 recent + 92 selected pages of 16 tokens, with
layers0/1 dense. Vortex's page/head selection differs from native selection;
matched retained-token budgets are not equal-quality evidence. FP8 checkpoint
quantization remains enabled; BF16 refers to activation/KV dtype.
Formal runs use unchanged 32K/128K inputs, 2K outputs and the shared window
defaults. Failed attempts remain explicit; only validated points can enter
`Vortex (QuEST)` curves. No old TP1 32K/512 observations are reused as new points.
