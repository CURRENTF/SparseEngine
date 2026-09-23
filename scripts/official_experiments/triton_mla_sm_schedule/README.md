# Persistent Triton MLA SM scheduling and external decode capacity

This package preserves the tuning recipe and adds Tangram SnapKV / HiSparse
QuEST PR-series curves to the canonical 128K-input / 2K-output capacity chart.
Unsupported model combinations are explicit N/A entries, not zero throughput.
It does not change either external sparse algorithm or claim equal quality.

## MLA rule

`head_tile = min(16, next_power_of_two(local_heads))`,
`target_splits = ceil(SM_count / ceil(local_heads / head_tile))`,
`programs = 4 * SM_count`.

Stage1 loops over requests inside each persistent CTA. Its split estimate does
not divide per-request parallelism by batch. Actual nonempty rows and lengths
produce a shared aligned block size on the GPU, within fixed Graph/workspace
capacities. The production heuristic has no H100-name profile; the existing
upstream-first provider portfolio and separate TileLang overlay are unchanged.
Only H100 has local performance evidence. The manual legacy configuration is
retained as a numerical/benchmark baseline, not a production batch table.

- `prepare.py` records the Git commit and prepares jobs for the existing
  GPU guard/queue, executing directly from `--repo`. Explicit repo, output root,
  conda and environment arguments are required.
- `micro.py --config micro.json --output-dir <NEW_OUTPUT>` compares the frozen
  legacy table with the production selector at 21 representative cases. `tune.json`
  adds the finite head-tile/target-wave diagnostic matrix. Timings include schedule,
  score reset, stage1 and stage2; independent FP32 Torch checks precede timing.
- `prepare_related.py` reuses historical H2O job settings with an explicitly
  prepared `--updated-source` checkout. Apply the intended schedule change there
  before preparing jobs. `summarize_related.py` validates the fixed native H2O
  traces, residency, outputs and windows used by this package.
  These 32K/512 edge-synchronized windows are separate from the 128K/2K stage plot.

## External capacity curves

`external_capacity.json` contains path placeholders and explicit native fork
parameters. `launch_external.py` resolves a new campaign from the original
baseline config and launches the canonical guarded `sweep_decode_capacity.py`
in tmux. Models, environments, scratch directory and GPUs are explicit arguments.
`run_external.sh` is the environment-variable-driven equivalent.
Choose a short persistent scratch path: multiprocessing uses Unix-domain sockets
under that directory and deeply nested paths can exceed the socket path limit.

Measurement still uses `benchmark/microbench.py`: synchronized full-batch stage
steps, actual computed decode tokens, 32 discarded full-batch steps, complete 2048
outputs, powers of two plus integer max/max+1 checks. Raw steps, outputs, configs,
source identities and capacity failures are retained. No failed/incomplete
measurement is replaced by a timing proxy.

- Tangram uses the vLLM stage adapter with an explicit backend label and protected
  constructor options. It retains 8192 tokens, sink64/recent512, observation32/pool7,
  uniform budget, compression/prefill chunk2048. The pinned fork requires at least
  two persistent-score workspace rows even for one submitted request. Its scoring
  granularity and prefill chunk differ from native SnapKV.
- HiSparse uses a scheduler-worker timing adapter, not HTTP. Overlap is disabled
  so scheduling through result processing and CUDA completion form one stage.
  SGLang's two reserved context positions are included in capacity, not workload.
  QuEST selects 96/8160 of history pages plus 32 recent pages: 2048 tokens at 128K,
  increasing slightly during decode. This full-GPU-KV PR implementation is not
  the paper's CPU-offload system. Scheduler-stage timing excludes IPC receive;
  vLLM stage timing additionally includes its all-worker synchronization RPC.
- Both external paths support the Qwen3-30B FP8 panel. The pinned implementations
  reject GLM MLA; those combinations are N/A. Actual supported execution still
  requires successful smoke and full measurement; static eligibility is not proof.

## Export and replot

`export.py --base-plot <RAW_VALIDATED_BASE> --portable-base <PORTABLE_BASE>
--external-root <CAMPAIGN> --output-dir <NEW_EXPORT>` revalidates all old/new raw
points, preserves the original 44 points, adds the two external curves and two
explicit N/A entries, and invokes the existing Matplotlib/Seaborn plotter.

The final `plot_data.json` / `portable-replot/points.csv` contain path-independent
data for later plotting. PNG/PDF/SVG use native constrained layout, unchanged
fonts, no titles, and normal/logarithmic y variants. No image-generation engine
or synthetic chart values are used.

When an interleaved baseline exhausts outputs during admission, the explicitly
approved extended-output variant uses `launch_external.py --input-len 124928
--output-len 8192` (122K/8K), keeping the original 133120-token total span. It
reruns the whole affected curve, rather than mixing standard and extended points.
`export.py --tangram-root <EXTENDED_CAMPAIGN>` requires unchanged model/topology/
memory/sparse settings and at least 2015 measured full-batch steps at each point.
The adjusted curve is labelled `Tangram*`; actual lengths and the reason are in
`config.curve_protocols` for the figure caption. Standard 128K/2K data remain
available separately. Outputs still complete in full; no truncation is introduced.

`capture_external.py` records imported package locations, Git versions/dirty
status, package versions and checkpoint identity. `bundle.py` packages raw
steps/output/config/capacity evidence and validation results under portable
artifact IDs. Source code, patches, compiled caches and weights are excluded.
All paths are supplied as arguments.
