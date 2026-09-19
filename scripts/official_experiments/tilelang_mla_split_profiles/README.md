# TileLang MLA split-profile comparison

Compare two launch profiles of the existing BF16 MLA decode kernel with FP32
per-head raw QK scores. The benchmark does not modify production dispatch. Both arms use
`TileMlaDecodeKernel(fixed_config=...)`; only `num_split` differs.

The rule-generated candidate is `sm_log_context_v1`:

```
head_tiles = padded_heads / legacy_block_h
target_ctas = SM_count * max(log2(max_actual_context / 64), 1)
raw_splits = target_ctas / (batch * head_tiles)
splits = next legal split >= ceil(raw_splits), capped at 32
legal_splits = [1, 2, 4, 8, 16, 32]
```

There is no additional minimum-work constraint or empirical adjustment. This is
a test of the proposed rule, not a claim that its generated profile is optimal.
The retired baseline is frozen in `config.json` with its source commit, so
promoting the rule in production cannot silently change the experimental baseline.
Other tiles and score semantics are preserved from that baseline. Head counts
5/10/20 correspond to GLM local-head shapes, not distributed TP/EP measurements.

## Run

From the repository root, set `RUN_ROOT` to a **new** directory on an output
volume, `CUDA_VISIBLE_DEVICES` to one idle GPU, `CONDA_EXE` to conda, and
`SPARSE_ENGINE_ENV` to the environment containing the project's CUDA dependencies:

```bash
bash scripts/official_experiments/tilelang_mla_split_profiles/run.sh
```

Use tmux for the long sweep. The controller retains a visible GPU reservation
through smoke and full runs, rejects foreign contention, and terminates its own
workers on failure. No automatic retry or candidate substitution is performed.
An independent repeat can use `--case-ids
scripts/official_experiments/tilelang_mla_split_profiles/repeat_cases.json`.

`config.json` defines 150 sampled shapes; the explicit repeat contains 27. Inputs
use distinct KV per request, shuffled 64-token pages, permuted slot-table request
indices, and strided BF16 views of packed 576-dimensional queries. Ragged cases
include empty, tiny, intermediate, and maximum-length rows.

## Measurement and evidence

Both profiles must pass an independent FP32 Torch raw-QK and softmax-times-V
oracle on up to three representative rows per case, before performance timing.
All output rows are checked for finite values; sampled rows also check untouched
masked score tails. Eager execution and CUDA Graph replay are validated.

The measured boundary is the existing wrapper's GPU work: split attention plus
combine, with its caller-owned contiguous score output. Invalid score tails are
not consumed and need no timed reset. The capture repeats eight calls to amortize
host graph-launch overhead; inputs and resident KV are reused without L2 flushing.
Every arm receives ten warmup calls and six block-averaged CUDA-event samples in
three balanced ABBA/BAAB rounds. Compilation/cold binding is recorded separately.
Identical configurations are controls, not evidence of a policy improvement.

Each run records Git commit/dirty status, resolved profiles, environment/device
identity, raw samples, per-case numerical status and medians. The aggregate is
an unweighted geometric mean over the sampled grid, **not** engine throughput
or a production-workload-weighted speedup. GPU clocks are observed, not locked.
No Nsight bottleneck attribution or end-to-end performance claim is made.

`confirmation_cases.json` selects the primary sweep's six lowest and six highest
speedups plus its three most deviant identical-config controls. These independent
process repeats test the extrema, but do not replace primary rows or alter the
primary aggregate. The repeat subset is intentionally selected, not a second
representative workload distribution.

## Replot

```bash
python scripts/official_experiments/tilelang_mla_split_profiles/summarize.py \
  --run "$PRIMARY_RUN/full" --repeat "$REPEAT_RUN/full" --output "$FIGURE_DIR"
```

Replotting needs NumPy, pandas, matplotlib and seaborn, but not CUDA. It verifies
completeness and reconstructs medians from raw samples before producing CSV/JSON,
per-head speedup heatmaps and 128K latency curves (linear/log y). Every figure is
saved as both PNG and PDF, without titles, using native constrained layout.
Ragged and BS256 supplemental heatmaps complete coverage of all sampled cases.
Use `--confirmation "$CONFIRMATION_RUN/full"` to include extrema repeat evidence.
The portable `data/{primary,repeat,confirmation}` directories contain everything
needed to replot without the original output volume; use those paths directly
instead of the corresponding `full` paths. Experiment configuration and measured
profiles are retained; source code is read from the selected checkout.

## Optional CPU precompilation

`precompile.py --run "$PRIMARY_RUN/full" --workers 6 --output "$COMPILE_RUN"`
can populate the same TileLang disk cache ahead of measurement. Explicitly set
`CUDA_VISIBLE_DEVICES=''`, `TILELANG_CACHE_DIR` to that run's cache, and `TMPDIR`
to its data-volume compiler directory. The target architecture is derived from
the measured manifest. Compiler workers never execute kernels or initialize a
CUDA context; compilation is not a scored sample. The installed TileLang cache
must support atomic publication of complete kernel directories.

The result trust-boundary tests run without CUDA:

```bash
python3 -m unittest discover \
  -s scripts/official_experiments/tilelang_mla_split_profiles -p test_artifacts.py
```
