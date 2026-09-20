# GLM-4.7-Flash conservative concurrency by context length

Independent experiment package for the 2026-09-19 H100 BF16 and RTX PRO6000
FP8 runs. All45 requested points completed successfully:25 H100 and20 PRO6000.
PRO6000 input173734 was cancelled by the user before execution; existing H100
173734 measurements remain included.

- [Full results](RESULTS.md): concurrency, decode throughput, TTFT, TPOT, E2E,
  and complete-workload output throughput.
- [CSV](results.csv) / [JSON](results.json): lightweight measurement exports.
- [Protocol](protocol.json): model, budgets, timing, repetitions and explicit settings.
- [Provenance](manifest.json): execution identity, raw artifact locations and export limitations.
- `remote_observed.tsv`: numeric results preserved from prior live remote queries.

## Measurement and interpretation

Each operating point uses single GPU, TP=EP=1, output2048, greedy/ignore EOS,
seed42, zero length jitter, chunk=max_num_batched_tokens32768, memory fraction.90,
Graph on and prefix cache off. Native staged admission is disabled. Each decode
and request experiment independently uses one discarded workload and three
measured workloads. Decode discards32 full-residency steps and times256 continuous
steps with boundary synchronization only; the full2048-token generation completes.
Request TTFT/TPOT/E2E use the canonical engine-adapter event boundary, not HTTP
client latency. vLLM and native adapter boundaries must be considered when
interpreting request metrics. No extra per-step synchronization is introduced.

Keep the two hardware/weight-dtype groups separate. Different concurrency and
precision mean these are capacity operating points, not matched-batch hardware
speedups or equal-quality evidence. Dense means Sparse-vLLM dense.

## Conservative candidate arithmetic

Use85% of the minimum method-specific KV slots observed in prior BS1 logs.
Let L=input tokens, O=2048, C=32768, U=floor(.85*slots).

- Dense/vLLM/OmniKV: floor(U/(L+O)); OmniKV retains full-context layers.
- H2O: D=min(L,8192)+O; P=min(L,8192+C)+O;
  candidate=1+floor((U-P)/D).
- SnapKV: D=min(L,8192)+O; P=L+O;
  candidate=1+floor((U-P)/D).

The actual launcher clamps candidates to[1,1024]. These are conservative heuristics,
not guaranteed admission bounds. H2O's estimate exceeds its steady-state decode
budget4096 plus eviction interval128. A single prefill peak estimate does not prove
multiple simultaneous prefill requests fit. Candidates require full measured
validation; failures are recorded without search or automatic downshift.
`maximum_verified=false`; no binary search, doubling, or max+1 proof was performed.

## Reproduction and export

Reuse `../sparse_decode_efficiency/run_context_capacity.py --config CONFIG
--repo CHECKOUT --gpus GPU_ID`; it invokes the canonical sweep and
`benchmark/efficiency/bench_probe.py`. Do not use a separate timing runner.
Use the source campaign config referenced by the manifest, supplying model,
conda/environment, checkout and output paths for the new host. The portable
`protocol.json` is a protocol record, not a complete executable host config.
Recompute candidates from that host's slots and populate
`conservative_candidates_by_length`; exclude PRO6000 input173734 to reproduce the
final authorized scope. Preserve method budgets and timing contracts.

Render the table with `python scripts/official_experiments/glm47_conservative_concurrency/render_results.py`.
The renderer only formats recorded metrics and never fills missing values.

## Export verification

All45 rows now contain all five metrics at source aggregate precision. Remote
access was restored in no-GPU mode and results/config/identity/status files were
downloaded without starting a benchmark. The20 remote aggregates passed success,
fixed input/output lengths, three decode repetitions, zero preemption and measured
request-count checks. Previously recorded values match within two-decimal rounding.
`remote_observed.tsv` retains the earlier rounded observations for provenance;
`results.json` and `results.csv` are the authoritative complete exports.
The manifest records the downloaded aggregate path and SHA256. Large logs and
token outputs remain outside Git; this export check is not a new GPU run.
