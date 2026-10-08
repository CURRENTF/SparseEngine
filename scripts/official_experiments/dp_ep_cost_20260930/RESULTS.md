# DP2 + EP2 cost diagnosis — 2026-09-30

The largest abnormal cost in this workload is waiting for the other attention
replica at MoE collectives when one replica prefills and the other decodes.
Decode execution also becomes substantially slower without CUDA Graph replay.

Device: two NVIDIA H100 80GB HBM3 GPUs, physical IDs 6 and 7. Model:
GLM-4.7-Flash BF16, vanilla BF16 MLA cache, TP1/DP2/EP2. Git commit:
`6feb77078ed25beae07cc977b52b9a9fe2d47b20`. Canonical entrypoint:
`benchmark/efficiency/bench_probe.py`; complete launch parameters are in
[arguments.json](arguments.json).

Each DP scheduler has an execution batch limit of 8, resident capacity 32,
token budget 8192, and prefill chunk size 1024. AG/RS resolves to
`flashinfer_mixed_comm_uc`. The two measured repetitions follow one warmup.
Fixed requests have 8192 input and 128 output tokens, global concurrency 16.
Churn has 64 requests per repetition, input jitter 0.5 and output jitter 0.75.
Generation is greedy and ignores EOS to validate exact requested output counts.

The following coordinator step RPC times come from the unprofiled churn run.
No additional CUDA synchronization was introduced. Step RPC time excludes
intervening benchmark-driver work and must not be interpreted as request TPOT.

| Rank phases | Steps | Share of steps | Mean step RPC (ms) | Share of step RPC time |
| --- | ---: | ---: | ---: | ---: |
| decode / decode | 662 | 78.9% | 12.17 | 27.8% |
| decode / prefill | 94 | 11.2% | 159.30 | 51.6% |
| prefill / prefill | 27 | 3.2% | 204.93 | 19.1% |
| decode / idle | 56 | 6.7% | 8.27 | 1.6% |

The fixed Graph/eager comparison uses identical per-iteration input token
digests, output lengths, provider bindings and sampling parameters. These are
unprofiled request metrics at native step token publication.

| Fixed execution | Successful measured requests | Request TPOT mean (ms) |
| --- | ---: | ---: |
| CUDA Graph | 32 | 12.85 |
| Eager | 32 | 60.34 |

Eager request TPOT is 4.70 times the Graph value. Both churn variants also
completed all 128 measured requests with identical traces. The unprofiled churn
request TPOT mean is 47.04 ms; its oversubscribed workload differs from fixed.

Nsight Systems attributes eager kernels by launch-time NVTX containment and
captured kernels by original graph-node lineage. Startup, capture and warmup
are excluded using measured-iteration markers. GPU category times are per-rank
kernel durations and include waiting inside communication kernels.

| GPU category | Balanced Graph decode, rank mean (ms/step) | Decode side of mixed phase, rank mean (ms/step) |
| --- | ---: | ---: |
| Attention, including projections | 3.86 | 4.12 |
| Routed experts | 4.89 | 39.58 |
| Dispatch, including padding and AllGather | 0.40 | 101.21 |
| Combine / ReduceScatter | 0.62 | 16.79 |
| Shared experts | 0.65 | 0.75 |
| Router | 0.58 | 6.25 |
| Sampling kernels | 0.01 | 0.01 |

On the decode side of mixed steps, the AllGather kernel alone averages
100.89 ms/step. Of that duration, 98.22 ms precedes the peer's entry into the
corresponding AllGather kernel: approximately 97.3%. This is evidence of
collective arrival skew, rather than an isolated transport bandwidth result.
Fixed-capacity padding increases the mixed-phase MoE input row volume by a mean
factor of 1.987 relative to the sum of the two agreed local row counts; this row
factor does not establish a speedup from removing padding.

Balanced Graph step coordination costs 0.56–0.57 ms/rank in the profiled run.
The coordinator step averages 16.63 ms, and its complement of GPU kernel
interval union is 4.92–5.34 ms/rank. That complement can contain copies, driver
work and host waits; it is not a direct measurement of pure CPU computation.

Nsight node tracing plus detailed instrumentation increases fixed Graph request
TPOT by 29.9% and eager TPOT by 38.3% relative to their unprofiled counterparts.
Use the unprofiled tables for latency comparisons and the timeline for diagnosis.
These H100 synthetic results do not establish RTX PRO6000 or MiniSWE attribution.

Validation: all six benchmark cases passed sample/output-count checks; matched
trace equality passed; the default two-rank BF16 AG/RS test passed an independent
additive/squared numerical oracle with unequal rows, idle participation and
Graph replay/replacement (`1 passed`). No production implementation was changed.

Compact source data: [summary.json](data/summary.json) and
[unprofiled_step_phases.csv](data/unprofiled_step_phases.csv).
