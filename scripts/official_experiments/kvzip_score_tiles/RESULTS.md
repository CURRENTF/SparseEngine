# Results: KVzip reconstruction score tiles

Single H100, Qwen3-4B-Instruct-2507 BF16, TP1; 32K input / 128 output tokens. All rows use the same traces, budget, token layout, and Graph settings. See [protocol and commands](README.md).

## Unprofiled serving measurements

| Concurrency | TTFT mean, before → after (s) | TPOT mean, before → after (ms) | Complete workload mean, before → after (s) | E2E speedup | Output throughput, before → after (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.220 → 3.160 | 5.981 → 5.982 | 8.980 → 3.920 | 2.29× | 14.25 → 32.66 |
| 4 | 32.699 → 12.584 | 9.350 → 9.338 | 33.892 → 13.775 | 2.46× | 15.11 → 37.17 |

Three measured workloads per row after warmup; C1 pools 3 requests and C4 pools 12 requests. Throughput uses total output tokens / total workload time. Each workload completed all outputs and expected compactions, with 127 decode Graph replays and no capture, recapture, or eager fallback.

## Nsight Systems attribution

Times below are GPU kernel sums inside inclusive NVTX scopes, not wall latency. The score scope is nested in prefill and reconstruction; do not add them together.

| Concurrency | Prefill kernels, before → after (ms) | Reconstruction score kernels, before → after (ms) | Score fraction of prefill, before → after | Remaining attention / GEMM fraction |
|---:|---:|---:|---:|---:|
| 1 | 8258.6 → 3077.5 | 5871.4 → 620.5 | 71.1% → 20.2% | 51.5% / 20.7% |
| 4 | 33007.2 → 12271.4 | 23462.4 → 2481.4 | 71.1% → 20.2% | 51.6% / 20.6% |

The common final-score grid changes from `512 × 542` to `512 × 32`; partial LSE changes from `512 × 542` to `512 × 66`. Short/tail launches and exact instance counts are in `summary.json`. Attention is now the largest prefill component. Hardware counters were unavailable, so this does not establish a compute- or bandwidth-bound hardware limit.

## Reconstruction scorer microbenchmark

| Source interval | Replay tokens | Median before → after (ms) | Speedup | Max absolute difference vs baseline |
|---|---:|---:|---:|---:|
| [64, 2064) | 2016 | 9.2432 → 0.9396 | 9.84× | 2.61e-08 |
| [16064, 18064) | 2016 | 9.2826 → 0.9388 | 9.89× | 4.1e-08 |
| [32064, 32640) | 592 | 2.7143 → 0.1225 | 22.15× | 5.22e-08 |

The common partial-LSE workspace changes from 133.38 MiB to 16.24 MiB. Both implementations pass the independent FP64 probability oracle; the optimized maximum absolute oracle error over these fixtures is `1.12e-7`. These are component timings, not serving speedups.

## Correctness and limits

- CUDA/CPU suite: 73 passed, 4 subtests passed; the initially skipped dual-GPU NCCL test passed in a separate two-device run (1 passed). Tests include BF16/FP16, padded GQA groups, irregular and shared-boundary tiles, ragged rows, non-contiguous tensors, reused workspaces, and Graph replay. Full-model TP>1 and other GPU architectures were not performance-validated.
- C1 model diagnostic: selected positions match exactly in all 36 layers for both the warmup and measured request; all 128 generated tokens match per request.
- C4 model diagnostic: all generated tokens match in 8 full-length requests (4 warmup + 4 measured). One selected position differs: request order 6, layer index 2, token position 748. Maximum absolute score difference across requests is `3.934e-6`.
- The differing token's head-7 score changes from `0.2873278558254242` to `0.2873273193836212`; the global threshold remains `0.2873273193836212`. The existing strict `>` rule therefore drops it. Compacting the FP32 LSE reduction changes floating-point summation order: the optimization preserves the mathematical scoring/selection rule but is not bitwise selection-equivalent at threshold ties.
- No downstream quality benchmark was run. Synthetic output agreement and numerical oracles are correctness evidence for the tested cases, not a general task-quality guarantee.

Source: [compact metrics, samples, grid counts, and validation evidence](summary.json). Raw profiler reports, request records, saved scores, commands, and logs are retained outside Git.
