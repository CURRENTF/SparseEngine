# 128K/2K decode-efficiency figure with GLM low-batch gaps filled

Device: NVIDIA H100 80GB HBM3, GPU0/1. GLM-4.7-Flash BF16 uses TP2/EP2. The six added measurements use `boundary_sync_v2`, 131072 input / 2048 output tokens, one discarded workload, three measured workloads, 32 full-batch warmup decode steps, and a continuous 256-step full-residency decode window per workload. The rate is pooled completed decode tokens divided by pooled boundary-synchronized window time, not serving throughput. Native SnapKV retains wave size 2 and a one-step decode gap.

| GLM method | BS | Decode throughput (tok/s) | Delta vs measured vLLM at same BS |
| --- | ---: | ---: | ---: |
| vLLM Vanilla | 3 | 225.704 | 0% |
| SparseEngine Vanilla | 3 | 217.936 | -3.441% |
| SparseEngine QuEST | 3 | 329.553 | +46.011% |
| SparseEngine OmniKV | 3 | 320.284 | +41.905% |
| SparseEngine SnapKV | 3 | 398.908 | +76.740% |
| SparseEngine SnapKV | 5 | 588.552 | +119.738% |

All six cases completed and passed raw-window, full-output, model/topology, and protocol validation; no contention or validity marker was present. Measurement Git commit: `a07c3e30be63646137cc6aeeff131adc987afe5d`. The exact run config and per-lane commands are under `/data2/haojitai/outputs/Sparse-vLLM/paper_decode_glm128_bs3_bs5_gap_20260924/`, with raw cases under its `runs/glm4.7-flash/` directory. The case runner used `--probe-only --probe-concurrency 3` for five lanes and `--probe-only --probe-concurrency 5` for SnapKV, with `--continue-after-contention --isolated-cache-root` on GPU0/1.

`128k/plot_data.json` is the latest portable formal 128K data. Its CSVs contain the exact plotted points, relative deltas, and largest-measured-batch bars. It extends the [Qwen BS3 update](../2026-09-24-qwen128-bs3-update/RESULTS.md) without changing any Qwen curve or figure configuration. The three 128K PNG/PDF/SVG figures are under `/data2/haojitai/outputs/Sparse-vLLM/paper_decode_figures_128k32k_20260924_glm_bs3_bs5_update/128k/`. The maximum-batch bar data and PNG are byte-identical to the previous formal 128K export because none of the added GLM points changes a method's largest measured batch. The [32K formal package](../2026-09-23-128k32k-boundary-sync-v2/RESULTS.md) is unchanged. GLM SnapKV's BS55 bar remains a measured lower bound, not a confirmed capacity maximum.
