# Qwen 128K/2K BS3 decode-efficiency update

Device: NVIDIA H100 80GB HBM3, GPU3, TP1/EP1. Protocol: `boundary_sync_v2`, 131072 input / 2048 output, one discarded workload and three measured workloads, 32 full-batch warmup decode steps and 256 measured steps per workload. The metric is pooled completed decode tokens divided by pooled boundary-synchronized window time. SparseEngine SnapKV uses wave size 2 and a one-step decode gap; Tangram uses its supported async queue. Graph is enabled and prefix cache disabled.

| Qwen method | BS | Decode throughput (tok/s) | Delta vs measured vLLM BS3 | Result |
| --- | ---: | ---: | ---: | --- |
| SparseEngine SnapKV | 3 | 518.019 | +208.604% | valid, 3/3 repetitions |
| Tangram SnapKV | 3 | 357.409 | +112.923% | valid, 3/3 repetitions |
| HiSparse QuEST | 3 | — | — | capacity exceeded; no throughput point |

HiSparse startup reported 396272 KV slots, below the 399360 tokens required by three complete requests. Its BS2 result remains 211.496 tok/s; the BS3 capacity failure confirms that integer boundary for this protocol. The earlier HiSparse BS3 worker crash remains preserved as a failed attempt and is not used as a throughput measurement.

Measurement Git commit: `a07c3e30be63646137cc6aeeff131adc987afe5d`. Run config and exact per-workload commands: `/data2/haojitai/outputs/Sparse-vLLM/paper_decode_qwen128_bs3_gap_20260924/config.json` and each lane's `bs3/run_manifest.json`. Raw results and capacity evidence: `/data2/haojitai/outputs/Sparse-vLLM/paper_decode_qwen128_bs3_gap_20260924/runs/qwen3-30b-fp8/` and `measurements/qwen3-30b-fp8/qwen128-bs3-gap-r4/hisparse-quest/bs3.json` under that run root.

`128k/plot_data.json` is the updated portable 128K data used for the new absolute and relative lines and largest-measured-batch bars. The accompanying CSVs contain their exact plotted values. The 32K panel is unchanged from [the 2026-09-23 package](../2026-09-23-128k32k-boundary-sync-v2/RESULTS.md). The 128K PNG/PDF/SVG figures are in `/data2/haojitai/outputs/Sparse-vLLM/paper_decode_figures_128k32k_20260924_bs3_update/128k/absolute/` and `relative/`; the prior figure directory was not overwritten.
