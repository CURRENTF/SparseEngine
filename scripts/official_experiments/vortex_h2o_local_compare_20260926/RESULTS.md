# Local SparseEngine / Vortex H2O request comparison

2026-09-26; NVIDIA H100 80GB HBM3 physical GPU 4; Qwen3-30B-A3B-Instruct-2507-FP8, BF16 KV; TP1/EP1/DP1. SparseEngine commit `7b32be0fa67fc06d0df8bfabab5a2700f8825591`; Vortex commit `c016fdfc871130481c76e3beead83e8c2b54cb99`.

Fixed batch 4, exactly 32,768 input and 512 output tokens per request, seed 42, prefix caching off, CUDA Graph on, one workload warmup and three measured workloads. All four cases used the same 12 prompt token digests and lengths. Vortex's request probe used [trace_v2_vortex_probe.py](trace_v2_vortex_probe.py) to match SparseEngine's `random-varlen-v2` generator at zero jitter; this is a probe adapter, not an algorithm change.

| Engine | Method | Full-workload output throughput, mean ± sample SD (token/s) |
| --- | --- | ---: |
| SparseEngine | Vanilla | 221.56 ± 1.11 |
| SparseEngine | H2O | 295.81 ± 0.37 |
| Vortex | `paper_dense` | 188.81 ± 0.13 |
| Vortex | `paper_h2o` | 79.38 ± 0.08 |

All 12 measured requests per case completed with 512 generated tokens. Each measured Vortex workload had 512 Graph replays; SparseEngine had 511 because the first token was produced by prefill. Neither engine recorded a measured Graph capture or eager decode. This table reports complete request workloads, not isolated decode-stage throughput.

SparseEngine H2O launch arguments: `--engine sparseengine --sparse-method h2o --tensor-parallel-size 1 --expert-parallel-size 1 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --sparse-prefill-score-mode logits --hyper-params '{"decode_graph":true,"engine_prefill_chunk_size":8192,"h2o_decode_budget":4096,"h2o_prefill_budget":8192,"h2o_recent_ratio":0.5,"h2o_prefill_score_window":128,"h2o_decode_eviction":false}'`. Native Vanilla used the same workload options without H2O parameters.

Vortex launch arguments: `--method paper_h2o --context-length 33408 --max-running-requests 4 --cuda-graph-max-bs 4`; page size 16, 128 heavy plus 128 recent pages, TRTLLM attention, BF16 KV, Triton FP8 GEMM, radix cache off, prefill chunk 8192. Its Dense reference used `--method dense` with the same server options. Both probes used `--model-path "$MODEL_PATH" --scenario fixed --prompt-lens 32768 --output-lens 512 --batch-sizes 4 --prompt-length-jitter 0 --output-length-jitter 0 --seed 42 --num-warmups 1 --num-iters 3 --monitor-gpus 4 --output-dir "$RUN_ROOT/<case>"`; Vortex additionally used `--server-url http://127.0.0.1:30486`.

The H2O algorithms differ: SparseEngine scores/selects tokens with a 128-token logits prefill window and physically reclaims dropped KV; Vortex `paper_h2o` uses cumulative attention mass and 16-token pages, retaining original physical KV slots. This run has no downstream quality evaluation or matched physical-residency claim. Vortex request times are observed at the HTTP client; SparseEngine observes engine token publication. Interpret the cross-engine throughput readings as results for these complete systems and settings.

Compact table data: [results.json](results.json). Raw samples, manifests, and logs are on persistent local output storage outside Git.
