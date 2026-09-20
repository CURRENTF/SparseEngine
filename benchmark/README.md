# Benchmark Entrypoints

This directory contains runnable benchmark code. The stable runbook lives in
[`docs/en/benchmarking/README.md`](../docs/en/benchmarking/README.md); keep this file
as a lightweight source-tree map.

| Directory | Main entrypoints | Notes |
| --- | --- | --- |
| `efficiency/` | `bench_probe.py`, `hardware_monitor.py`, `validate_unified_suite.py` | Matched synthetic efficiency, request-churn, directly sampled GPU activity, and unified-suite validation. See the [runbook](../docs/en/benchmarking/efficiency.md). |
| `long_bench/` | `pred.py`, `eval.py` | Existing LongBench v1 prediction and scoring through the native SparseEngine runtime. |
| `long_bench_v2/` | `pred.py`, `upstream/` | Independent LongBench v2 runner with the official repository pinned as a submodule, untruncated token-stratified selection, and native SparseEngine inference. |
| `math_bench/` | `pred.py`, `eval.py` | GSM8K, AIME 2024, MATH-500, and HMMT Nov tasks. |
| `scbench/` | `run_scbench.py`, `run_scbench_preprocessed.py`, `compute_scores.py`, `run_kvzip_preprocessed.py` | SCBench standard, preprocessed, scoring, and KVZip routes. |
| `claw_eval/` | `run_sparseengine_claw_eval.sh` | Claw-Eval through the shared SparseEngine OpenAI-compatible server. |
| `swe_bench_lite/` | `run.py` | Thin mini-SWE-agent generator and official SWE-bench Lite evaluator. |
| `microbench.py` | `microbench.py` | Synthetic prompt-length throughput benchmark for TTFT, prefill/decode tok/s, ITL, and peak memory. |
| `simulated_deep_research/` | `run.py` | Synthetic multi-round main-agent/subagent workload through the non-uniform OpenAI smart router. |
| `multimodal/` | `video_qa/`, `image_qa/` | Video QA and image QA benchmark runners. |
| `ruler_vt/` | `pred.py`, `tasks.py` | Self-contained RULER core runner for NIAH retrieval, variable tracking, CWE, and FWE; the historical directory name is retained for compatibility. |
| `niah/` | `test_niah.py`, `gen_niah.py` | Needle-in-a-haystack generation and evaluation utility. |
| `sparseengine_regression/` | `run_suite.py` | Fixed LongBench v1/v2, RULER quality, performance, and stress regression harness. |

Do not store local experiment ledgers in this directory. Put reproducible
commands, stable runbook notes, and result-interpretation rules in
`docs/en/benchmarking/`.
