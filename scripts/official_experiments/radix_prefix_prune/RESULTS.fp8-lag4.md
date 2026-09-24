# OmniKV delayed tool-result pruning: FP8 rerun

SWE-bench Lite `test`, 300 tasks, GLM-4.7-Flash-FP8 on 4× RTX 4090 48GB.
SparseEngine ran four TP1/EP1/DP1 OmniKV replicas through one router, with
24 MiniSWE agents and at most six decoding sequences per replica. Each new tool
result triggered pruning of the fifth most recent tool result, retaining 20% of
its eligible aligned tokens (`kvzip_global`, `tool_result_lag=4`). This is a
separate FP8 condition from the BF16 results in [`RESULTS.md`](RESULTS.md).

| Resolved | Final sample statuses | Generation exits | API calls |
| ---: | --- | --- | ---: |
| **75/300 (25.00%)** | 148 success, 149 model failed, 3 metric failed | 151 submitted, 148 limits exceeded, 1 repeated format error | 18,853 |

The 24 tasks affected by an infrastructure outage were regenerated before
scoring. The final files contain 300 unique generation and 300 unique
per-sample rows. Official evaluation and summarization exited successfully on
2026-09-25 at 03:38 +08; the GPU host shut down at 03:39 +08 after result and
log archive checks.

The engine source base commit was `a07c3e30be63646137cc6aeeff131adc987afe5d`.
The GPU and Worker-A checkouts had run-scoped uncommitted changes, so the base
commit alone is insufficient for exact reproduction. The full launch settings,
source patch identity, and raw artifact roots are in
[`results.fp8-lag4.json`](results.fp8-lag4.json). This condition changes both
precision/topology and prune timing relative to the BF16 campaign; the score
difference does not isolate any one change.
