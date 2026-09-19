# SparseVLLM Regression Rubrics

This file records the stable ABCD rubrics for the SparseVLLM regression
harness. It should not contain dated campaign results, open bug lists, run IDs,
or remote log paths.

The executable grading rules live in
`benchmark/sparseengine_regression/grading.py`. This document is the human-facing
summary of those rules.

## Common Semantics

- `A`, `B`, and `C` are passing grades with decreasing confidence.
- `D` is a required-gate failure; the harness writes the failure artifacts,
  marks the run failed, and exits nonzero.
- `N/A` means the gate was skipped by policy.
- Dated run results and blockers belong in run artifacts, issue notes, or a
  campaign-specific report outside this rubric file.

## Quality

Quality first requires the vanilla LongBench-mini score to meet the manifest's
absolute baseline floor, then compares sparse score against vanilla from the
same run.

| Grade | Rule |
| --- | --- |
| `A` | Sparse score loss `< 0.1` vs vanilla. |
| `B` | Sparse score loss `<= 0.5`. |
| `C` | Sparse score loss `<= 1.0`. |
| `D` | Vanilla is below the absolute baseline floor, sparse score loss `> 1.0`, missing aggregate score, or failed quality run. |

## RULER Core Quality

Every configured RULER task is graded independently at every configured context
length. Vanilla must meet the manifest baseline floor in every task/length
bucket. Sparse score loss is compared with the exactly aligned vanilla samples
from the same run. The default self-contained tasks are `niah_single_1`,
`niah_multikey_2`, `vt`, `cwe`, and `fwe`.

| Grade | Rule |
| --- | --- |
| `A` | No score loss vs vanilla at this context length. |
| `B` | Score loss is at most half of `maximum_score_loss`. |
| `C` | Score loss is at most `maximum_score_loss`. |
| `D` | Vanilla is below its floor, sparse loss exceeds `maximum_score_loss`, samples are incomplete/misaligned, or generation fails. |

When prefix caching is enabled, the RULER runner also requires nonzero replay
hit requests and tokens plus exact deterministic output equality. These are
cache-correctness conditions and do not grade cache performance.

## LongBench V2 Quality

LongBench v2 uses exactly aligned vanilla and sparse samples selected by the
model tokenizer in fixed prompt-token buckets. Prompts are never truncated.
Every selected sample must reach an explicit evaluated terminal state, and the
source dataset hash must match between the paired runs. An unparseable non-empty
model response is retained as `parse_failed` and scored as incorrect; it is not
an execution failure.

| Grade | Rule |
| --- | --- |
| `A` | No score loss vs vanilla. |
| `B` | Score loss is at most half of `maximum_score_loss`. |
| `C` | Score loss is at most `maximum_score_loss`. |
| `D` | Vanilla is below its floor, sparse loss exceeds `maximum_score_loss`, samples are incomplete/misaligned, source hashes differ, or generation fails. |

## Performance

Performance grades decode speedup and, when required, decode CUDA graph
activation. A method may also declare `performance.minimum_prefill_speedup`;
falling below that method-specific prefill floor is a `D`.

| Grade | Rule |
| --- | --- |
| `A` | Speedup `>= 2.0` and required decode CUDA graph is active. |
| `B` | Speedup `>= 1.5`. |
| `C` | Speedup `> 1.0`. |
| `D` | Speedup `<= 1.0`, failed performance run, or expected decode CUDA graph is inactive. |

When the performance policy sets `require_speedup=false`, speedup is recorded
but not required; the gate records `A` if the decode CUDA graph requirement is
satisfied.

## Memory

Memory compares expected and observed memory savings.

| Grade | Rule |
| --- | --- |
| `A` | Observed saving is positive and absolute error from expected saving is `<= 0.05`. |
| `B` | Observed saving is positive and absolute error is `<= 0.10`. |
| `C` | Observed saving is positive and absolute error is `<= 0.20`. |
| `D` | Missing memory accounting, non-positive observed saving, or absolute error `> 0.20`. |

## Stress

Stress grades the fixed-length admission/decode stress test.

| Grade | Rule |
| --- | --- |
| `A` | Completed, no crash, no preemptions, full admission window reached, and utilization is OK. |
| `B` | Completed with no preemptions, but at least one `A` condition is not fully met. |
| `C` | Completed with one or more preemptions. |
| `D` | Crashed, stuck, failed rows were emitted, or the run did not finish. |

## Stress V2

Stress V2 grades synthetic serving-trace runs with shared-prefix and multi-turn
workloads.

| Grade | Rule |
| --- | --- |
| `A` | All cases succeed, at least one prefix-cache case observes cache hits, prompt lengths vary, and minimum eligible cache-hit rate is `>= 0.80`. |
| `B` | All `A` structural conditions hold and minimum eligible cache-hit rate is `>= 0.50`. |
| `C` | All `A` structural conditions hold but minimum eligible cache-hit rate is `< 0.50`. |
| `D` | Missing summary, no cases, any failed case, no prefix-cache-enabled case, no observed prefix-cache hit, or no variable prompt lengths. |

## Worst Required Grade

The harness reports `worst_required_grade` as the minimum non-`N/A` grade across
the required gates in a run.
