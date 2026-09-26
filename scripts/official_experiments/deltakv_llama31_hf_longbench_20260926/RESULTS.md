# DeltaKV author HF quality results

Llama-3.1-8B-Instruct BF16; six RTX 4090 48 GiB GPUs, one HF process per GPU.
DeltaKV author commit `11a837b21acfa475957210c5b2d339b89c1b7e3b` with the
two compatibility/correctness edits described in [README.md](README.md).

| Benchmark | DeltaKV author HF | Samples | Failed samples | Reference |
| --- | ---: | ---: | ---: | --- |
| LongBench v1, six-category macro average | **49.65** | 3750 | 0 | HF FullKV 50.47 under the same v1 decoding protocol |
| LongBench v2, exact accuracy | **30.22** (152/503) | 503 | 0 model failures, 18 parse failures | SparseEngine DeltaKV 29.42 (148/503) |

The v1 category scores are SDQA 43.51, MDQA 44.28, SUM 28.03, FewShot
67.79, Syn 53.84 and Code 60.47. All 16 task scores and scorer identity are
in [longbench_v1.json](longbench_v1.json).
The HF FullKV v1 reference comes from
`SparseEngine-Baselines/experiments/llama31_longbench_v1/results/full_v1_fullkv.json`.

The v2 result is 0.80 percentage points, or four correctly answered questions,
above the earlier SparseEngine DeltaKV v2 result. Its parse-failure count is
18 versus 22. The full aggregate, including domain, length and difficulty
groups, is in [longbench_v2.json](longbench_v2.json).

| v2 domain | HF Correct / Total | HF Score | SparseEngine Score |
| --- | ---: | ---: | ---: |
| Code Repository Understanding | 15 / 50 | 30.00 | 32.00 |
| Long In-context Learning | 26 / 81 | 32.10 | 22.22 |
| Long Structured Data Understanding | 9 / 33 | 27.27 | 30.30 |
| Long-dialogue History Understanding | 6 / 39 | 15.38 | 10.26 |
| Multi-Document QA | 40 / 125 | 32.00 | 34.40 |
| Single-Document QA | 56 / 175 | 32.00 | 32.57 |

| v2 official length | HF Correct / Total | HF Score | SparseEngine Score |
| --- | ---: | ---: | ---: |
| long | 33 / 108 | 30.56 | 26.85 |
| medium | 60 / 215 | 27.91 | 26.51 |
| short | 59 / 180 | 32.78 | 34.44 |

| v2 difficulty | HF Correct / Total | HF Score | SparseEngine Score |
| --- | ---: | ---: | ---: |
| easy | 57 / 192 | 29.69 | 31.77 |
| hard | 95 / 311 | 30.55 | 27.97 |

SparseEngine subgroup counts are recorded in its
[earlier result](../deltakv_llama31_longbench_v2_20260925/RESULTS.md).

For v2, all 503 sample IDs and prompt token counts match the existing
SparseEngine DeltaKV v2 result. Both use greedy decoding and a 128-token
generation cap. The author HF implementation uses sparse-reference FP8;
the existing SparseEngine v2 run disabled it, and the cache implementations
also differ. The historical HF FullKV v2 score used sampling decoding and
serves as a separate reference.

The campaign started at 2026-09-26 15:05:52 and completed at 17:02:30
(Asia/Shanghai). Each of the twelve shard processes exited 0; the merged raw
outputs contain 3750 and 503 unique successful sample IDs. v1 scoring used
LongBench commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`; the exact
scorer script hashes are in the two aggregate JSON files. The scored raw
outputs and all logs remain at the remote path in [README.md](README.md).
