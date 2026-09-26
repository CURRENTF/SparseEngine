# Palu author HF versus SparseEngine Palu

Both runs use the same Llama-3.1-8B-Instruct source model and frozen BF16
group-size-1, K/V rank-96 factors. The author HF model was run on 4 x
NVIDIA GeForce RTX 4090 48 GB with one model per GPU (TP1).

| Benchmark | Samples | Author HF Palu | SparseEngine Palu | HF minus SparseEngine |
|---|---:|---:|---:|---:|
| LongBench v1 overall category average | 3750 | 41.60 | 41.77 | -0.17 |
| LongBench v2 accuracy (%) | 503 | 27.038 | 28.032 | -0.994 |

## LongBench v1 categories

HF FullKV is a separate control condition, recorded in the
[FullKV result package](../palu_llama31_longbench_hf_fullkv_20260925/RESULTS.md).

| Category | Author HF Palu | SparseEngine Palu | HF FullKV |
|---|---:|---:|---:|
| SDQA | 36.37 | 36.64 | 43.41 |
| MDQA | 35.97 | 36.41 | 46.60 |
| SUM | 25.27 | 25.23 | 28.84 |
| FewShot | 65.81 | 66.42 | 69.36 |
| Syn | 47.00 | 47.00 | 55.04 |
| Code | 39.17 | 38.95 | 59.56 |
| Six-category average | 41.60 | 41.77 | 50.47 |

## LongBench v2 groups

All entries below are accuracy percentages. Domain, length, and difficulty
are separate partitions of the same 503 samples.

| Domain | Samples | Author HF Palu | SparseEngine Palu | HF FullKV |
|---|---:|---:|---:|---:|
| Code Repository Understanding | 50 | 16.00 | 18.00 | 36.00 |
| Long In-context Learning | 81 | 25.93 | 28.40 | 24.69 |
| Long Structured Data Understanding | 33 | 30.30 | 30.30 | 33.33 |
| Long-dialogue History Understanding | 39 | 20.51 | 17.95 | 10.26 |
| Multi-Document QA | 125 | 28.80 | 28.80 | 30.40 |
| Single-Document QA | 175 | 30.29 | 32.00 | 31.43 |

| Official length | Samples | Author HF Palu | SparseEngine Palu | HF FullKV |
|---|---:|---:|---:|---:|
| short | 180 | 33.33 | 33.33 | 33.89 |
| medium | 215 | 25.12 | 25.58 | 26.05 |
| long | 108 | 20.37 | 24.07 | 26.85 |

| Difficulty | Samples | Author HF Palu | SparseEngine Palu | HF FullKV |
|---|---:|---:|---:|---:|
| easy | 192 | 31.77 | 33.33 | 30.73 |
| hard | 311 | 24.12 | 24.76 | 27.97 |

The unchanged v2 scorer records 38, 35, and 23 answer parse failures for
Author HF Palu, SparseEngine Palu, and HF FullKV respectively. All three
scores use all 503 samples as the denominator; there were no model failures.

Author HF source commit: `bb22666e2ef96707e8dd21d93fc00146c2e0d615`.
SparseEngine Palu run commit: `a07c3e30be63646137cc6aeeff131adc987afe5d`.
The precise protocol and runtime cache semantics are in [README.md](README.md).
Raw per-sample outputs and logs remain on the remote persistent storage.
