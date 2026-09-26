# Qwen3-4B-Thinking-2507 AIME 2024: local HiSparse QuEST, C20

Status: complete, **quality anomaly**. Device: one NVIDIA H100 80GB HBM3,
physical GPU4 (`GPU-74a210c8-c8d6-f512-a42b-9afa5ef8f79c`), TP1.
SparseEngine checkout: `7b32be0fa67fc06d0df8bfabab5a2700f8825591`.
The SGLang/HiSparse checkout is based on
`6ef651c0e83e0e6333e2e080e3865812e620de0e`; the applied source diff
against that commit has SHA-256
`b5f962693f87f912f5ab43e81b075e7e7a70bd120f2b1d1c3eb07424d135172c`.
The full diff is saved as `source.patch` in the raw run root.

The same 60 tokenized AIME 2024 requests as the
[two-try comparison](../2026-09-25-qwen3-4b-thinking-official-sampling-2try-c32c64/RESULTS.md)
were submitted in one `Engine.generate()` call. Input SHA-256:
`cb269f382ebd5fdded3d4fa8824611f0f68856d44ec2472462b2a3ce7ff35022`.
Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42,
40,960 new tokens maximum, 41,984 context tokens maximum. SGLang used BF16,
FA3, page size 16, `max_running_requests=20`, `mem_fraction_static=0.9`,
prefix cache off, and breakable decode CUDA Graph. QuEST selected at most 128
historical pages plus 5 recent pages, or 2,128 attended tokens. The framework
kept full KV in GPU memory; the selected page budget was an attention budget.
Unlike the native SparseEngine QuEST run, the first two layers were not exempt
from sparse selection.

| Result | Value |
|---|---:|
| Correct | 4/60 (6.67%) |
| Trial 0 / trial 1 | 4/30 / 0/30 |
| Questions with at least one / both correct | 4/30 / 0/30 |
| Complete 60-request `Engine.generate()` time | 1716.0 s |
| Generated completion tokens | 2,438,993 |
| Outputs stopped by the 40,960-token cap | 58/60 |
| Logged KV-full request retraction events | 39 |

All 60 generations and MathBench evaluations returned success. The time
includes admission, prefill, decode, selection, scheduler waits, retractions,
and recomputation inside the generation call; it excludes engine startup and
answer evaluation. Long output reduced actual residency below the C20 limit,
and retracted requests were re-prefilled. This run overlapped with the official
vLLM R-KV run on GPU2 of the same host. Treat the time as a run observation,
not an isolated cross-engine speedup.

The quality result needs a correctness check before it can represent QuEST's
method quality. In a deterministic 512-token diagnostic on AIME request 1
(214 prompt tokens), dense SGLang and HiSparse with sparse selection disabled
produced identical text. Enabling QuEST selection changed the text after 443
shared characters with decode CUDA Graph, or after 664 with eager decode. The
entire diagnostic context was below the 2,128-token attention budget, so all
history pages should be selected. This narrows the discrepancy to the enabled
sparse path; it does not by itself prove which kernel, page metadata, or
floating-point difference caused it. Raw diagnostic outputs are under
`/data2/haojitai/outputs/Sparse-vLLM/aime_hisparse_short_diagnostic_20260925`.

The earlier native SparseEngine QuEST C20 run with a 2,128-token budget scored
40/60 in 907.2 s, but its selection rule and first-two-layer policy differ.
The observed 4/60 here should therefore not be used as an equal-method
quality or efficiency comparison.

Launch: `scripts/tmp/run_aime_hisparse_quest_c20_20260925.sh`.
Generation entrypoint: `scripts/official_experiments/aime/run_hisparse_quest.py`.
Raw command, manifest, per-sample results, logs, and patch are under
`/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_hisparse_quest_c20_20260925`.
