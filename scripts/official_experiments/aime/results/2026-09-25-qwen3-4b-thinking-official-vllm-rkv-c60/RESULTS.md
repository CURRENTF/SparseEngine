# Qwen3-4B-Thinking-2507 AIME 2024: official vLLM R-KV, C60

Status: complete. Device: one NVIDIA H100 80GB HBM3, physical GPU2
(`GPU-566a0ffb-6f93-413f-cdf1-bab7f7acd20e`), TP1. SparseEngine checkout:
`7b32be0fa67fc06d0df8bfabab5a2700f8825591`. The official R-KV checkout is
`6715468b9872442be72e5c97322e4d9c9a2abf55`; its vLLM patch was applied
to the pinned vLLM revision `752a3a504485790a2e8491cacbb35c137339ad34`.

The exact 60 tokenized requests from the [two-try AIME run](../2026-09-25-qwen3-4b-thinking-official-sampling-2try-c32c64/RESULTS.md)
were submitted in one `LLM.generate()` call. Input SHA-256:
`cb269f382ebd5fdded3d4fa8824611f0f68856d44ec2472462b2a3ce7ff35022`.
Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42,
40,960 new tokens maximum, 41,984 context tokens maximum. The engine used
`max_num_seqs=60`, `max_num_batched_tokens=65536`, block size 16, BF16,
`gpu_memory_utilization=0.87`, prefix cache off, async scheduling on, and
PIECEWISE CUDA Graph. R-KV used total retained budget 4,176, buffer/compaction
interval 1,024, observation window 8, kernel size 7, mix lambda 0.1,
retain ratio 0.2, batched scoring, and physical block release.

| Result | Value |
|---|---:|
| Correct | 34/60 (56.67%) |
| Trial 0 / trial 1 | 17/30 / 17/30 |
| Questions with at least one / both correct | 19/30 / 15/30 |
| Complete 60-request `LLM.generate()` time | 805.9 s |
| Generated completion tokens | 1,651,188 |
| Actual R-KV compactions | 1,328 |

All 60 generations and MathBench evaluations succeeded. The time includes
admission, prefill, decode, scoring, compaction, scheduling, and any wait inside
the generation call; it excludes engine initialization and answer evaluation.
All 73 periodic engine log samples recorded waiting queue 0, and the log has
no preemption message. This is evidence of no observed preemption, rather than
a per-request event count. The official run overlapped with a HiSparse QuEST
run on GPU4 of the same host, so its 805.9 s is a run observation, not an
isolated cross-engine efficiency measurement.

The official vLLM port's `VLLM_V1_R_KV_SCORE_CHUNK_MB` caps an entire
per-request similarity matrix; it does not tile the sequence dimension. With
the requested 512 MB, a first compaction needs at least about 3.7 GB under the
port's own 2x admission estimate (8 KV heads and at least 5,204 cached tokens).
It therefore skips compaction, fills the KV pool, and causes scheduler waits.
That C60 attempt was stopped and is excluded from the result. The completed
run raised this **scoring workspace cap** to 8,192 MB and enabled compression
trace logging; it did not change the retained budget, scoring formula, or
sampling. A preflight retry after a GPU-query timeout also produced no samples.

For context, the earlier SparseEngine RKV run scored 31/60 at C64 and took
867.5 s; Vanilla scored 49/60 at C32 and took 1087.8 s. The official port's
34/60 does not establish whether the three-answer difference comes from
selection semantics or sampling randomness. The methods have different cache
selection and RNG implementations, and the timing runs were not isolated
against one another.

Launch: `scripts/tmp/run_aime_rkv_vllm_c60_scorecap8192_20260925.sh` with
`RKV_VLLM_RUN_ROOT=/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_official_vllm_rkv_c60_scorecap8192_retry1_20260925`.
Generation entrypoint: `scripts/official_experiments/aime/run_rkv_vllm.py`.
The raw run root contains the exact command, environment, status, model output,
per-sample scores, and logs. The stopped 512 MB attempt is retained under
`/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_official_vllm_rkv_c60_20260925`.
