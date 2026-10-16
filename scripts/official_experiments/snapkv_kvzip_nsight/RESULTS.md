# SnapKV and KVzip Nsight results

- Code: `5d96738429c16bbd2f2ba29f392b3fd74f084f02`; 2026-10-16.
- Device: one idle H100 80GB HBM3; Qwen3-4B-Instruct-2507 BF16, TP=EP=DP=1. Torch 2.11.0+cu130, Triton 3.6.0, SGL kernel 0.4.5, driver 580.65.06; SGL FA3 paged prefill/decode providers.
- Protocol: synthetic random-token matched prompts, seed 42, 32768 input / 128 output, concurrency 1 and 4, zero length jitter, greedy decoding, ignore EOS. Context=32640, protected prefix=64, question=128. Prefill chunk=8192, max model length=40832, GPU memory fraction=0.80.
- Storage: explicit `token` layout, prefix caching off, Graph on with buckets [1,2,4]. SnapKV retains 64 sink + 3968 selected + 64 recent tokens, observation window 64, max pool 7, decode eviction off. KVzip budget=4096 unprotected tokens averaged across layers/heads, reconstruction chunk=2000, previous postfix=8.
- Each arm: one warmup, three unprofiled measured workloads, one separately profiled matched workload after its own warmup. SnapKV logits is the default score mode; probability is an explicit supplementary arm. This is not a quality comparison or equal-physical-retention comparison. Head-slot storage was not measured.

## Unprofiled request measurements

TTFT and TPOT are pooled per-request means over three workloads, observed at engine token publication. TPOT=(finish-first)/127. E2E output throughput counts all 128 output tokens per request over total measured workload time.

| Method / score | Concurrency | TTFT (s) | TPOT (ms) | E2E output (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| SnapKV / logits | 1 | 1.010 | 5.125 | 77.05 |
| SnapKV / probability | 1 | 1.014 | 5.121 | 76.92 |
| KVzip / reconstruction | 1 | 8.216 | 5.983 | 14.26 |
| SnapKV / logits | 4 | 3.931 | 6.170 | 108.48 |
| SnapKV / probability | 4 | 3.979 | 6.175 | 107.36 |
| KVzip / reconstruction | 4 | 32.700 | 9.370 | 15.10 |

## Nsight Systems attribution

Values below sum GPU kernel execution time assigned by launch correlation to each NVTX scope. They are diagnostic, not unprofiled latency. Nested scopes are inclusive. Decode sums include overlapping kernels and therefore are not execution-stage wall time.

| Method | Concurrency | Prefill kernel sum (ms) | Score kernels (% prefill) | Attention (% prefill) | GEMM (% prefill) | Attention (% decode) | GEMM (% decode) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SnapKV / logits | 1 | 974.52 | 0.76 | 54.31 | 32.73 | 17.90 | 67.84 |
| SnapKV / probability | 1 | 994.73 | 1.58 | 54.12 | 32.31 | 17.89 | 67.84 |
| KVzip / reconstruction | 1 | 8258.55 | 71.08 | 18.55 | 7.56 | 29.95 | 57.52 |
| SnapKV / logits | 4 | 3892.85 | 0.74 | 54.77 | 32.51 | 26.45 | 58.49 |
| SnapKV / probability | 4 | 3926.42 | 1.52 | 54.28 | 32.35 | 26.36 | 58.54 |
| KVzip / reconstruction | 4 | 33007.22 | 71.06 | 18.58 | 7.55 | 52.95 | 37.28 |

## Bottlenecks and next experiments

1. **KVzip reconstruction scoring is the primary measured cost.** At concurrency 1, 17 reconstruction forwards score 36 layers each. `_prefill_score_final_kernel` takes 3118.96 ms, `_prefill_score_partial_lse_kernel` 2709.69 ms, and LSE reduction 41.16 ms. The score scope totals 5871.43 ms; reconstruction attention adds 1001.99 ms and reconstruction GEMM 305.29 ms. Global selection and compaction kernels total only 0.96 ms and 1.46 ms. At concurrency 4, the score scope totals 23462.40 ms, retaining the same approximately 71% prefill share.
2. **The first optimization hypothesis is to restrict reconstruction scorer launch domains.** Common measured score grids are 512 x 542 blocks; the final kernel needs only the current roughly 2000-token source chunk (about 32-33 of 542 key tiles), while partial LSE needs the protected prefix, current source chunk and causal replay. [prefill_score.py](../../../src/sparseengine/kernels/triton/prefill_score.py) launches over the full score width and masks positions inside each tile, with unconditional dot products. About 94% of final-kernel key tiles contain no candidate positions. Compact launch domains could remove unnecessary tiles without changing the score definition; speedup is unmeasured. Preserve absolute causal positions, reconstruction masks, FP32 normalization and per-head maximum semantics. Ordinary attention LSE has a different normalization domain and cannot simply replace reconstruction LSE.
3. **KVzip token-layout unions increase decode work.** The measured retained context averages are 20284.72 tokens/layer at concurrency 1 and 20412.09 at concurrency 4, including the protected prefix and excluding the question. Global per-head budget 4096 does not imply 4096 physical tokens after head unions. At concurrency 4, attention contributes 52.95% of KVzip decode kernel time versus 26.45% for SnapKV logits. A separate head-slot experiment should quantify its storage/provider tradeoff; this run does not establish a head-slot speedup or a quality-equivalent comparison.
4. **SnapKV overhead is small for these shapes.** Default-mode scoring is under 0.8% of prefill kernel time. Ordinary attention and GEMM dominate prefill; GEMM accounts for 58-68% of decode kernel time. Prioritize matched provider/shape analysis of these costs before optimizing selection or compaction. Findings are specific to this dense 4B model, context, budget and concurrency.

## Validation and limits

- All 18 unprofiled workloads and six profiled workloads completed the expected request counts and 128 output tokens/request. Input trace digests match across methods and corresponding profiles. Every measured workload had 127 Graph replays, zero capture/recapture and zero eager fallbacks. KVzip executed every expected context compaction.
- Nsight Systems 2025.3.2 collected CUDA/NVTX/OS runtime traces. Nsight Compute 2026.1.0 failed its counter access check with `ERR_NVGPUCTRPERM`; achieved bandwidth, occupancy, roofline and compute-versus-memory limitations are unmeasured.
- Profiler perturbation is visible: SnapKV logits concurrency-1 TPOT is 5.125 ms unprofiled and 6.219 ms profiled. Do not use profile wall time as serving latency. Full captures include teardown; CUDA API totals outside measured NVTX scopes must not be mistaken for workload bottlenecks.

Reproduce with [run.sh](run.sh); validate and reaggregate with [analyze.py](analyze.py). Compact values are in [summary.json](summary.json). The general workflow and targeted Nsight Compute command are in [README.md](README.md). Raw case directories retain `run_manifest.json`, `raw_samples.jsonl`, `request_samples.jsonl`, `timeline.nsys-rep`, `timeline.sqlite` and exported `stats_*.csv` outside Git.
