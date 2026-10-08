# MiniMax M2.7 32K request decode rerun (2026-09-26)

## Identity and protocol

- Device: guest-KR6288, 4 × NVIDIA H100 80GB (physical GPUs 4–7), TP4/EP4.
- Model: MiniMax-M2.7 block-FP8 checkpoint; SparseEngine first-launch commit `7f99bc77b55e5b24099110fdea074a66b74ebe2d`, repeat commit `6feb77078ed25beae07cc977b52b9a9fe2d47b20`; Vortex fork commit `c016fdfc871130481c76e3beead83e8c2b54cb99`. The SparseEngine commit difference changes README and supported-model documentation only.
- Launch: `benchmark/efficiency/bench_probe.py --engine sparseengine --sparse-method quest --prompt-lens 32768 --output-lens 512 --batch-sizes 4 --scenario fixed --seed 42 --tensor-parallel-size 4 --expert-parallel-size 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --num-warmups 1 --num-iters 3`, with QuEST `decode_keep_tokens=4672`, `quest_chunk_size=16`, `quest_skip_layers=2`, `decode_graph=true`, and prefill chunk 8192. QuEST was measured in two separate launches.
- Vortex launch: FlashInfer/TRTLLM attention, `gqa_quest_sparse_attention`, page size 16, top-k 291 previous pages plus one reserved EOS page, dense layers 0 and 1, context limit 33408, prefill chunk 8192, max 4 requests, CUDA Graph max batch 4, static memory fraction 0.90, followed by the Vortex fixed-batch probe with the same prompt/output/batch/seed/warmup/iteration arguments. The benchmark trace generator was aligned to `random-varlen-v2`; the fork used a rank-local DeepGEMM cache adaptation.
- All three runs completed. Each of the three measured iterations has identical prompt lengths, output lengths, and prompt digests across runs; each produced 4 × 512 output tokens. SparseEngine recorded 511 CUDA Graph replays and zero forced-eager steps in each iteration. Vortex completed CUDA Graph capture on all four ranks.

## Results

The TPOT diagnostic is `batch_size × 1000 / TPOT_ms` per iteration, then the arithmetic mean of three iteration rates. Higher diagnostic and end-to-end output throughput are better; lower TPOT and TTFT are better.

| Run | TPOT observation | TPOT by iteration (ms) | Mean batch/TPOT diagnostic (tok/s) | TTFT P50 (s) | End-to-end output (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| SparseEngine QuEST, first launch | engine request first token to finish | 15.23 / 17.66 / 19.89 | 230.08 | 2.56 | 170.77 |
| SparseEngine QuEST, repeat | engine request first token to finish | 15.23 / 15.26 / 15.20 | 262.67 | 2.56 | 197.26 |
| Vortex QuEST | HTTP client stream | 13.76 / 13.74 / 13.51 | 292.63 | 12.22 | 100.47 |

The first QuEST launch slowed in its second and third measured iterations; the repeat did not. The reason is undetermined, so both launches are retained. SparseEngine timestamps token publication inside the engine, while Vortex timestamps HTTP responses at the client. These request-level TPOT values are diagnostic proxies, not continuous decode-stage throughput; the differing observation boundaries and toolchains do not support a strict speedup claim. TTFT and end-to-end rates include prefill and serving overhead.

Source artifacts: the paired QuEST/Vortex `raw_samples.jsonl` and `summary.json` are under `/data1/haojitai/outputs/Sparse-vLLM/minimax_m27_quest_32k_rerun_20260926_181210/{quest,vortex}/`; the QuEST repeat and its `operator_runtime_stats.json` are under `/data1/haojitai/outputs/Sparse-vLLM/minimax_m27_quest_32k_repeat_20260926_183227/quest/`.

## Historical comparison boundary

The 2026-09-01 MiniMax QuEST figure of 302.98 tok/s used `batch_size / mean synchronized decode-step duration`: its probe called `torch.cuda.synchronize()` inside every `llm.step()` and averaged steps labeled decode. The old Vortex figure of 296.03 tok/s came from HTTP client stream TPOT, so that historical QuEST/Vortex ratio also mixed observation boundaries. The present QuEST diagnostic uses mean per-request `(finish − first token) / 511` without per-step synchronization. The old trace generator was `random-varlen-v1`; the present one is `v2`. Therefore 302.98 versus 262.67 is not a measured regression.

Both artifacts also contain a first-token-to-last-completion event window: the old QuEST run reports 189.66 tok/s, and the stable present repeat reports 221.59 tok/s. Whole-workload output throughput is 173.24 versus 197.26 tok/s. These directionally different figures also cannot establish a code speedup because the traces and synchronization policy changed. The August optimized-versus-pre-optimization QuEST latency result was already included in the September 1 code. The later QuEST scoring/FP8-MoE optimization does not exercise its tensor-core scoring or Triton fused-MoE path in this MiniMax run: runtime statistics select scalar Triton scoring and FlashInfer CUTLASS FP8 MoE.

Historical source: `/data1/haojitai/outputs/Sparse-vLLM/quest_vortex_decode_glm47_minimax_32k_final4_20260901_0112/` (`matched_decode_analysis.json` and `minimax/sparsevllm/{raw_samples.jsonl,summary.json}`).
