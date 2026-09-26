# Qwen3-30B-A3B-Instruct-2507-FP8: Gasai agent trace

This run replays 100 seed-42 Gasai trajectories as 5,210 requests with
1,036,727 forced completion tokens. The client uses deterministic synthetic
tool pauses in `[0, 2)` seconds (seed 42), requires observed cache reuse,
and measures the complete replay after the full server is ready. Model loading,
CUDA Graph capture, and the separate smoke server are outside the timed replay.
The trace manifest SHA-256 is
`4126a043d03dfc394ffaa10b6b8ebc5fc8cae31cfa52b6e48c60c802e6ec455c`;
the prepared Qwen forced-workload digest is
`4d6ebda674123e274fe0c3807f279786afca5e43ad83be58d29446b235c34bda`.

All cases used the same two NVIDIA H20 GPUs, TP2/EP2/DP1, CUDA Graph,
`max_decoding_seqs=max_num_seqs_in_batch=C`, and
`max_num_seqs_in_gpu=1.5*C`. GPU UUIDs, per-case engine launch arguments,
latency percentiles, cache reuse, and artifact locations are in
[gasai_qwen3_fp8_results.json](gasai_qwen3_fp8_results.json). The runner
recorded Git HEAD `a07c3e30be63646137cc6aeeff131adc987afe5d`.

| Method and cache setting | C | Resident rows | Replay time (s) | Output token/s | Vanilla C32 time / case time |
|---|---:|---:|---:|---:|---:|
| Vanilla, radix prefix | 32 | 48 | 2,993.70 | 346.30 | 1.00× |
| QuEST, radix prefix | 32 | 48 | 3,532.43 | 293.49 | 0.85× |
| OmniKV, radix prefix | 32 | 48 | 1,837.36 | 564.25 | 1.63× |
| SnapKV 16K, Chain Cache, decode eviction | 52 | 78 | 1,800.23 | 575.88 | 1.66×* |
| SnapKV 8K, Chain Cache, decode eviction | 72 | 108 | 1,333.80 | 777.27 | 2.24×* |
| H2O, Chain Cache, probability score, decode eviction | 80 | 120 | 1,502.06 | 690.20 | 1.99×* |

Every listed case completed all 5,210 requests and 1,036,727 output tokens,
with observed prefix or chain reuse, zero active-request preemptions, and zero
recompute replays. An asterisk marks a comparison across different concurrency
settings; these ratios describe the requested end-to-end serving configurations
and do not isolate a method-only speedup.

An earlier SnapKV 16K C32 diagnostic also completed 5,210 requests with zero
preemptions and recompute replays: 1,832.04 s, 565.89 output token/s, and
1.63× Vanilla C32 replay-time ratio. The requested 16K configuration is C52,
listed in the main table. The C32 artifact and launch arguments are retained
under `diagnostic_results` in the result JSON.

SnapKV 16K retains `64+512+15,808` tokens; SnapKV 8K retains
`64+512+7,616`. Both enable decode eviction at a 1,024-token interval.
H2O uses the default probability-score setting, 16K prefill and 8K decode
budgets, with decode eviction enabled at a 128-token interval. This H2O run
does not use the separate SWE-lite FP32-logits/no-eviction override.

QuEST, OmniKV, and H2O ran in an isolated checkout that extended the
Qwen3-MoE TP2/EP2 prefix-cache validation gate for these methods. Each passed
GPU smoke, complete replay, cache-hit checks, and the zero-preemption gate;
this measurement does not by itself establish support beyond the tested model
and topology. Large per-request artifacts remain in the persistent run output.
