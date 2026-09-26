# Gasai longest-60: Qwen3 FP8 time to 2,000 completed requests

The selected 60 Gasai trajectories contain 7,844 requests. This experiment
stopped each replay once it had at least 2,000 successful HTTP responses. The reported time
starts at the first server `request_start` and ends at the 2,000th
`request_finish`, with one-second log timestamp precision. Model loading and
server CUDA Graph capture are excluded; runtime compilation during replay and
deterministic synthetic tool pauses in `[0, 2)` seconds are included. Forced
recorded tokens preserve each trajectory's prompt continuation. These are
**short-window serving results**, not complete-trajectory end-to-end times.

The model was `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` on two 96 GiB NVIDIA
H20s, TP2/EP2/DP1. Every case used `max_decoding_seqs=C` and
`max_num_seqs_in_gpu=1.5*C`. The trace manifest SHA-256 is
`4013570c0c4f22d5440441e023f05967cd90aa13f6af222084553bfd02e0ca3a`;
the target-model forced-workload SHA-256 is
`1e6ed6523ff3d239f25751889d6ea8b69e1aea151b6fa1f6340fd65d9df7c563`.
GPU UUIDs, Git HEAD, exact case artifact paths, archive hashes, and per-case
settings are in [the result JSON](gasai_longest60_qwen3_fp8_target2000_results.json).

| Method and cache | C | Time to 2,000 (s) | Prompt tokens in 2,000 | Output tokens in 2,000 | Ratio to Vanilla C24 |
|---|---:|---:|---:|---:|---:|
| Vanilla, radix prefix | 24 | 1,105 | 72,387,541 | 356,215 | 1.00× |
| QuEST, radix prefix | 24 | 943 | 72,375,836 | 356,040 | 1.17× |
| OmniKV, radix prefix | 24 | 607 | 72,394,745 | 356,705 | 1.82× |
| SnapKV 16K, Chain Cache, decode eviction | 52 | 538 | 41,830,343 | 317,585 | — |
| H2O, Chain Cache, probability score, decode eviction | 80 | 487 | 38,202,920 | 305,639 | — |
| SnapKV 8K, Chain Cache, decode eviction | 72 | 447 | 38,186,575 | 305,904 | — |

The C24 cases processed nearly equal token work at this boundary, so their
time ratios are useful short-window comparisons. The higher-concurrency cases
completed a different mix of turns and substantially fewer prompt tokens by
their 2,000th response. Their times cannot establish a speedup against the C24
Vanilla case. There are only 60 agents, so configured C72/C80 never means more
than 60 concurrently active agents.

Every listed case reached 2,000 HTTP 200 responses, with zero recorded errors
and cancellations before the target. The historical monitor recorded no
recompute replay starts, but it did not count preemptions of requests that had
not generated an output token. Active-request preemption counts are therefore
unverified until the archived server logs are checked; the result JSON records
them as `null`. Prefix methods had observed cached tokens or radix hits; chain
methods had observed resumed requests. Cancellations after the target were caused
by intentionally stopping the replay client. QuEST's initial monitor did not
record the HTTP 200 count in its summary, so the selected server log was checked
separately: the 2,000th
HTTP 200 immediately followed the 2,000th `request_finish`, before any cancel.

The first SnapKV 16K C52 attempt failed after 1,842 `request_finish` lines:
the remote filesystem exhausted its inode quota, causing 60 HTTP 500 responses.
That attempt is excluded. Earlier request logs were archived to release inodes;
the valid C52 result comes from a fresh output directory. The original Vanilla
full replay was deliberately stopped after its valid 2,000-request boundary.

The remote persistent artifact root is
`/root/autodl-fs/outputs/Sparse-vLLM/gasai_agent_5k_longest60_turns_qwen_fp8_20260926`.
The result JSON gives each case directory relative to this root. Request logs
are retained as verified `server_requests.tar` archives there.

To run one case with the same short-window protocol, set the checkpoint, trace,
forced-workload, output, GPU, port, scratch, and environment paths explicitly:

```bash
VENV_ACTIVATE="$VENV_ACTIVATE" SPARSEVLLM_MASTER_PORT="$MASTER_PORT" \
AGENT_TRACE_SCRATCH_BASE="$SCRATCH_ROOT" \
AGENT_TRACE_COMPLETION_TARGET=2000 AGENT_TRACE_REQUEST_TIMEOUT_S=86400 \
AGENT_TRACE_ALLOW_PREEMPTIONS=1 AGENT_TRACE_SKIP_SMOKE=1 \
bash scripts/official_experiments/agent_trace_sparse_methods/run_case.sh \
  qwen3_30b_a3b_fp8 "$MODEL_DIR" "$METHOD" "$C" "$OUTPUT_ROOT" \
  "$TRACE_DIR" "$GPU_PAIR" "$PORT" full "$FORCED_WORKLOAD"
```

For either SnapKV case, also set `AGENT_TRACE_SNAPKV_DECODE_EVICTION=1`; for
the 8K case, set `AGENT_TRACE_SNAPKV_TOTAL_BUDGET=8192`. All other method
settings come from `chain_cache_miniswe/setting.json`. The successful remote
run used Git HEAD `a07c3e30be63646137cc6aeeff131adc987afe5d` with the
runner file digests recorded in the result JSON.
