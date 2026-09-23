# GLM-4.7-Flash MiniSWE official results

Common launch arguments: SWE-bench Lite `test` 300, GLM-4.7-Flash BF16,
`max_model_len=202752`, `max_tokens=16384`, `temperature=0.7`, `top_p=1`,
`step_limit=80`, `wall_time_limit_seconds=7200`, preserved thinking,
`max_num_batched_tokens=65536`, `engine_prefill_chunk_size=4096`, and CUDA Graph,
except where a row explicitly overrides them.

| Method | Device | Method launch arguments | Final result | Git commit |
|---|---|---|---:|---|
| Vanilla | 2× RTX PRO 6000 | vLLM; TP1/DP2/EP2; agents=16; max_num_seqs=8 per replica; native prefix cache; prefix pruning disabled | 75/300 (25.00%) | `3b232adb1f86112f939ec61a06dbdb3ac4da3174` |
| SnapKV | 2× RTX PRO 6000 | SparseEngine; TP2/EP2/DP1; agents/decode/resident=40/40/64; favor_min_decoding_seqs=32; Chain Cache; sink/recent/selected=64/512/15808; prefix pruning disabled | 74/300 (24.67%) | `58bb6b26c8c5e1267be66b2e7c0ca6860247dc83` |
| H₂O | 2× H100 80GB | SparseEngine; TP2/EP2/DP1; agents/decode/resident=64/64/64; favor_min_decoding_seqs=48; Chain Cache; prefill/decode budgets=16384/8192; probability scoring; online decode eviction every 128 steps; prefix pruning disabled | 39/300 (13.00%) | `8611054c264d346e5e6dfc5e288ea192a045bd14` |
| H₂O, logits/no decode eviction | 2× H20 | SparseEngine; TP2/EP2/DP1; agents/decode/resident=80/80/80; favor_min_decoding_seqs=48; Chain Cache; prefill/decode budgets=16384/8192; prefill chunk=16384; FP32 logits scoring; decode eviction disabled; decode reservation=1024; prefix pruning disabled | 64/300 (21.33%) | `58138120c4cbb55dbce7f7fb7525bbbb64ac0b8d` |
| OmniKV | 2× H100 80GB | SparseEngine; TP2/EP2/DP1; agents/decode/resident=16/16/16; favor_min_decoding_seqs=10; radix prefix cache; sink/recent/selected=64/512/1472; full layers=auto; prefix pruning disabled | 93/300 (31.00%) | `58bb6b26c8c5e1267be66b2e7c0ca6860247dc83` |
| QuEST | 2× H20 | SparseEngine; TP2/EP2/DP1; agents/decode/resident=16/16/16; favor_min_decoding_seqs=10; radix prefix cache; sink/recent/selected=64/512/1472; page size=16; first 2 layers full; prefix pruning disabled | 91/300 (30.33%) | `e720f93864881ce841f6da6e5e64381df2eb5be6` |

`*-prefix` denotes prefix caching, not prefix pruning. None of the listed runs
used prefix-cache pruning. The devices and concurrency settings differ, so the
table records deployed end-to-end outcomes rather than a matched algorithm-only
comparison.

The H₂O logits/no-eviction result combines 215 validated completed predictions
from the earlier run with 85 newly generated predictions in the recovery run.
The official SWE-bench Lite `test` evaluator scored all 300: 112 `success`,
183 `model_failed`, 5 `metric_failed`, and 0 incomplete. Source metric:
`resolved_instances=64`, `total_instances=300`, `score=0.21333333333333335`
in the recovery run's
`h2o-chain/full/benchmark/final_summary.json`; per-instance statuses are in
`h2o-chain/full/benchmark/per_sample_results.jsonl`. Its 6,106 collected server
request logs cover only the 85 newly generated tasks, so they are not a
300-task request-latency or throughput result.
