# GLM-4.7-Flash MiniSWE radix-pruning results

Common launch arguments: SWE-bench Lite `test` 300, GLM-4.7-Flash BF16,
SparseEngine TP2/EP2/DP1 on 2× H20 96GB, agents/decode/resident=24/24/64,
`favor_min_decoding_seqs=18`, `max_model_len=202752`,
`max_num_batched_tokens=65536`, prefill chunk 4096, CUDA Graph through batch 24,
GPU memory fraction 0.95, and prefix-cache offload disabled. MiniSWE uses
`max_tokens=16384`, `temperature=0.7`, `top_p=1`, preserved thinking, 80 steps,
and a 7200-second task limit.

All methods use a radix prefix cache without Chain Cache. After each eligible
turn, `kvzip_global` jointly scores the accumulated aligned tool-result ranges
and keeps 20% of their tokens; user input, assistant reasoning, and assistant
output are not pruning targets.

| Sparse method | Method launch arguments | Resolved | Final status counts | Git commit |
|---|---|---:|---|---|
| Vanilla | Dense attention | 71/300 (23.67%) | 135 success, 161 model failed, 4 metric failed | `ed632bc3eb21a32e502ffb31d47e4c59a6eacfd2` |
| QuEST | sink/recent/selected=64/512/1472; page size 16; first 2 layers full | 64/300 (21.33%) | 120 success, 173 model failed, 7 metric failed | `ed632bc3eb21a32e502ffb31d47e4c59a6eacfd2` |
| OmniKV | sink/recent/selected=64/512/1472; full layers=auto | 67/300 (22.33%) | 127 success, 169 model failed, 4 metric failed | `ed632bc3eb21a32e502ffb31d47e4c59a6eacfd2` |

Vanilla and OmniKV apply the requested token budget directly. QuEST selects
16-token pages and rounds the shared keep budget down to whole pages.

These are closed-loop application outcomes: each method generated its own tool
trajectory. The shared launch protocol does not make the token traces identical
or isolate the effect of decode sparsity from prefix pruning.
