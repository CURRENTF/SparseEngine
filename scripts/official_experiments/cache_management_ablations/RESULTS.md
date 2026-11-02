- Launch: `ssh pro5000 'cd /data/haojitai/code/SparseEngine-ContextCompaction && bash scripts/tmp/paper_ablations/run.sh'` (RTX PRO 5000 72GB GPU 1, authorized sharing). Full per-case arguments, engine settings and retained-prediction scoring commands are in the `launch_command` field of [results.json](results.json).
- Git commit: `2c984f63a5a091ffb5c8045cb39ea980c30a3151` (PRO5000 runtime; dirty worktree).
- Final results: completed all 16 cases; validated 4,000 successful replay requests and 300 quality outcomes against raw artifacts.

| Chain Cache | Off E2E (s) | On E2E (s) | E2E speedup | Off TTFT (ms) | On TTFT (ms) | TTFT speedup | On cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SnapKV | 901.92 | 616.41 | 1.463x | 1996.39 | 1287.40 | 1.551x | 96.82% |
| H2O | 872.39 | 578.04 | 1.509x | 1931.87 | 1192.84 | 1.620x | 96.82% |

| Prefix pruning | Sampled prompt KV (GiB) | KV reduction | Replay E2E (s) | Runtime JIT events | Quality resolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.180 | 0.00% | 1158.01 | 16 | 15/50 (30%) |
| keep10_lag4 | 0.710 | 39.82% | 1277.82 | 25 | 9/50 (18%) |
| keep20_lag4 | 0.761 | 35.50% | 1288.35 | 0 | 10/50 (20%) |
| keep50_lag4 | 0.916 | 22.35% | 1305.22 | 2 | 11/50 (22%) |
| keep20_lag0 | 0.715 | 39.45% | 1308.50 | 0 | 10/50 (20%) |
| keep20_lag2 | 0.741 | 37.24% | 1294.69 | 0 | 11/50 (22%) |

At keep=20%, lag=4, sampled prompt KV falls by 35.50%, while replay E2E rises from 1,158.01 s to 1,288.35 s and quality changes from 15/50 to 10/50. The sweep demonstrates a memory/quality/maintenance tradeoff; it does not support a no-quality-loss or E2E-speedup claim.

[Chain Cache turn curves](chain_cache_ablation.pdf) · [Prefix pruning sensitivity](prefix_pruning_sensitivity.pdf) · [Per-turn measurements](turn_metrics.csv)

Shared-GPU measurements on RTX PRO 5000 72GB GPU 1; Qwen and GLM weights are FP8, while the GLM latent KV payload is BF16.

Replay uses 10 fixed trajectories and all 40 turns per trajectory, concurrency 1, forced recorded responses, and zero tool waits. Quality uses independent closed-loop runs on the same fixed 50-task SWE-bench Lite cohort.

Pruning reduces sampled resident prompt KV payload; these values do not measure allocated GPU memory, workspace, generated suffixes, or maximum concurrency.

Replay end-to-end times include pruning and validation RPCs and reported first-use runtime JIT; pruning did not improve end-to-end elapsed time in this workload.

Unpruned GLM quality was 30%; pruned settings scored 18-22%. These results do not establish quality preservation. Each setting has one stochastic run on 50 tasks.

All 300 quality outcomes remain in their original denominators, including empty or invalid model patches and context/step/format limits. The two metric_failed outcomes are invalid model patches, not unresolved scorer infrastructure failures.
