# 模拟 Deep Research Serving Benchmark

此 benchmark 是确定性的 serving workload，不是研究质量评估。BrowseComp-Plus 仍是独立的端到端质量 benchmark。

## Workload

默认 run 模拟一个 research job。`--num-jobs` 控制完整 job 总数，`--job-concurrency` 限制同时执行的完整 job 数。每个 job 独立执行以下 workload：

- 10 个顺序 research round。
- 每轮 20 个并行 subagent request。
- 可选参数 `--min-subagents-per-round` 和 `--max-subagents-per-round` 会替代固定数量，为每个 job 的每一轮独立采样一个闭区间内的数量。
- 每个 subagent 接收 64 个 query token 加一个 heavy-tailed article length：60% 从 1,000–8,000 token 采样，25% 从 8,001–16,000 采样，10% 从 16,001–32,000 采样，5% 从 32,001–64,000 采样。
- Subagent output 使用第二个 heavy-tailed distribution：90% 从 100–600 token 采样，10% 从 800–1,500 采样，并设置 `ignore_eos=true`。
- 每个 subagent barrier 之后，一个 main-agent request 将当前 round 的回答压缩成均匀采样的 512–1,024 token round summary。
- Main-agent prompt 使用稳定的 synthetic system/query prefix、累计的前序 round summary 和当前 round 回答。这样无需保留每个 raw answer，也能产生确定性的跨轮 prefix reuse。
- 最后一个 main-agent request 使用所有 round summary，均匀采样生成 1,000–2,000 token。

因此，默认配置下每个 job 发出 211 个模型 request：200 个 SnapKV subagent request、10 个 OmniKV/vanilla round-summary request 和 1 个 OmniKV/vanilla final request。Synthetic token ID 可以精确控制所需 prompt length，避免把 serving benchmark 变成 tokenizer-content benchmark。默认 article 和 subagent output 的期望长度分别约为 10,500 和 430 token。最大 request 至少需要 65,564 token 的 context limit。

## Non-Uniform Router Contract

runner 调用 smart router 的 `/v1/completions` endpoint。

- Subagent 发送 `sengine_method_preference=snapkv`。
- Main-agent request 发送 `sengine_method_preference=omnikv,vanilla`。
- 可选的 `--subagent-required-tags` 和 `--main-agent-required-tags` 会作为 `sengine_required_tags` 转发。这样两个使用相同方法的 worker（例如双 worker vanilla baseline）仍可分配给不同角色。
- Preflight 要求所选模型至少有两个 healthy worker，并验证 worker 广告的方法和 context capacity 覆盖两个 agent role。使用同一模型但方法无关的 worker 不约束 benchmark。
- 每个符合 main-agent routing 条件的 worker 都必须启用 prefix cache，并报告正整数 `prefix_cache_block_size`，且不能大于 workload 保证可复用的 main-agent prefix：`main_overhead_tokens + max(0, rounds - 1) * min_round_summary_tokens`。
- 每个 response 必须包含 `X-SparseVLLM-Worker`、`X-SparseVLLM-Route-Reason`、`X-SparseVLLM-Sparse-Method` 和 `X-SparseVLLM-Prefix-Matched-Tokens`。
- request 被路由到错误方法时记录为 `metric_failed`，active round 记录完成后 run 停止。
- 如果 router 或 benchmark-eligible worker 报告 `git_dirty=true`，preflight 会拒绝。提供 `git_commit` 的 source revision 必须报告 `git_dirty=false`，即使同时提供 package version。仅当 installed package 的 `git_commit=null` 且提供 package version 时，才允许 `git_dirty=null`。

该 contract 要求使用独立 worker，因为一个 worker 只广告一种 sparse method。典型部署使用一张 GPU 运行 main-agent OmniKV/vanilla worker，另一张运行 SnapKV subagent worker。由于 workload 有意复用不断增长的 prefix，main-agent worker 必须启用 prefix cache；preflight 会拒绝未启用的 eligible main-agent worker。它也会拒绝大于保证 reusable-prefix bound 的 block size，并指示 operator 增加 round/minimum summary length 或减小 block size。默认 bound 为 4,736 token，因此默认 workload 不允许 8,192-token prefix-cache block。benchmark 不会启动或重新配置这些 service。

## 运行

按照 OpenAI serving runbook 启动 worker 和 router，然后运行：

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/<RUN_NAME>
```

client timeout 必须比 router upstream timeout 至少多 `--router-timeout-margin-s`。默认 client timeout 为 930 秒、margin 为 30 秒，systemd router 使用 `SPARSEENGINE_ROUTER_REQUEST_TIMEOUT_S=900`。独立 router control-plane timeout 应保持较短，默认 5 秒。

worker 根据 immutable dispatcher snapshot 响应内部 routing-load 和 prefix-match probe，因此很长的同步 prefill step 不会让较短的 control timeout 误判 worker failure。完整 `/v1/worker/load` control endpoint 仍与 engine 同步，router 的 short-timeout probe 不使用它。

router 必须有 worker 广告的方法满足 `--subagent-methods snapkv` 和 `--main-agent-methods omnikv,vanilla`。默认 profile 下，每个 eligible subagent worker 需要 `max_model_len >= 65564`，每个 eligible main-agent worker 需要 `max_model_len >= 40368`。也可以使用 69,632 等更大的共享限制。preflight 从每个 eligible worker 读取这些限制，而不是使用 router model card 的 cross-method minimum。

synthetic workload 可由 `seed` 精确复现；runner 为每个 job 使用 `seed + job_index`，不注入 per-run nonce。配置随机 subagent-count range 时，独立的确定性 random stream 为每个 `(job_index, round_index)` 采样一个数量。相同 seed 的 variant 会得到相同 sampled fanout，同时不改变 fixed-count run 使用的现有 token-length stream。采样数量保存在 `round_metrics.jsonl`，每个 job 的完整计划数量列表保存在 `job_metrics.jsonl`。

因此，使用相同 seed 重新运行但不清理 worker，可能命中前一次 run 的 cache entry。runner 不会静默测量 warm cache，而会检测该情况：实际 match 大于当前 job 的 block-aligned expectation（包括 job 第一个 main-agent request 的任何 hit）时，状态为 `metric_failed`。重新运行固定 seed 前，应清理或重启 prefix-cache worker。

要测量多个并发 job 数下的完整 job 吞吐量，应保持 per-job workload 不变，并使用 clean cache 分别运行，例如：

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/jobs_4 \
  --num-jobs 4 \
  --job-concurrency 4
```

client request pool 大小为 `job_concurrency * max_subagents_per_round`；fixed-count 模式下为 `job_concurrency * articles_per_round`。提高 subagent 数量会改变 per-job workload 和 main-agent context length；实验变量是同时执行的完整 job 数时，应使用 `--job-concurrency`。

为避免并发 research job 的 barrier 同步，可为每个 job 和 round 独立采样 10 到 40 个 subagent：

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/random_10_40_jobs_16 \
  --num-jobs 16 \
  --job-concurrency 16 \
  --min-subagents-per-round 10 \
  --max-subagents-per-round 40
```

两个 range 参数必须同时提供，取值为正数、闭区间且顺序正确。未提供时，`--articles-per-round` 保留原 fixed-count 行为。preflight 使用配置的最大值计算 main-agent context requirement，确保每个 sampled round 都有效。

为了公平比较双 worker vanilla baseline，通过 `SPARSEENGINE_WORKER_TAGS=subagent` 和 `SPARSEENGINE_WORKER_TAGS=main-agent` 标记 worker，在 main-agent worker 上启用 prefix cache，然后运行：

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/vanilla_2_jobs_4 \
  --num-jobs 4 \
  --job-concurrency 4 \
  --subagent-methods vanilla \
  --main-agent-methods vanilla \
  --subagent-required-tags subagent \
  --main-agent-required-tags main-agent
```

runner 从 source tree 执行，preflight 前要求 Git inspection 成功。如果无法检查 client commit 或 worktree status，run 会以 `invalid_input` 失败并写入 failure artifact，而不是产生 client provenance 未知的结果。

weighted distribution 可配置：

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/<RUN_NAME> \
  --article-token-buckets 60:1000:8000,25:8001:16000,10:16001:32000,5:32001:64000 \
  --subagent-output-token-buckets 90:100:600,10:800:1500
```

如需明确标注的 single-worker baseline，禁用 multi-worker router 检查：

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:8000/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/<BASELINE_NAME> \
  --allow-direct-server
```

`--allow-direct-server` 会省略 router-only method hint，并禁用 router header 与 actual-method validation。声称为 non-uniform router run 的结果不得使用它。

## Artifact

每个 output directory 都必须是新目录，已存在的目录会被拒绝。run 写入：

- `run_info.json`：命令、解析后的 workload、client Git state、环境、router health 与 code revision、role-specific context requirement，以及每个 worker 报告的 benchmark-critical config 和 code revision。
- `client_worktree.patch`：tracked client 文件为 dirty 时生成。其路径和 SHA-256 记录在 `run_info.json`；untracked client 文件会被拒绝，因为 tracked Git patch 无法复现它们。
- `raw_outputs.jsonl`：准确 request URL、timeout、prompt seed 和 payload，以及 raw HTTP response、response header 和 parsed router observation。
- `parsed_outputs.jsonl`：提取的 text、finish reason、usage 和 parse status。
- `per_sample_results.jsonl`：phase、requested/actual token count、latency、method preference、actual worker、actual method、expected reusable main-agent prefix token、selected-worker block size、block-aligned expected token、该 worker 上先前成功 prompt 数、selected worker 的 actual matched prefix token，以及显式 status。
- `round_metrics.jsonl`：sampled subagent count、barrier time、p50/p95/max subagent latency、straggler gap、main-agent latency，以及每个尝试 round 的显式 status，包括 main-agent request 失败的 round。
- `job_metrics.jsonl`：每个请求 job 的显式 job status、elapsed time、request 和 token count、完整计划 subagent-count sequence、route distribution、prefix-cache observation 和 completed-round count。
- `aggregate_metrics.json`：端到端 research-job throughput、token throughput、job 和 request latency percentile、job/request status count、route distribution，以及分别统计的 attempted/completed job 和 round 数。phase elapsed time 是 request interval 的并集，因此不会重复计算 overlapping job。其 top-level 和 per-phase `prefix_cache` object 包含 raw expected、block-aligned expected 和 actual token total，expected/actual hit count、partial hit、unexpected zero hit 和 unexpected excess hit。

任何 HTTP request 失败、malformed response、token-count mismatch、错误 method route、缺失 route header、无效 matched-token header，或超出本次 run 预期的实际复用，都会保留在 artifact 中并使 run 失败。

`expected_reusable_prefix_tokens` 是当前 main-agent prompt 与本次 run 中由 selected worker 处理的所有成功先前 main-agent prompt 之间，最大的 raw common-prefix length。其他 worker 处理的 prompt 不会提高该 expectation。对于 block size `b`，每个先前 prompt 的贡献取以下三者最小值：其 whole-block common prefix；当前 prompt 的 cacheable portion `floor((current_length - 1) / b) * b`；先前 prompt 的 materialized portion `floor(prior_length / b) * b`。

`block_aligned_expected_reusable_prefix_tokens` 是最大贡献值。只有当前 prompt 会保留最后一个 token 用于 logits；如果先前 prompt 正好结束在 block boundary，则最后一个完整 block 已被 materialize，仍可复用。因此，小于一个 block 的 raw prefix 允许 actual zero hit。partial hit 和 zero hit 会被记录但不会单独导致失败，因为 cache capacity 可能降低或消除先前 request 后的复用。实际复用超过本次 run 的 cacheable expectation 时为 `metric_failed`。

router run 要报告成功，至少一个成功 main-agent record 必须同时具有正数 block-aligned expected reusable prefix 和正数 actual matched-prefix count。该 run-level 检查防止把每个 main-agent prompt 路由到不同 worker 后，将 zero-hit benchmark 报告为成功。它不适用于 `--allow-direct-server`。
