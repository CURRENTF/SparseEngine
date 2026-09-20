# SWE-bench Lite

## 目的

`benchmark/swe_bench_lite/run.py` 是两个外部 upstream component 的轻量 adapter：

- mini-SWE-agent 通过 OpenAI-compatible model API 为每个 issue 生成 patch。
- `swebench.harness.run_evaluation` 在 Docker 中应用每个 patch，并计算官方 SWE-bench result。

adapter 不会重新实现 SWE-bench dataset、environment、test 或 metric。它支持 `SWE-bench/SWE-bench_Lite` 的 `test` split，包括完整 300-instance benchmark 和较小的 smoke selection。

Upstream source：

- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)
- [SWE-bench Lite dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite)

## 前置条件

使用独立 Python 环境，同时安装带 SWE-bench extra 的 `mini-swe-agent` 和官方 SWE-bench harness。upstream checkout 应放在本仓库之外，例如 `../SWE-bench`。

将 `SWE_BENCH_PYTHON` 设为该环境的 Python。shell 入口还会把同一环境的 `bin/` 目录放到 `PATH` 最前面，避免 `mini-extra` 与导入的 SWE-bench harness 意外来自不同环境。

adapter 默认使用 cached/offline Hugging Face access，并要求所选 Docker image 全部已存在于本地。它不会静默下载 dataset 或数百 GB image。只有明确需要使用网络和存储时才使用 `--allow-dataset-download` 或 `--allow-image-pulls`。

所选 instance 和 image name 会写入 run directory 中的 `instances.txt` 与 `images.txt`，本地 image 检查失败时也会写入。

## 容器内存隔离

可写层限制管的是磁盘，不是内存。为避免某条样本 OOM 耗尽整个 worker，
需同时设置单容器内存上限和本次运行的父 cgroup 总上限，协调进程放在该组之外；
不必因此降低并发。

可选的内存保护要求 Linux cgroup v2、本地 Docker daemon，以及已启用的可写层保护。
导出 `SPARSEENGINE_DOCKER_MEMORY_LIMIT_BYTES`、`SPARSEENGINE_DOCKER_MEMORY_PARENT`
和 `SPARSEENGINE_DOCKER_MEMORY_EVENTS`（OOM JSONL 路径）。在 MiniSWE 附加配置的
`environment.run_args` 中设置匹配的 `--memory`、`--memory-swap` 和
`--cgroup-parent`；memory 与 memory-swap 相等表示容器不使用 swap。
不要加 `--rm`：检测失败前需要保留 Docker OOM 元数据，之后由保护逻辑显式清理容器。
启动前先创建父 cgroup 并设置其总上限。

生成容器发生 OOM 时抛出 `DockerMemoryLimitExceeded`，该样本记为失败，不重试，
也不从评测分母中排除。官方评分容器同样施加内存上限，失败仍由上游评分器报告。
内存限制会影响资源失败率，必须随实验记录。

## SparseEngine Server

以独立 long-running process 启动 `sparseengine.entrypoints.openai.api_server`。典型命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=$PWD/src \
python -m sparseengine.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --served-model-name sparseengine-swe \
  --host 127.0.0.1 \
  --port 18000 \
  --engine-kwargs /path/to/engine_kwargs.json \
  --request-log-dir /path/to/server_requests
```

例如，`engine_kwargs.json` 可以包含：

```json
{
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.88,
  "max_model_len": 131072,
  "engine_prefill_chunk_size": 4096,
  "sparse_method": "vanilla"
}
```

在 server log 旁创建 JSON manifest。不要在其中保存 API key 或其他 secret。

```json
{
  "command": "python -m sparseengine.entrypoints.openai.api_server --model <MODEL_PATH> --served-model-name sparseengine-swe --host 127.0.0.1 --port 18000 --engine-kwargs /path/to/engine_kwargs.json",
  "model_path": "<MODEL_PATH>",
  "served_model_name": "sparseengine-swe",
  "cuda_visible_devices": "2",
  "server_port": 18000,
  "engine_kwargs": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.88,
    "max_model_len": 131072,
    "engine_prefill_chunk_size": 4096,
    "sparse_method": "vanilla"
  }
}
```

对于本地 API，adapter 要求提供该 manifest，并将其 snapshot 到 benchmark run 中。它还会检查 `/v1/models`，确认广告的模型与 `--served-model-name` 匹配。adapter 不启动、重启或停止 server。

## 单 Instance Smoke Test

运行全部 300 个 instance 前，先运行一个。`openai/` 是 LiteLLM provider prefix；`sparseengine-swe` 是 server 广告的准确 model name。SparseEngine 不要求认证，但 LiteLLM 需要非空 OpenAI key，因此使用本地 dummy value。

```bash
export OPENAI_API_KEY=local-sparseengine

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-smoke \
  --model openai/sparseengine-swe \
  --api-base http://127.0.0.1:18000/v1 \
  --served-model-name sparseengine-swe \
  --server-manifest /path/to/server_manifest.json \
  --slice 0:1 \
  --batch-size 1 \
  --mini-workers 1 \
  --eval-workers 1 \
  --step-limit 80 \
  --cost-tracking ignore_errors \
  --cost-limit 0
```

本地模型没有 provider billing metadata。使用 `--cost-tracking ignore_errors` 时，cost 记录为 0，`--cost-limit` 无法强制 budget；有界控制项为 `--step-limit` 和 `--wall-time-limit-seconds`。

## 完整 Lite Run

使用新的 run directory 并移除 `--slice 0:1`。模型、API、prompt、decode、dataset 和 server 设置应与 smoke run 相同。先使用保守的 generation concurrency，只有在 server 处理 concurrent tool-calling request 稳定后再提高 `--mini-workers`。

```bash
export OPENAI_API_KEY=local-sparseengine

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-lite300 \
  --model openai/sparseengine-swe \
  --api-base http://127.0.0.1:18000/v1 \
  --served-model-name sparseengine-swe \
  --server-manifest /path/to/server_manifest.json \
  --batch-size 50 \
  --mini-workers 1 \
  --eval-workers 6 \
  --step-limit 80 \
  --cost-tracking ignore_errors \
  --cost-limit 0
```

只有在 prediction 和 `batch_done.json` prediction hash 都通过验证后，completed batch 才会被跳过。partial mini-SWE-agent batch directory 会重新传给 mini-SWE-agent；其默认行为是跳过 completed trajectory。adapter 只合并声明的 numeric batch directory，因此 backup directory 不会引入重复 prediction。

## 分离 Generation 与 Evaluation

只有 `prepare` 和 `generate` 需要 LLM server。官方 evaluation 不调用 model API。分阶段运行时，重复相同 semantic argument，只改变 `--stage` 和 operational worker count：

```bash
COMMON_ARGS=(
  --swe-bench-dir ../SWE-bench
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-lite300
  --model openai/sparseengine-swe
  --api-base http://127.0.0.1:18000/v1
  --served-model-name sparseengine-swe
  --server-manifest /path/to/server_manifest.json
  --batch-size 50
  --step-limit 80
  --cost-tracking ignore_errors
  --cost-limit 0
)

# Requires the model API and API key.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage generate --mini-workers 1 "${COMMON_ARGS[@]}"

# Requires Docker images, but not the model API or API key.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage evaluate --eval-workers 6 "${COMMON_ARGS[@]}"

# Requires only completed artifacts in the run directory.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage summarize \
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-lite300
```

adapter 拒绝在已有 run directory 中改变 semantic config。改变模型、dataset selection、step limit、decode setting、API endpoint 或 server config 时应使用新 run directory。对于所有非 `summarize` stage，它还会将当前 adapter source、SWE-bench source、Python executable 和 package version 与 `run_manifest.json` 对比；source 或 toolchain drift 要求使用新 run directory。

官方 evaluation 使用与 prediction 绑定的 run ID，例如 `<RUN_ID>-pred-<HASH>`。其 report 和 per-instance cache 位于 `<RUN_DIR>/official/`。adapter 在允许 cache reuse 前写入 identity marker，并拒绝无法关联到当前 prediction 与 runtime provenance 的已有 cache directory。

共享 adapter 默认固定 `temperature=0`、`top_p=1` 和 `max_tokens=4096`。由于该 OpenAI-compatible path 当前没有共享 seed control，因此记录 `seed=null`。

## 外部 API Provider

使用 DeepSeek 或其他 LiteLLM provider 时，省略 `--api-base`、`--served-model-name` 和 `--server-manifest`，然后显式选择 provider key 环境变量。默认直接进行 model call：mini-SWE-agent process 会移除 proxy 环境变量。只有 provider 要求该 proxy 时才使用 `--api-proxy-from-environment`。

```bash
export DEEPSEEK_API_KEY=<KEY>

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/deepseek-lite300 \
  --model deepseek/deepseek-v4-flash \
  --api-key-env DEEPSEEK_API_KEY \
  --mini-extra-config /path/to/deepseek_nonthinking.yaml \
  --cost-tracking default \
  --cost-limit 0.05 \
  --step-limit 80
```

上面使用的可选 provider config 只包含 DeepSeek 特定 request field：

```yaml
model:
  model_kwargs:
    extra_body:
      thinking:
        type: disabled
```

DeepSeek thinking control 等 provider-specific request field 不属于共享 adapter config。通过 provider-specific `--mini-extra-config` 传入；adapter 会 hash 并 snapshot 该文件。不要为 SparseEngine 复用此类 config，也不要在其中保存 credential。Config 和 server-manifest validation 会在 snapshot 前拒绝敏感 field name、常见 provider token format、authorization header 和 URL credential。

## 输出

每个 run directory 包含：

| Artifact | 含义 |
| --- | --- |
| `run_config.json` | Immutable semantic experiment config 和 selected instance ID。 |
| `run_manifest.json` | Code revision、package version、Python、credential variable name 和 runtime policy。 |
| `server_manifest.json` | 适用时，本地 SparseEngine server config 的 snapshot。 |
| `evaluation_identity.json` | 用于 cache ownership 的 prediction hash、official run ID 和 runtime-provenance hash。 |
| `invocations.jsonl` | Stage invocation 和 operational concurrency setting。 |
| `status.jsonl` | Append-only stage 和 batch status event。 |
| `batches/*/*traj.json` | Raw mini-SWE-agent trajectory 和 model response。 |
| `preds_all.json` | 经过严格验证、使用官方 SWE-bench prediction format 的 patch。 |
| `generation_results.jsonl` | 每个 instance 的 parsed generation status 和 model statistic。 |
| `official/*.json` | 未修改的官方 SWE-bench aggregate report。 |
| `official/logs/run_evaluation/` | 与 prediction 绑定的官方 per-instance evaluation cache 和 log。 |
| `per_sample_results.jsonl` | 规范化的 per-instance status 和官方 outcome。 |
| `final_summary.json` | Aggregate score、status count、API call、cost 和 artifact path。 |

规范 `status` 值包括 `success`、`invalid_input`、`model_failed`、`parse_failed`、`metric_failed` 和 `skipped_by_policy`。`success` 表示官方 harness 完成了该 instance；独立的 `resolved` 字段记录其 test 是否通过。benchmark score 始终是 `resolved_instances / total_instances`，分母包括 empty patch 和 failure。

## 只剪枝工具返回的 KV

在上面的 smoke 命令中添加以下参数：

```bash
--no-chain-cache \
--prefix-prune-policy kvzip_global \
--prefix-prune-target tool_results \
--prefix-prune-tokenizer /path/to/server-tokenizer \
--prefix-prune-keep-ratio 0.5
```

服务端须启用 Vanilla 或 OmniKV 的 radix prefix cache，并支持多区间 prune。
`--prefix-prune-tokenizer` 是 harness 本地可访问、与服务端一致的 tokenizer 目录，
包括 chat template；不会自动下载。harness 环境需要 `transformers`、fast tokenizer
及共享 chat renderer 的依赖。

harness 只标注 `role=tool` 的返回正文，保留用户输入、assistant 思考、调用参数和
模板标记。它对完整渲染文本定位 token，将完整落在正文内的 token 向内按服务端
block size 对齐，再通过 `/prefix_cache/match` 比较 token 路径 ID；不匹配则报错。
`block_size=1` 可减少边界损失，但仍保留跨正文边界的 token。engine 只收到原始
`token_ids`、`ranges` 和总 `keep_tokens`，不接收工具类别。

`keep_ratio=0.5` 表示在所有候选工具区间中总共保留一半 token（向下取整），
不是每个工具返回分别保留一半。允许 `0 <= keep_ratio < 1`；不剪枝基线不传
`--prefix-prune-policy`。此模式忽略静态区间及 `--prefix-prune-keep-tokens` 参数。

工具模式在每个有新增工具返回的回合完成推理后剪枝：只选择尚未处理过的
工具正文，同轮新增区间共用 `floor(候选 token 数 * keep_ratio)` 的保留预算。
旧工具区间不再压缩；`--prefix-prune-trigger-tokens` 仅用于静态区间模式，工具
模式不等待总长度阈值。空正文或没有完整对齐块时记录 `skipped_by_policy`，
并推进消息游标；失败时不推进。历史消息被改写时显式报错。

下一轮完成推理后，将上次剪枝的复用检查合并到本轮 prefix match 中。
`run_config.json` 保存配置；`prefix_prune_events.jsonl` 保存消息游标、区间、
预算、释放量和复用检查，原始聊天记录由 agent trace 保存。没有后续回合的
最后一次剪枝不算已验证复用。工具区间定位仍需完整模板渲染和一次分词，以
保持 BPE 边界及模型模板语义；不会仅分词新增文本后直接拼接 token。
