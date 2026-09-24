# SparseVLLM 回归测试

## 真实 Agent 轨迹性能回归

`--layer agent_trace` 可向已启动的 SparseEngine vanilla 或 upstream vLLM
OpenAI-compatible 服务回放 MiniSWE 轨迹，不启动 GPU 服务、不执行工具，也不评解题分数。
先检查 GPU 空闲并启动服务，保存真实 `server_manifest.json`；基线和候选运行保持
相同缓存冷热状态、客户端和网络位置。该层使用外部 trace manifest，不使用合成负载矩阵。

当前公开标准语料是 [`JitaiHao/SWE-lite-trace100` 的 main](https://huggingface.co/datasets/JitaiHao/SWE-lite-trace100)
上的可追加 100 条、8,000 次请求版本。复现实验请固定 revision
`6bce0bb91ba68948248fd44b914ef77664b6b4e6`。比较时须对齐 trace
manifest 哈希。

优先从旧记录恢复：

```bash
python -m benchmark.sparseengine_regression.agent_trace \
  --run-dir "$MINISWE_RUN" --expected-instances 100 \
  --legacy-server-requests "$SERVER_REQUESTS" --output "$TRACE_DIR"
```

恢复运行的请求日志可重复传 `--legacy-server-requests`。默认按冻结的 `instances.txt`
取前 100 条；加 `--selection longest_turns` 则按每条轨迹的模型请求轮数降序选取，
轮数相同按原 `instances.txt` 顺序确定。也可用 `total_prompt_tokens` 或
`max_prompt_tokens` 按原响应中的输入 token 数排序。均不按成功与否筛选；
用 response ID 精确关联轨迹和请求，保留原始输入、
输出、输出 token 数与轮间等待。超时/步数耗尽仍是原来的终止状态，不能叫成功。
请求缺失/重复、API 次数不符、单条轨迹跨服务器运行或出现负间隔时明确失败。

旧日志的等待时间按 `下一轮日志时间 - 下一轮请求耗时 - 上一轮日志时间` **估算**。
日志序列化开销未知，不能宣称精确计时；工具、网络和重试等待均保留，不做推测性扣除。
旧生成耗时只参与间隔计算，不作为回放字段保存，更不会在回放时等待这段时间。

未来精确采集：在 `python -m benchmark.swe_bench_lite.run` 的常规命令上增加
`--record-agent-trace --slice 0:100`，保留模型、API、manifest、数据集和 Docker 参数。
默认 `mini-extra` 自动通过 `timed_mini.py` 包装，同步非流式 HTTP 请求按 instance
写入 `agent_trace.jsonl` 并逐条 flush。未结束或包含失败 HTTP 调用的轨迹不能静默导出。
导出时去掉 legacy 参数即可。使用既有 MiniSWE/httpx 环境；不记录认证头，但输入和
工具输出可能敏感，语料存放在数据盘，不提交 Git。

```bash
python benchmark/sparseengine_regression/run_suite.py --layer agent_trace \
  --agent_trace "$TRACE_DIR" --agent_api_base "$API_BASE" \
  --agent_server_manifest "$SERVER_MANIFEST" --agent_concurrency 16 \
  --agent_allow_estimated_timing \
  --output_root "$OUTPUT_ROOT" --run_id agent_baseline
# 用匹配的新服务和新 run_id 重测，并追加：
# --agent_baseline "$OUTPUT_ROOT/sparseengine_regression/agent_baseline/agent_trace.json"
# --agent_max_slowdown 1.10
```

只有旧日志估算语料需要显式接受 `--agent_allow_estimated_timing`。
每个 worker 跑完整条 agent 再接下一条；第一轮立即发送，之后每轮都在**当前响应结束后**
等待录下的间隔，再发送原始下一轮输入，不复用原绝对到达时间。
模型别名替换为目标服务别名；移除 stop 条件、设置 `ignore_eos=true` 和记录的输出
token 数，固定解码工作量并核对实际 token 数。新生成的文本不执行、不参与后续输入。

`agent_trace.json` 记录本次 HTTP 延迟 P50/P95/P99，以及全部输出 token / 含等待的
完整回放时间，不是纯 GPU 吞吐或 TTFT/TPOT。每条 agent 保存新原始响应、失败及
依赖失败而跳过的请求。任何请求失败则 gate 失败；匹配基线时检查 P95 延迟与总时间，
默认最多慢 10%。trace hash、模型、GPU UUID、后端、引擎配置、并发与超时须一致。
未传基线仅建立基线，不宣称通过性能回归；`--dry_run` 只验证数据和配置、不访问服务。

## 目的

本文说明如何运行 `benchmark/sparseengine_regression/` 下固定的 SparseVLLM regression harness。

harness 用于对以下层进行可复现的方法/模型检查：

- `quality`：LongBench v1 mini、LongBench v2 与按上下文长度分桶的 RULER core generation quality。
- `longbench_v2`：只运行 LongBench v2 quality gate。
- `ruler`：只运行 RULER core quality gate。
- `perf`：prefill/decode throughput 和 memory accounting。
- `stress`：固定长度、高并发的 SparseVLLM admission/decode stress。
- `stress_v2`：带 shared-prefix 和 multi-turn workload、可变 prompt length，并对支持方法验证 prefix-cache hit 的 synthetic serving trace stress。
- `validate`：manifest 和 output artifact validation。

test plan 由 `benchmark/sparseengine_regression/manifest.json` 控制。

## 前置条件

为运行 suite 的机器配置以下路径：

- Working directory：`<REPO_ROOT>`
- Conda env：`<CONDA_ENV>`
- Output root：`<OUTPUT_ROOT>`
- LongBench data：`<LONGBENCH_ROOT>`
- LongBench v2 JSON/JSONL export：`<LONGBENCH_V2_DATA>`
- 模型：
  - `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
  - `<MODEL_ROOT>/Qwen3-4B-Instruct-2507`
  - `<MODEL_ROOT>/Llama-3.1-8B-Instruct`
- Compressor checkpoint：
  - `<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor`
  - `<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor`
  - `<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor`

运行 suite 前设置环境：

```bash
cd <REPO_ROOT>

export SPARSEENGINE_OUTPUT_DIR=<OUTPUT_ROOT>
export SPARSEENGINE_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>
export SPARSEENGINE_LONGBENCH_V2_DATA=<LONGBENCH_V2_DATA>

export DELTAKV_MODEL_QWEN25_7B=<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M
export DELTAKV_MODEL_QWEN3_4B=<MODEL_ROOT>/Qwen3-4B-Instruct-2507
export DELTAKV_MODEL_LLAMA31_8B=<MODEL_ROOT>/Llama-3.1-8B-Instruct

export DELTAKV_COMPRESSOR_QWEN25_7B=<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor
export DELTAKV_COMPRESSOR_QWEN3_4B=<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor
export DELTAKV_COMPRESSOR_LLAMA31_8B=<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor

export PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}
```

第一次运行 V2 前初始化固定版本的官方 LongBench submodule：

```bash
git submodule update --init benchmark/long_bench_v2/upstream
```

submodule 提供官方 prompt、参考实现和版本 provenance。官方
`THUDM/LongBench-v2` 数据集独立分发；将 train split 导出为一个本地 `.json`
或 `.jsonl` 文件，并让 `SPARSEENGINE_LONGBENCH_V2_DATA` 指向这个不可变输入。

manifest 还包含 `qwen25_32b`；除非 GPU memory 足够且设置了相应 model/checkpoint 环境变量，否则省略它。

## 快速 Unit Test

运行保护 regression harness、RULER generator/grading、manifest policy 和 OmniKV
full-layer selector 的 unit test：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python -m pytest \
  tests/test_sparseengine_regression_grading.py \
  tests/test_omnikv_full_layer_selector.py \
  tests/test_ruler_tasks.py \
  tests/test_ruler_vt_regression.py \
  tests/test_longbench_v2.py \
  -q
```

当前 harness 的预期结果：所有 test 通过。

## Manifest Validation

长 GPU run 前使用 `validate`。它解析 runtime path，写入 resolved manifest，并创建空的必需 artifact 文件。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer validate \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id validate_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

如果缺少 model/checkpoint path 时应使 run 失败，而不是记录为 skipped，请使用 `--no-allow_skipped_policy`。

## 常用运行命令

所有命令写入：

```text
<OUTPUT_ROOT>/sparseengine_regression/<run_id>/
```

### Quality

默认情况下，`--layer quality` 同时运行 LongBench v1 mini、LongBench v2 和
RULER core。只有明确需要缩小范围时，才传入如
`--quality_benchmarks longbench_v2` 的子集。

LongBench-mini 使用：

- task：`qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc`
- LongBench batch size：`100`
- SparseVLLM `max_num_seqs_in_batch`：`16`
- SparseVLLM `max_decoding_seqs`：`16`
- 每个 task 的 sample：`50`

将 OmniKV 与 vanilla baseline 对比：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id omnikv_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

运行不含 32B 的完整 quality：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,quest,deltakv,deltakv-less-memory \
  --run_id quality_3models_all_methods_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

TP decode CUDA Graph v1 quality validation 使用默认 LongBench data-worker parallelism，并通过 regression-suite override 传入 engine TP：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_quality_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

该 run 内 sparse method 与 TP vanilla 对比。记录 A/B/C grade；crash 或 D grade 使 TP graph quality gate 失败。

快速 TP prefix-cache + decode-graph regression gate 保持相同方法覆盖，但使用显式小 sample override 和 child-command timeout。它仍覆盖 LongBench quality、SCBench quality 和 stress，但属于 smoke/regression gate，不是完整的每 task 50 sample quality suite：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer quality \
  --quality_benchmarks longbench \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --quality_tasks qasper,hotpotqa \
  --quality_batch_size 2 \
  --quality_samples_per_task 2 \
  --quality_min_required_samples 2 \
  --quality_sparseengine_max_num_seqs_in_batch 2 \
  --quality_sparseengine_max_decoding_seqs 2 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_quality_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

然后运行匹配的 SCBench quality 和 prefix-hit stress layer：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer scbench \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --scbench_decode_cuda_graph \
  --scbench_tasks scbench_kv \
  --scbench_num_eval_examples 1 \
  --scbench_max_turns 2 \
  --scbench_max_seq_length 1024 \
  --scbench_batch_size 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_scbench_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>

conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer stress \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --require_prefix_cache_hit \
  --stress_length 256 \
  --stress_request_counts 2 \
  --stress_output_len 2 \
  --stress_max_num_seqs_in_batch 2 \
  --stress_max_decoding_seqs 2 \
  --stress_max_decode_steps_after_full 1 \
  --stress_admission_wave_size 1 \
  --stress_wave_decode_gap_steps 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_stress_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### LongBench V2 Quality

LongBench v2 是新增项，不替换现有 LongBench v1 mini gate。原始 benchmark
覆盖约 8K 到 2M words。标准 regression profile 使用 120 个自然样本，按应用
chat template 后的 token 数分为 `32K-64K`、`64K-96K`、`96K-127K` 三桶，
每桶 40 个样本，
`max_model_len=131072`；因此既能支持 128K-class 模型，也超过 V1 runner 的
121K 上限。给定 dataset、tokenizer 和 seed，选样完全确定。

runner 使用 submodule 中的官方 zero-shot direct-answer prompt，但不复制上游 API
client，也不采用上游的 head/tail truncation。超出配置容量的 prompt 在确定性选样
前排除；任一桶样本不足会直接失败。gate 还要求同一 run 的 vanilla/sparse 样本
完全对齐、source-data hash 一致，并为每个样本保留显式状态。非空 response 若没有
包含官方答案格式，会保留为 `parse_failed` 并按错误答案计分，与官方 evaluator
一致；model/runtime 执行失败仍会使整个 run 无效。

该固定 120-sample profile 使用 greedy decoding 以减少 run-to-run noise。它是 repo
quality regression gate，不是官方完整 503-sample leaderboard protocol 的复现。

只运行标准 V2 gate：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer longbench_v2 \
  --models qwen25_7b \
  --methods vanilla,omnikv \
  --command_timeout_s 7200 \
  --run_id longbench_v2_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

对于已经验证支持 128K 以上上下文的模型，应同时显式扩展 runtime limit 和 token
bucket。这是独立 profile，不会静默改变标准 gate：

```bash
python benchmark/sparseengine_regression/run_suite.py \
  --layer longbench_v2 \
  --models qwen25_7b \
  --methods vanilla,omnikv \
  --longbench_v2_max_model_len 262144 \
  --longbench_v2_token_buckets_json \
  '[{"name":"128k-192k","min_prompt_tokens":131072,"max_prompt_tokens":196607,"samples":4},{"name":"192k-255k","min_prompt_tokens":196608,"max_prompt_tokens":261888,"samples":4}]' \
  --output_root <OUTPUT_ROOT>
```

### RULER Core Quality

固定的自包含集合运行 `niah_single_1`、`niah_multikey_2`、`vt`、`cwe` 和
`fwe`，覆盖 retrieval、multi-hop tracing 和两种 aggregation contract。matrix
使用 `16K,32K,64K,98K` target context length，并在每个 task、每个长度上用
同一 run 中完全对齐的 vanilla dataset 独立评分，因此单个 task/length 回归不会被
平均分掩盖。每个 sample 必须至少达到目标 sequence length 的 90%。runner 保留
raw、parsed、per-sample、dataset、aggregate 和 grade artifact。
默认每个 task/length bucket 取 10 个 sample，即每个 model/method 运行 200 次
generation。

task contract 和 prompt 对齐 NVIDIA RULER；确定性 synthetic word pool 替代可选的
`wonderwords` 和大型 word asset，因此这是 repo regression set，不是官方
leaderboard dataset。依赖外部语料的 essay-based NIAH 和 `qa_1`/`qa_2` 不包含。

只运行 RULER core、不运行 LongBench-mini：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer ruler \
  --models qwen25_7b \
  --methods vanilla,omnikv \
  --ruler_tasks niah_single_1,niah_multikey_2,vt,cwe,fwe \
  --ruler_context_lengths 16384,32768,65536,98304 \
  --ruler_samples_per_length 10 \
  --command_timeout_s 7200 \
  --run_id ruler_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

对于支持 prefix cache 的方法，启用后会在同一 engine 中立即原样重放每个
deterministic batch。gate 要求 cache hit request 和 hit token 都非零，并要求
重放输出和得分与 primary pass 完全一致：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer ruler \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --ruler_tasks vt \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --ruler_context_lengths 16384,32768 \
  --ruler_samples_per_length 2 \
  --command_timeout_s 1800 \
  --run_id ruler_prefix_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

检查 `ruler.json`、`grade_summary.json`，以及各 task/method 目录中的
`prefix_cache_summary.json`。这是 quality 与 cache-correctness gate，不是
prefix-cache performance benchmark。

### Performance

Performance 使用：

- prompt length：`16000,64000`
- batch size：`1,4`
- output token：`256`
- 方法支持时请求 decode CUDA Graph

对于 sparse method，benchmark 还会以相同 shape 运行 vanilla，以便 suite 计算 decode speedup。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_perf_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

TP decode CUDA Graph v1 performance validation：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_perf_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

检查 `perf.jsonl` 中的 `decode_graph_expected=true` 和 `decode_graph_active=true`。

### Stress

Stress 当前使用：

- prompt length：`16000`
- request count / batch size：`80`
- output token：`64`
- `max_num_seqs_in_batch=80`
- `max_decoding_seqs=80`
- full admission 后最大 decode step：`32`

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer stress \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_stress80_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Stress V2

`stress_v2` 使用 `scripts/benchmarks/bench_prefix_cache.py` 作为 serving-like trace 的 regression layer。与固定 `stress` 不同，它运行有 seed 的 synthetic request：

- workload：`shared_prefix,multiturn`
- 支持的方法：`vanilla`、`omnikv`、`quest`
- `vanilla` case：`baseline_full,prefix_full`
- `omnikv` case：`prefix_omnikv`
- `quest` case：`prefix_quest`
- session / turn：`8 / 4`
- shared-prefix request：`8`
- output token：`64`
- 最大 active request：`8`
- 可变 multi-turn user length：`128..1024`
- 可变 session-prefix length：`1024..4096`
- 可变 shared suffix length：`512..4096`
- 最大 prompt length：multi-turn 约 `16.5k` token，shared-prefix 约 `12.3k`
- prefix-cache block size：`16`

启用 prefix cache 的 case 未观察到 cache hit，或实际 prompt length 没有变化时，gate 失败。不支持的方法记录为 `skipped_by_policy`，因为该 layer 专门验证 prefix-cache serving 行为。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer stress_v2 \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --run_id stress_v2_qwen3_serving_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### 组合 Layer

`nightly` 运行 quality 和 performance，不运行 stress。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparseengine_regression/run_suite.py \
  --layer nightly \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id nightly_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

`pre-refactor` 运行 quality、performance 和 stress。

## 结果记录

将此文件作为稳定 regression runbook。不要在此添加按时间排列的实验记录或本地 result index。面向仓库的结果声明需要证据时，直接引用原始 run artifact path。

## OmniKV Full-Layer Selection

OmniKV full layer 与模型相关。发布新模型的 OmniKV 或与 OmniKV 对齐的 DeltaKV regression 数值前，使用 `python -m sparseengine.utils.select_omnikv_full_layers`。

selector 在 LongBench task 上运行离线 decode-attention coverage calibration，选择 `--num-full-layers` 个 layer，并把选中的 layer 字符串写入 `selected_full_layers.json`。这不是 online runtime mode：必须将该字符串作为 `full_attention_layers` 传回。

Qwen2.5-7B 选择六个 full layer 的示例：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python -m sparseengine.utils.select_omnikv_full_layers \
  --model-path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --longbench-root <LONGBENCH_ROOT> \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir <OUTPUT_ROOT>/omnikv_full_layer_calibration_$(date -u +%Y%m%d)/qwen25_7b_full6 \
  --num-full-layers 6 \
  --num-samples 32 \
  --topk 2048 \
  --random-decode-points-per-sample 8 \
  --num-sink-tokens 0 \
  --num-recent-tokens 32 \
  --prefill-chunk-size 512 \
  --torch-dtype bfloat16 \
  --device cuda
```

主要输出：

- `selected_full_layers.json`：选中的 layer ID 和 runtime config 使用的 `full_attention_layers` 字符串。
- `per_sample_points.jsonl`：calibration 使用的 sampled decode point。
- `pair_scores.npy` 和 `segment_scores.npy`：用于审计的 raw coverage matrix。
- `run_info.json`：命令、Git state、model/data path 和 calibration setting。

在临时 Sparse-VLLM run 中使用选中 layer 时，将 `full_attention_layers` 值复制到 `--hyper_params`：

```bash
PYTHONPATH=$PWD:$PWD/src python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <MODEL_DIR> \
  --methods omnikv \
  --lengths 131072 \
  --batch_sizes 4 \
  --output_len 128 \
  --hyper_params '{"sparse_method":"omnikv","full_attention_layers":"0,2,4,11,16,22","decode_keep_tokens":4096,"recent_keep_tokens":32,"sink_keep_tokens":0,"engine_prefill_chunk_size":512}'
```

对于 regression run，更新 `benchmark/sparseengine_regression/manifest.json` 中的 `methods.omnikv.model_configs`。如果 DeltaKV regression config 有意与 OmniKV observation/full layer 对齐，在同一 manifest 中更新匹配的 DeltaKV model config，并在 run summary 中记录。当前 manifest 使用：

```text
qwen25_7b:  0,2,4,11,16,22
qwen3_4b:   0,1,3,9,13,16,21,28
llama31_8b: 0,2,7,13,16,26
```

更改这些 layer 后，运行 `validate`，并重新运行 OmniKV quality/perf/stress。

## 输出

每次 run 写入：

- `resolved_manifest.json`：环境变量解析后的 manifest。
- `grade_summary.json`：command record、grade 和 final status。
- `metrics.json`：quality aggregate record。
- `perf.jsonl`：展平的 performance row。
- `memory.json`：根据 performance row 得到的 memory grade。
- `stress.json`：stress row 和 stress grade。
- `stress_v2.json`：serving-trace stress row 和 stress_v2 grade。
- `ruler.json`：按 task、model 和 method 保存的 RULER aggregate record。
- `longbench_v2.json`：按 model 和 method 保存的 LongBench v2 aggregate record。
- `raw_outputs.jsonl`、`parsed_outputs.jsonl`、`sample_results.jsonl`：运行 quality 时的 generation artifact。
- Layer-specific log：
  - `quality/<model>/<method>/run.log`
  - `ruler/<task>/<model>/<method>/run.log`；同目录保存 dataset、aggregate 和可选的
    prefix-cache replay artifact
  - `longbench_v2/<model>/<method>/run.log`；同目录保存选样、source hash、
    per-sample result 和 aggregate metric
  - `perf/<model>/<method>.log`
  - `stress/<model>/<method>.log`

快速 summary 命令：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("<OUTPUT_ROOT>/sparseengine_regression/<run_id>")
data = json.loads((root / "grade_summary.json").read_text())
print("status:", data["status"])
print("worst_required_grade:", data.get("worst_required_grade"))
for grade in data.get("grades", []):
    print(grade.get("task"), grade.get("model"), grade.get("method"), grade.get("context_length"), grade["name"], grade["grade"], grade["status"], grade["metrics"])
PY
```

## Regression Rubric

可执行 gate rule 位于 `benchmark/sparseengine_regression/grading.py`。稳定的人类可读 rubric 位于 `benchmark/sparseengine_regression/rubrics.md`。

只有在稳定 ABCD rubric 定义改变时，才更新 `benchmark/sparseengine_regression/rubrics.md`。不要向 rubric 文件添加带日期的 campaign result、open blocker、run ID 或 remote log path。

## 故障排查

- 缺少 model 或 compressor path：
  - 运行 `validate`。
  - 检查 `resolved_manifest.json`。
  - 如果缺少 path 时 run 应失败，传入 `--no-allow_skipped_policy`。
- Import error：
  - 确保 `PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}`。
  - 使用包含[快速开始](../getting_started/README.md)所列依赖的环境。
- Quality dataset error：
  - 设置 `SPARSEENGINE_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>`。
- GPU memory failure：
  - 不要在 harness 内添加 fallback。
  - 在 issue note 中记录准确 run ID、model、method、layer、log path 和 error。
- 命令提前退出：
  - 检查 `<run_id>/grade_summary.json`；failed command 会记录 `returncode`、`cmd` 和 `log_path`。
  - 检查 command record 中对应 layer-specific log path。
