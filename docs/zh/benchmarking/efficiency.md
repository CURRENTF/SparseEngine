# 效率与吞吐性能基准套件

[English](../../en/benchmarking/efficiency.md) | 简体中文

本手册维护测量定义、入口及支持限制。具体实验的参数、扫描和重绘说明放在
[scripts/official_experiments/](../../../scripts/official_experiments/)；
历史结果不随默认协议变化而改写。

## 入口与最小示例

| 目的 | 入口 |
| --- | --- |
| 请求 TTFT/TPOT、E2E、fixed/churn 对照 | `benchmark/efficiency/bench_probe.py`；可用 `scripts/benchmarks/run_efficiency_probe.sh` 编排 |
| 论文主图的连续 decode 吞吐 | 同一 probe，显式开启 `--decode-only-steps` |
| 逐步同步的阶段诊断 | `benchmark/microbench.py --synchronize_step_timing`，不混入主图 |
| 共享统计及逐请求离线重聚合 | `benchmark/efficiency/metrics.py` |

从仓库根目录运行。先激活正确环境（conda 用 activate 或 conda run），确认模型、
依赖和输出盘可用，并用 `nvidia-smi` 检查所有参与 GPU 空闲。Wrapper 有空卡检查，
独立 Python CLI 需要手动检查。每次运行使用新的持久输出目录。

以下是请求模式的功能 smoke，不是论文测量点：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:$PWD/src" python3 \
  benchmark/efficiency/bench_probe.py \
  --engine sparseengine --sparse-method vanilla --model-path "<MODEL_PATH>" \
  --tensor-parallel-size 1 --monitor-gpus 0 \
  --scenario fixed --prompt-lens 4096 --output-lens 32 --batch-sizes 1 \
  --num-warmups 0 --num-iters 1 --output-dir "<NEW_RUN_DIR>"
```

完整参数用 `python3 benchmark/efficiency/bench_probe.py --help` 查询。
`--monitor-gpus` 是物理 GPU ID，须与 `CUDA_VISIBLE_DEVICES` 对齐。
Probe 默认每个 scheduler 的 `max_num_batched_tokens=65536`。
Probe 为 SparseEngine 显式设置 `engine_prefill_chunk_size=8192`，可通过
`--hyper-params` 覆盖；调度 token 总预算与 prefill chunk size 是两个独立参数。
当前 vLLM 适配器没有对应的独立 chunk size 参数，其切块受可用调度 token 预算约束。
跨拓扑比较须记录这一差异，以及每 replica 和全局的预算。
跨系统运行时对齐模型、权重/KV dtype、trace、GPU/TP/DP/EP、预算、Graph、
seed、长度、jitter、warmup 和重复次数；不同方法的预算需分别说明
sink/recent/selected/full layers，同名预算不保证相同工作量或质量。

连续窗口必须显式指定 `--scenario fixed --prompt-length-jitter 0
--output-length-jitter 0 --decode-only-steps 256 --decode-only-warmup-steps 32`。
这组窗口参数不是普通请求模式的默认值。正式容量扫描及其 smoke 使用
[稀疏 decode 效率实验入口](../../../scripts/official_experiments/sparse_decode_efficiency/README.md#boundary-sync-rerun)，
不要复制计时 runner。窗口须覆盖方法的周期评分/驱逐，必要时统一加长所有对照。

其他入口按需使用：

- [Probe wrapper](../../../scripts/benchmarks/run_efficiency_probe.sh)：
  `SYSTEMS MODEL_NAME_OR_PATH PHYSICAL_GPU_IDS`；变量及模型别名以脚本为准。
  别名固定 TP，自定义路径默认 TP2；显式拓扑用 Python CLI。
  稀疏参数来自 [regression manifest](../../../benchmark/sparseengine_regression/manifest.json)；
  OmniKV 无校准条目会失败，可显式提供 `BENCH_MANIFEST_MODEL_ID` 或
  `OMNIKV_FULL_ATTENTION_LAYERS`，单层配置需要显式消融开关。
- [Unified suite](../../../scripts/benchmarks/run_unified_efficiency_suite.sh)：
  参数顺序是 `GPUS SYSTEMS MODEL_NAME`，运行 synthetic 与 LongBench；
  数据目录用 `SPARSEENGINE_LONGBENCH_DATA_DIR`，最终检查 `suite_status.json`。
- [Nsight 诊断](../../../scripts/benchmarks/run_efficiency_profile.sh)：
  标准测试发现可疑 case 后使用，参数为 `SYSTEM MODEL_PATH GPUS`。
  当前 wrapper 支持 vanilla/SnapKV/vLLM vanilla、默认 TP2；
  需要 `nsys` 与 performance-counter 权限，输出 `timeline.nsys-rep`。

<a id="measurement-contract"></a>

## 指标与计时契约

### 连续 decode：论文主结果

吞吐 = 窗口实际完成的 decode tokens / 协调端完整窗口秒数。

- Prefill、wave admission 和满批 warmup 在窗口之前完成。窗口内保持同一组请求，
  不允许 prefill、抢占、换请求、掉批、Graph capture、意外 eager 或窗口不足。
- 只在窗口两端等待所有参与 rank/设备完成相关工作，并处理异步队列中的在途工作。
  保留引擎兼容的 async/overlap 和算法必需同步，不添加逐步 CUDA sync 或同步 RPC。
  按实际完成而非提交次数计 token，分子与分母覆盖同一窗口，逐 rank 校验 Graph 状态。
- 包含调度、驱动、采样、评分、驱逐和通信，不是孤立 kernel 时间。
  保存首尾每请求上下文长度、admission、warmup 和实测步数；
  wave 入场可能使各请求起始上下文不同。
- 请求仍须完成指定输出并验证 token 数。不能以缩短窗口、降低该点 BS 或截断掩盖失败；
  改变输入/输出或截断需要明确批准，并按新协议重新测量相关对照。
- 当前每个 workload 重建引擎；舍弃的 workload 预热持久编译缓存，
  每次实测另外舍弃满批 warmup 步数，不代表跨 workload 复用热引擎。
- 窗口边界扰动请求延迟，此模式不报告 TTFT/TPOT。整体服务效率另测请求/E2E，
  不用排除了 admission 的窗口结果替代。逐步同步与边界同步结果不得混图或混合汇总。

满批窗口不足时，先区分请求过早结束与真实 KV 容量不足。经允许，可在独立试点中
减少 input、等量增加 output；不缩短 warmup/测量窗口、不关闭 async。
保存原/新长度、调整原因和窗口内实际上下文；总长度相同不代表 decode 工作量相同。
调整后的曲线须重测并明确标注，不与原设置混点，也不代表原设置的最大容量。

### 请求模式：TTFT/TPOT 与 E2E

默认 synthetic probe 使用确定性的随机 token trace，各 iteration 更新 trace，
同 seed/case 的系统匹配；默认有长度 jitter，churn 包含超额请求及替换。
连续 decode 入口使用其 manifest 记录的固定 trace，不能与默认 probe 假定同源。

固定请求模式可显式加 `--enable-prefix-caching --shared-prompt
--prompt-length-jitter 0` 测组内完整前缀复用。每轮使用新前缀，预热不预先填充
实测前缀；所有请求一起提交，由原生调度器决定缓存命中和入场。
逐请求 `num_cached_tokens` 记录实际复用量，不能只凭开关断言仅执行了一次 prefill。
该模式仅支持 SparseEngine/vLLM 请求模式、DP1（vLLM），不支持 prefill wave 或
连续 decode 窗口。TPOT 和 decode 事件窗口仍不是纯执行阶段耗时。
若需让所有实测请求命中已建立的前缀，另加 `--prime-shared-prompt`：每轮先以
相同输入生成 1 token，再计时该轮请求；原始记录的 `prefix_prime` 单独保留建立
前缀的耗时和命中量，不计入请求 workload。此结果不能直接当作冷缓存 E2E。

| 指标 | 定义与边界 |
| --- | --- |
| TTFT | 请求到达至首 token |
| TPOT | `(完成时间 - 首 token 时间) / (输出 tokens - 1)`；单 token 为 null，保留调度等待和 prefill 干扰 |
| `output_token_throughput_tps` | 全部输出 tokens / 完整 workload 时间，即 E2E |
| `first_token_window_throughput_tps` | 输入 tokens / 首 token 事件窗口 |
| `batch_decode_token_throughput_tps` | 后续输出 tokens / 最早首 token 至最后完成的窗口 |

后两项是可能重叠的事件窗口，不是执行阶段吞吐。默认
`stage_metrics_status=not_measured`；旧 `prefill_token_throughput_tps` /
`decode_token_throughput_tps` 仅是兼容别名。`tpot_concurrency_proxy_tps`
是 concurrency × 1000 / mean TPOT，不是观测吞吐。

请求统计契约 `per_request_distribution_v3` 合并各 iteration 的逐请求样本，
报告 mean/P50/P95/P99；旧批次最大 TTFT 均值为 `batch_max_ttft_ms_mean`。
SparseEngine 在 step 返回观测 token，不额外逐步同步，标记
`sparseengine_step_token_publication_no_extra_sync_v1`；vLLM 的
legacy finished_time / V1 last_token_ts 按 `timing_source` 区分。
它们是引擎事件，不是 HTTP 客户端延迟；观测边界不一致时不能直接比较。

### 逐步同步：阶段诊断

`--synchronize_step_timing` 测实际计算 tokens / 累计完整同步 step 时间，
包含 step 内工作，但不包含 step 之间的驱动开销。Prefill 排除 prefix hits，
decode 排除 prefill 产生的 token；逻辑输入不等于实际计算量。
既不开同步、也不选连续窗口时，阶段吞吐为 null。旧 `ttft/itl` 是批次观测或代理，
不是请求分布。

固定满批诊断可加 `--require_full_decode_batch --decode_warmup_steps_after_full N`：
保留完整输出，无抢占；排除 warmup 与掉批尾部，拒绝截断。
记录 admission、warmup 和截断策略。GPU activity 来自 `nvidia-smi` 采样，
不是理论 MFU/MBU，也不能据此归因 CPU/launch 开销。

<a id="support"></a>

## 支持范围与验证状态

下表描述适配路径，不保证任意模型/拓扑/依赖版本均已通过 GPU 验证。
新增能力时同步更新本节；每个目标组合先验证计时边界、完整输出、实际 Graph
及 async/overlap 状态，再做正式测量，不能仅以“代码已实现”声称性能已验证。

| 模式 | 当前范围与限制 |
| --- | --- |
| 默认请求 probe | SparseEngine / vLLM，fixed/churn，显式 TP；模型能力另行约束 |
| 原生连续 decode | 已实现逐 rank 边界同步，不以 TP1 为永久限制；当前编排 DP1，TP/EP 受模型和引擎能力约束 |
| vLLM / Tangram 连续 decode | 已实现 async 队列边界排空；外部版本和模型须逐组合 smoke；无 wave admission |
| HiSparse QuEST 连续 decode | 已实现 TP1 overlap 队列边界排空；不是 MLA 适配，无 wave admission |
| Vortex QuEST 连续 decode | 复用 SGLang overlap 完成计数，DP1、EP1 或 EP=TP；已实现 TP 组边界同步和逐 rank Graph/工作量校验，模型及拓扑须分别 smoke 验证；无 wave admission |
| 同步 vLLM 阶段诊断 | 既有验证版本 v0.26.0；设 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；DP1，EP1 或 EP=TP，无 prefix cache/wave；chunk 如指定须等于 scheduler token budget |

连续入口当前要求 fixed、零 jitter、seed 42，开启 Graph、关闭 prefix cache。
原生支持 wave admission；无法满足契约时报告不支持/测量失败，不关闭 async、
回退 step-sync 或减少测量工作量来获取“成功”结果。

vLLM-compatible fork 用 `--engine-kwargs @config.json` 和 `--backend-label`。
参数不能覆盖入口控制的模型、容量、seed、prefix-cache 和计时设置；
`--sparse-method` 写实际算法，保存 fork 的精确版本与最终配置。

<a id="artifacts"></a>

## 验收、留存与排错

先检查终态与原始证据，不仅看终端吞吐：

| 模式 | 必查产物 |
| --- | --- |
| 默认请求 probe | `run_status.json`、`run_manifest.json`、`summary.json` 均为 `success`；`raw_samples.jsonl` 与 `request_samples.jsonl` 完整有效 |
| 连续 decode | `run_status.json` 为 `completed`，`performance.jsonl` 每行 `success`；校验 `repetitions[].decode_window`、逐步完成记录及完整输出，不套用请求模式文件清单 |
| Unified suite | 另需 `suite_status.json` 为 `success`，匹配 source ID coverage 和样本数量 |

请求模式另保存 `comparison_report.md`、`case_hardware/*.json` 及适用时的
`operator_runtime_stats.json`。后者用于 Provider 绑定和实际路径核查。
离线重聚合：`python3 benchmark/efficiency/metrics.py <RUN_DIR>/request_samples.jsonl`，
无需 CUDA，JSON 输出到 stdout，不覆盖原文件；缺字段/失败样本会报错。
缺 `timing_source` 须先核实边界；仅有批次聚合无法恢复请求分位数。

在原始运行产物中记录命令、配置、Git commit、依赖、模型、trace identity、
GPU/拓扑和失败。
除非用户明确要求，不复制或归档源码、不保存工作区补丁、不生成逐文件源码指纹，
不以源码哈希相等作为运行、续跑或复用条件，也不因未跟踪源码而拒绝运行。
保留配置、输入数据、模型和结果校验；源码变化是否需要重测由实验者判断。
正式结果不承担工作区恢复；无法重建时按记录的 commit 和启动参数重跑。
原始输出、逐次测量与聚合分开保存；吞吐按总 tokens / 总时间聚合，保留离散程度。
最大并发须验证整数边界及 max+1；普通崩溃不是容量不足证据。
多方法队列中，方法的 smoke/测量失败应保留证据、跳过该方法并继续其他方法，
最终仍报告失败；GPU 冲突、资源失效、存储错误或用户停止中止整组。
契约、硬件或 Graph/backend 政策变化须重测 baseline，不覆盖旧数据。

脚本、可复用配置、绘图代码和精简的正式结果包都保存在
`scripts/official_experiments/<experiment>/`。结果包只记录设备、启动参数、
最终结果和 Git commit，以及重现正式表格或图所必需的精简 JSON/CSV。
不保存工作区状态、patch、源码快照、逐文件 hash 或恢复材料。大体积日志和
原始输出保存在 Git 外的持久数据盘。Research-Vault 可以索引私有或临时证据，
但不能替代仓库内的正式结果包。不放 tmp、不覆盖旧实验。

| 现象 | 处理 |
| --- | --- |
| GPU busy | 等待或选择其他空卡，不终止他人进程 |
| 依赖/模型不可用 | 检查激活环境、导入、模型 config 及访问权限 |
| OOM | 保存失败；固定矩阵不要静默降 BS/长度，容量扫描按明确策略定位边界 |
| 输出目录冲突 | 换新目录，不覆盖历史 run |
| 窗口/Graph/异步验证失败 | 保留日志，排查适配；不能回退同步协议冒充成功 |
| 硬件采样失败或 Nsight 权限不足 | 检查采样 JSON、工具和权限；不能用粗粒度 activity 替代 counter 数据 |
