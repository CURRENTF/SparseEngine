---
name: paper-efficiency
description: Standardize SparseEngine paper efficiency comparisons, capacity sweeps, and reuse of results in figures. Use for 论文效率测试、跨引擎吞吐对比、最大并发扫描 and their measurement protocols; not isolated kernel microbenchmarks or figure-only styling.
---

# Paper Efficiency

**维护规则：使用前核对当前代码；入口、参数或支持范围变化时，及时更新手册及本 skill 中受影响的说明。**
例如新增 TP2/EP2 或外部引擎适配后，更新实现与验证状态，不能沿用旧缺口阻断。
手册集中维护测量定义和支持范围；本 skill 维护执行流程及论文默认值，不复制支持列表。
代码与文档不符时先核实、修正文档，不把“已实现”当作“GPU 已验证”。

## 执行前必读

读取 `AGENTS.md` 及效率手册的以下章节：

- [指标与计时契约](../../../docs/zh/benchmarking/efficiency.md#measurement-contract)：选择正确测量模式，遵守窗口、异步完成和计数要求。
- [支持范围与验证状态](../../../docs/zh/benchmarking/efficiency.md#support)：核对当前模型、拓扑和外部版本的适配限制。
- [验收、留存与排错](../../../docs/zh/benchmarking/efficiency.md#artifacts)：确认原始证据、聚合和可重绘数据要求。

用户明确指定的协议优先；偏离默认值须写入配置及结果。复现实验沿用其显式协议，
不追溯修改历史数据。论文主结果测正常执行效率，逐步同步仅用于诊断或附录。

## 复用入口

所有代码路径相对仓库根目录：

- 连续 decode：`benchmark/efficiency/bench_probe.py --scenario fixed
  --decode-only-steps 256 --decode-only-warmup-steps 32`，长度 jitter 显式设为 0。
- 请求 TTFT/TPOT、E2E：同一 probe，不开 decode-only；可用
  `scripts/benchmarks/run_efficiency_probe.sh` 编排。
- 用户指定共享前缀时，请求模式可用 `--enable-prefix-caching --shared-prompt
  --prompt-length-jitter 0`；每轮新前缀，核对逐请求 `num_cached_tokens`，
  不凭配置假定只执行一次 prefill，也不把事件窗口称为纯 decode 阶段。
  `--prime-shared-prompt` 可先生成 1 token 建立每轮前缀；`prefix_prime` 单独计时，
  实测 workload 排除这段时间，不能作为冷缓存 E2E。
- 阶段诊断：`benchmark/microbench.py --synchronize_step_timing`。
- 计时与统计：`benchmark/efficiency/metrics.py`；连续窗口复用 `PipelinedDecodeWindow`。
- 扫描、配置和绘图：在 `dev-paper-branch` 复用 `scripts/official_experiments/` 对应目录；
  128K/2K 重跑使用 `sparse_decode_efficiency/config.boundary-sync.json`，
  按[实验说明](https://github.com/CURRENTF/SparseEngine/blob/dev-paper-branch/scripts/official_experiments/sparse_decode_efficiency/README.md#boundary-sync-rerun)先跑 `--smoke-only`。
  32K/2K 历史变体使用外部数据目录下的 `configs/config.boundary-sync.32k2k.json`，
  通过完整路径传给 `--config`；跨长度面板复用绘图入口
  `--grid-config`，各面板独立校验协议。允许显式标为未完成的实测曲线，不能伪造容量边界。
- Vortex QuEST 补测复用 `prepare_vortex.py` 冻结代码/参数和同一 capacity runner，
  `bench_probe.py --engine vortex` 复用 SGLang overlap 完成计数；TP 多 rank 边界已实现，
  每个模型/拓扑仍须通过 GPU smoke 才能声明已验证。不能沿用旧 TP1 HTTP 观察器冒充 TP2 证据。
  在首次 `generate` 前读取 SGLang 启动 readiness 的 `scheduler_info` 获取 KV slots；
  不能假定状态查询 RPC 已有响应处理循环，避免把查询等待误判成 kernel 卡死。

缺能力时扩展共享入口或已有适配器，不按模型/方法复制 runner。
实验脚本只编排参数、队列和导出，不另写计时公式。没有实现授权时报告缺口，
不启动不符合协议的替代实验，不静默关闭 async 或回退 step-sync。

## 默认参数：冻结到实验配置

| 项目 | 新建固定并发实验的默认值 |
| --- | --- |
| 输入 / 输出 | 131072 / 2048 tokens，两种长度 jitter 均为 0 |
| 随机性 | seed=42，greedy，忽略 EOS，生成完整指定输出 |
| 重复 | 1 次完整 workload warmup，3 次独立测量 |
| decode 窗口 | 满批后舍弃 32 步，连续测 256 步；所有对照一致 |
| 执行 | Graph 开启，保留兼容的 async/overlap，不额外逐步同步 |
| 内存 / 缓存 | gpu_memory_utilization=0.90，prefix cache 关闭 |
| Prefill | max_num_batched_tokens=65536；支持独立设置时 chunk size=8192 |
| 并发 | 折线只测所需低 BS；容量先按实测 KV slots 选候选。BS≤30 要求整数最大值及 max+1；BS>30 可接受距最大值不超过 5 的已验证近似值 |

显式填写模型/checkpoint、权重与 KV dtype、GPU/拓扑及稀疏预算。
同一面板匹配 trace 和测量契约；预算分别列出 sink/recent/selected/full layers。
必要的 chunk/wave 差异声明为例外，不把相同 budget 名称当作等量工作或等质量证据。
窗口须覆盖周期评分/驱逐；不足时统一加长所有对照。Warmup 的引擎生命周期按手册说明记录，
不能声称未实际测量的热引擎状态。

### 满批窗口不足时

先区分请求过早结束与真实 KV 容量不足。经用户允许，可小幅减少 input、等量增加
output，保持总长度不变，为满批 warmup 和测量窗口留足输出；先做有界目标 BS 试点。
这不能保证突破容量上限，也不能掩盖适配器错误。失败时保留证据，不持续缩短输入追求成功。
调整用独立配置/attempt，记录原值、新值、原因及窗口内实际 context lengths；
总长度相同不代表 decode 上下文相同。正式采用时重测该曲线并在图例或 caption 标注，
不得混入原设置的点，或将调整后的容量宣称为原设置的最大并发。

## 执行与交付

1. 检查空卡，固定配置，记录 Git commit 和 dirty 状态，按手册验证每个目标组合的 smoke，再启动持久队列。
   除非用户明确要求，不复制或归档源码、不保存工作区补丁、不生成逐文件源码指纹，
   不以源码哈希相等作为运行、续跑或复用条件，也不因未跟踪源码而拒绝运行；
   源码变化由实验者判断是否需要重测。保留配置、输入数据、模型和结果校验。
   独立冻结编排时，用显式 `--repo` 预检统计模块导入；不能假定脚本父目录是源码根。
2. 校验实际完成工作、全部输出和逐 rank Graph/async 状态；分别记录容量失败、
   测量无效与实现错误，普通崩溃不能作为容量上界证据。
   容量失败按下述精度目标继续定位；方法的 smoke/测量失败保留证据并跳到其他方法，
   最终队列仍返回非零。GPU 冲突、保留进程失效、存储错误和用户停止中止整组。
   续接旧队列只启动未开始的方法，不自动重跑失败点或覆盖已有数据。
   经批准补缺失边界时，复用 probe-only 多点入口；重验旧原始点并记录源码差异，
   只有补测达到所声明的容量精度后才发布相应标签；完整容量曲线仍要求整数 max/max+1，
   近似或仅有下界的曲线走显式 partial 路径。
3. 按手册聚合并保留重复间离散程度；复用数据前核对协议与原始证据。
   容量扫描优先利用实际 KV slots 估算候选，低 BS 只保留图中需要的点。
   BS≤30 的精确最大值仍须实测 max 成功及 max+1 容量失败。BS>30 可停止在
   已验证成功的 B：若有独立的容量上界 U≤B+5（例如已分类的容量失败 F 给出
   U=F-1），报告“近似最大 BS（误差≤5）”及上下界，不必二分至 max+1；
   若只有 KV-slot 估算而无可信上界，则只报告“已测可用 BS≥B”，不得声称误差≤5。
   估算值不是实测吞吐点；压缩/wave 路径还需考虑峰值占用与满批驻留。
   step-sync 与 boundary-sync 不混图、不估算修正；异常未解释时不宣称性能或质量优越。
   分类修复后的同协议续跑遵循实验 README 的参数、硬件和原始结果校验，不放宽跨协议复用。
4. 脚本、可复用配置、绘图代码和正式结果包留在
   `dev-paper-branch` 的 `scripts/official_experiments/<experiment>/`。结果包只记录设备、启动参数、
   最终结果和 Git commit；若有正式图表，同目录保存其必需的精简 JSON/CSV。
   不保存工作区状态、patch、源码快照、源码 hash 或恢复材料。大体积日志/
   原始输出留在持久数据盘。Research-Vault 只用于可选的私有运行历史、
   失败记录和外部证据索引，不能替代仓库内的正式结果。
5. 图例标注外部系统及具体方法；unsupported 不伪造为 0。
   声称整体服务效率时另报请求/E2E 指标，不只交付 decode 窗口图。
