# 稀疏方法运行时架构

本文说明 SparseEngine 如何组织不同的稀疏方法，以及新增方法时应该把代码放在
哪里。它也适用于接入模型原生的动态稀疏注意力（DSA）。

最重要的原则只有三条：

1. `CacheManager` 负责 KV Cache 的实际存储和长期状态。
2. `SparseMethodRuntime` 负责当前推理步骤中的打分、选择和跨层协调。
3. `SparseController` 只提供统一入口，不直接实现某一种稀疏算法。

这样做的目的，是让每种稀疏方法都能使用适合自己的存储方式，同时保持
Attention、Scheduler 和 ModelRunner 的通用路径简单、稳定。

## 整体关系

```mermaid
flowchart TD
    A["配置与方法注册表"] --> B["ModelRunner"]
    B --> C["SparseController 统一入口"]
    C --> D["SparseMethodRuntime 方法逻辑"]
    C --> E["ActivationController 激活值逻辑"]
    B --> F["CacheManager 物理缓存"]
    G["Attention"] --> C
    G --> F
    D --> H["SparseSelection 选择结果"]
    H --> F
    F --> I["Prefill/Decode 计算视图"]
    I --> J["Attention Provider"]
```

这里有两层统一接口：

- 推理引擎始终调用同一个 `SparseController`。
- `SparseController` 再把工作交给当前方法对应的
  `SparseMethodRuntime`。

`CacheManager` 是另一套独立接口，专门处理物理缓存。Runtime 和
CacheManager 可以各自复用代码，不要求使用相同的继承关系。

Decode batch 可以包含不同长度的请求。各方法根据逐行长度对有效 KV 进行评分、选择或
压缩；未超过方法实际保留预算的请求即使执行评分与 selection，也保留全部有效 KV。
CUDA Graph 对每个 batch
bucket 捕获一组容量为 `max_model_len` 的图，请求跨过稀疏预算边界时不切换图。
Prefill 仍按执行模式和兼容性分组。

OmniKV decode selection 直接使用设备端候选长度。历史候选未超过保留预算的行直接保留
全部历史，不执行分数归约或 top-k 扫描；长行只归约、选择有效历史。未写入的分数尾部
不具备有效值，消费者不得读取。不同 provider 可以输出不同排列，但逻辑索引与物理
slot 必须逐项对应。捕获容量和图身份保持不变。

## 各组件负责什么

| 组件 | 主要职责 | 不应负责 |
| --- | --- | --- |
| `method_registry.py` | 方法名称、别名、默认调度方式、打分要求、模型兼容性、Prefix Cache 和 CUDA Graph 支持范围。 | 运行时张量、缓存分配。 |
| `SparseController` | 向推理引擎提供统一调用接口，把调用转交给 Runtime 和 ActivationController。 | 具体方法的判断分支和算法实现。 |
| `SparseMethodRuntime` | 当前步骤的逐层状态、注意力分数准备、稀疏位置选择、跨层结果传递，以及触发压缩或淘汰。 | 物理 slot 的所有权、与 Prefix Cache 绑定的长期元数据。 |
| `ActivationController` | 保存和使用隐藏层激活值，例如基于激活值的复用或跳层逻辑。 | KV Cache 分配和物理读视图。 |
| `CacheManager` | KV 或 latent cache、slot、page、压缩池、长期元数据、空间分配、压缩、重建、Prefix Cache 生命周期和计算视图。 | 调度策略和模型层中的方法判断。 |
| `MemoryOracle` / `RuntimeState` | 向 Scheduler 提供容量、临时空间需求和执行方式。 | 稀疏算法本身。 |
| `Attention` | 写入本层 KV、请求计算视图、运行 attention 算子并调用通用 hook。 | 识别稀疏方法名称或保存方法状态。 |
| Operator / Provider | 执行已经准备好的算子，并管理算子需要的工作区。 | 缓存分配和稀疏策略。 |

判断状态归属时，可以按下面的规则处理：

- 需要经历 append、Prefix Cache 命中、fork、恢复、回滚、offload 或释放的
  状态，放在 `CacheManager`。
- 每次 forward 都会重新准备，或者只负责把选择结果传到后续层的状态，放在
  `SparseMethodRuntime`。
- 从隐藏层激活值产生的状态，放在 `ActivationController`。
- 会影响请求是否能进入、能否组 batch 或需要多少显存的信息，通过
  `MemoryOracle` 或 `RuntimeState` 提供给 Scheduler。

## SparseController：统一入口

`src/sparseengine/engine/sparse_controller.py` 应保持轻量。推理引擎主要使用以下
接口：

```python
class SparseController:
    def prepare_forward(self, seqs, is_prefill): ...
    def get_prefill_selection(self, layer_idx): ...
    def get_decode_selection(self, layer_idx, query): ...
    def on_layer_attention_end(self, layer_idx): ...
    def on_layer_end(self, layer_idx, context): ...
    def post_forward(self, seqs, is_prefill): ...
```

它还负责汇总 CUDA Graph 需要长期保留的张量、重置打分缓冲区、输出调试信息，
以及把 tokenizer 信息交给 `ActivationController`。

不要在这里增加 `if sparse_method == ...`。如果一个方法需要特殊行为，应在
对应 Runtime、CacheManager 或其他职责明确的通用接口中实现。

## SparseMethodRuntime：方法逻辑

`src/sparseengine/engine/sparse_methods/base.py` 定义了统一的输入类型：

- `SparseStepContext`：一次 prefill 或 decode 的上下文。
- `PrefillSelectionRequest`：某层的 prefill 选择请求。
- `DecodeSelectionRequest`：某层的 decode 选择请求，其中包含当前 query。
- `AttentionEndEvent`：某层 attention 完成后的通知。
- `LayerEndEvent`：整个模型层完成后的通知。

Runtime 在不同阶段执行以下工作：

| 方法 | 调用时机 | 作用 |
| --- | --- | --- |
| `prepare_step` | 进入模型层之前 | 读取当前 batch 的缓存信息，准备逐层状态和打分缓冲区。 |
| `needs_attention_score` | 准备阶段 | 判断当前层是否需要收集注意力分数。 |
| `build_prefill_selection` | Prefill attention 之前 | 生成本层的逻辑选择结果。 |
| `build_decode_selection` | Decode attention 之前 | 根据当前状态和 query 生成逻辑选择结果。 |
| `on_attention_end` | Attention 完成后 | 完成必须等待 attention 结束才能做的打分处理。 |
| `on_layer_end` | 模型层结束后 | 处理分数并把选择结果传给后续层。 |
| `finish_step` | 整次 forward 结束后 | 触发压缩、淘汰或其他收尾操作。 |

`LayerBatchSparseState` 只表示当前推理步骤中的逐层逻辑状态。它可以引用
CacheManager 中的稳定张量，但不负责这些张量的分配和释放。

## Runtime 如何复用代码

`engine/sparse_methods/factory.py` 在初始化时，根据解析后的物理 cache method 选择
Runtime。通常它就是 `sparse_method`；例外是
`prefill_sparse_method="h2o_prefill"`，即使 decode 为 vanilla，也会解析为 H2O
CacheManager/Runtime ownership。逐层执行时不再查询注册表，也不再按方法名分支。

当前 Runtime 按实际处理方式进行少量继承：

`omnikv_prefill` 在 factory 中通过 `PrefillOverrideRuntime` 组合：prefill 交给
`OmniKVPrefillRuntime`，decode 保留原 vanilla/OmniKV runtime。`finish_step`
结束后立即恢复 decode 状态，因为 Graph replay 恢复引用时不会调用 `prepare_step`。
这种互斥阶段委托仅用于没有 prefill 评分/压缩职责的 decoder。

Chain 模式跨轮保留 Standard/OmniKV 的共享物理槽位池；逐层驻留量重复同一个分配
长度，准入取预留量的最大值，不把它们当成独立池相加。Chain fingerprint 包含
prefill 观察层、预算和 chunk 限制。由于 chunk 后面的 query 能改变前面 token 的
深层 KV，radix 仍不适用。OmniKV offload 复用已有 prefill gather：选中的历史从
host backing 读取，完整当前 chunk 直接来自 GPU 张量。选择表将当前 token 放在
末尾，满足已有 gather 的索引契约。

Prefill 复用 `context_attention_fwd` 的 causal raw-QK 累加，除以 query 数后取
head max，并在 attention TP 组内做 MAX 归约，然后选择历史。
`build_omnikv_keep_and_slots` 构造共享逻辑视图，完整物理 KV 不变。
Operator 的可选评分契约在初始化时分别准备评分和无评分 provider，普通 attention
仍遵循 upstream-first 选择。检查过的 SGL context kernel 和 vLLM FA 接口没有提供
该逐 head raw-QK 累加输出，attention output/LSE 不能直接替代这一评分契约。

MLA 使用已有分块 prefill 评分器，对 raw QK 取 query/head 最大值。视图中的
float32 `[B, L]` 输出映射为覆盖当前全部 query 的评分请求；历史评分复用每个
有界 block 已展开的 K，观察层之间将共享 atomic-max 缓冲重置为负无穷。
Gemma 4 仅在全局层使用已有逐 head raw-QK 评分。层选择遍历逻辑 KV 消费者，
包括共享 KV 的别名层；跳过滑动窗口层时保留前一个全局观察层。因此滑动层继续
使用完整位置域，不需要修改稀疏 window mask。

设 batch 为 `B`，本地 query heads 为 `H`，上下文为 `L`，chunk 为 `C`，
历史预算为 `K`，head dimension 为 `D`。每个观察层需要 `O(B H C L D)` attention，
以及对 `B H L` 分数的归约/选择。后续层最多读取 `sink + K + recent + C` 个 token。
共享 float32 评分缓冲占 `4 B H L` 字节，归约分数占 `4 B L` 字节；每个观察区间
保留两张 int32 选择表，共最多 `8 B (sink + K + recent + C)` 字节，另有行长度和
top-k workspace。`B=8, H=32, L=32768` 时共享评分缓冲为 32 MiB。
Startup prefill profiling 包含这些分配。观察层之间清零评分，step 结束释放；
不物化完整 QK 矩阵、不新增持久 KV pool。收益取决于观察层数及 chunk/budget，需实测。

| Runtime | 当前方法 | 共同点 |
| --- | --- | --- |
| `PassThroughRuntime` | vanilla、QuEST | Controller 侧返回完整逻辑选择，特殊物理视图由 CacheManager 或 Provider 构造。 |
| `StreamingLLMRuntime` | StreamingLLM | Attention 使用普通视图，结束后按 sink 和 recent window 物理淘汰。 |
| `ScoredCompactionRuntime` | SnapKV、PyramidKV | 共用打分和物理压缩流程；PyramidKV 使用逐层预算。 |
| `H2ORuntime` | H2O prefill 和/或 H2O decode | 准备 prompt 分数并独立触发 prefill/final-prompt 压缩；可选 decode 概率累计和周期驱逐。 |
| `JointDecodeRuntime` | R-KV、SkipKV | 共用 decode 压缩流程，但分数来源和选择算法不同。 |
| `DynamicSelectionRuntime` | OmniKV、DeltaKV | 在观察层收集分数，并把动态选择结果传给后续层。 |

H2O 的可选 decode 开关复用现有评分 kernel：prefill 概率评分先按 head
汇总 query 观测，再取 head-wise max；decode 概率则对 query heads 求和。
强制 probability 避免混加 raw logits 与概率，但不统一这两种 head reduction，
也不代表已经与原始 H2O 评分完全对齐。

MLA decode 是显式近似：provider 输出 head-max raw QK，H2O runtime 再对保留
token 做 softmax 和累计。它复用 reduced-score 缓冲区与驱逐生命周期，不受
explicit-KV 的逐 head score-fusion 设置影响；TODO 和一次性警告标明尚未实现
逐 head 概率等价性。

Explicit-KV H2O 的 score fusion 由 provider 持有一份
`[batch, heads, capacity]` raw-QK workspace，供顺序执行的各层和 Graph 复用。
每层立即利用 attention 的自然对数 LSE 将 logits 归一化并对 head 求和，
写入该层独立的 `[batch, capacity]` 概率后，下一层才复用 workspace。
只有 reduced 分数保留到 `finish_step`；历史累计和物理驱逐仍由 CacheManager
负责。这样消除逐层、逐 Graph 的 raw-logit 副本，不改变评分和驱逐策略，
也不改变已捕获的地址与 launch 容量。

只有以下行为确实一致时，才应继承同一个 Runtime：

- 分数的含义和形状；
- 触发压缩或选择的时机；
- 选择的是 token、slot、page 还是压缩索引；
- 修改 CacheManager 的顺序；
- Prefix Cache 行为；
- CUDA Graph 缓冲区的使用方式。

如果这些行为不一致，应新增独立 Runtime，只抽取真正共用的小函数。

CacheManager 的继承体系处理的是另一类问题，例如物理布局、slot 分配、Prefix
Cache 和显存核算。因此，两个方法即使共用 Runtime，也不一定应该共用同一个
CacheManager 基类；反过来也一样。

## 从选择结果到 Attention 计算

一次 attention 调用按照下面的顺序进行：

1. Runtime 生成 `SparseSelection`，说明逻辑上要看哪些内容。
2. Attention 把选择结果以及当前 query、K、V 交给 `CacheManager`。
3. `CacheManager` 将逻辑位置转换成物理 slot、page、latent cache 或临时重建结果。
4. `CacheManager` 返回 `PrefillComputeView` 或 `DecodeComputeView`。
5. Attention Provider 使用这个计算视图运行算子，不需要知道稀疏方法名称。

常用的数据类型包括：

- `SparseSelection`：逻辑选择结果；
- `AttentionViewMeta`、`PagedDecodeViewMeta`：物理位置和长度信息；
- `PrefillComputeView`、`DecodeComputeView`：算子使用的完整计算视图；
- `ExplicitKVPayload`、`MlaLatentPayload`：实际缓存数据。

不要通过 `layers/attention.py` 传递方法专用 tuple、整个配置对象、隐藏的全局张量
或方法名称。

QuEST 是一个典型例子。它的 Runtime 返回普通的完整逻辑选择，但
`QuestCacheManager` 和选择算子会使用当前 query 构造原生物理页视图。QuEST 的
page metadata 长期跟随物理缓存，因此应留在 CacheManager，而不是复制到
Runtime。

## 有序提交与结果回收

同步和异步执行使用相同的 attention、层和步骤 hook。`post_forward` 与 cache
`on_forward_end` 在提交阶段执行，早于 forward context 清理和下一批准备。
它们可以只提交 GPU 工作而不等待完成；依赖 CPU 读取结果的操作保留必要等待。
Graph replay 在图外更新激活策略输入，已捕获的层内 hook 保持原有顺序。

结果回收只处理已复制的 token/logprobs 和延迟的 prefix 记录发布，不能对当前批次的
可变状态再次执行计算 hook。请求存储在所有未完成使用者退出后才释放。

需要在 CPU 读取尚未确认 token 历史的 runtime、激活控制器或 cache manager 声明
`requires_committed_token_history`。调度器在准备下一批之前收回待确认的 token。
不产生输出 token 的中间 prefill 分块仍可连续提交。GPU 消费者使用有序的设备 token 接续。

Cache 自有的可复用 CPU 上传区在写入前调用 `acquire_step_host_buffer`。
异步提交借用独占区域，直到结果回收才归还；同步执行继续使用原缓冲区。
GPU graph 地址保持固定。辅助队列的拥有者负责在依赖读取或空间复用前完成队列交接，
包括步骤结束时的交接。

## Prefix Cache 和 CUDA Graph

Prefix Cache 保存的是物理缓存状态，所以由 `CacheManager` 负责。与某段 KV
对应的长期元数据，必须和 KV 一起支持：

- 分配和追加；
- Prefix Cache 命中和挂接；
- fork；
- 恢复和回滚；
- 淘汰和 offload；
- 释放。

Prefix Cache 挂接完成后，Runtime 从 CacheManager 提供的当前 batch 状态重新
构建逻辑状态。Runtime 不应长期保存可能在缓存转移后失效的物理位置。

Runtime 可以持有 CUDA Graph 使用的打分或选择工作区，但必须满足：

- 在 capture 前完成分配；
- 通过 `decode_graph_keepalive_tensors` 保证张量不会被释放；
- replay 前正确重置输入分数；
- replay 期间不临时分配，不查询方法注册表，也不切换到其他实现；
- eager 和 CUDA Graph 得到相同的选择和输出。

## 接入新方法或 DSA 模型

建议按以下顺序接入：

1. 在配置和 `method_registry.py` 中登记方法名、支持范围和默认行为。
2. 明确支持哪些模型、缓存布局、并行方式和 Attention Provider，并拒绝不支持的
   组合。
3. 在 CacheManager 或存储模块中实现长期物理状态及其完整生命周期。
4. 选择已有 Runtime，或者新增 Runtime，并在 Runtime factory 中登记。
5. 使用统一的选择类型和计算视图连接 Runtime、CacheManager 与 Attention。
6. 通过 `MemoryOracle` 或 `RuntimeState` 告诉 Scheduler 实际容量和临时空间需求。
7. 只有确实需要时，才增加 ActivationController 或新算子。
8. 对声明支持的 Prefix Cache 和 CUDA Graph 路径逐项验证。

接入模型原生 DSA 时，先判断模型实际提供了什么：

- 已选 token 的索引；
- 根据 query 选择 page 的规则；
- 压缩或 latent cache；
- 需要长期保存的路由信息。

随后仍按相同方式分工：

- 模型和 layout 代码说明存储及算子需要的数据格式；
- Runtime 负责打分、选择和跨层传递；
- CacheManager 负责物理缓存和长期元数据；
- Provider 使用统一计算视图或模型原生数据运行算子；
- Scheduler 只读取容量和执行限制。

不要在 Attention、ModelRunner 或 Scheduler 中直接增加某个 DSA 方法的判断。
如果现有类型无法表达新的缓存数据，应扩展最小的统一数据类型，而不是绕开现有
接口。

## 提交前需要确认

- `SparseController`、Attention、ModelRunner 和 Scheduler 没有新增方法名分支。
- 物理缓存元数据和 slot 生命周期仍由 CacheManager 负责。
- Runtime factory 中的方法映射清楚、可检查。
- 分数形状、类型、初始值、归一化顺序、Top-K 相同分数处理、触发边界和修改
  顺序符合算法原定义。
- 声明支持 Prefix Cache 时，已经覆盖首次运行、命中、追加、fork、恢复和释放。
- 声明支持 CUDA Graph 时，已经验证张量地址稳定，并与 eager 结果一致。
- 先用固定输入比较生成 token，再进行质量评测。
- 性能比较使用相同请求、Provider、预算和有效实验产物。

完整的接入和验证步骤见仓库内的 `$add-sparse-method` skill。
