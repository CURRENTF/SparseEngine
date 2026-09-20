# 核心稀疏方法

SparseEngine 围绕 cache-manager-first sparse runtime 构建。engine 支持 physical eviction、logical masking 和 hybrid compression，而不强迫 `attention.py` 持有方法特定状态。

## 支持的方法

将 `sparse_method` 设置为下列方法名之一。

| 方法 | 类别 | 说明 | 主要 Runtime 参数 |
| --- | --- | --- | --- |
| `palu` | 低秩 KV | [基于激活 whitening 的分组 K/V 压缩](palu.md)，decode 融合重建。 | `palu_checkpoint_path` |
| `vanilla` | Dense baseline | Full attention baseline，用于验证正确性并测量非稀疏 engine path。 | 仅使用通用 engine 参数。 |
| `streamingllm` | Physical eviction | StreamingLLM 风格的固定 sink 加 recent-window cache。保留 prefix/tail 策略之外的 token 会从 active KV cache 中被物理淘汰。 | `sink_keep_tokens`, `recent_keep_tokens` |
| `attention-sink` | Physical eviction | attention-sink alias policy，使用相同的 sink-token 和 recent-window 保留模型。适合将 sink-window 行为与其他 physical eviction 方法对比。 | `sink_keep_tokens`, `recent_keep_tokens` |
| `snapkv` | Physical eviction | SnapKV 风格的 token selection 使用 prompt 末尾的 observation window，在生成前选出并保留紧凑的重要 prompt KV。当前与论文对齐的 decode 路径不再评分，也不会再次执行 SnapKV selection，只追加生成 token。 | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `kvzip` | Physical eviction | 仓库现有的 token 共享 KVzip 重建打分，在 prefill 完成后统一压缩 prompt KV。 | `kvzip_token_budget`, `kvzip_score_chunk_size`, `kvzip_prev_postfix_size` |
| `h2o` | Physical eviction | 中间 prefill chunk 可压缩到 `h2o_prefill_budget`，最终 prompt 压缩到 `h2o_decode_budget`。默认 decode 不评分或驱逐，物理 row 随生成 token 增长；开启 `h2o_decode_eviction` 后逐步累计概率分数并周期驱逐。 | `h2o_decode_eviction`, `h2o_decode_budget`, `h2o_decode_eviction_interval`, `h2o_prefill_budget`, `h2o_recent_ratio`, `h2o_prefill_score_window`, `sparse_prefill_score_mode` |
| `pyramidkv` | Physical eviction | PyramidKV 风格、依赖 layer 的 KV 保留方式。它在 layer 之间分配 sparse budget，并物理存储选中的 context token。 | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `omnikv` | Logical masking，可选 offload | 跨层共享 token 选择；可将稀疏层完整历史保存在 pinned CPU 内存，decode 精确取回当前选择的 KV。 | `full_attention_layers`, `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `enable_omnikv_offload` |
| `quest` | Query-aware page selection | QuEST 根据持久化的 page min/max summary 选择 token page，prefill 保持 dense。显式 KV 模型在 key 坐标中评分；GLM-4.7-Flash 使用匹配的 absorbed decode query 对融合 MLA latent/RoPE cache 评分，同时 compute payload 继续保持 latent。 | `quest_chunk_size`, `quest_skip_layers`, `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens` |
| `deltakv` | Hybrid compression | 依赖 compressor 的精简 DeltaKV runtime。旧配置中的 `deltakv-less-memory*` 名称会规范到此方法，但实际 benchmark run 仍需要匹配的 compressor checkpoint。 | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `full_layer_kv_quant_bits` |

SparseEngine 在 public command、`LLM(...)`、runtime config 与内部消费者中统一使用 `sparse_method`。



## KVzip

使用 `LLM(model, sparse_method="kvzip", kvzip_token_budget=4096)`，
在上下文重建后最多保留 4096 个 prompt token。较短的 prompt 保留完整 KV。
首个输出 token 来自原始 dense prefill；后续 decode 读取保留的 prompt KV
和全部生成 token 的 KV。

这是仓库 `kvzip_global` 的 token 共享变体，与原论文逐 head 非均匀淘汰不同。
各层、各 head 和 attention TP rank 共享选中的 token 位置，无需额外 checkpoint。
`kvzip_token_budget` 是完整的 prompt 保留预算；`sink_keep_tokens`、
`recent_keep_tokens`、`decode_keep_tokens` 不额外增加保护区域。

支持显式 KV 存储的 Qwen2、Qwen3 和 Llama，可使用分块 prefill、批量请求、
tensor parallel、eager decode 与 decode CUDA Graph。当前拒绝 radix/chain
prefix cache、offload、稀疏 prefill 组合，以及 recurrent、MLA、共享 KV 和 MoE 模型。

`kvzip_score_chunk_size` 控制每次重建的原文 token 数（默认 2048），
`kvzip_prev_postfix_size` 控制附带的前文 token 数（默认 64）。重建指令、
前文与当前 chunk 的总长度必须不超过 `engine_prefill_chunk_size`；完整 prompt
加重建输入必须不超过 `max_model_len`，需预留额外容量。重建会增加 prefill 开销，
物理压缩生效不等于已证明性能提升。


## OmniKV KV offload

OmniKV offload 将全注意力层 KV 保留在 GPU，稀疏层历史保存在 CPU，
按精确选择的索引取回 attention 所需 KV，并复用 GPU 缓存中的 KV。
建议在显存受限时开启。当并发超过 GPU KV 容量时，可减少请求等待，明显降低 TTFT；
但搬运开销可能使 decode TPS 回退，尤其是在全部请求本来就能装入显存时。

| 参数 | 说明 |
|---|---|
| `enable_omnikv_offload` | 默认 `False`。在 `sparse_method="omnikv"` 时设为 `True` 开启。 |
| `omnikv_offload_cache_tokens` | 每个稀疏层、每个请求的 GPU 缓存 token 数。默认 `None` 自动设置；`0` 关闭 LRU 缓存；正数必须覆盖所选 token 预算。增大缓存可减少搬运，但会占用更多显存。 |

Prefill 加速由 `prefill_sparse_method` 独立选择。当前支持三种方法：
`h2o_prefill` 用于中间 chunk 的物理 KV 压缩，`flashprefill_v2` 用于稀疏化
prefill attention 计算，`omnikv_prefill` 用于分块 prefill 的跨层历史选择。
它们是同一条轴上的备选项，可以分别与兼容的 cache/decode
方法组合。H2O prefill/decode 组合矩阵及“省略”和“显式空字符串”的兼容规则见
[runtime 参数语义](../configuration/runtime-parameter-semantics.md#prefill-sparsity)。

`omnikv_prefill` 可搭配 vanilla（`sparse_method=""`）或 OmniKV decode。
它保留完整 KV 存储，只改变 prefill attention 的读取范围。
`omnikv_prefill_full_attention_layers="auto"`（默认）优先使用已登记的 prefill
profile，否则复用模型的 OmniKV decode profile，并仅保留全局 KV 层；
未登记的模型会报错，需要先校准。也可显式指定
层列表，必须包含第一个全局 KV 层。滑动窗口层不参与选择，始终使用原有
attention。这些层配置及以下预算均独立于 decode 配置：

| 参数 | 含义 |
|---|---|
| `omnikv_prefill_keep_tokens` | 选中的历史 token 数，不含 sink、recent 和当前 chunk；默认 4096。 |
| `omnikv_prefill_sink_keep_tokens` | 始终保留的开头 token 数；默认 8。 |
| `omnikv_prefill_recent_keep_tokens` | 当前 chunk 之前始终保留的最近历史 token 数；默认 128。 |

当前 chunk 始终完整保留，并使用 causal mask。评分固定为完整 chunk 的 raw QK，
使用 float32：显式 KV 对 query 取均值后取 head 最大值；MLA 对 query 和 head
一起取最大值。`sparse_prefill_score_mode` 不改变此方法。
`engine_prefill_chunk_size` 同时影响质量和可减少的历史 attention 计算量；
整个 prompt 只有一个 chunk 时，没有历史计算可裁减。即使搭配 vanilla decode，
稀疏 prefill 仍可能影响质量。

支持显式 KV、MLA latent，以及 Gemma 4 全局 attention（包括共享 KV 的后续层）。
Gemma 4 滑动窗口层既不评分，也不使用 OmniKV prefill 的选择结果。连续显式 KV
和 MLA 可开启 `enable_omnikv_offload=true`，使用 OmniKV 的 CPU KV backing；
prefill 和 decode 的完整 attention 层列表可以不同。异构 Gemma 4 KV 存储仍受
原有 offload 限制。

开启 `enable_prefix_caching=true` 后，`prefix_cache_mode="auto"` 自动选择 `chain`，
也可显式指定 `chain`。下一轮发送完整逻辑上下文和返回的 `chain_id`，续接已完成的
同一条链。复用的是实际计算过的 KV，不会按新的 chunk 划分重算历史轮次。
跨整个 chunk 的 query 选择可能改变共享前缀的 KV，因此拒绝 `radix`。
保持 `enable_prefix_cache_offload=false`：空闲链快照尚不支持这种共享槽位布局；
独立的 `enable_omnikv_offload` 仍可启用。

> [!NOTE]
> 两种 score-free decode contract 的论文来源不同。[SnapKV 论文](https://arxiv.org/abs/2404.14469)
> 使用 prompt 末尾的 observation window 选择 prompt KV；增加 decode-time
> 重新评分和淘汰属于 SparseEngine 增强。[H2O 论文](https://arxiv.org/abs/2306.14048)
> 则定义了跨连续 decode step 的动态保留策略。SparseEngine 对中间 chunk 的 H2O
> 压缩是自己提出的 prefill 扩展。最终 prompt 压缩虽然发生在 final-prefill boundary，
> 但它准备的是生成阶段消费的短 cache，因此属于 decode contract。可选的在线评分更新
> 向原始 H2O 算法靠近；周期性的 batch 淘汰和有限 prefill observation window
> 仍属于系统或算法变体。

SnapKV 的 `sparse_prefill_score_mode` 默认为 `logits`，PyramidKV 默认为
`probability`。H2O（含独立的 `h2o_prefill`）默认使用
`sparse_prefill_score_mode="logits"` 和 `h2o_prefill_score_window=128`。
显式指定的 score mode 和 window 会覆盖默认值，但 `h2o_decode_eviction=True`
会强制使用 `sparse_prefill_score_mode="probability"`。

H2O 默认采用有限 query window 的近似评分。如需使用完整 chunk 的归一化
attention mass 评分，请显式设置 `sparse_prefill_score_mode="probability"`
和 `h2o_prefill_score_window=0`。此时每个 KV layer 独立地对完整当前 query
chunk 的归一化 attention probability 求和，并在 prefill chunk 之间累计
attention mass。H2O probability 模式会输出性能警告，因为即使复用 attention
LSE，仍需额外计算 QK 评分。两种模式都要求每个 H2O KV layer 独立保存评分；
decode score 收集与周期淘汰默认关闭，可通过 `h2o_decode_eviction=True` 开启。

该开关要求 `sparse_method="h2o"`。开启后每个 decode step 累计归一化 attention
mass，物理 row 达到 `h2o_decode_budget + h2o_decode_eviction_interval` 时保留
heavy hitters 和 recent tokens，压缩回 decode budget；显存容量压力可使超预算的
active row 提前驱逐。即使显式设置 `logits`，也会强制改为 `probability` 并输出警告。
`h2o_prefill_score_window` 保持用户设置，允许非零值；沿用概率模式的 `[0, 128]`
范围。MLA latent 模型使用显式近似：对跨 head 取 max 的 decode logits 计算
`softmax(scale * RAW_QK_REDUCED)`，再累计和驱逐。这不等价于每个 head 先归一化
再归约，尚未与原 H2O 完全对齐；启用时每个进程仅警告一次。MLA prefill 评分和
默认的 score-free decode 路径不受影响。

## Prefill Scheduling Policy

Prefill scheduling 是方法 contract 的一部分，由 registry 管理。唯一事实来源是 `src/sparseengine/method_registry.py`；benchmark script 和用户配置不应重新定义方法语义。

| Policy | Runtime 语义 | 当前默认方法 |
| --- | --- | --- |
| `all_chunked` | 每个 prefill request 都受 `engine_prefill_chunk_size` 和 scheduler 常规 batch 限制约束；忽略 `long_prefill_offload_threshold`。 | `vanilla`, `streamingllm`, `attention-sink`, `snapkv`, `h2o`, `quest`, `omnikv` |
| `long_bs1full_short_batch` | 在附加受支持的 prefix 后，residual 不超过 `long_prefill_offload_threshold` 时使用 atomic full prefill，并且可以互相 batch；更大的 residual 被隔离，并使用不超过 `engine_prefill_chunk_size` 的 RawKV offload chunk。 | `pyramidkv` 和 DeltaKV family 方法 |

DeltaKV family 方法和 PyramidKV 只对外提供 `long_bs1full_short_batch` policy。threshold 默认是 `65536` token（64K）。未设置 `engine_prefill_chunk_size` 时，它默认等于 threshold；显式值必须为正数且不大于 threshold。必要时，`Config` 会提高 `max_num_batched_tokens`，使一个 threshold 大小的 full prefill 能够原子容纳。PyramidKV 根据 chain prefix attach 后的 residual 进行分类。DeltaKV 不支持 prefix caching，并会在修改 compressed 或 quantized row metadata 前拒绝 attached-prefix prefill。

启用 full-layer KIVI 时，DeltaKV 的 decode 常驻 raw 尾部池与 `max_model_len` 大小的 prefill staging buffer 是两块独立容量。多个 short prefill 通过互不重叠的 request range 共享 staging buffer；常驻 raw 尾部的 slot 数不是 prefill batch 上限。

## Prefix Cache 模式

`enable_prefix_caching=true` 支持两种有意分离的布局。
`prefix_cache_mode=auto` 为 vanilla/OmniKV/QuEST 选择 radix，为
SnapKV/H2O/PyramidKV/R-KV/SkipKV 选择线性 chain。选择 `omnikv_prefill` 后，
vanilla/OmniKV 改用 chain 模式。也可以显式请求 `radix`
或 `chain`，但不兼容的方法/模式组合会快速失败。
GLM-4.7-Flash latent QuEST 支持设备驻留的 radix Prefix Cache 与 decode CUDA
Graph 组合，prefix block 大小须等于 `quest_chunk_size`。页选择保持 TP rank
本地 query head 打分，因此不保证与单卡结果等价。支持 Prefix CPU offload，
恢复时同时搬运 latent/RoPE cache 和 page summary；
沿用 attention TP=1 或 TP=2 的 offload 限制。
已有 vanilla/OmniKV radix tree 可通过
[Prefix cache 修剪](prefix-cache-pruning.md)中的 SnapKV 或 KVzip 打分维护接口
进行物理压紧；QuEST tree 会明确拒绝修剪。

Chain 布局跨 turn 保留同一个 owner `seq_id`，且永不分支。调用方发送完整逻辑
上下文和服务端返回的 `chain_id`；服务端验证 processed boundary 后只转发新增
suffix。方法 KV 与 metadata 仍由 cache manager 持有。Idle chain 采用严格
LRU 回收，active writer 保持 pinned。Rank 0 使用紧凑 32-bit storage 保存
processed logical token ID，以便文本 continuation 保持驻留的 BPE tokenization。
该 CPU 历史受 `max_model_len * max_num_seqs_in_gpu` 限制，并随 chain 一起回收。

### Chain CPU offload

设置 `enable_prefix_caching=true`、`enable_prefix_cache_offload=true`、
`prefix_cache_mode="chain"`（或 `auto`），并显式设置
`prefix_cache_host_size_gb`。沿用现有 offload 参数，适用于 StreamingLLM、
SnapKV、H2O、PyramidKV、R-KV、SkipKV 各自支持的模型路径，TP 为 1 或 2。
支持连续的显式 KV 和 MLA latent/RoPE 存储；包含 recurrent/linear layer 的
模型在初始化时明确拒绝。需要 pinned host memory、device stream 和已安装的
SGL cache-transfer 接口。

每轮正常结束后，异步将全部保留缓存及方法状态复制到 CPU，同时保留 GPU
副本供快速续轮。GPU 容量不足时，已有完整 CPU 快照的 idle chain 可以释放
GPU slot 和 row，保留 chain ID。仅有 CPU 副本的 chain 会先完整恢复，再处理
新增 suffix。下一轮写入前会等待上一轮拷贝完成，并使旧 CPU 快照失效；本轮
结束后重新复制，包括旧 token 和累计 score 的变化。

Host size 限制**每个 rank** 的快照 tensor 字节数。CPU 容量不足时按 LRU
回收 CPU 副本：仍有 GPU 副本的 chain 保留 ID，仅有 CPU 副本的被淘汰 chain
变为 `chain_gone`。单条快照超过预算时，在轮次结束阶段明确报错。逻辑 token
历史与 Python metadata 不计入快照 tensor 预算；offload 会按每 4 字节配置
host 容量增加 1 个 token，扩展 driver 的有界历史配额。CUDA Graph 使用恢复后
的同一存储，传输与分配均在 capture/replay 外执行。

`Config` 会把 `None`、空字符串和 `auto` 解析为 registry default。与方法默认值不一致的显式 policy 会快速失败，避免实验静默改变 scheduler 语义。任何 policy override 都应视为显式 ablation，并随 benchmark result 一起记录。

## Runtime 所有权

- 持久物理缓存和跟随 Prefix Cache 的元数据属于
  `src/sparseengine/engine/cache_manager/`。
- 当前步骤的逐层逻辑状态、打分、跨层选择以及压缩/淘汰触发属于
  `src/sparseengine/engine/sparse_methods/` 下的
  `SparseMethodRuntime`。
- `src/sparseengine/engine/sparse_controller.py` 是稳定、与方法无关的统一入口，
  不得包含方法名热路径分支。
- `src/sparseengine/layers/attention.py` 应保持通用，只调用 shared hook。
- 新的一等方法必须在 `src/sparseengine/method_registry.py` 中注册默认 prefill policy，并在 `tests/test_prefill_schedule_policy.py` 中覆盖。

完整的接口、职责划分、Prefix Cache、CUDA Graph 和扩展规则参见
[稀疏方法运行时架构](../design/sparse-method-runtime.md)。

## Query-Aware 参数

`quest` runtime 参数：

- `quest_chunk_size`：QuEST page/chunk 的 token 数量；
- `sink_keep_tokens`、`decode_keep_tokens`、`recent_keep_tokens`：QuEST 在 config 构造期间将三者相加，一次性得到 decode token budget；
- `quest_skip_layers`：在 decode 中保持前 N 个 layer 为 dense。

`quest_token_budget` 已不再是 runtime input。传入该参数会快速失败；请删除它，改为配置上述三个通用 keep-token 字段。
