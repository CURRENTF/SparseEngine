# 运行时参数

以下列出常用运行时参数，传入 `LLM(model, **kwargs)` 或 `Config`；JSON 配置和 benchmark manifest 使用相同名称。默认值为配置初始值，`None` 表示自动解析或未设置。

## 模型与执行

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model` | str | 必填 | 模型路径。 |
| `max_model_len` | int / None | `None` | 最大上下文长度；自动值受模型和运行时容量限制，显式值不能超过模型上限。 |
| `gpu_memory_utilization` | float | `0.9` | 引擎可使用的 GPU 显存比例。 |
| `tensor_parallel_size` | int | `1` | Attention 张量并行度。 |
| `decode_graph` | bool | `True` | 启用 decode CUDA Graph；需要 eager 执行或方法不支持时设为 `False`。 |
| `decode_graph_capture_sizes` | `auto` / list[int] | `auto` | 捕获的 decode batch size。自动计划在图数量预算内优先覆盖小 batch，并将其余图分布到 `max_decoding_seqs`；显式列表须包含 `max_decoding_seqs`。 |
| `decode_graph_startup_capture_limit` | int / None | `None`（32） | 启动时 decode graph 的数量上限；自动计划最多使用该数量。 |

GLM-4.7-Flash、Qwen3.5/3.6 MoE 和 Gemma 4 MoE（稠密 MLP 分支）的
DP=1 decode 默认并行执行 shared 和 routed 分支，包括已支持的 FP8 配置。
Prefill 保持串行；已有的 shared expert 融合路径继续使用融合。Qwen3 MoE 与 MiniMax M2
没有共享分支，不受该优化影响。

该优化自动启用，独立于 `async_scheduling`。支持的 decode CUDA Graph 会捕获分支依赖，
不改变请求调度，以及各模型的门控、归一化和规约顺序。

## 调度与 Token 预算

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `max_num_batched_tokens` | int / `auto` | `auto` | 每轮调度的 token 预算；启动时按模型、TP 和设备显存估算。 |
| `max_num_seqs_in_batch` | int | `32` | 单个 batch 的最大请求数。 |
| `max_decoding_seqs` | int / None | `None` | decode batch 的最大请求数；默认跟随 `max_num_seqs_in_batch`，也是 graph 计划的最大捕获尺寸。 |
| `engine_prefill_chunk_size` | int / `auto` / None | `auto` | 单请求每步的 prefill token 上限；`auto`（或 `None`）跟随最终 batch token 预算，并遵守方法约束。 |
| `long_prefill_offload_threshold` | int | `65536` | 长请求阈值，单位 token；用于长请求整段 prefill、短请求批处理策略。 |
| `mla_prefill_history_chunk_size` | int | `16384` | MLA prefill 每次处理的历史 KV token 上限；调小可减少历史工作区显存。 |
| `decode_reservation_tokens` | int | `1024` | 每次为后续 decode 预留的最大 token 窗口；须为正整数，不是总输出上限。 |

两个参数默认在启动时解析为整数，运行中保持固定。普通分块策略下，两个 `auto`
使用同一预算。只显式指定 batch token 预算时，自动 chunk 跟随该值；只显式指定
chunk 时，自动 batch 预算至少容纳该 chunk，否则启动报错。显式整数仍可分别指定。

`long_bs1full_short_batch` 的自动预算必须容纳整个 `long_prefill_offload_threshold`，
自动 chunk 不超过该阈值；两个参数都为 `auto` 时，通常都取该阈值（更大的 decode
并发上限可能要求更大的 batch 预算）。显存估算不足时会报错，不会自动改变 offload
阈值；可降低阈值或显式设置 batch token 预算。自动预算需要预留 activation 显存，
因此 `gpu_memory_utilization=1.0` 时须显式设置 batch token 预算。

自动选择是保守估算，不保证最优吞吐或避免所有 OOM。启动日志和 worker 信息记录
最终整数；可复现实验应固定并记录这两个值。引擎默认值不替代 benchmark runner
显式传入的参数。

## 异步执行

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `async_scheduling` | bool / None | `None` | 对兼容的 CUDA 配置自动启用异步调度。`False` 使用同步执行；显式设为 `True` 时，不兼容的配置会报错。 |
| `async_max_inflight` | int | `2` | 异步调度中已提交但尚未取回结果的 batch 数上限，最小为 `2`。 |

异步调度支持 DP=1、没有循环注意力的纯文本模型，以及 radix prefix cache 或关闭 prefix cache，
不支持 prefix offload。自动模式下，其他配置保留同步调度。稀疏 decode 和独立稀疏 prefill
沿用原有生命周期 hook；方法、存储和 CUDA Graph 原有的兼容性约束仍然有效。
Eager 和受支持的 decode Graph 路径都可使用异步调度。

GPU 计算与 KV 更新保持顺序。需要 CPU 已确认 token 历史的方法会在相应依赖边界等待；
异步调度不会取消必要的等待，也不会改变选择和压缩语义。
输出仍按顺序发布。遇到 EOS 或取消时，已提交的少量后续计算可能继续执行，但不会发布其输出；
相关存储在所有使用者完成后释放。容量抢占会先取回在途结果，再执行同步重算恢复。
此开关不会启用 prefill/decode 混合 batch。

## 稀疏方法与共享预算

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `sparse_method` | str | `""` | Cache/decode 方法；空字符串或 `vanilla` 表示全量注意力。可选方法见[稀疏方法](../features/sparse-methods.md)。 |
| `sink_keep_tokens` | int | `64` | 保留的开头 token 数。 |
| `recent_keep_tokens` | int | `512` | 保留的最近 token 数。 |
| `decode_keep_tokens` | int | `4096` | 稀疏选择的 token 预算，具体含义随方法而定。 |
| `full_attention_layers` | str / list[int] | `"auto"` | 完整注意力层；支持自动配置、逗号分隔字符串或层索引列表。未登记的 OmniKV / DeltaKV 模型需先校准或显式指定。 |
| `sparse_prefill_score_mode` | str / None | `None` | 评分方式：自动选择、`probability` 或 `logits`；SnapKV/PyramidKV 的 prefill 和 decode observation window 使用同一种方式。`logits` 仅适用于 SnapKV、PyramidKV、H2O，且要求 float32 分数。 |

Token 数预算须为非负整数，不接受比例。QuEST 的总选择预算为 `sink_keep_tokens + decode_keep_tokens + recent_keep_tokens`，不能直接设置 `quest_token_budget`。

<a id="prefill-sparsity"></a>

## Prefill 稀疏

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `prefill_sparse_method` | str / None | `None` | 独立选择 prefill 加速：`h2o_prefill`、`flashprefill_v2`、`omnikv_prefill`；`""` 关闭。省略时仅 H2O 默认启用 `h2o_prefill`。 |
| `omnikv_prefill_full_attention_layers` | str / list[int] / None | `"auto"` | OmniKV prefill 的完整注意力层，独立于 decode 配置。 |
| `omnikv_prefill_keep_tokens` | int | `4096` | OmniKV prefill 的历史选择预算。 |
| `omnikv_prefill_sink_keep_tokens` | int | `8` | OmniKV prefill 保留的开头 token 数。 |
| `omnikv_prefill_recent_keep_tokens` | int | `128` | OmniKV prefill 保留的最近 token 数。 |
| `flashprefill_v2_abs_threshold` | float / None | `None` | FlashPrefill V2 稀疏阈值，范围 `[0, 1]`；启用时必须显式提供模型校准值。 |

`flashprefill_v2` 仅支持显式 KV 模型；`omnikv_prefill` 不支持 radix 前缀复用。组合限制见[稀疏方法](../features/sparse-methods.md)，FlashPrefill 调参见 [FlashPrefill V2](../features/flashprefill-v2.md)。

## SnapKV 与 PyramidKV

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `observation_window_size` | int | `32` | 正整数。prefill 评分使用 prompt 末尾的 query；每次 decode 驱逐最多使用最近这么多个 query，压缩后重新累计。 |
| `snapkv_decode_eviction` | bool | `False` | 开启 SnapKV 的周期性 decode 驱逐；PyramidKV 始终开启。 |
| `decode_eviction_interval` | int | `1024` | 正整数。SnapKV/PyramidKV 每层物理 KV 长度达到该层预算加此值时驱逐；仅在边界步的 decode Graph replay 后评分。 |

## KVzip

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `kvzip_token_budget` | int | `4096` | 重建后保留的 prompt token 总数，必须为正。生成 token 继续追加，不再次淘汰。 |
| `kvzip_score_chunk_size` | int | `2048` | 每次前向重建的原文 token 数，必须为正。 |
| `kvzip_prev_postfix_size` | int | `64` | 重建输入中附带的前文 token 上限，必须非负。 |

重建临时容量和与原论文逐 head KVzip 的差异见 [KVzip 说明](../features/sparse-methods.md#kvzip)。

## H2O

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `h2o_prefill_budget` | int | `8192` | 中间 prefill chunk 压缩后的 KV 预算；不能小于 decode 预算。 |
| `h2o_decode_budget` | int | `4096` | 最终 prompt 压缩后的 KV 预算；须为正整数。 |
| `h2o_recent_ratio` | float | `0.5` | 预算中 recent token 的比例，范围 `(0, 1)`。 |
| `h2o_prefill_score_window` | int | `128` | Prefill 评分的 query 窗口；`0` 表示整个当前 chunk，probability 模式范围为 `[0, 128]`。 |
| `h2o_decode_eviction` | bool | `False` | 启用 decode 持续评分和驱逐；开启后强制使用 probability 评分。 |
| `h2o_decode_eviction_interval` | int | `128` | Decode 驱逐间隔；容量紧张时可能提前驱逐。 |
| `h2o_decode_score_fusion` | bool | `True` | 启用 decode 驱逐时复用 attention 分数，减少独立评分开销。 |

默认只在 prefill 压缩，decode 不持续驱逐。MLA 的 decode 驱逐使用近似分数，尚未与原始 H2O 完全对齐。

## DeltaKV

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `deltakv_checkpoint_path` | str / None | `None` | 兼容的 compressor checkpoint 路径；正式推理需提供。 |
| `deltakv_neighbor_count` | int | `4` | Reference neighbor 数量。 |
| `deltakv_center_ratio` | float | `0.1` | Reference center 比例。 |
| `deltakv_latent_dim` | int | `128` | Compressor 隐空间维度，需与 checkpoint 匹配。 |
| `deltakv_latent_quant_bits` | int | `4` | 隐状态量化位数：`0`、`2` 或 `4`；`0` 关闭量化。 |
| `deltakv_latent_quant_group_size` | int | `0` | 隐状态量化分组大小；默认 4-bit 配置下自动设为 `32`。 |

模型与 checkpoint 配置见 [DeltaKV](../features/deltakv.md)。

## RoPE 与上下文限制

RoPE 从模型 checkpoint 的 `rope_parameters` 读取，兼容旧字段 `rope_scaling`。

| 模型路径 | 支持范围 |
| --- | --- |
| 普通一维 MHA/GQA | `default`、`linear`、`yarn`、`llama3`；不支持 `dynamic`、`longrope`。 |
| DeltaKV | 仅 `default`、`llama3`。 |
| Qwen3.5 MRoPE、GLM-4.7-Flash MLA、Gemma4 per-layer RoPE | 使用各自的 RoPE 配置，不接受普通一维 scaling 参数。 |

YaRN 的有效上下文长度为 `original_max_position_embeddings × factor`；显式上下文上限不能超过该值。

## Benchmark adapter

文本 benchmark 共用 `benchmark/model_adapters/sparseengine.py`。它接收相同的
public 参数，构造原生 engine，并为 LongBench、MathBench、NIAH 和 RULER core
提供轻量 generation callable。SCBench 使用原生 `sparseengine` attention
type，不存在 `--backend hf` 选项。

## Decode 预留窗口

`decode_reservation_tokens` 是所有缓存方法和 prefix 模式共用的正整数参数，
默认 `1024`。它限制下一段 decode 步数，不是总输出上限，也不是稀疏驱逐间隔。
首个输出 token 来自 prefill，后续 decode 每次预留最多一个窗口或剩余输出
上限所需的容量。各方法按自己的资源单位计入页取整、驱逐峰值和压缩池需求。

Scheduler 在接纳更多 prefill 工作前为活动 decode 续租。失败时使用现有
抢占/recompute 恢复；只剩一个请求时，只要还能执行一步，就允许缩小窗口。
窗口边界不会强制轮转队列或改变评分与驱逐。较小窗口可能增加高负载下的
抢占，较大窗口可能延迟新请求。Chain 仍单独检查 suffix prefill 和 CPU
快照恢复所需容量。

## R-KV 保留策略

`rkv` 在所有 KV 层和 attention TP ranks 间使用同一组保留 token。
总保留预算仍为 `sink_keep_tokens + decode_keep_tokens + recent_keep_tokens`，
但这三个字段对 RKV 只贡献总量，不再表示分别保护的区域。RKV 不固定保留 sink，
只保护最后 `rkv_observation_tokens` 个 token；该参数也决定评分使用的最近 decode
query 数量，不包含 prompt queries。

`rkv_compression_interval` 指定 decode buffer 边界，必须不小于观察窗口。
压缩还要求物理缓存长度达到“总预算 + interval”。长 prompt 会保留到 decode
边界，不在 prefill 阶段压缩。

`rkv_kernel_size` 是重要性评分的正奇数 pooling 宽度，默认 7。
`rkv_alpha` 对应 `alpha * importance - (1-alpha) * redundancy`。
`rkv_score_chunk_mb` 限制评分分块工作区，默认 512 MiB；启动时会检查它能否
处理一个长度为 `max_model_len` 的请求，包括展开后的 K 和 query。
保留预算不限制首次 decode 压缩前的完整 prompt 长度；长上下文若未通过
工作区检查，需要按报错增加此参数。cache allocator 会单独
预留。若一个评分分块都无法容纳，需要增大该参数；评分失败不会自动退回 Full-KV。

旧近似算法的非零 `rkv_redundancy_window`、非空 `rkv_max_redundancy_tokens`、
非 0.5 的 `rkv_similarity_threshold` 和非 1 的 `rkv_recent_similar_keep` 会被拒绝。
选择语义已变化，旧 RKV 结果需要重新运行后才能比较。
