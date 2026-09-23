# GLM-4.7-Flash 运行时支持设计

本文说明 `glm4_moe_lite` 的稳定运行时契约、组件所有权和支持边界。模型与
稀疏方法的汇总矩阵见[支持的模型](../features/supported-models.md)。

## 设计边界

GLM-4.7-Flash 使用 BF16 latent MLA。每层、每个 token 的持久化 attention
cache 由 512 维 latent 和 64 维 RoPE key 组成，不持久化展开后的多头 K/V。

实现遵循以下所有权边界：

- 模型层负责 GLM 投影、部分 RoPE、K/V absorption、Dense/MoE topology、
  biased-sigmoid routing 和 checkpoint 语义。
- CacheManager 负责 slot、请求生命周期、prefix 生命周期和 logical view。
- Storage strategy 负责显式 K/V 或 MLA latent 的物理张量、写入和显存核算。
- MLA operator 通过 `OpSpec -> OpResolver -> Provider -> kernel` 在初始化时
  绑定；模型不直接选择 kernel。
- 稀疏方法继续通过 cache-manager-first 接口工作，不在通用
  `attention.py` 中增加 GLM 方法分支。

Attention compute view 使用公共 metadata 与 tagged payload。显式 K/V 和
MLA latent payload 是不同类型；消费者收到错误 payload 时必须 fail fast，
不得通过互斥 optional tensor 或 metadata 字典绕过类型契约。

## Attention 数据路径

Prefill 按以下顺序执行：

1. 模型产生当前 chunk 的 latent 和 RoPE key。
2. Storage 将有效 token 写入持久化 latent cache，并跳过 padding slot。
3. CacheManager 按 active slot gather 完整可见历史。
4. MLA layer 临时展开 prefill 所需的 K/V，并调用共享 prefill attention。
5. 临时 workset 在该次调用结束后释放。

因此，多 chunk prefill 的后续 chunk 必须能看到此前所有可见 token，而不是只
看到当前 chunk。

Decode 使用 absorbed query 直接读取 latent cache。Provider 拥有 Triton
workspace、调度和数值 kernel；输出经过 value projection 重建到模型 hidden
维度。静态 batch 中的 padding row 不得读写 `slot=-1`。

## Cache 与前缀复用

MLA storage 的显存容量按 `512 + 64` 个 BF16 value/token/layer 核算。所有
allocation、free、reuse、eviction、slot copy 和 prefix replay 都必须同时处理
latent 与 RoPE cache，且显存统计必须覆盖两个物理张量。

Prefix Cache 的模式由稀疏方法决定：

- vanilla 和 OmniKV 使用 radix prefix cache。
- StreamingLLM、SnapKV、H2O 和 R-KV 使用 chain prefix cache。

QuEST 使用独立的 latent-aware page contract：每个物理 page、每层保存
融合 `[latent, RoPE]` 576 维 key 的逐通道 min/max；decode 使用
`[absorbed NoPE query, RoPE query]` 在相同坐标中计算 sign-split upper bound，
并在每个 TP rank 的本地 query head 间取均值后共享一套 page selection。
Attention compute view 仍携带 `MlaLatentPayload`，不会持久化或伪装成展开后的
显式 K/V。该路径当前不支持 Prefix Cache 或 Prefix offload。

Prefix hit 只有在请求实际复用了 token、cache 状态和方法特定 metadata 时才算
成功；仅完成请求不能证明 prefix 路径生效。

## MoE 与并行布局

模型第 0 层为 Dense，后续层为 routed MoE，并包含 shared expert。GLM 复用
Qwen3-MoE 的 packed-expert 物理执行，但保留自己的 router、模型 topology 和
checkpoint loader。

已支持的 `(TP, EP)` 布局为：

| TP | EP | Attention | MoE | World size |
| ---: | ---: | --- | --- | ---: |
| 1 | 1 | 单 rank | 单 rank | 1 |
| 2 | 1 | TP=2 | MoE TP=2 | 2 |
| 4 | 1 | TP=4 | MoE TP=4 | 4 |
| 1 | 2 | 每个 EP rank 复制 | EP=2 | 2 |
| 1 | 4 | 每个 EP rank 复制 | EP=4 | 4 |
| 2 | 2 | TP=2 | EP=2，MoE TP=1 | 2 |
| 4 | 2 | TP=4 | EP=2，MoE TP=2 | 4 |
| 4 | 4 | TP=4 | EP=4，MoE TP=1 | 4 |

联合 TP/EP 使用 outer-TP topology：attention TP 为 `T`，MoE EP 为 `E`，
MoE TP 为 `T/E`，world size 为 `T`。该布局要求 `DP=1`、`T % E == 0`，
专家数量能被 `E` 整除，MoE intermediate dimension 能被 `T/E` 整除。

## Operator 与平台边界

当前 MLA provider 支持 NVIDIA H100 80GB HBM3、BF16、SM90，以及 TP 1、2、4。
Provider 在模型初始化时解析并绑定；kernel 执行失败必须直接暴露，不能在
forward 中静默切换到 Torch 或其他 backend。

Vendor kernel 固定来源为 LightLLM commit
`65c174ee95ac6a6fd36b18b63d0b33d97e76b770`。本地 vendor 目录保留
Apache-2.0 license、来源映射和修改说明。模型和 CacheManager 不直接依赖
LightLLM Python runtime。

## Serving 边界

Chat Completions 与 Responses API 都使用 Transformers response parser。本地
只提供 GLM 缺失的声明式 response template。Terminal EOS、stop boundary、
流式增量和 raw parser text 的映射由通用 dispatcher/detokenizer 处理，不在
GLM parser 中加入模型特判。

## 不支持的组合

以下组合必须在配置或模型构造前明确拒绝，不能落入默认实现：

- H100 以外的 GPU、非 BF16 checkpoint 和量化权重。
- `DP>1`、上述矩阵以外的 TP/EP 布局。
- MTP/speculative decoding；loader 只精确跳过 checkpoint 中的 MTP 层。
- Prefix offload，以及 QuEST 与 Prefix Cache 的组合。
- PyramidKV、SkipKV、DeltaKV，以及多个稀疏方法叠加。
- 128K/202K 长上下文容量或吞吐支持声明。

## 验证门禁

支持声明至少需要覆盖以下可复现门禁：

- Kernel：BF16 数值 oracle、ragged/non-contiguous slot、边界长度、padding 和
  workspace 容量检查，入口见 `tests/test_mla_kernels.py`。
- Operator/layer：resolver 拒绝原因、初始化时绑定、prefill 完整历史和 decode
  数值契约，入口见 `tests/test_mla_attention_operator.py` 与
  `tests/test_mla_attention_layer.py`。
- Storage/lifecycle：allocation、free、reuse、copy、eviction、prefix replay 和
  显存核算，入口见 `tests/test_attention_cache_storage.py` 与
  `tests/test_glm_mla_prefix_cache.py`。
- Model/MoE：projection、RoPE、router、packed experts、loader allowlist 和
  tiny 多步 decode，入口见 `tests/test_glm4_moe_lite.py`。
- CUDA Graph：真实 capture/replay counter、零 eager fallback、全词表 logits 和
  方法特定状态证据，入口见 `tests/test_glm_cuda_graph.py` 与
  `tests/test_glm_mla_sparse_methods.py`。
- Serving：非流式、SSE、reasoning、tool call、EOS 和 stop boundary，入口见
  `tests/test_openai_api_server.py`。

验证产物应分别保存 resolved config、命令、Git commit、环境与
checkpoint 信息、raw/parsed output、逐样本状态和聚合结果。公开文档只维护稳定
契约、支持边界与自动化门禁，不记录单次运行的通过数量、设备占用或本地路径。
