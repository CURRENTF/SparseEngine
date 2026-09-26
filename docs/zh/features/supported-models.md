# 支持的模型

本页汇总 SparseEngine 支持的模型、精度、并行方式和稀疏方法组合。

`精度`指 checkpoint 的权重格式。`TP`、`DP` 和 `EP` 分别表示 tensor parallelism、data parallelism 和 expert parallelism。勾号表示支持该模式；数字限制表示对应的并行规模必须使用该值。

## 模型与并行支持

| 模型 | `model_type` | 精度 | TP | DP | EP |
| --- | --- | --- | :---: | :---: | :---: |
| Qwen2.5 | `qwen2` | BF16 / FP16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3 Dense | `qwen3` | BF16 / FP16 / 块级 FP8 | ✅（FP8：1/2/4/8） | 仅支持 1 | 仅支持 1 |
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / 块级 FP8 | ✅ | ✅ | ✅ |
| Qwen3.5 / 3.6 / 3.8 | `qwen3_5` | BF16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3.6 MoE | `qwen3_5_moe` | BF16 / 块级 FP8 | ✅ | 仅支持 1 | ✅ |
| GLM-4.7-Flash | `glm4_moe_lite` | BF16 / 实验性逐张量 FP8 | ✅ | ✅⁵ | ✅ |
| Gemma 4 Dense / MoE | `gemma4` | BF16 / FP16 | ✅ | 仅支持 1 | ✅（仅 MoE） |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| [DeepSeek V4 Flash-0731](deepseek-v4.md)（实验性） | `deepseek_v4` | 块级 FP8 / MXFP4 / BF16 / FP32 | 仅支持 1 | DP=EP | EP=DP |
| MiniMax M2.7 | `minimax_m2` | 块级 FP8，非量化权重使用 BF16 | ✅ | ✅ | ✅ |

TP 规模限制为 1 到 8，并且 checkpoint 维度（包括 attention head 数和 vocabulary
大小）必须能被所选 TP 规模整除。

`tensor_parallel_size` 和 `data_parallel_size` 描述 attention 拓扑，总进程数为
`world_size = DP * TP`。`expert_parallel_size` 描述同一个 world 内的 routed expert
分布，专家内部 TP 为 `world_size / EP`，因此 EP 必须整除 world size，且不会额外
增加进程。例如 `TP=4, DP=1, EP=2` 使用四个进程，attention TP=4、专家内部 TP=2。

GLM、Qwen3MoE 和 MiniMax 支持 attention TP×DP，要求 `moe_backend="agrs"`
且 `EP=world_size`（专家内部 TP=1）。例如 `TP=2, DP=2, EP=4` 使用四张卡；
更大的合法拓扑也可使用，具体取决于模型维度和硬件约束。Dense MLP 和 shared expert
在每个 DP 副本内使用 attention TP。每个副本拥有自己的调度器和 prefix cache，
支持 decode CUDA Graph；原有稀疏方法和 prefix-cache 限制继续适用。

AG/RS 是默认 DP 通信方式。`moe_backend="deepepv1"` 仍要求 attention TP=1
以及受支持的 NVLink 硬件，不支持的组合会在启动阶段报错。DP=1 的默认后端仍为 `all-reduce`。

块级 FP8 要求使用 E4M3 权重、动态激活量化以及 `128 x 128` 的权重块大小。
Llama、Qwen2 和 Qwen3 Dense FP8 checkpoint 还要求每个 TP-local dense
projection 维度按 128 对齐，且非量化参数保持 BF16。

GLM 逐张量 FP8 加载支持 E4M3 权重以及每个量化投影对应的 BF16 标量
`weight_scale`，例如 `marksverdhei/GLM-4.7-Flash-FP8` 使用的格式。
它要求动态激活量化，并需要设备提供兼容的原生 FP8 provider。
权重精度不会改变 MLA KV cache 的 dtype。该路径仍为实验性支持，不代表比 BF16
更快；部署前应使用匹配工作负载进行对比。

DeepSeek V4 仅使用原生 `deepseek_v4` 方法，不适用下方通用稀疏方法矩阵，具体限制见[运行要求](deepseek-v4.md)。

## 稀疏方法支持

| 模型 | Vanilla | StreamingLLM | SnapKV | H2O | PyramidKV | OmniKV | QuEST | RetroInfer | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | 实验性⁷ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| Qwen3 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | 实验性⁷ | ✅ | — | 需要压缩器² |
| Qwen3MoE | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | 实验性⁷ | ✅ | — | — |
| Qwen3.5 / 3.6 / 3.8 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | — | ✅ | — | 匹配的 checkpoint³ |
| Qwen3.6 MoE | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | — | ✅ | — | — |
| GLM-4.7-Flash | ✅ | ✅ | ✅ | 实验性⁴ | 实验性⁸ | ✅ | ✅⁵ | — | ✅ | — | — |
| Gemma 4 Dense / MoE | ✅ | ✅⁶ | — | — | — | ✅ | — | — | — | — | — |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | 实验性⁷ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | 实验性⁷ | ✅ | — | — |

¹ SkipKV 仅支持已发布 steering vector 的模型：
`DeepSeek-R1-Distill-Qwen-7B`、`DeepSeek-R1-Distill-Qwen-14B` 和
`DeepSeek-R1-Distill-Llama-8B`。

² DeltaKV 需要针对 base model 训练的 compressor checkpoint。

³ Qwen3.5、Qwen3.6 和 Qwen3.8 需要 mixed-attention runtime 可接受的匹配 DeltaKV checkpoint。

⁴ H2O 的 tensor-parallel 执行可能产生与 TP=1 不同的稀疏选择。各模型原有的
TP、EP、DP 限制仍然适用。

⁵ GLM QuEST 支持 radix Prefix Cache（含 CPU offload）与 decode CUDA Graph。
打分使用 TP rank 本地 heads，不保证与 TP=1 等价；CPU offload 支持 attention
TP=1 或 TP=2。

⁶ 带共享 KV 层的 Gemma 4 checkpoint 不支持逐层 StreamingLLM eviction；
Vanilla 和 OmniKV 仍受支持。

⁷ RetroInfer 仅支持 GPU、统一的 FP16/BF16 显式 KV、attention TP=1、eager decode，
不支持 prefix cache 或 async scheduling。Llama 3.1 已有
完整 LongBench 质量结果；
其他模型条目表示代码兼容，尚未完成各自的模型验证。这些结果不构成服务吞吐证据。

⁸ GLM PyramidKV MLA 已有 LongBench v1/v2 质量结果，
条件为 FP8 权重、BF16 KV、attention TP=1；CUDA Graph 与 prefix cache 均关闭。
该结果不覆盖其他拓扑。

Palu（Llama/Qwen3）、KVzip（Qwen2/Qwen3/Llama）和 KV 量化方法
（`kivi`、`turboquant`、`fp8_kv`：Llama/Qwen2/Qwen3/Qwen3MoE）另有运行要求，
详见 [Palu](palu.md)、[KVzip](sparse-methods.md#kvzip) 与
[KV cache 量化](quantized-kv.md)。

## 原生多模态支持

设置 `enable_multimodal=True` 后，可通过 OpenAI 兼容的 Chat 和 Responses API
传入受支持的图片和视频。不受支持的媒体会返回错误。

| 模型家族 | 图片 | 视频 | 音频 |
| --- | :---: | :---: | :---: |
| Qwen3.5 / 3.6 / 3.8 Dense 与 Qwen3.6 MoE | ✅ | ✅ | — |
| Gemma 4 Dense 与 MoE | ✅ | ✅ | — |

`—` 表示当前不支持该组合。
