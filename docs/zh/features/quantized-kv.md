# KV cache 量化压缩

`kivi`、`turboquant`、`fp8_kv` 保留所有 token，只压缩 KV 表示，不量化模型权重。

独立 `kivi` 和 `turboquant` 方法目前均存在已知的 decode 效率问题：功能和
CUDA Graph 验证通过，不代表延迟已达到可用水平。请将两者视为实验性实现；
用于延迟敏感场景前，务必实测目标工作负载。KIVI 的寄存器溢出修复尚未解决
整体效率问题；仅启用 CUDA Graph 也未解决 TurboQuant 的性能问题。

```python
from sparseengine import LLM

llm = LLM(
    model_path,
    sparse_method="fp8_kv",  # 也可选 "kivi" 或 "turboquant"
    decode_graph=True,  # False 以 eager 执行同一量化 decode 计算路径
    enable_prefix_caching=False,
    tensor_parallel_size=1,
    max_model_len=4096,
    max_num_seqs_in_batch=4,
    kv_quant_page_size=32,
)
```

| 方法 | 参数 | 表示 |
| --- | --- | --- |
| `kivi` | `kivi_bits=2` 或 `4`（默认） | K 按通道、V 按 token 内通道组做非对称整数量化 |
| `turboquant` | `turboquant_bits=2/3/4`（默认 4）；`turboquant_seed=0` | 固定种子正交旋转及非均匀标量量化 |
| `fp8_kv` | 无位宽参数 | E4M3，K/V 分别使用逐 token、逐 KV head 动态 scale |

支持 CUDA、FP16/BF16 的 Llama/Qwen2/Qwen3/Qwen3-MoE、head dimension 64/128/256，
以及模型合法的 TP head 分片。Qwen3-MoE 也支持 EP 和现有 outer-TP/EP 布局。
量化只处理本 rank 的 KV heads，不新增通信；模型已有的 TP/EP 通信不变。
独立推理副本各自维护 KV cache。本 KV 量化路径支持的模型要求
`data_parallel_size=1`；GLM DP attention 使用单独的 MLA latent cache。
FP8 需要 GPU 原生 FP8 支持。页大小必须是 16～128 的 2 的幂，
且能整除 head dimension。支持 decode CUDA Graph；prefix cache/offload 和 sparse prefill
组合会明确报错。[Palu](palu.md) 使用独立的低秩缓存路径。

整页量化，未满的一页保留模型精度；prefill 在有界工作区内恢复历史 KV，decode
直接读取压缩页。若启动提示工作区内存不足，可减小 prefill batch 或最大上下文长度。

这些是服务引擎适配版，不宣称与官方完全一致：KIVI 只保留未满页，不额外配置独立
residual window；TurboQuant 使用 MSE 风格高斯码本，不包含 QJL 残差修正；FP8
动态计算 scale，不需要校准文件。scale、整数打包对齐、未满页和工作区会降低实际
压缩收益。FP8 存储不等于 FP8 Tensor Core attention，也不保证端到端加速；
尤其在使用更低位宽前，应验证目标任务精度。
