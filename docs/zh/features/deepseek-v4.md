# DeepSeek V4 Flash 原生推理

[English](../../en/features/deepseek-v4.md)

实验性支持 CUDA SM90 上的原始 DeepSeek-V4-Flash-0731 checkpoint。
使用 `sparse_method="deepseek_v4"` 选择模型原生注意力和缓存格式。
Checkpoint 的块级 FP8 投影、MXFP4 routed experts 以及 BF16/FP32 参数
按原始格式加载。MTP 权重会跳过，暂不支持投机解码。

## 环境

先按仓库 README 安装基础 CUDA 环境，再在已激活的环境中安装固定版本的
Hadamard 依赖：

```bash
pip install --no-build-isolation -e ".[deepseek-v4]"
```

原生路径要求 FlashInfer 0.6 系列且不低于 0.6.18.post1、SGL Kernel 0.4.5，
并需要 CUDA 编译器构建上游 JIT 算子。AGRS 不需要可选的 DeepEP 包，也不需要
安装 vLLM 运行时。

## 运行

Attention TP 和专家内部 TP 都必须为 1，因此 `expert_parallel_size`
必须等于 `data_parallel_size`。示例使用两个 attention 副本和 AGRS。
Sequence 和 token 预算按每个副本计算。

使用 checkpoint 的 `encoding/encoding_dsv4.py` 编码聊天消息，传入 token IDs
时不要再次添加 tokenizer special tokens。

```python
from pathlib import Path
import sys

from sparseengine import LLM, SamplingParams

model_dir = Path("<MODEL_PATH>")
sys.path.insert(0, str(model_dir / "encoding"))
from encoding_dsv4 import encode_messages

llm = LLM(
    str(model_dir),
    sparse_method="deepseek_v4",
    tensor_parallel_size=1,
    data_parallel_size=2,
    expert_parallel_size=2,
    moe_backend="agrs",
    max_model_len=4352,
    max_num_batched_tokens=512,
    engine_prefill_chunk_size=512,
    max_num_seqs_in_batch=2,
    max_decoding_seqs=2,
    decode_graph=True,
    enable_prefix_caching=True,
    prefix_cache_block_size=128,
)
try:
    prompt = encode_messages(
        [{"role": "user", "content": "What is the capital of France?"}],
        thinking_mode="chat",
    )
    tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    outputs = llm.generate([tokens], SamplingParams(temperature=0, max_tokens=64))
    print(outputs[0]["text"])
finally:
    llm.exit()
```

## 缓存与执行

分块 prefill、eager decode、decode CUDA Graph 和 radix prefix 复用共享原生
滑动窗口及压缩历史缓存。压缩比例和稀疏选择预算由 checkpoint 决定；通用参数
`sink_keep_tokens`、`recent_keep_tokens` 和 `decode_keep_tokens` 不用于调整
此方法。不能组合其他稀疏方法或独立 sparse prefill override。

Prefix 复用保留滑动窗口、压缩 carry 和压缩 page。可复用 block 常驻 GPU，
计入缓存容量。`prefix_cache_max_blocks` 限制保留的快照数量；快照或 page
无法保留时，缓存复用按可用容量进行。暂不支持 chain prefix cache 或 prefix
offload。设置 `decode_graph=False` 可运行 eager decode。
