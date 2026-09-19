# 快速开始

本页介绍环境安装、checkpoint 下载和最小 Sparse-Engine 使用示例。

## 使用 Conda 安装

```bash
conda create -n sparse-engine-cu130-py312 python=3.12 -y
conda activate sparse-engine-cu130-py312

python -m pip config --site set global.extra-index-url \
  "https://download.pytorch.org/whl/cu130 https://flashinfer.ai/whl"
python -m pip install "transformers==5.13.1" -e ".[cu130]"
python -m pip check

# Optional
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

## 使用 uv 安装

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ".[cu130]"

# Optional
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
```

CUDA 12.9 环境将 `cu130` 换成 `cu129`。

`einops`、`sglang-kernel==0.4.5` 以及训练、benchmark 和测试包均已是主依赖，
不再需要工作流专用 extra。SGL kernel package 固定到已经验证的 PyTorch/CUDA ABI；
其他版本会在 Provider 准备阶段明确失败，不会静默 fallback。

CUDA extra 同时要求 `flashinfer-python>=0.6.15,<0.7`。FlashInfer 或 SGL kernel
缺失、package metadata 不兼容或无法加载时，GPU engine 会在启动阶段直接失败；
请按 CUDA 版本执行 `pip install -e ".[cu129]"` 或
`pip install -e ".[cu130]"` 修复环境。

Sparse-Engine 当前支持未量化 BF16 和 block-scaled FP8 格式的
Qwen3.5/Qwen3.6/Qwen3.8 checkpoint。三者共享 `qwen3_5` 运行时架构和支持矩阵。

其 causal Conv1D 与 decode packing path 仍使用仓库自有 Triton kernel。
GDN core 在模型准备阶段只解析一次。安装受支持的
`flashinfer-python>=0.6.15,<0.7` 后，满足合同的
SM90、SM100/SM103 与 SM120/SM121 环境会绑定 FlashInfer 公开 prefill dispatcher
和仓库本地 fused Triton decode kernel；其他受支持合同绑定本地 Triton 实现。
SM100/SM103 要求 CUDA 13，所有 Blackwell path 均要求 head dim 128，且 value
head 数必须等于 key head 数或是其整数倍；adapter 在 FlashInfer 边界使用 FP32
initial/final state，同时保留仓库配置的 BF16 或 FP32 runtime state。Provider
resolution 会在执行前验证公开 dispatcher 的签名和对应架构的 kernel symbol。

`flashinfer-jit-cache` 是可选加速 package：

```bash
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130
```

Block-scaled FP8 Linear 会根据当前 CUDA device capability，从本地 operator
registry 选择实现。SM90 上匹配 BF16 block-scale 契约的算子使用 FlashInfer
实现，其他受支持的 SM90 契约使用通用 Triton；RTX PRO 6000 上任何模型只要
FP8 Linear 的 shape 与语义契约命中 profile，都会绑定同一个模型无关
dispatch plan：`M < 512` 使用 Triton，`M >= 512` 使用由 SGL
per-token-group activation quantization 与
FlashInfer 公开 `gemm_fp8_nt_groupwise` CUTLASS kernel 组成的 atomic
Provider。未命中 profile、但仍处于该上游 atomic contract 内的 SM120 shape，
默认 portfolio 会使用上游 groupwise Provider；只有上游契约不适用或可选依赖
缺失时，才绑定通用 Triton portable baseline。binding report 会记录 selection
basis、profile 判断和 route；warmup 期间不会下载 Hub kernel。当前 profile 的
测量 workload 是 Qwen3-30B TP1，但模型名只属于 provenance，不参与选择。

## DeltaKV Checkpoint

基于 compressor 的 DeltaKV run 需要本地 checkpoint 目录。传入 `deltakv_checkpoint_path` 前，请下载与 base model 匹配的 compressor。

| Base model | Compressor checkpoint |
| --- | --- |
| `Qwen/Qwen2.5-7B-Instruct-1M` | [`JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor`](https://huggingface.co/JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor) |
| `Qwen/Qwen2.5-32B-Instruct` | [`JitaiHao/Qwen2.5-32B-Instruct-Compressor`](https://huggingface.co/JitaiHao/Qwen2.5-32B-Instruct-Compressor) |
| `meta-llama/Llama-3.1-8B-Instruct` | [`JitaiHao/Llama-3.1-8B-Instruct-Compressor`](https://huggingface.co/JitaiHao/Llama-3.1-8B-Instruct-Compressor) |

```bash
export DELTAKV_CKPT_ROOT=<CHECKPOINT_ROOT>/compressor
mkdir -p "$DELTAKV_CKPT_ROOT"

huggingface-cli download JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Qwen2.5-7B-Instruct-1M-Compressor"

huggingface-cli download JitaiHao/Qwen2.5-32B-Instruct-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Qwen2.5-32B-Instruct-Compressor"

huggingface-cli download JitaiHao/Llama-3.1-8B-Instruct-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Llama-3.1-8B-Instruct-Compressor"
```

将下载后的本地目录用作 `deltakv_checkpoint_path`。除非 compressor 是针对另一个 base model 训练且 layer/head dimension 匹配，否则不要跨 base model 复用 compressor checkpoint。

## 最小用法

```python
from sparseengine import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen2.5-7B-Instruct-1M",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    engine_prefill_chunk_size=4096,
    sparse_method="omnikv",
    full_attention_layers="0,1,2,4,7,14",
    decode_keep_tokens=2096,
)

outputs = llm.generate(
    prompts=["Write a short story about sparse attention."],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0]["text"])
llm.exit()
```

## 关键参数

Sparse-Engine runtime 参数定义在 `src/sparseengine/configs/groups.py` 和 `runtime.py` 中，可原样作为 keyword argument 传给 `LLM(...)`。`sparse_method` 与 `engine_prefill_chunk_size` 是规范名称；`sparse_method`、`engine_prefill_chunk_size`、`num_top_tokens`、`model_cls` 和 `compressor_path` 等旧名称会在 runtime boundary 被拒绝。

常用参数：

- `tensor_parallel_size`：启动的 GPU rank 数。
- `gpu_memory_utilization`：分配给 KV cache 的 GPU 总显存比例。
- `max_model_len`：允许的最大 prompt 加生成 token 数。
- `engine_prefill_chunk_size`：Sparse-Engine prefill scheduling 与 memory admission 的 chunk size。
- `max_num_batched_tokens`：单 step 的 token 预算。
- `max_num_seqs_in_batch`：prefill batch 上限，同时作为默认 decode batch 上限。
- `max_decoding_seqs`：可选的 decode batch 覆盖值；显式设置后，其精确值会纳入 decode CUDA Graph 预捕获 buckets。

稀疏参数：

- `sparse_method`：方法 selector。
- `deltakv_checkpoint_path`：本地 DeltaKV compressor checkpoint 目录或文件。
- `sink_keep_tokens`：始终保留的 prefix/sink token。
- `recent_keep_tokens`：始终保留的 recent tail token。
- `decode_keep_tokens`：共享的 sparse top/important token budget。
- `full_attention_layers`：运行 full attention 的 layer index，以逗号分隔的字符串或列表表示。

## 文档导航

- [核心稀疏方法](../features/sparse-methods.md)
- [基准测试](../benchmarking/README.md)
- [DeltaKV](../features/deltakv.md)
- [故障排查](troubleshooting.md)

### Scheduler 阶段亲和性

默认启用阶段亲和性，`favor_min_decoding_seqs` 取
`ceil(max_decoding_seqs * 0.75)`。可显式设置 `0..max_decoding_seqs` 的整数覆盖默认值；
设为 `0` 时恢复严格的 prefill 优先。Python API 将它作为 LLM 参数；
OpenAI 服务 CLI 使用 `--favor-min-decoding-seqs 4`；效率 probe 使用
`--hyper-params '{"favor_min_decoding_seqs":4}'`。

Decode 阶段中，下一步可执行 decode 数达到阈值时保持 decode，否则尝试 prefill。
计数包含混合长度请求，并遵守 decode batch 上限和可写 KV 容量规则，
包含可执行的 recompute decode；不使用 resident KV 行数。
没有 prefill 时继续 decode，不等待凑批。最老 prefill 等待达到 60 秒时，
取消 decode 偏好并优先尝试最老可接纳的 prefill；容量和 replay 恢复约束仍然有效。
这是执行机会的出口，不是完成时间保证。

进入 prefill 阶段后持续推进 chunk，也接纳符合容量和入场限制的新请求，
不固定请求集合、不因 decode 数重新达到阈值而逐 chunk 切换。
没有可执行 prefill 时回到 decode；入场上限不阻止已开始的 partial prefill。
不设置 prefill 阶段最长时间。连续到达、立即完成并释放容量的短请求仍可能延后 decode。

新请求从 scheduler 入队开始计时；partial 从前一 chunk 完成、重新等待时计时；
recompute prefill 从抢占入队计时。扫描／容量不足重新入队不会重置计时，
获得 prefill 执行机会后结束本次等待，取消时清理。策略仅作用于单个 scheduler，
不实现 DP 副本协调或 mixed prefill/decode。
