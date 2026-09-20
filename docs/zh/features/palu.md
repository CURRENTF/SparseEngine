# Palu 低秩 KV 缓存

使用 `sparse_method="palu"` 和离线生成的因子目录。Palu 保留所有 token，
把 K/V 存为低秩表示。Decode 在 attention kernel 内分块重建 K，V 的重建
矩阵融合进输出投影，不生成完整的 dense 历史缓存。

## 准备因子

准备具有代表性的校准语料：JSONL 每行包含非空的 `text` 字段。
评测数据应与校准语料分开。在推理环境中，从仓库根目录运行：

```bash
PYTHONPATH=src:. python scripts/compression/prepare_palu.py \
  --model "<MODEL_PATH>" --output "<PALU_FACTOR_DIR>" \
  --group-size 1 --rank-ratio 0.75 \
  --calibration-jsonl "<CALIBRATION_JSONL>" \
  --calibration-samples 32 --calibration-seqlen 1024
```

工具执行基于激活统计 whitening 的分组 SVD，输出 `palu.json` 和
`palu.safetensors`，记录源投影权重与校准数据的哈希、实际 token 数和秩。
运行时仍需原始 checkpoint；架构、K/V 权重或权重 dtype 不匹配会报错。

`--group-size` 表示一组包含多少个 KV head，必须整除 KV head 数。
保留秩为 `floor(group_size * head_dim * rank_ratio / 16) * 16`，最低 16，
且不得超过 `min(group_size * head_dim, 256)`。也可用 `--ranks-json` 指定
每层的 `[K_rank, V_rank]` 列表，覆盖统一比例；同一层各组使用相同秩。

默认要求校准语料。只有显式传入 `--uncalibrated` 才执行普通 SVD 消融，
这种方式即使压缩比例不大也可能严重损坏生成质量。当前不包含自动 Fisher
秩分配和 latent 量化。压缩比例和校准样本数需要按任务验证。

## 推理

```python
from sparseengine import LLM, SamplingParams

llm = LLM(
    "<MODEL_PATH>",
    sparse_method="palu",
    palu_checkpoint_path="<PALU_FACTOR_DIR>",
    tensor_parallel_size=1,
    max_model_len=8192,
    decode_graph=True,
)
outputs = llm.generate(["解释天空为什么是蓝色的。"],
                       SamplingParams(temperature=0, max_tokens=64))
llm.exit()
```

支持未量化的 Llama/Qwen3、FP16/BF16、head_dim 为 64/128/256、TP=EP=DP=1，
以及分块／批量 prefill、eager／CUDA Graph decode。需要 FlashInfer 和
Triton。Prefix cache/offload、稀疏 prefill 组合、模型权重量化和
`tiny_random` 会明确报错。

缓存更小不等于推理更快：K 重建增加计算，V rank 大于 head_dim 时还会
扩大输出投影。使用[效率测试入口](../benchmarking/efficiency.md)，通过
`--hyper-params` 传入 `palu_checkpoint_path`，在一致设置下比较质量、
请求延迟、端到端吞吐和连续 decode 吞吐。
