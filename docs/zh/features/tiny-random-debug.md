# Tiny Random 调试模式

Tiny random 模式根据源 checkpoint 的 `config.json` 构造更小的模型，并初始化确定性的伪造权重，不读取任何 checkpoint tensor 文件。它用于模型开发、TP、prefill/decode 和数值对齐调试，其输出不代表模型质量。

实现位于显式启用的 `sparseengine.debug.tiny_random` 模块。普通推理不会导入该模块。核心集成点仅有模型配置和启动时权重初始化；scheduler、attention、cache 和模型 forward 热路径均不改变。

## 配置

使用以下环境变量启用该模式：

```bash
export SPARSEENGINE_TINY_RANDOM=1
export SPARSEENGINE_TINY_RANDOM_CONFIG="$PWD/configs/debug/qwen3_tiny_random.json"
export SPARSEENGINE_TINY_RANDOM_SEED=17
```

JSON override 文件仅接受：

- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `vocab_size`
- `max_position_embeddings`

`num_hidden_layers`、`hidden_size` 和 `intermediate_size` 是必填项，且不能扩大源模型。无效的 head dimension，以及与 TP 不兼容的 attention head、KV head 或 vocabulary size，会在模型构造前失败。仍需提供源模型目录以读取配置和 tokenizer metadata，但不会打开其中的 `.safetensors` 文件。

## Qwen3-8B 双 GPU 检查

使用项目根目录的 uv 环境：

```bash
CUDA_VISIBLE_DEVICES=5,6 \
PYTHONPATH="$PWD:$PWD/src" \
.venv/bin/python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <MODEL_ROOT>/Qwen3-8B \
  --lengths 128 \
  --batch_sizes 1 \
  --methods vanilla \
  --output_len 2 \
  --hyper_params '{"tensor_parallel_size":2,"max_model_len":2048,"engine_prefill_chunk_size":256,"max_num_batched_tokens":512,"max_num_seqs_in_batch":1,"max_decoding_seqs":1,"gpu_memory_utilization":0.02,"mlp_chunk_size":256}'
```

该命令只验证原生 SparseEngine tiny-random 的 model construction、TP、prefill
和 decode 路径，不代表模型质量。

## 限制

- 不支持量化 base-model 权重。
- 暂不支持 Qwen3.5 mixed attention。
- 不支持 DeltaKV learned compressor 权重。
- 此模式不测试 checkpoint 加载或下游任务质量。
