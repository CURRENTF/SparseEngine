# DeltaKV

DeltaKV 为长上下文推理压缩 KV cache。本仓库只包含原生 SparseEngine
inference path 和原生 benchmark 集成。

## 推理

将 `sparse_method` 设置为以下值之一：

- `"deltakv"`
- `"residualkv"` 是 `"deltakv"` 的别名，默认配置相同。
- `"deltakv-less-memory"` 和 `"deltakv-less-memory-cudagraph"` 等 legacy alias
  仍会被规范为 `"deltakv"`，以兼容旧配置。

默认 `use_compression=true` 时，还需传入
`deltakv_checkpoint_path="/path/to/local/trained_compressor_dir_or_file"`。
compressor 模式的推理必须使用已训练的权重。

只量化残差、不使用学习式 compressor 时，配置为：

```json
{
  "sparse_method": "residualkv",
  "use_compression": false,
  "deltakv_latent_quant_bits": 4,
  "deltakv_latent_quant_group_size": 32,
  "deltakv_checkpoint_path": null
}
```

此模式保留中心选择，对 K normalization 和 RoPE 之前的完整维度残差
`KV - mean(reference KV)` 量化。重建时将反量化残差加回参考均值，不创建或加载
compressor。`deltakv_latent_dim` 在此模式下不生效，同时传入 checkpoint 会报错。
原有 full layer 选择和 token 保留预算仍然适用。将残差位宽设为 `0` 可保留浮点残差，
用于诊断。完整维度残差通常比量化后的 compressor latent 占用更多缓存；暂不支持 int2 残差。

无 compressor 模式也支持 Qwen3.5/3.6 MoE 的 TP=1 纯文本路径；
`full_attention_layers` 必须选择模型实际的 KV attention 层，不能选择 GDN 层。
此路径不支持多模态位置重建。

DeltaKV 支持独占 chain cache：设置 `enable_prefix_caching=true` 和
`prefix_cache_mode="chain"`（`auto` 也会选择 chain），并使用返回的 `chain_id`
续接会话。中心、残差、full-layer KV 和模型 recurrent state 随 chain 保留。
Raw 和 KIVI full layer 共用这一生命周期。暂不支持 radix 共享和 chain CPU offload。
续轮读取重建后的历史 KV；有损存储意味着输出不保证与完整历史重新 prefill 一致。

可能需要使用的 DeltaKV 参数：

- `deltakv_checkpoint_path`：已训练 compressor 权重的本地路径。
- `use_compression`：使用已训练 compressor（`true`）或直接存储 KV 残差（`false`）。
- `deltakv_latent_dim`：使用 compressor 时的 latent dimension。
- `deltakv_center_ratio`、`cluster_metric`：reference selection 和 clustering 行为。
- `deltakv_neighbor_count`：重建时使用的 center/reference token 数量。
- `deltakv_latent_quant_bits`：在支持的路径上，设为 `4` 会以 int4 打包 DeltaKV cache state。

## 精简 Runtime

SparseEngine DeltaKV runtime 使用 `src/sparseengine/engine/cache_manager/` 下的 cache-manager 实现。它按照 `full_attention_layers` 保留 full layer，为 sparse layer 存储残差，并在 decode 中使用 graph-stable metadata。

BF16/FP16 full layer 可搭配浮点或 int4 sparse residual；KIVI int4 full layer 必须搭配
int4 sparse residual。compressor 和直接残差模式均使用这些存储组合。

快速吞吐量 smoke test：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --temperature 0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"deltakv_checkpoint_path":"<COMPRESSOR_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor","deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_dim":256,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"enable_full_layer_kivi_quant":true,"deltakv_full_pool_reserve_ratio":0.2}'
```

## Compressor 训练

Compressor 训练代码由独立仓库
[CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV) 维护。请在该仓库中准备训练数据、训练 compressor checkpoint 和运行训练消融实验；SparseEngine 仅消费兼容 checkpoint，用于推理和 benchmark。

## 在 LongBench 上评估

`benchmark/long_bench/pred.py` 运行 LongBench prediction，并将 JSONL 输出写入本地 output directory。

```bash
python benchmark/long_bench/pred.py \
  --model all \
  --model_path <PATH_TO_BASE_MODEL> \
  --tokenizer_path <PATH_TO_TOKENIZER_OR_MODEL> \
  --ws 1 \
  --batch_size 1 \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "<LOCAL_PATH_TO_TRAINED_COMPRESSOR_DIR>" \
  --hyper_param '{"engine_prefill_chunk_size":16384,"decode_keep_tokens":2048,"full_attention_layers":"0,1,2,8,18","recent_keep_tokens":128,"sink_keep_tokens":8,"use_compression":true,"deltakv_center_ratio":0.1}'
```

说明：

- `--hyper_param` 接受 JSON 字符串或 JSON 文件路径。
- `full_attention_layers` 以逗号分隔的 layer index 字符串传入。
- 原生 runtime 中 keep budget 必须使用显式 token 数。

## Checkpoint

- 公开 compressor checkpoint 列在[快速开始](../getting_started/README.md#deltakv-checkpoint)中。
- `deltakv_checkpoint_path` 可以指向本地目录或单个 checkpoint 文件。
- loader 优先扫描 `*.safetensors`，随后扫描 `*.bin` 和 `*.pt`。
- SparseEngine loader 不支持 split-KV checkpoint（`k_compress_*` / `v_compress_*`）。
