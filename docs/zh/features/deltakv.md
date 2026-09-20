# DeltaKV

DeltaKV 为长上下文推理压缩 KV cache。本仓库只包含原生 SparseEngine
inference path 和原生 benchmark 集成。

## 推理

将 `sparse_method` 设置为以下值之一：

- `"deltakv"`
- `"deltakv-less-memory"` 和 `"deltakv-less-memory-cudagraph"` 等 legacy alias
  仍会被规范为 `"deltakv"`，以兼容旧配置。

进行 DeltaKV 推理时，还需传入
`deltakv_checkpoint_path="/path/to/local/trained_compressor_dir_or_file"`。
当前 SparseEngine DeltaKV runtime 依赖 compressor；缺少 checkpoint 的情况仅用于构造测试，不能用于可报告的 benchmark run。

可能需要使用的 DeltaKV 参数：

- `deltakv_checkpoint_path`：已训练 compressor 权重的本地路径。
- `deltakv_latent_dim`：压缩 KV 的 latent dimension。
- `deltakv_center_ratio`、`cluster_metric`：reference selection 和 clustering 行为。
- `deltakv_neighbor_count`：重建时使用的 center/reference token 数量。
- `deltakv_latent_quant_bits`：在支持的路径上，设为 `4` 会以 int4 打包 DeltaKV cache state。

## 精简 Runtime

SparseEngine DeltaKV runtime 使用 `src/sparseengine/engine/cache_manager/` 下的 cache-manager 实现。它按照 `full_attention_layers` 保留 full layer，为 sparse layer 存储 compressor residual state，并在 decode 中使用 graph-stable metadata。

支持两类存储路径：BF16/FP16 full layer 加 BF16/FP16 compressor latent residual，或者 KIVI int4 full layer 加 int4 compressor residual。

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
