# 可复现性

使用本页的稳定 checklist 复现 SparseEngine 实验。不要把本地 run ledger 放入仓库；面向仓库的结果需要证据时，应引用原始 run artifact path。

## 环境

README 包含当前安装命令。预期 baseline 为：

- Python 3.10 或更高版本。
- 从 `pyproject.toml` 中选择一个 CUDA extra 安装：`cu129` 或 `cu130`。
- 将 `pyproject.toml` 视为当前依赖契约；仓库不再提供冻结环境 lock，
  因此每次实验都要记录实际解析出的 package 版本。
- 需要预编译 JIT 模块时，可从对应 CUDA 版本的 FlashInfer wheel index 安装匹配的 `flashinfer-jit-cache`。
- 使用 `MAX_JOBS=8 pip install flash-attn --no-build-isolation` 安装 `flash-attn`。
- 在仓库根目录运行 `pip install -e ".[cu129]"` 或
  `pip install -e ".[cu130]"`。训练、benchmark 和测试依赖均包含在主安装中。
- 记录 operator binding report 中的 `selected_provider`、`selected_profile`、
  `selection_basis`、provider metadata 和 CUDA compute capability。数值正确性与
  性能结论必须引用独立 validation artifact；上游默认选择不等价于本地性能
  benchmark 证据。FP8 provider 在 warmup 期间不会下载 Hub kernel。
- RMSNorm 默认使用 `SPARSEENGINE_RMSNORM_PROVIDER=auto`，在已安装时优先选择 FlashInfer。设为 `triton` 可强制使用本地 Triton kernel；设为 `flashinfer` 可明确要求 FlashInfer。

每次报告 benchmark 时，都应记录 CUDA 版本、GPU 类型和数量、visible GPU ID、branch、commit 以及相关未提交改动。

## 模型与 Checkpoint

Base model 与 DeltaKV compressor checkpoint 必须匹配。公开 compressor checkpoint 列在 README 的 [DeltaKV checkpoint 下载](README.md#deltakv-checkpoint)一节。

将下载后的本地目录作为 `deltakv_checkpoint_path` 传入。当前 loader 读取本地 `model.safetensors` 文件；不要假设所有位置都可以直接传入 Hugging Face repo ID。

## 数据路径

LongBench 和 MathBench 从环境变量读取数据根目录：

- `SPARSEENGINE_OUTPUT_DIR`：benchmark prediction 和 log 的 output root。
- `SPARSEENGINE_DATA_DIR`：通用 benchmark dataset root。
- `SPARSEENGINE_LONGBENCH_DATA_DIR`：包含 `data/*.jsonl` 的 LongBench root。
- `SCBENCH_LOCAL_DATA_DIR`：standard SCBench 文件的可选 local root。
- `SCBENCH_PREPROCESSED_ROOT`：包含 SCBench 预处理 `<task>.parquet` 文件的 root。

Benchmark 入口不假设 host-specific dataset path。缺少必需 data root 或文件时，命令应快速失败并打印必须设置的环境变量或 CLI flag。

如果命令使用 `<DATA_ROOT>`、`<MODEL_ROOT>` 或 `<OUTPUT_ROOT>` 等本地 placeholder，请按目标机器改写，并在 run record 中记录最终路径。

## 参数规则

使用规范 public 参数名：

- `sparse_method`
- `deltakv_checkpoint_path`
- `decode_keep_tokens`
- `sink_keep_tokens`
- `recent_keep_tokens`
- `full_attention_layers`
- `engine_prefill_chunk_size`

命令、manifest、`LLM(...)` 与内部配置都应原样使用上述名称。不要再使用 `engine_prefill_chunk_size`、`sparse_method`、`model_cls`、`compressor_path`、`deltakv_checkpoint_path`、`num_top_tokens` 或 `seq_chunk_size` 等旧 key。规范 contract 参见[运行时参数语义](../configuration/runtime-parameter-semantics.md)。

SparseEngine 要求显式 integer keep budget；ratio 必须在启动前换算为 token count。

## Smoke Check

运行长 benchmark 前先执行小规模命令：

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 1 \
  --methods vanilla \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.8,"engine_prefill_chunk_size":512}'
```

基于 compressor 的 DeltaKV SparseEngine smoke test：

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"deltakv_checkpoint_path":"<LOCAL_COMPRESSOR_CHECKPOINT>","deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_dim":256,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"enable_full_layer_kivi_quant":true,"deltakv_full_pool_reserve_ratio":0.2}'
```

确认 loader log 显示 compressor 权重已加载。只有明确设置 `allow_missing_deltakv_path` 的构造测试才能省略 checkpoint。

## Artifact 要求

对于报告的结果，保存或记录：

- 准确命令和 working directory。
- Runtime config 和规范 sparse 参数。
- 模型、tokenizer、checkpoint、精度、backend 和量化设置。
- Dataset path、split、sample count、filtering/truncation 和 seed。
- benchmark 支持时，保存 raw output、parsed output、per-sample record、aggregate metric 和 run info。
- Log path 和 result file path。
- 失败或结论不确定的 run 应保存 failure status 和关键 error line。

没有 source log 或 result artifact 时，不要报告 metric。
