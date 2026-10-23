# DeltaKV

DeltaKV compresses the KV cache for long-context inference. This repo contains
only the native SparseEngine inference path and native benchmark integration.

## Inference

Set `sparse_method` to one of:

- `"deltakv"`
- `"residualkv"` is an alias of `"deltakv"` with identical defaults.
- Legacy aliases such as `"deltakv-less-memory"` and
  `"deltakv-less-memory-cudagraph"` still normalize to `"deltakv"` for older
  configs.

With the default `use_compression=true`, also pass
`deltakv_checkpoint_path="/path/to/local/trained_compressor_dir_or_file"`.
Compressor mode requires trained weights for inference.

To quantize residuals without a learned compressor, use:

```json
{
  "sparse_method": "residualkv",
  "use_compression": false,
  "deltakv_latent_quant_bits": 4,
  "deltakv_latent_quant_group_size": 32,
  "deltakv_checkpoint_path": null
}
```

This retains center selection and quantizes the full-dimensional residual
`KV - mean(reference KV)` before K normalization and RoPE. Reconstruction adds
the dequantized residual to the reference mean; no compressor is created or
loaded. `deltakv_latent_dim` is unused in this mode. Passing a checkpoint is an
error. The existing full-layer selection and retention budgets still apply.
Setting residual bits to `0` retains floating-point residuals for diagnostics.
The full-dimensional payload generally uses more cache space than a quantized
compressor latent; int2 residuals are unsupported.

Compressor-free mode also supports the TP=1 text path for Qwen3.5/3.6 MoE.
`full_attention_layers` must select actual KV attention layers, not GDN layers.
Multimodal position reconstruction is unsupported.

DeltaKV supports exclusive chain caching with `enable_prefix_caching=true` and
`prefix_cache_mode="chain"` (`auto` also selects chain). Reuse the returned
`chain_id` for an append-only continuation. Raw centers, residuals, full-layer
KV and model recurrent state remain owned by that chain. Both raw and KIVI full
layers use the same chain lifecycle. Radix sharing and chain CPU offload are
unsupported. A resumed turn reads reconstructed historical KV; lossy storage
means it need not match recomputing the entire conversation from scratch.

DeltaKV knobs you may need:

- `deltakv_checkpoint_path`: local path to trained compressor weights.
- `use_compression`: use trained compressors (`true`) or direct KV residuals (`false`).
- `deltakv_latent_dim`: latent dimension when using a compressor.
- `deltakv_center_ratio`, `cluster_metric`: reference selection and clustering behavior.
- `deltakv_neighbor_count`: number of selected center/reference tokens used for reconstruction.
- `deltakv_latent_quant_bits`: `4` packs DeltaKV-style cached state as int4 where supported.

## Slim Runtime

The SparseEngine DeltaKV runtime uses the cache-manager implementation under
`src/sparseengine/engine/cache_manager/`. It keeps full layers according to
`full_attention_layers`, stores residual state for sparse layers, and
uses graph-stable metadata for decode.

Raw BF16/FP16 full layers support floating-point or int4 sparse residuals.
KIVI int4 full layers require int4 sparse residuals. Both compressor and direct
residual modes use these storage combinations.

Quick throughput smoke:

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

## Compressor Training

Compressor training is maintained in
[CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV). Use that repository to
prepare training data, train compressor checkpoints, and run training
ablations. SparseEngine consumes compatible checkpoints for inference and
benchmarking.

## Evaluate on LongBench

`benchmark/long_bench/pred.py` runs LongBench prediction and writes JSONL
outputs under a local output directory.

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

Notes:

- `--hyper_param` accepts either a JSON string or a path to a JSON file.
- `full_attention_layers` is passed as a comma-separated string of layer indices.
- Keep budgets are explicit token counts in the native runtime.

## Checkpoints

- Public compressor checkpoints are listed in [Getting Started](../getting_started/README.md#deltakv-checkpoints).
- `deltakv_checkpoint_path` can point to a local directory or a single checkpoint file.
- The loader scans `*.safetensors` first, then `*.bin` and `*.pt`.
- Split-KV checkpoints (`k_compress_*` / `v_compress_*`) are not supported by the SparseEngine loader.
