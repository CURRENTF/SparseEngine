# Quantized KV cache

`kivi`, `turboquant`, and `fp8_kv` compress KV representations without dropping
tokens. They do not quantize model weights.

The standalone `kivi` and `turboquant` methods currently have known
decode-efficiency problems:
functional and CUDA Graph validation does not imply acceptable latency.
Treat both as experimental and do not select them for latency-sensitive
workloads without measuring your workload. The KIVI register-spill repair
does not resolve its overall efficiency issue; CUDA Graph support alone does
not resolve TurboQuant's performance issue either.

```python
from sparseengine import LLM

llm = LLM(
    model_path,
    sparse_method="fp8_kv",  # alternatively "kivi" or "turboquant"
    decode_graph=True,  # False uses the same quantized decode computation eagerly
    enable_prefix_caching=False,
    tensor_parallel_size=1,
    max_model_len=4096,
    max_num_seqs_in_batch=4,
    kv_quant_page_size=32,
)
```

| Method | Controls | Representation |
| --- | --- | --- |
| `kivi` | `kivi_bits=2` or `4` (default) | Asymmetric integer quantization: per-channel K, grouped per-token V |
| `turboquant` | `turboquant_bits=2`, `3`, or `4` (default); `turboquant_seed=0` | Seeded orthogonal rotation and nonuniform scalar quantization |
| `fp8_kv` | No bit-width option | E4M3 with separate dynamic per-token/per-head K and V scales |

The implementation supports CUDA, FP16/BF16 Llama/Qwen2/Qwen3/Qwen3-MoE models,
and head dimensions 64/128/256. TP is supported with the model's legal head
sharding; Qwen3-MoE also supports EP and its existing outer-TP/EP layout.
Quantization uses rank-local KV heads and adds no communication. Model TP/EP
collectives are unchanged. Independent inference replicas keep separate KV
caches. The models supported by this KV quantization path require
`data_parallel_size=1`; GLM DP attention uses its separate MLA latent cache.
FP8 requires a GPU with native FP8
support. `kv_quant_page_size` must be a power of two from 16 to 128 and divide
the head dimension. Decode CUDA Graphs are supported. Prefix caching/offload and sparse prefill
combinations are rejected. [Palu](palu.md) uses a separate low-rank cache path.

Full pages are compressed; incomplete pages stay in the activation dtype.
Prefill uses a bounded dense history workspace, so reduce prefill batch size
or maximum context length if startup reports insufficient workspace memory.
Decode reads compressed pages directly.

These are serving adaptations, not exact official implementations. KIVI has
one incomplete-page residual rather than an independently configured residual
window. TurboQuant uses the MSE-style Gaussian-codebook recipe, without QJL
residual correction. FP8 uses dynamic scales and needs no calibration file.
Scale metadata, integer packing padding, raw tails, and workspaces reduce the
effective memory savings. FP8 storage does not imply FP8 tensor-core attention
or an end-to-end speedup. Validate quality on your workload before relying on
lower bit widths.
