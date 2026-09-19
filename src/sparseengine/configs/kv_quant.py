"""Validation for token-preserving compressed explicit KV methods."""

import torch

from sparseengine.method_registry import QUANTIZED_KV_METHODS
from sparseengine.models.layout import resolve_attention_qk_head_dim


def validate_quantized_kv(config) -> None:
    if config.sparse_method not in QUANTIZED_KV_METHODS:
        return
    for name, allowed in (("kivi_bits", {2, 4}), ("turboquant_bits", {2, 3, 4})):
        value = getattr(config, name)
        if type(value) is not int or value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    size = config.kv_quant_page_size
    if type(size) is not int or size < 16 or size > 128 or size & (size - 1):
        raise ValueError("kv_quant_page_size must be a power of two from 16 through 128.")
    if type(config.turboquant_seed) is not int or config.turboquant_seed < 0:
        raise ValueError("turboquant_seed must be a non-negative integer.")
    if config.hf_config.model_type not in {"llama", "qwen2", "qwen3", "qwen3_moe"}:
        raise ValueError("Quantized KV methods currently support llama, qwen2, qwen3, and qwen3_moe.")
    if config.attention_cache_layout != "explicit_kv":
        raise ValueError("Quantized KV methods require homogeneous explicit KV storage.")
    dim = resolve_attention_qk_head_dim(config.hf_config)
    if dim not in {64, 128, 256} or dim % size:
        raise ValueError("Quantized KV requires head_dim 64/128/256 divisible by kv_quant_page_size.")
    if config.hf_config.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Quantized KV methods require FP16 or BF16 model activations.")
    # ModelSpec validates topology and head/expert divisibility. Quantization
    # operates on each rank's local heads and adds no collective operations.
    if config.enable_prefix_caching or config.enable_prefix_cache_offload:
        raise ValueError("Quantized KV methods do not yet support prefix caching/offload.")
    if config.prefill_sparse_method:
        raise ValueError("Quantized KV methods currently require dense prefill attention.")
