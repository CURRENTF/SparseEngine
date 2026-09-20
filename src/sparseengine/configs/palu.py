"""Validation of offline grouped low-rank projection artifacts."""

import json
from pathlib import Path

import torch

from sparseengine.models.layout import resolve_attention_qk_head_dim


def read_palu_manifest(path, hf_config):
    root = Path(path)
    with (root / "palu.json").open() as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "sparsevllm.palu.v1":
        raise ValueError("Unsupported Palu artifact format.")
    expected = {
        "model_type": hf_config.model_type,
        "hidden_size": int(hf_config.hidden_size),
        "num_attention_heads": int(hf_config.num_attention_heads),
        "num_key_value_heads": int(hf_config.num_key_value_heads),
        "num_hidden_layers": int(hf_config.num_hidden_layers),
        "head_dim": resolve_attention_qk_head_dim(hf_config),
    }
    if manifest.get("model") != expected:
        raise ValueError(f"Palu artifact architecture mismatch: expected {expected}.")
    group = manifest.get("group_size")
    if type(group) is not int or group <= 0 or expected["num_key_value_heads"] % group:
        raise ValueError("Palu group_size must divide the number of KV heads.")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != expected["num_hidden_layers"]:
        raise ValueError("Palu requires one K/V rank pair per transformer layer.")
    for pair in ranks:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(r) is not int or r < 16 or r % 16
                       or r > min(group * expected["head_dim"], 256) for r in pair)):
            raise ValueError("Palu ranks must be multiples of 16 in [16, min(group_size*head_dim, 256)].")
    if not (root / "palu.safetensors").is_file():
        raise FileNotFoundError(root / "palu.safetensors")
    return manifest


def validate_palu(config):
    path = config.palu_checkpoint_path
    if config.sparse_method != "palu":
        if path is not None:
            raise ValueError("palu_checkpoint_path requires sparse_method='palu'.")
        return
    if not path:
        raise ValueError("Palu requires palu_checkpoint_path; generate the offline factor artifact first.")
    if config.hf_config.model_type not in {"llama", "qwen3"}:
        raise ValueError("Palu currently supports Llama and Qwen3.")
    if config.quantization_config.enabled or config.tiny_random:
        raise ValueError("Palu requires real unquantized model weights.")
    if any(int(getattr(config, name)) != 1 for name in
           ("tensor_parallel_size", "expert_parallel_size", "data_parallel_size")):
        raise ValueError("Palu currently supports TP=EP=DP=1; distributed projection sharding is not implemented.")
    if config.enable_prefix_caching or config.enable_prefix_cache_offload:
        raise ValueError("Palu prefix caching/offload is not implemented.")
    if config.prefill_sparse_method:
        raise ValueError("Palu requires its native dense prefill path.")
    if getattr(config.hf_config, "attention_bias", False):
        raise ValueError("Palu currently requires bias-free attention projections.")
    if config.hf_config.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Palu requires FP16 or BF16 activations.")
    if resolve_attention_qk_head_dim(config.hf_config) not in {64, 128, 256}:
        raise ValueError("Palu requires head_dim 64, 128 or 256.")
    config.palu_manifest = read_palu_manifest(path, config.hf_config)
    config.attention_cache_layout = "low_rank_kv"
