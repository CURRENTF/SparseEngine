from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext, ParallelGroup


def _glm_hf_config(**overrides):
    values = {
        "model_type": "glm4_moe_lite",
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "dtype": torch.bfloat16,
        "max_position_embeddings": 4096,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 47,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "vocab_size": 128,
        "q_lora_rank": 768,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "moe_intermediate_size": 64,
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "num_experts_per_tok": 4,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.8,
        "mlp_layer_types": ["dense"] + ["sparse"] * 46,
        "num_nextn_predict_layers": 1,
        "rope_interleave": True,
        "rope_scaling": None,
        "attention_bias": False,
        "quantization_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _glm_config(*, hf_overrides=None, **overrides) -> Config:
    kwargs = {
        "model": str(Path(__file__).resolve().parents[1]),
        "max_model_len": 128,
        "max_num_batched_tokens": 64,
        "engine_prefill_chunk_size": 64,
    }
    kwargs.update(overrides)
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=_glm_hf_config(**(hf_overrides or {})),
    ):
        return Config(**kwargs)


def _single_rank_parallel_context() -> ParallelContext:
    singleton = ParallelGroup(None, (0,), 0, 1)
    return ParallelContext(
        world=singleton,
        moe_tp=singleton,
        attn_tp=singleton,
        moe_ep=singleton,
        attn_dp=singleton,
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = (
        tensor.detach()
        .contiguous()
        .cpu()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()
