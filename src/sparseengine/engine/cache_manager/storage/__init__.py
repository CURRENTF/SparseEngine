from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AttentionCacheStorage, CacheLayout
from .explicit_kv import ExplicitKVStorage
from .heterogeneous_explicit_kv import HeterogeneousExplicitKVStorage

if TYPE_CHECKING:
    from .mla_latent import MlaLatentStorage


def create_attention_cache_storage(
    config: Any,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> AttentionCacheStorage:
    configured_layout = config.attention_cache_layout
    layout = (
        configured_layout
        if isinstance(configured_layout, CacheLayout)
        else CacheLayout(str(configured_layout))
    )
    dtype = config.hf_config.dtype
    if layout is CacheLayout.EXPLICIT_KV:
        runtime_layout = getattr(config, "runtime_layout", None)
        parallel_topology = getattr(config, "parallel_topology", None)
        layer_shapes = (
            runtime_layout.local_kv_shapes(parallel_topology.attn_tp_size)
            if runtime_layout is not None and parallel_topology is not None
            else ()
        )
        if len(set(layer_shapes)) > 1:
            return HeterogeneousExplicitKVStorage(
                layer_shapes=layer_shapes,
                dtype=dtype,
            )
        return ExplicitKVStorage(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
    if layout is CacheLayout.MLA_LATENT:
        from .mla_latent import MlaLatentStorage

        return MlaLatentStorage(
            kv_lora_rank=int(config.hf_config.kv_lora_rank),
            rope_dim=int(config.hf_config.qk_rope_head_dim),
            dtype=dtype,
            validate_runtime_invariants=bool(
                getattr(config, "validate_runtime_invariants", False)
            ),
        )
    raise AssertionError(f"Unhandled attention cache layout: {layout!r}")


def __getattr__(name: str):
    if name == "MlaLatentStorage":
        from .mla_latent import MlaLatentStorage

        return MlaLatentStorage
    raise AttributeError(name)


__all__ = [
    "AttentionCacheStorage",
    "CacheLayout",
    "ExplicitKVStorage",
    "HeterogeneousExplicitKVStorage",
    "MlaLatentStorage",
    "create_attention_cache_storage",
]
