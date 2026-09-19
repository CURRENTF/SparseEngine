"""DeltaKV configuration normalization and runtime validation."""

import importlib.util
import json
import os
from typing import Any

from sparseengine.configs.common import (
    _normalize_float_attr,
    _normalize_int_attr,
    _normalize_positive_int,
    _normalize_positive_multiple,
)
from sparseengine.utils.log import log_once

def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_deltakv_sparse_decode_backend(value: Any) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in {"auto", "custom", "fa2"}:
        raise ValueError(
            "deltakv_sparse_decode_backend must be one of 'auto', 'custom', or 'fa2', "
            f"got {value!r}."
        )
    if backend == "auto":
        resolved = "fa2" if _flash_attn_available() else "custom"
        reason = "flash_attn available" if resolved == "fa2" else "flash_attn not available"
        log_once(
            f"DeltaKV sparse decode backend auto-selected {resolved!r} ({reason}).",
            level="INFO",
        )
        return resolved
    if backend == "fa2" and not _flash_attn_available():
        raise ValueError(
            "deltakv_sparse_decode_backend='fa2' requires the flash_attn package; "
            "use 'custom' or leave it as 'auto' when flash_attn is not installed."
        )
    return backend


def _normalize_deltakv_kernel_options(config) -> None:
    if int(config.deltakv_cluster_gather_chunk_size) <= 0:
        raise ValueError(
            "deltakv_cluster_gather_chunk_size must be > 0, "
            f"got {config.deltakv_cluster_gather_chunk_size}."
        )
    _normalize_int_attr(config, "deltakv_cluster_gather_chunk_size")
    config.sparse_attn_score_dtype = str(config.sparse_attn_score_dtype or "float32").strip().lower()
    if config.sparse_attn_score_dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError(
            "sparse_attn_score_dtype must be 'float32', 'bfloat16', or 'float16', "
            f"got {config.sparse_attn_score_dtype!r}."
        )


def _normalize_deltakv_quantization(config) -> None:
    _normalize_int_attr(config, "full_layer_kv_quant_bits", fallback=0)
    if config.full_layer_kv_quant_bits not in (0, 2, 4):
        raise ValueError(
            "full_layer_kv_quant_bits must be 0, 2, or 4, "
            f"got {config.full_layer_kv_quant_bits}."
        )
    _normalize_float_attr(config, "full_layer_cluster_ratio", fallback=0.0)
    if config.full_layer_cluster_ratio < 0.0:
        raise ValueError(f"full_layer_cluster_ratio must be >= 0, got {config.full_layer_cluster_ratio}.")
    _normalize_int_attr(config, "deltakv_latent_quant_bits", fallback=0)
    if config.deltakv_latent_quant_bits not in (0, 2, 4):
        raise ValueError(f"deltakv_latent_quant_bits must be 0, 2, or 4, got {config.deltakv_latent_quant_bits}.")
    _normalize_int_attr(config, "deltakv_latent_quant_group_size", fallback=0)
    if config.deltakv_latent_quant_group_size < 0:
        raise ValueError(f"deltakv_latent_quant_group_size must be >= 0, got {config.deltakv_latent_quant_group_size}.")


def _normalize_full_layer_kivi(config) -> None:
    _normalize_positive_int(config, "full_layer_kivi_group_size", fallback=32)
    config.full_layer_kivi_residual_length = int(
        config.full_layer_kivi_residual_length or config.full_layer_kivi_group_size
    )
    if config.full_layer_kivi_residual_length <= 0:
        raise ValueError(
            "full_layer_kivi_residual_length must be > 0, "
            f"got {config.full_layer_kivi_residual_length}."
        )
    _normalize_positive_multiple(
        config, "full_layer_kivi_decode_block_seq", multiple=16, fallback=256
    )
    _normalize_positive_multiple(
        config, "full_layer_kivi_decode_block_n", multiple=16, fallback=16
    )
    _normalize_int_attr(config, "full_layer_kivi_decode_num_warps", fallback=2)
    if config.full_layer_kivi_decode_num_warps not in {1, 2, 4, 8}:
        raise ValueError(
            "full_layer_kivi_decode_num_warps must be one of 1, 2, 4, or 8, "
            f"got {config.full_layer_kivi_decode_num_warps}."
        )
    _normalize_positive_int(config, "full_layer_kivi_decode_num_stages", fallback=3)
    config.enable_full_layer_kivi_fused_decode = bool(config.enable_full_layer_kivi_fused_decode)
    config.enable_full_layer_kivi_grouped_decode = bool(config.enable_full_layer_kivi_grouped_decode)
    config.enable_full_layer_kivi_dense_decode = bool(config.enable_full_layer_kivi_dense_decode)
    if config.enable_full_layer_kivi_fused_decode:
        raise ValueError(
            "enable_full_layer_kivi_fused_decode was removed; full-layer KIVI decode now "
            "uses the direct packed backend."
        )
    if config.enable_full_layer_kivi_grouped_decode:
        raise ValueError(
            "enable_full_layer_kivi_grouped_decode was removed; full-layer KIVI decode now "
            "uses the direct packed backend."
        )


def _normalize_deltakv_capacity(config) -> None:
    _normalize_float_attr(config, "deltakv_full_pool_reserve_ratio", fallback=0.0)
    if config.deltakv_full_pool_reserve_ratio < 0.0 or config.deltakv_full_pool_reserve_ratio >= 1.0:
        raise ValueError(
            "deltakv_full_pool_reserve_ratio must be in [0, 1), "
            f"got {config.deltakv_full_pool_reserve_ratio}."
        )
    _normalize_float_attr(config, "deltakv_cache_capacity_margin", fallback=1.0)
    if config.deltakv_cache_capacity_margin < 1.0:
        raise ValueError(
            "deltakv_cache_capacity_margin must be >= 1.0, "
            f"got {config.deltakv_cache_capacity_margin}."
        )
    _normalize_float_attr(config, "deltakv_center_capacity_margin", fallback=1.0)
    if config.deltakv_center_capacity_margin < 1.0:
        raise ValueError(
            "deltakv_center_capacity_margin must be >= 1.0, "
            f"got {config.deltakv_center_capacity_margin}."
        )


def normalize_deltakv_storage(config) -> None:
    _normalize_deltakv_kernel_options(config)
    _normalize_deltakv_quantization(config)
    _normalize_full_layer_kivi(config)
    _normalize_deltakv_capacity(config)


def _checkpoint_targets_model(path: str | None, model_types: frozenset[str]) -> bool:
    config_path = os.path.join(path, "config.json") if path and os.path.isdir(path) else None
    if config_path is None or not os.path.isfile(config_path):
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        checkpoint_config = json.load(f)
    return any(
        str(checkpoint_config.get(field, "")).strip().lower() in model_types
        for field in (
            "model_type",
            "base_model_type",
            "target_model_type",
            "runtime_model_type",
        )
    )


def validate_deltakv_runtime(config) -> None:
    # Normalize compressor type strings.
    for attr in ("compressor_down_type", "compressor_up_type"):
        v = getattr(config, attr, "auto")
        if v is None:
            v = "auto"
        v = str(v).strip().lower()
        setattr(config, attr, v if v else "auto")

    if config.sparse_method == "deltakv":
        log_once(
            "DeltaKV support in Sparse-Engine is still experimental and not fully mature; "
            "verify results carefully before treating them as final.",
            level="WARNING",
        )
        checkpoint_model_types = config.model_spec.deltakv_checkpoint_model_types
        if checkpoint_model_types and not _checkpoint_targets_model(
            config.deltakv_checkpoint_path,
            checkpoint_model_types,
        ):
            raise ValueError(
                f"DeltaKV for {config.model_spec.name} requires a compatible "
                "deltakv_checkpoint_path. Use sparse_method='' to run vanilla inference."
            )
        if not bool(getattr(config, "use_compression", True)):
            raise ValueError("DeltaKV runtime is compressor-only; set use_compression=True.")
        if bool(getattr(config, "enable_sparse_ref_fp8", False)):
            raise ValueError("enable_sparse_ref_fp8 was removed from the slim DeltaKV runtime.")
        if config.deltakv_checkpoint_path is None and not config.allow_missing_deltakv_path:
            raise ValueError(
                "DeltaKV requires deltakv_checkpoint_path for compressor sparse layers. "
                "Set allow_missing_deltakv_path=True only for construction-only tests."
            )
        if config.deltakv_latent_quant_bits not in (0, 4):
            raise ValueError(
                "DeltaKV slim runtime supports sparse compressor residual bits 0 or 4 only, "
                f"got deltakv_latent_quant_bits={config.deltakv_latent_quant_bits}."
            )
        if config.full_layer_kv_quant_bits not in (0, 4):
            raise ValueError(
                "DeltaKV slim runtime supports full-layer storage bits 0 or 4 only, "
                f"got full_layer_kv_quant_bits={config.full_layer_kv_quant_bits}."
            )
        if config.deltakv_latent_quant_bits == 4 and config.deltakv_latent_quant_group_size == 0:
            config.deltakv_latent_quant_group_size = 32
        _normalize_positive_multiple(
            config,
            "deltakv_triton_materialize_block_tokens",
            multiple=8,
            fallback=16,
        )
        config.deltakv_sparse_decode_backend = _resolve_deltakv_sparse_decode_backend(
            config.deltakv_sparse_decode_backend
        )
        is_bf16_full_compressor_sparse = (
            config.full_layer_kv_quant_bits == 0 and config.deltakv_latent_quant_bits == 0
        )
        is_bf16_full_int4_compressor_sparse = (
            config.full_layer_kv_quant_bits == 0 and config.deltakv_latent_quant_bits == 4
        )
        is_kivi4_full_int4_compressor_sparse = (
            config.full_layer_kv_quant_bits == 4
            and config.deltakv_latent_quant_bits == 4
            and bool(getattr(config, "enable_full_layer_kivi_quant", True))
        )
        if not (
            is_bf16_full_compressor_sparse
            or is_bf16_full_int4_compressor_sparse
            or is_kivi4_full_int4_compressor_sparse
        ):
            raise ValueError(
                "DeltaKV slim runtime supports exactly three paths: "
                "(full_layer_kv_quant_bits=0, deltakv_latent_quant_bits=0) and "
                "(full_layer_kv_quant_bits=0, deltakv_latent_quant_bits=4) and "
                "(full_layer_kv_quant_bits=4, deltakv_latent_quant_bits=4, enable_full_layer_kivi_quant=True)."
            )
