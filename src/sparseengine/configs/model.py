"""Model metadata loading and validation orchestration."""

import json
import os
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig

from sparseengine.distributed.topology import ParallelTopology
from sparseengine.method_registry import (
    is_deltakv_method,
    validate_model_runtime_compatibility,
    validate_sparse_method_assets,
)
from sparseengine.models.checkpoint import validate_checkpoint
from sparseengine.models.layout import RuntimeLayout
from sparseengine.models.rope import (
    resolve_rope_max_position,
    resolve_rope_parameters,
)
from sparseengine.models.spec import ModelSpec, resolve_model_spec
from sparseengine.quantization import QuantizationConfig
from sparseengine.utils.config import config_get
from sparseengine.utils.log import logger, log_once


_CONTEXT_LENGTH_KEYS = (
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
)


def _config_to_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def _load_model_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as error:
        load_error = error
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise RuntimeError(
            "AutoConfig.from_pretrained failed and no config.json exists for an "
            f"explicit raw-config fallback. model={model_path} "
            f"error={type(load_error).__name__}: {load_error}"
        ) from load_error
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    model_type = str(config_get(raw_config, "model_type", "") or "")
    model_spec = resolve_model_spec(model_type)
    if not model_spec.allow_raw_config:
        raise RuntimeError(
            "AutoConfig.from_pretrained failed. Refusing to silently fall back to raw "
            f"config.json for {model_spec.name}. model={model_path} "
            f"error={type(load_error).__name__}: {load_error}"
        ) from load_error
    log_once(
        f"AutoConfig.from_pretrained failed for {model_spec.name}; loading raw "
        "config.json through its explicit model specification.",
        level="WARNING",
    )
    return _config_to_namespace(raw_config)


def _extract_text_config(config: Any) -> Any:
    text_config = config_get(config, "text_config", None)
    if text_config is None:
        return config
    if isinstance(text_config, dict):
        return _config_to_namespace(text_config)
    return text_config


def _normalize_hf_config_dtype(config: Any, fallback: Any = None) -> torch.dtype:
    dtype = config_get(config, "dtype", None)
    if dtype is None:
        legacy_values = config if isinstance(config, dict) else vars(config)
        dtype = legacy_values.get("torch_dtype", fallback)
    if isinstance(dtype, str):
        normalized = dtype.strip().lower().replace("torch.", "")
        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(normalized)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Model config dtype must be a torch.dtype, got {dtype!r}.")
    if isinstance(config, dict):
        config["dtype"] = dtype
    else:
        setattr(config, "dtype", dtype)
    return dtype


def _model_context_length(hf_config: Any) -> int:
    for key in _CONTEXT_LENGTH_KEYS:
        value = config_get(hf_config, key, None)
        if value is not None:
            context_length = int(value)
            if context_length <= 0:
                raise ValueError(f"Model config {key} must be positive, got {value!r}.")
            break
    else:
        raise ValueError(
            "Model config does not declare a supported context length; expected one of "
            f"{', '.join(_CONTEXT_LENGTH_KEYS)}. Set max_model_len explicitly."
        )

    rope_parameters = resolve_rope_parameters(hf_config)
    rope_type = str(rope_parameters["rope_type"])
    if rope_type != "yarn":
        if (
            rope_type == "linear"
            and "original_max_position_embeddings" not in rope_parameters
        ):
            context_length = int(
                context_length * float(rope_parameters.get("factor", 1.0))
            )
            if context_length <= 0:
                raise ValueError(
                    "Model config linear RoPE context length must be positive, "
                    f"got {context_length}."
                )
        return context_length

    rope_context_length = resolve_rope_max_position(
        hf_config,
        model_name="Model config",
    )
    if key == "max_position_embeddings":
        return rope_context_length
    if context_length > rope_context_length:
        raise ValueError(
            f"Model config {key} exceeds the YaRN context length: "
            f"declared={context_length} supported={rope_context_length}."
        )
    return context_length


def _validate_sparse_method_rope_compatibility(config: Any) -> None:
    if not is_deltakv_method(config_get(config, "sparse_method", None)):
        return

    rope_type = str(resolve_rope_parameters(config.hf_config)["rope_type"])
    if rope_type not in {"default", "llama3"}:
        raise NotImplementedError(
            "DeltaKV reconstruction supports only default and llama3 RoPE; "
            f"got rope_type={rope_type!r}. Its de-RoPE/re-RoPE CUDA kernels need "
            "separate correctness validation before enabling other scaling types."
        )


def _finalize_model_config(config, model_spec: ModelSpec) -> None:
    _validate_sparse_method_rope_compatibility(config)
    model_context_length = _model_context_length(config.hf_config)
    config.max_model_len_auto = config.max_model_len is None
    if config.max_model_len_auto:
        config.max_model_len = model_context_length
    else:
        config.max_model_len = int(config.max_model_len)
        if config.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {config.max_model_len}.")
        if config.max_model_len > model_context_length:
            raise ValueError(
                "max_model_len exceeds the model context length: "
                f"requested={config.max_model_len} supported={model_context_length}."
            )

    config.runtime_layout = RuntimeLayout.from_config(
        config.hf_config,
        require_mixed=model_spec.mixed_attention,
    )

    if config.max_num_seqs_in_batch > 32:
        logger.warning('max_num_seqs_in_batch 过大或许会占用太多显存')


def load_and_validate_model(config) -> None:
    validate_sparse_method_assets(config.sparse_method, config.model)
    if isinstance(config.deltakv_checkpoint_path, str):
        deltakv_checkpoint_path = config.deltakv_checkpoint_path.strip()
        config.deltakv_checkpoint_path = (
            None
            if deltakv_checkpoint_path.lower() in {"", "none", "null"}
            else deltakv_checkpoint_path
        )
    if config.tiny_random and config.sparse_method == "deltakv":
        raise NotImplementedError(
            "Tiny random mode does not support DeltaKV compressor weights yet."
        )
    config.outer_hf_config = _load_model_config(config.model)
    model_type = str(config_get(config.outer_hf_config, "model_type", "") or "")
    model_spec = resolve_model_spec(model_type)
    config.hf_config = _extract_text_config(config.outer_hf_config)
    outer_dtype = _normalize_hf_config_dtype(config.outer_hf_config, torch.bfloat16)
    _normalize_hf_config_dtype(config.hf_config, outer_dtype)
    config.model_spec = model_spec
    config.parallel_topology = ParallelTopology(
        attn_tp_size=config.tensor_parallel_size,
        attn_dp_size=config.data_parallel_size,
        moe_ep_size=config.expert_parallel_size,
    )
    model_spec.validate_parallel_execution(config.parallel_topology)
    if config.tiny_random:
        from sparseengine.debug.tiny_random import apply_tiny_random_overrides

        if not model_spec.supports_tiny_random:
            raise NotImplementedError(
                f"Tiny random mode does not support {model_spec.name} yet."
            )
        config.tiny_random_overrides = apply_tiny_random_overrides(
            config.hf_config,
            config.tiny_random_config,
            validate_standard_head_shape=(
                model_spec.tiny_random_requires_standard_head_shape
            ),
        )
        log_once(
            "TINY RANDOM MODE is enabled: checkpoint weights will not be read and "
            f"model-quality results are invalid. config={config.tiny_random_config} "
            f"seed={config.tiny_random_seed} overrides={config.tiny_random_overrides}",
            level="WARNING",
        )
    model_spec.validate_sharding(config.hf_config, config.parallel_topology)

    raw_quantization_config = config_get(
        config.hf_config,
        "quantization_config",
        config_get(config.outer_hf_config, "quantization_config", None),
    )
    config.quantization_config = QuantizationConfig.from_hf_config(
        raw_quantization_config,
        required_fp8=model_spec.requires_fp8,
        model_name=model_spec.name,
        activation_dtype=config.hf_config.dtype,
    )
    if (
        config.tiny_random
        and config.quantization_config.enabled
        and not model_spec.supports_quantized_tiny_random
    ):
        raise NotImplementedError(
            "Tiny random mode does not support quantized model weights."
        )
    setattr(config.hf_config, "quantization_config", config.quantization_config)
    config.attention_cache_layout = model_spec.attention_cache_layout
    validate_checkpoint(
        model_type,
        outer_config=config.outer_hf_config,
        config=config.hf_config,
        raw_quantization_config=raw_quantization_config,
        quantization=config.quantization_config,
        topology=config.parallel_topology,
    )

    validate_model_runtime_compatibility(
        model_type=model_type,
        sparse_method=config.resolved_cache_sparse_method,
        decode_sparse_method=config.sparse_method,
        topology=config.parallel_topology,
        decode_graph=config.decode_graph,
        enable_prefix_caching=config.enable_prefix_caching,
    )
    _finalize_model_config(config, model_spec)
