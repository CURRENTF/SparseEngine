"""Prefix-cache normalization and validation."""

import math

from sparseengine.configs.common import (
    _coerce_bool_config,
    _coerce_optional_positive_int,
)
from sparseengine.engine.chain_cache import normalize_prefix_cache_mode
from sparseengine.engine.prefix_cache import resolve_prefix_cache_block_size
from sparseengine.method_registry import PREFIX_CACHE_SUPPORTED_METHODS
from sparseengine.utils.log import log_once

def normalize_prefix_cache(config) -> None:
    cache_method = str(
        getattr(config, "resolved_cache_sparse_method", config.sparse_method)
        or ""
    )
    config.enable_prefix_caching = _coerce_bool_config("enable_prefix_caching", config.enable_prefix_caching)
    config.prefix_cache_mode = str(config.prefix_cache_mode or "auto").strip().lower()
    config.resolved_prefix_cache_mode = normalize_prefix_cache_mode(
        config.prefix_cache_mode,
        enabled=config.enable_prefix_caching,
        method=cache_method,
        prefill_method=getattr(config, "prefill_sparse_method", ""),
    )

    config.prefix_cache_block_size = _coerce_optional_positive_int(
        "prefix_cache_block_size",
        config.prefix_cache_block_size,
    )
    config.prefix_cache_max_blocks = _coerce_optional_positive_int(
        "prefix_cache_max_blocks",
        config.prefix_cache_max_blocks,
    )
    config.chain_cache_max_tombstones = int(config.chain_cache_max_tombstones)
    if config.chain_cache_max_tombstones <= 0:
        raise ValueError(
            "chain_cache_max_tombstones must be > 0, got "
            f"{config.chain_cache_max_tombstones}."
        )
    config.enable_prefix_cache_offload = _coerce_bool_config(
        "enable_prefix_cache_offload",
        config.enable_prefix_cache_offload,
    )
    if config.prefix_cache_host_size_gb is not None:
        config.prefix_cache_host_size_gb = float(config.prefix_cache_host_size_gb)
        if (
            not math.isfinite(config.prefix_cache_host_size_gb)
            or config.prefix_cache_host_size_gb <= 0
        ):
            raise ValueError(
                "prefix_cache_host_size_gb must be positive when set, "
                f"got {config.prefix_cache_host_size_gb}."
            )
    if config.enable_prefix_cache_offload:
        if getattr(config, "prefill_sparse_method", "") == "omnikv_prefill":
            raise ValueError(
                "OmniKV prefill chain cache retains its shared slot pool; "
                "enable_prefix_cache_offload does not support this layout. "
                "Use enable_omnikv_offload for OmniKV CPU KV backing."
            )
        if not config.enable_prefix_caching:
            raise ValueError(
                "enable_prefix_cache_offload requires enable_prefix_caching=True."
            )
        if cache_method not in ("", "omnikv", "quest") and config.resolved_prefix_cache_mode != "chain":
            raise ValueError(
                "prefix cache offload requires a supported radix or chain cache; "
                f"got cache method={cache_method!r}."
            )
        if int(config.tensor_parallel_size) not in (1, 2) and not getattr(config, "enable_omnikv_offload", False):
            raise ValueError(
                "prefix cache offload currently supports tensor_parallel_size=1 or 2."
            )
        if config.prefix_cache_host_size_gb is None:
            raise ValueError(
                "enable_prefix_cache_offload requires an explicit "
                "prefix_cache_host_size_gb."
            )
    recurrent_state_max_bytes = _coerce_optional_positive_int(
        "recurrent_state_max_bytes",
        config.recurrent_state_max_bytes,
    )
    config.prefix_cache_max_recurrent_bytes = _coerce_optional_positive_int(
        "prefix_cache_max_recurrent_bytes",
        config.prefix_cache_max_recurrent_bytes,
    )
    if config.prefix_cache_max_recurrent_bytes is not None:
        log_once(
            "prefix_cache_max_recurrent_bytes is deprecated; use "
            "recurrent_state_max_bytes instead. The budget is an explicit "
            "hard limit for the live recurrent-state pool.",
            level="WARNING",
        )
        if (
            recurrent_state_max_bytes is not None
            and recurrent_state_max_bytes != config.prefix_cache_max_recurrent_bytes
        ):
            raise ValueError(
                "conflicting recurrent state budgets: "
                f"recurrent_state_max_bytes={recurrent_state_max_bytes} and "
                "prefix_cache_max_recurrent_bytes="
                f"{config.prefix_cache_max_recurrent_bytes}."
            )
        recurrent_state_max_bytes = config.prefix_cache_max_recurrent_bytes
    config.recurrent_state_max_bytes = recurrent_state_max_bytes
    if config.enable_prefix_caching and cache_method not in PREFIX_CACHE_SUPPORTED_METHODS:
        raise ValueError(
            "prefix caching only supports vanilla, streamingllm, omnikv, quest, "
            "snapkv, h2o, pyramidkv, rkv, and skipkv."
        )
    config.prefix_cache_salt = str(config.prefix_cache_salt or "")

def finalize_prefix_cache(config) -> None:
    block_multiple = config.model_spec.prefix_cache_block_size_multiple
    if (
        block_multiple is not None
        and config.resolved_prefix_cache_mode == "radix"
        and config.prefix_cache_block_size is None
    ):
        config.prefix_cache_block_size = block_multiple
    config.prefix_cache_block_size = resolve_prefix_cache_block_size(config)
    if block_multiple is not None and config.resolved_prefix_cache_mode == "radix":
        if (
            config.prefix_cache_block_size < block_multiple
            or config.prefix_cache_block_size % block_multiple
        ):
            raise ValueError(
                f"{config.model_spec.name} prefix cache requires "
                f"prefix_cache_block_size to be {block_multiple}*N, "
                f"got {config.prefix_cache_block_size}."
            )
