"""Batch-capacity and prefill-scheduling normalization."""

from sparseengine.configs.common import (
    _coerce_optional_positive_int,
    _resolve_long_prefill_offload_threshold,
)
from sparseengine.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparseengine.method_registry import (
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    normalize_sparse_method,
    resolve_prefill_schedule_policy,
)


def normalize_scheduling(config) -> None:
    window = config.decode_reservation_tokens
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("decode_reservation_tokens must be a positive integer.")
    config.max_num_seqs_in_batch = int(config.max_num_seqs_in_batch)
    if config.max_num_seqs_in_batch <= 0:
        raise ValueError(
            "max_num_seqs_in_batch must be > 0, "
            f"got {config.max_num_seqs_in_batch}."
        )
    config.max_decoding_seqs = (
        config.max_num_seqs_in_batch
        if config.max_decoding_seqs is None
        else int(config.max_decoding_seqs)
    )
    if config.max_decoding_seqs <= 0:
        raise ValueError(
            f"max_decoding_seqs must be > 0, got {config.max_decoding_seqs}."
        )
    if config.favor_min_decoding_seqs is None:
        config.favor_min_decoding_seqs = (3 * config.max_decoding_seqs + 3) // 4
    favor = config.favor_min_decoding_seqs
    if isinstance(favor, bool) or not isinstance(favor, int) or not 0 <= favor <= config.max_decoding_seqs:
        raise ValueError("favor_min_decoding_seqs must be an integer between 0 and max_decoding_seqs.")
    configured_max_num_seqs_in_gpu = _coerce_optional_positive_int(
        "max_num_seqs_in_gpu",
        config.max_num_seqs_in_gpu,
    )
    if configured_max_num_seqs_in_gpu is None:
        configured_max_num_seqs_in_gpu = max(
            config.max_num_seqs_in_batch * REDUNDANCY_BATCH_SIZE_FACTOR,
            config.max_decoding_seqs,
        )
    if configured_max_num_seqs_in_gpu < config.max_num_seqs_in_batch:
        raise ValueError(
            "max_num_seqs_in_gpu must be >= max_num_seqs_in_batch: "
            f"{configured_max_num_seqs_in_gpu} < {config.max_num_seqs_in_batch}."
        )
    if configured_max_num_seqs_in_gpu < config.max_decoding_seqs:
        raise ValueError(
            "max_num_seqs_in_gpu must be >= max_decoding_seqs: "
            f"{configured_max_num_seqs_in_gpu} < {config.max_decoding_seqs}."
        )
    config.max_num_seqs_in_gpu = int(configured_max_num_seqs_in_gpu)

    config.prefill_schedule_policy = resolve_prefill_schedule_policy(
        config.sparse_method,
        config.prefill_schedule_policy,
    )
    for name in ("max_num_batched_tokens", "engine_prefill_chunk_size"):
        value = getattr(config, name)
        is_auto = value == "auto" or (name == "engine_prefill_chunk_size" and value is None)
        setattr(config, f"{name}_auto", is_auto)
        if not is_auto:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a positive integer or 'auto', got {value!r}.") from None
            if isinstance(value, bool) or parsed <= 0 or (not isinstance(value, str) and parsed != value):
                raise ValueError(f"{name} must be > 0 and an integer, got {value!r}.")
            setattr(config, name, parsed)
    if config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
        config.long_prefill_offload_threshold = _resolve_long_prefill_offload_threshold(
            config.long_prefill_offload_threshold
        )

    if int(config.mlp_chunk_size) <= 0:
        raise ValueError(f"mlp_chunk_size must be > 0, got {config.mlp_chunk_size}.")
    config.mlp_chunk_size = int(config.mlp_chunk_size)
    config.mla_prefill_workspace_bytes = int(config.mla_prefill_workspace_bytes)
    config.mla_prefill_history_chunk_size = int(config.mla_prefill_history_chunk_size)
    if config.mla_prefill_history_chunk_size <= 0:
        raise ValueError("mla_prefill_history_chunk_size must be > 0.")
    if config.mla_prefill_workspace_bytes <= 0:
        raise ValueError(
            "mla_prefill_workspace_bytes must be > 0, got "
            f"{config.mla_prefill_workspace_bytes}."
        )


def estimate_prefill_token_capacity(
    hf_config, *, total_memory_bytes: int, tensor_parallel_size: int, gpu_memory_utilization: float,
) -> int:
    """Reuse the activation-headroom heuristic; this is not an OOM guarantee."""
    import torch

    intermediate_size = int(getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4))
    intermediate_size_per_rank = intermediate_size // int(tensor_parallel_size)
    dtype_size = torch.empty((), dtype=hf_config.dtype).element_size()
    utilization = float(gpu_memory_utilization)
    if not 0 < utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")
    if intermediate_size_per_rank <= 0 or total_memory_bytes <= 0:
        raise ValueError("Prefill capacity estimation requires positive memory and per-rank intermediate size.")
    reserved_bytes = int(total_memory_bytes * (1 - utilization))
    return reserved_bytes // (intermediate_size_per_rank * dtype_size * 16)


def resolve_prefill_token_budget(config) -> None:
    """Resolve once, after model metadata and before model/workspace construction."""
    from sparseengine.platforms import current_platform
    from sparseengine.utils.log import logger

    budget_auto = config.max_num_batched_tokens_auto
    chunk_auto = config.engine_prefill_chunk_size_auto
    long_policy = config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
    threshold = int(config.long_prefill_offload_threshold) if long_policy else 0
    budget = config.max_num_batched_tokens
    chunk = config.engine_prefill_chunk_size
    if budget_auto:
        # Keep the former token-budget default as the automatic ceiling, unless
        # an explicit chunk or the atomic full-prefill contract requires more.
        total_memory = min(
            current_platform.get_total_memory(rank) for rank in range(config.world_size)
        )
        estimated = estimate_prefill_token_capacity(
            config.hf_config, total_memory_bytes=total_memory,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        if estimated <= 0:
            raise ValueError(
                "Auto prefill token capacity is zero. Reduce gpu_memory_utilization "
                "to reserve activation headroom, or set max_num_batched_tokens explicitly."
            )
        required = max(threshold, 0 if chunk_auto else int(chunk), config.max_decoding_seqs)
        if estimated < required:
            raise ValueError(
                f"Auto max_num_batched_tokens estimate {estimated} is below the required "
                f"{required} tokens (chunk={chunk}, long_prefill_offload_threshold={threshold}, "
                f"max_decoding_seqs={config.max_decoding_seqs}). Reduce these limits or "
                "set max_num_batched_tokens explicitly; the offload threshold is not changed automatically."
            )
        budget = min(estimated, max(65536, required))
        if long_policy and chunk_auto:
            budget = max(threshold, config.max_decoding_seqs)
        logger.info(
            "Auto prefill token budget: total_memory_bytes={} estimated_tokens={} resolved_tokens={}.",
            total_memory, estimated, budget,
        )
    elif long_policy and budget < threshold:
        # Preserve the existing explicit-budget normalization for atomic prefill.
        logger.warning(
            "long_bs1full_short_batch requires one full residual at the offload "
            "boundary to fit; raising max_num_batched_tokens from {} to {}.",
            budget, threshold,
        )
        budget = threshold
    if chunk_auto:
        chunk = min(budget, threshold) if long_policy else budget
    if long_policy and chunk > threshold:
        raise ValueError(
            "long_bs1full_short_batch requires 0 < engine_prefill_chunk_size <= "
            "long_prefill_offload_threshold: "
            f"engine_prefill_chunk_size={chunk}, long_prefill_offload_threshold={threshold}."
        )
    method = normalize_sparse_method(config.sparse_method)
    if method in {"snapkv", "pyramidkv"} and chunk < int(config.snapkv_window_size):
        raise ValueError(
            f"{method} requires engine_prefill_chunk_size >= snapkv_window_size so the "
            f"final score window fits in one prefill step: engine_prefill_chunk_size={chunk}, "
            f"snapkv_window_size={config.snapkv_window_size}."
        )
    config.max_num_batched_tokens = int(budget)
    config.engine_prefill_chunk_size = int(chunk)
