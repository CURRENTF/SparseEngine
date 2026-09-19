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
    config.max_num_batched_tokens = int(config.max_num_batched_tokens)
    if config.max_num_batched_tokens <= 0:
        raise ValueError(
            "max_num_batched_tokens must be > 0, "
            f"got {config.max_num_batched_tokens}."
        )
    configured_chunk_prefill_size = (
        None if config.engine_prefill_chunk_size is None else int(config.engine_prefill_chunk_size)
    )
    if config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
        config.long_prefill_offload_threshold = _resolve_long_prefill_offload_threshold(
            config.long_prefill_offload_threshold
        )
        config.engine_prefill_chunk_size = (
            int(config.long_prefill_offload_threshold)
            if configured_chunk_prefill_size is None
            else configured_chunk_prefill_size
        )
        if config.engine_prefill_chunk_size > config.long_prefill_offload_threshold:
            raise ValueError(
                "long_bs1full_short_batch requires 0 < engine_prefill_chunk_size <= "
                "long_prefill_offload_threshold: "
                f"engine_prefill_chunk_size={config.engine_prefill_chunk_size}, "
                f"long_prefill_offload_threshold={config.long_prefill_offload_threshold}."
            )
    else:
        config.engine_prefill_chunk_size = (
            8192
            if configured_chunk_prefill_size is None
            else configured_chunk_prefill_size
        )
    if config.engine_prefill_chunk_size <= 0:
        raise ValueError(
            f"engine_prefill_chunk_size must be > 0, got {config.engine_prefill_chunk_size}."
        )
    score_window_method = normalize_sparse_method(config.sparse_method)
    if score_window_method in {"snapkv", "pyramidkv"}:
        snapkv_window_size = int(config.snapkv_window_size)
        if config.engine_prefill_chunk_size < snapkv_window_size:
            raise ValueError(
                f"{score_window_method} requires engine_prefill_chunk_size >= "
                "snapkv_window_size so the "
                "final score window fits in one prefill step: "
                f"engine_prefill_chunk_size={config.engine_prefill_chunk_size}, "
                f"snapkv_window_size={snapkv_window_size}."
            )
    if (
        config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
        and config.max_num_batched_tokens < config.long_prefill_offload_threshold
    ):
        from sparseengine.utils.log import log_once

        log_once(
            "long_bs1full_short_batch requires one full residual at the offload "
            "boundary to fit; raising max_num_batched_tokens from "
            f"{config.max_num_batched_tokens} to "
            f"{config.long_prefill_offload_threshold}.",
            level="WARNING",
        )
        config.max_num_batched_tokens = config.long_prefill_offload_threshold

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
