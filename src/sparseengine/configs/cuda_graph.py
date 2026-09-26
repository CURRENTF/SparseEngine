"""Decode CUDA Graph normalization and validation."""

from collections.abc import Callable
from typing import Any

from sparseengine.configs.common import _coerce_bool_config
from sparseengine.method_registry import (
    DECODE_CUDA_GRAPH_SUPPORTED_METHODS,
    decode_graph_path_id,
    is_decode_cuda_graph_supported,
    is_tp_decode_cuda_graph_supported,
)
from sparseengine.utils.log import log_once


def _default_decode_cuda_graph_capture_sizes(
    max_batch_size: int, capture_limit: int = 32
) -> list[int]:
    """Fill the graph budget, with shorter padding gaps at small batches."""
    max_batch_size = int(max_batch_size)
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}.")
    capture_limit = int(capture_limit)
    if capture_limit <= 0:
        raise ValueError(f"decode graph capture limit must be positive, got {capture_limit}.")
    if max_batch_size <= capture_limit:
        return list(range(1, max_batch_size + 1))
    if capture_limit == 1:
        return [max_batch_size]

    dense_limit = min(8, max(1, capture_limit // 4))
    sizes = list(range(1, dense_limit + 1))
    remaining_bucket_budget = capture_limit - dense_limit
    span = max_batch_size - dense_limit
    stride = (span + remaining_bucket_budget - 1) // remaining_bucket_budget
    slack = stride * remaining_bucket_budget - span
    current = dense_limit
    for _ in range(remaining_bucket_budget):
        gap = stride - min(slack, stride - 1)
        slack -= stride - gap
        current += gap
        sizes.append(current)
    return sizes


def _resolve_positive_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    *,
    name: str,
    default_factory: Callable[[], list[int]],
) -> list[int]:
    if value is None:
        sizes = default_factory()
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto"}:
            sizes = default_factory()
        else:
            try:
                sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be 'auto' or a comma-separated "
                    f"integer list, got {value!r}."
                ) from exc
    elif isinstance(value, int):
        sizes = [int(value)]
    elif isinstance(value, (list, tuple)):
        sizes = [int(item) for item in value]
    else:
        raise ValueError(
            f"{name} must be 'auto', an int, a list/tuple of ints, "
            f"or None, got {type(value).__name__}."
        )

    sizes = sorted(set(sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"{name} must contain positive integers, got {sizes}.")
    return sizes


def _resolve_decode_cuda_graph_capture_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    max_real_batch_size: int,
    capture_limit: int = 32,
) -> list[int]:
    sizes = _resolve_positive_sizes(
        value,
        name="decode_graph_capture_sizes",
        default_factory=lambda: _default_decode_cuda_graph_capture_sizes(
            max_real_batch_size, capture_limit
        ),
    )
    max_real_batch_size = int(max_real_batch_size)
    if sizes[-1] < max_real_batch_size:
        raise ValueError(
            "decode_graph_capture_sizes must cover max_decoding_seqs: "
            f"max capture size {sizes[-1]} < max_decoding_seqs "
            f"{max_real_batch_size}."
        )
    if max_real_batch_size not in sizes:
        raise ValueError(
            "decode_graph_capture_sizes must contain max_decoding_seqs as an "
            f"exact batch bucket, got max_decoding_seqs={max_real_batch_size} "
            f"and capture sizes {sizes}."
        )
    return sizes


def _select_decode_cuda_graph_batch_size(
    real_batch_size: int,
    capture_sizes: list[int] | tuple[int, ...],
) -> int:
    real_batch_size = int(real_batch_size)
    if real_batch_size <= 0:
        raise ValueError(
            f"decode batch size must be > 0, got {real_batch_size}."
        )
    sizes = sorted(set(int(size) for size in capture_sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(
            "decode_graph_capture_sizes must contain positive integers, "
            f"got {sizes}."
        )
    for size in sizes:
        if size >= real_batch_size:
            return size
    raise ValueError(
        "decode_cuda_graph capture sizes do not cover current decode batch: "
        f"batch_size={real_batch_size}, capture_sizes={sizes}."
    )


def _decode_cuda_graph_max_real_batch_size(
    *,
    max_decoding_seqs: int,
) -> int:
    """Return the largest decode batch that the scheduler can execute in one step."""

    return int(max_decoding_seqs)


def _resolve_decode_static_batch_capacity(
    capture_sizes: list[int] | tuple[int, ...],
    *,
    max_decoding_seqs: int,
) -> int:
    """Return the largest padded decode batch reachable by the scheduler."""

    max_real_batch_size = _decode_cuda_graph_max_real_batch_size(
        max_decoding_seqs=max_decoding_seqs,
    )
    return _select_decode_cuda_graph_batch_size(
        max_real_batch_size,
        capture_sizes,
    )


def build_decode_cuda_graph_startup_plan(config) -> list[tuple[int, int]]:
    """Capture each batch bucket once, with capacity for every request length."""
    batches = sorted(set(int(size) for size in config.decode_graph_capture_sizes))
    limit = int(config.decode_graph_startup_capture_limit)
    if len(batches) > limit:
        raise ValueError(
            "decode CUDA Graph startup capture must cover every batch bucket: "
            f"required={len(batches)}, limit={limit}."
        )
    return [(size, int(config.max_model_len)) for size in reversed(batches)]


def normalize_decode_cuda_graph(config) -> None:
    startup_capture_setting = config.decode_graph_startup_capture
    startup_capture_auto = startup_capture_setting is None
    if startup_capture_auto:
        config.decode_graph_startup_capture = bool(config.decode_graph)
    else:
        config.decode_graph_startup_capture = _coerce_bool_config(
            "decode_graph_startup_capture",
            startup_capture_setting,
        )

    if config.decode_graph_startup_capture_limit is None:
        config.decode_graph_startup_capture_limit = 32
    config.decode_graph_startup_capture_limit = int(
        config.decode_graph_startup_capture_limit
    )
    if config.decode_graph_startup_capture_limit <= 0:
        raise ValueError(
            "decode_graph_startup_capture_limit must be a positive integer, "
            f"got {config.decode_graph_startup_capture_limit}."
        )
    if config.decode_graph_startup_capture and not config.decode_graph:
        raise ValueError("decode_graph_startup_capture requires decode_graph=True.")
    if config.decode_graph and not config.decode_graph_startup_capture:
        raise ValueError(
            "decode_graph requires startup capture so the complete graph plan "
            "is sealed before serving."
        )
    if config.decode_graph_capture_sampling and not config.decode_graph:
        raise ValueError("decode_graph_capture_sampling requires decode_graph=True.")
    if not config.decode_graph:
        return

    if config.data_parallel_size > 1 and config.decode_graph_capture_sampling:
        raise ValueError(
            "DP attention does not support decode_graph_capture_sampling=True; "
            "sampling runs outside the captured decode graph."
        )
    if config.enable_prefix_caching and config.decode_graph_capture_sampling:
        raise ValueError(
            "prefix caching with decode_graph does not support "
            "decode_graph_capture_sampling=True yet."
        )
    if config.tensor_parallel_size > 1:
        if config.decode_graph_capture_sampling:
            raise ValueError(
                "decode_graph_capture_sampling is disabled when tensor_parallel_size > 1 "
                "because TP workers do not materialize rank-0 gathered logits."
            )
        if not is_tp_decode_cuda_graph_supported(config.sparse_method):
            supported = ", ".join(
                repr(method)
                for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS)
                if method and is_tp_decode_cuda_graph_supported(method)
            )
            raise ValueError(
                "decode_graph with tensor_parallel_size > 1 supports these methods only: "
                f"'', {supported}. DeltaKV is not supported."
            )
        if config.sparse_method and config.sparse_method not in {"rkv", "kvzip"}:
            log_once(
                "decode_graph with tensor_parallel_size > 1 uses TP-local sparse selection: "
                "each rank selects sparse tokens from its local heads/KV heads without cross-rank "
                "sparse-index aggregation, so sparse behavior is not guaranteed equivalent to TP=1 "
                "or global-head sparse selection.",
                level="WARNING",
            )
    elif not is_decode_cuda_graph_supported(config.sparse_method):
        supported = ", ".join(
            repr(method)
            for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS)
            if method
        )
        raise ValueError(f"decode_graph supports these methods only: '', {supported}.")

    capture_sizes_setting = config.decode_graph_capture_sizes
    max_real_batch_size = _decode_cuda_graph_max_real_batch_size(
        max_decoding_seqs=config.max_decoding_seqs,
    )
    config.decode_graph_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
        capture_sizes_setting,
        max_real_batch_size,
        int(config.decode_graph_startup_capture_limit),
    )

    startup_plan = build_decode_cuda_graph_startup_plan(config)
    path_summary = [{
        "path_id": decode_graph_path_id(config.sparse_method),
        "context_capacity": int(config.max_model_len),
    }]
    log_once(
        "Decode CUDA Graph startup precapture enabled "
        f"({'default' if startup_capture_auto else 'explicit'}): "
        f"budget={config.decode_graph_startup_capture_limit}, "
        f"planned_graphs={len(startup_plan)}, "
        f"batch_buckets={config.decode_graph_capture_sizes}, "
        f"topology_paths={path_summary}."
    )
