"""Model-path and parallelism validation."""

import os

from sparseengine.configs.common import _coerce_bool_config


def normalize_platform(config) -> None:
    if not os.path.isdir(config.model):
        raise FileNotFoundError(f"Model directory does not exist: {config.model}")
    config.tensor_parallel_size = int(config.tensor_parallel_size)
    config.expert_parallel_size = int(config.expert_parallel_size)
    config.data_parallel_size = int(config.data_parallel_size)
    config.weight_loading_workers = int(config.weight_loading_workers)
    if config.async_scheduling is not None:
        config.async_scheduling = _coerce_bool_config("async_scheduling", config.async_scheduling)
    config.async_max_inflight = int(config.async_max_inflight)
    if not 1 <= config.tensor_parallel_size <= 8:
        raise ValueError(f"tensor_parallel_size must be in [1, 8], got {config.tensor_parallel_size}.")
    if config.expert_parallel_size <= 0:
        raise ValueError(
            f"expert_parallel_size must be positive, got {config.expert_parallel_size}."
        )
    if config.data_parallel_size <= 0:
        raise ValueError(
            f"data_parallel_size must be positive, got {config.data_parallel_size}."
        )
    if config.weight_loading_workers <= 0:
        raise ValueError(
            "weight_loading_workers must be positive, "
            f"got {config.weight_loading_workers}."
        )
    config.gpu_memory_utilization = float(config.gpu_memory_utilization)
    config.decode_graph = _coerce_bool_config(
        "decode_graph",
        config.decode_graph,
    )
    config.decode_graph_capture_sampling = _coerce_bool_config(
        "decode_graph_capture_sampling",
        config.decode_graph_capture_sampling,
    )
