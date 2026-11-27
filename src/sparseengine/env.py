"""Read SparseEngine environment settings when requested."""

import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def env_is_one(name: str) -> bool:
    return os.environ.get(name) == "1"


def env_bool(name: str, default: bool = False, *, strict: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    if strict:
        # Keep the existing error text for the force-eager flag.
        if name == "SPARSEENGINE_DELTAKV_CUDAGRAPH_FORCE_EAGER":
            raise ValueError(f"{name} must be a boolean value, got {value!r}.")
        raise ValueError(
            f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}."
        )
    return default


def env_truthy(name: str) -> bool:
    """Read a legacy opt-in that is case insensitive and whitespace sensitive."""
    return env_str(name).lower() in _TRUE_VALUES


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_optional_str(name: str) -> str | None:
    return os.environ.get(name)


def debug_slots_enabled() -> bool:
    return env_is_one("SPARSEENGINE_DEBUG_SLOTS")


def debug_runtime_enabled() -> bool:
    return env_is_one("SPARSEENGINE_DEBUG_RUNTIME")


def graph_capture_diagnostics_enabled() -> bool:
    return env_is_one("SPARSEENGINE_GRAPH_CAPTURE_DIAGNOSTICS")


def debug_prefill_bounds_enabled() -> bool:
    return env_is_one("SENGINE_DEBUG_PREFILL_BOUNDS")


def debug_decode_bounds_enabled() -> bool:
    return env_is_one("SENGINE_DEBUG_DECODE_BOUNDS")


def allow_large_prefill_chunk() -> bool:
    return env_is_one("SPARSEENGINE_ALLOW_LARGE_PREFILL_CHUNK")


def profiler_reset_after_warmup() -> bool:
    return env_is_one("SPARSEENGINE_PROFILER_RESET_AFTER_WARMUP")


def tp_rpc_ack_timeout_s() -> float:
    return env_float("SPARSEENGINE_TP_RPC_ACK_TIMEOUT_S", 30.0)


def tp_rpc_status_timeout_s() -> float:
    return env_float("SPARSEENGINE_TP_RPC_STATUS_TIMEOUT_S", 300.0)


def dp_rpc_timeout_s() -> float:
    return env_float("SPARSEENGINE_DP_RPC_TIMEOUT_S", 1800.0)


def engine_exit_timeout_s() -> float:
    return env_float("SPARSEENGINE_ENGINE_EXIT_TIMEOUT_S", 10.0)


def worker_join_timeout_s() -> float:
    return env_float("SPARSEENGINE_WORKER_JOIN_TIMEOUT_S", 5.0)
