"""Shared coercion helpers for configuration modules."""

import os
from typing import Any


_MISSING = object()


def _normalize_int_attr(config: Any, name: str, *, fallback: Any = _MISSING) -> int:
    value = getattr(config, name)
    if fallback is not _MISSING:
        value = value or fallback
    normalized = int(value)
    setattr(config, name, normalized)
    return normalized


def _normalize_float_attr(
    config: Any,
    name: str,
    *,
    fallback: Any = _MISSING,
) -> float:
    value = getattr(config, name)
    if fallback is not _MISSING:
        value = value or fallback
    normalized = float(value)
    setattr(config, name, normalized)
    return normalized


def _normalize_positive_int(
    config: Any,
    name: str,
    *,
    fallback: Any = _MISSING,
) -> int:
    value = _normalize_int_attr(config, name, fallback=fallback)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _normalize_positive_multiple(
    config: Any,
    name: str,
    *,
    multiple: int,
    fallback: Any = _MISSING,
) -> int:
    value = _normalize_int_attr(config, name, fallback=fallback)
    if value <= 0 or value % multiple != 0:
        raise ValueError(
            f"{name} must be a positive multiple of {multiple}, got {value}."
        )
    return value


def _coerce_bool_config(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or explicit true/false string, got {value!r}.")


def _coerce_optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw.isdecimal():
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        parsed = int(raw)
    else:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0 when set, got {parsed}.")
    return parsed


def _resolve_long_prefill_offload_threshold(configured: Any) -> int:
    raw = os.getenv("SPARSEENGINE_LONG_PREFILL_OFFLOAD_MIN_TOKENS")
    legacy_raw = os.getenv("SPARSEENGINE_DEFERRED_PREFILL_MIN_TOKENS")
    if raw is not None and legacy_raw is not None and raw != legacy_raw:
        raise ValueError(
            "SPARSEENGINE_LONG_PREFILL_OFFLOAD_MIN_TOKENS and "
            "SPARSEENGINE_DEFERRED_PREFILL_MIN_TOKENS are both set with different values."
        )
    value = raw if raw is not None else legacy_raw
    resolved = _coerce_optional_positive_int(
        "long_prefill_offload_threshold",
        configured if value is None else value,
    )
    if resolved is None:
        raise ValueError("long_prefill_offload_threshold must be a positive integer.")
    return int(resolved)
