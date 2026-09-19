from __future__ import annotations

from typing import Literal


PREFILL_EXECUTION_CHUNKED = "chunked"
PREFILL_EXECUTION_FULL = "full"
PREFILL_EXECUTION_RAW_OFFLOAD = "raw_offload"

PrefillExecutionMode = Literal["chunked", "full", "raw_offload"]

SUPPORTED_PREFILL_EXECUTION_MODES = frozenset(
    {
        PREFILL_EXECUTION_CHUNKED,
        PREFILL_EXECUTION_FULL,
        PREFILL_EXECUTION_RAW_OFFLOAD,
    }
)


def validate_prefill_execution_mode(mode: str) -> PrefillExecutionMode:
    normalized = str(mode)
    if normalized not in SUPPORTED_PREFILL_EXECUTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_PREFILL_EXECUTION_MODES))
        raise ValueError(
            f"Unsupported prefill execution mode={mode!r}; expected one of {supported}."
        )
    return normalized  # type: ignore[return-value]
