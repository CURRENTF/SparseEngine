from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Literal

import torch

from sparseengine.kernels.external.flashinfer.support import flashinfer_kernel_support
from sparseengine.kernels.external.support import ExternalKernelContractError


_GDN_PREFILL_REQUIRED_ARGUMENTS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "initial_state",
    "output_final_state",
    "cu_seqlens",
    "use_qk_l2norm_in_kernel",
    "use_cp",
)


_GDN_PREFILL_KERNEL_BY_CAPABILITY = {
    (9, 0): "chunk_gated_delta_rule_sm90",
    (10, 0): "chunk_gated_delta_rule_sm100",
    (10, 3): "chunk_gated_delta_rule_sm100",
    (12, 0): "chunk_gated_delta_rule_sm120",
    (12, 1): "chunk_gated_delta_rule_sm120",
}
FLASHINFER_GDN_PREFILL_CAPABILITIES = frozenset(
    _GDN_PREFILL_KERNEL_BY_CAPABILITY
)


@lru_cache(maxsize=None)
def _gdn_prefill_op(compute_capability: tuple[int, int]):
    feature = f"GDN prefill on SM{compute_capability[0]}{compute_capability[1]}"
    _, reason = flashinfer_kernel_support(feature)
    try:
        module = importlib.import_module("flashinfer.gdn_prefill")
        function = getattr(module, "chunk_gated_delta_rule")
        parameters = inspect.signature(function).parameters
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"failed to load: {type(error).__name__}: {error}",
        ) from error
    if not callable(function):
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            "flashinfer.gdn_prefill.chunk_gated_delta_rule is not callable",
        )
    missing_arguments = sorted(
        set(_GDN_PREFILL_REQUIRED_ARGUMENTS).difference(parameters)
    )
    positional_only_arguments = sorted(
        name
        for name in _GDN_PREFILL_REQUIRED_ARGUMENTS
        if name in parameters
        and parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY
    )
    if missing_arguments or positional_only_arguments:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            "unsupported schema: "
            f"missing keyword parameters={missing_arguments}, "
            f"positional-only parameters={positional_only_arguments}",
        )
    kernel_name = _GDN_PREFILL_KERNEL_BY_CAPABILITY.get(compute_capability)
    if kernel_name is None:
        raise ValueError(
            f"Unsupported FlashInfer GDN compute capability {compute_capability}."
        )
    if not callable(getattr(module, kernel_name, None)):
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"{kernel_name} is unavailable",
        )
    return function, reason


def flashinfer_gdn_prefill_support(
    compute_capability: tuple[int, int],
) -> tuple[bool, str]:
    _, reason = _gdn_prefill_op(compute_capability)
    return True, reason


def flashinfer_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_exp: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    use_cp: Literal["auto"] | bool = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_capability = torch.cuda.get_device_capability(q.device)
    function, _ = _gdn_prefill_op(compute_capability)
    result = function(
        q=q,
        k=k,
        v=v,
        g=g_exp,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        # The public non-CP paths cast integer offsets, but the CP dispatcher
        # in FlashInfer 0.6.18 passes them through to kernels requiring int64.
        # Normalize here so both dispatch paths accept the engine's int32 view.
        cu_seqlens=cu_seqlens.to(dtype=torch.int64).contiguous(),
        use_qk_l2norm_in_kernel=False,
        use_cp=use_cp,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "FlashInfer GDN prefill violated its output contract: expected "
            "(output, final_state)."
        )
    if result[1].dtype != torch.float32:
        raise RuntimeError(
            "FlashInfer GDN prefill violated its state contract: expected "
            f"FP32 final_state, got {result[1].dtype}."
        )
    return result


__all__ = [
    "FLASHINFER_GDN_PREFILL_CAPABILITIES",
    "flashinfer_chunk_gated_delta_rule",
    "flashinfer_gdn_prefill_support",
]
