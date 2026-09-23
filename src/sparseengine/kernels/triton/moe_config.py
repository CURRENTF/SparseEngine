from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from sparseengine.utils.device_name import device_name_contains


@dataclass(frozen=True)
class MoeGemmConfig:
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int
    swap_ab: bool = False

    def as_triton_kwargs(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
            "GROUP_SIZE_M": self.group_m,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


@dataclass(frozen=True)
class MoeGemmShape:
    hardware: str
    capability: tuple[int, int]
    dtype: torch.dtype
    top_k: int
    num_local_experts: int
    hidden_size: int
    intermediate_size: int


TUNED_TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


@lru_cache(maxsize=None)
def device_info(
    device_type: str,
    device_index: int,
) -> tuple[str, tuple[int, int]]:
    if device_type != "cuda":
        raise ValueError(f"MoE Triton config requires a CUDA device, got {device_type}.")
    return (
        torch.cuda.get_device_name(device_index),
        torch.cuda.get_device_capability(device_index),
    )


def _profiled_hardware(device_name: str) -> str:
    for keyword in ("H100", "H20"):
        if device_name_contains(device_name, keyword):
            return keyword.lower()
    return "unprofiled"


def token_bucket(num_tokens: int) -> int:
    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    return min(TUNED_TOKEN_COUNTS, key=lambda value: abs(value - num_tokens))


def _heuristic_config(
    *,
    num_tokens: int,
    top_k: int,
    output_size: int,
) -> MoeGemmConfig:
    assignments = num_tokens * top_k
    small = assignments <= 32
    return MoeGemmConfig(
        block_m=16,
        block_n=128 if small or output_size > 4096 else 64,
        block_k=32 if small else 64,
        group_m=8,
        num_warps=4,
        num_stages=4 if small else 3,
    )


_A = MoeGemmConfig(16, 64, 64, 8, 4, 3)
_B = MoeGemmConfig(16, 128, 64, 8, 4, 3)
_C = MoeGemmConfig(16, 128, 64, 8, 8, 3)
_D = MoeGemmConfig(16, 128, 32, 8, 4, 4)
_F = MoeGemmConfig(64, 64, 64, 8, 8, 3)
_G = MoeGemmConfig(16, 32, 64, 8, 4, 4)
_H = MoeGemmConfig(16, 32, 64, 8, 4, 3)
_I = MoeGemmConfig(16, 64, 64, 8, 4, 4)
_J = MoeGemmConfig(16, 64, 64, 8, 4, 2)
_K = MoeGemmConfig(16, 128, 64, 8, 4, 2)
_GLM_DECODE_32 = MoeGemmConfig(4, 32, 128, 8, 4, 3)
_GLM_DECODE_64 = MoeGemmConfig(64, 128, 64, 8, 8, 3)
_GLM_MID_BATCH = MoeGemmConfig(64, 128, 64, 1, 8, 3)
_GLM_LARGE_BATCH = MoeGemmConfig(128, 128, 64, 1, 8, 3)
_GLM_H100_LARGE_PREFILL = MoeGemmConfig(128, 256, 64, 1, 8, 4)
_GLM_EP2_TINY_BATCH = MoeGemmConfig(16, 64, 128, 1, 4, 3)
_GLM_EP2_SMALL_BATCH = MoeGemmConfig(16, 64, 128, 1, 4, 4)
_GLM_H20_PACKED_DECODE = MoeGemmConfig(16, 64, 128, 1, 4, 3)


def _glm_h20_packed_decode_config(
    shape: MoeGemmShape,
    *,
    num_tokens: int,
    stage: str,
) -> MoeGemmConfig | None:
    """Measured H20 BF16 decode profile with the shared expert packed in."""
    if (
        shape != MoeGemmShape("h20", (9, 0), torch.bfloat16, 5, 65, 2048, 1536)
        or stage not in {"w13", "w2"}
        or not 1 <= num_tokens <= 16
    ):
        return None
    # Use the measured interval directly: nearest-bucket lookup would also
    # apply the decode tuning to larger, unmeasured prefill batches.
    return _GLM_H20_PACKED_DECODE


def _glm_h100_tp2_config(
    shape: MoeGemmShape,
    *,
    num_tokens: int,
    stage: str,
) -> MoeGemmConfig | None:
    """Return measured BF16 configs for the GLM TP2 expert shape."""

    profiled_shapes = {
        MoeGemmShape(
            "h100",
            (9, 0),
            torch.bfloat16,
            4,
            64,
            2048,
            768,
        ),
        MoeGemmShape(
            "h100",
            (9, 0),
            torch.bfloat16,
            5,
            65,
            2048,
            768,
        ),
    }
    if (
        stage not in {"w13", "w2", "gate_up_swiglu"}
        or shape not in profiled_shapes
    ):
        return None
    # Offline H100 BF16 profile for routed-only TP2; packed shared experts
    # and the fused gate/up activation kernel retain their existing profiles.
    if (
        num_tokens >= 4096
        and shape.num_local_experts == 64
        and shape.top_k == 4
        and stage in {"w13", "w2"}
    ):
        return _GLM_H100_LARGE_PREFILL
    if num_tokens <= 32:
        return _GLM_DECODE_32
    if num_tokens <= 64:
        return _GLM_DECODE_64
    if num_tokens <= 512:
        return _GLM_MID_BATCH
    return _GLM_LARGE_BATCH


def _glm_sm90_tp2_ep2_config(
    shape: MoeGemmShape,
    *,
    num_tokens: int,
    stage: str,
) -> MoeGemmConfig | None:
    """Return measured BF16 configs for the GLM outer-TP2/EP2 shape."""

    profiled_shape = MoeGemmShape(
        shape.hardware,
        (9, 0),
        torch.bfloat16,
        4,
        32,
        2048,
        1536,
    )
    if (
        shape.hardware not in {"h100", "h20"}
        or stage not in {"w13", "w2"}
        or shape != profiled_shape
    ):
        return None
    if shape.hardware == "h20":
        if not 1 <= num_tokens <= 16:
            return None
        return _GLM_EP2_TINY_BATCH if num_tokens == 4 else _GLM_EP2_SMALL_BATCH
    if num_tokens >= 4096:
        return _GLM_H100_LARGE_PREFILL
    if num_tokens <= 4:
        return _GLM_EP2_TINY_BATCH
    if num_tokens <= 128:
        return _GLM_EP2_SMALL_BATCH
    return _GLM_LARGE_BATCH


def _stage_table(
    w13: tuple[MoeGemmConfig, ...],
    w2: tuple[MoeGemmConfig, ...],
) -> dict[str, dict[int, MoeGemmConfig]]:
    if len(w13) != len(TUNED_TOKEN_COUNTS) or len(w2) != len(TUNED_TOKEN_COUNTS):
        raise ValueError("Each tuned MoE stage must cover every token count.")
    return {
        "w13": dict(zip(TUNED_TOKEN_COUNTS, w13)),
        "w2": dict(zip(TUNED_TOKEN_COUNTS, w2)),
    }


# BF16 profiles are keyed by kernel-relevant hardware and GEMM shape rather
# than by model name.
_TUNED_CONFIGS = {
    MoeGemmShape("h20", (9, 0), torch.bfloat16, 8, 256, 2048, 512): {
        "w13": {1: _G, 2: _G, 4: _A, 8: _C},
        "w2": {1: _G, 2: _A, 4: _K, 8: _I},
    },
    MoeGemmShape("h20", (9, 0), torch.bfloat16, 8, 256, 2048, 256): {
        "w13": {1: _G, 2: _G, 4: _G, 8: _A},
        "w2": {1: _H, 2: _J, 4: _K, 8: _K},
    },
    MoeGemmShape("h20", (9, 0), torch.bfloat16, 8, 128, 2048, 512): {
        "w13": {1: _G, 2: _I, 4: _I, 8: _C},
        "w2": {1: _I, 2: _H, 4: _A, 8: _I},
    },
    MoeGemmShape("h20", (9, 0), torch.bfloat16, 8, 128, 2048, 768): _stage_table(
        (_D, _D, _D, _A, _A, _A, _B, _B, _B, _F, _F, _F),
        (_D, _D, _D, _B, _B, _B, _B, _B, _B, _F, _F, _F),
    ),
    MoeGemmShape("h20", (9, 0), torch.bfloat16, 8, 64, 2048, 768): _stage_table(
        (_D, _D, _D, _C, _B, _A, _A, _B, _B, _F, _F, _F),
        (_D, _D, _D, _B, _A, _B, _A, _A, _B, _F, _F, _F),
    ),
    MoeGemmShape("h20", (9, 0), torch.bfloat16, 8, 32, 2048, 768): _stage_table(
        (_D, _D, _D, _A, _C, _A, _B, _B, _B, _F, _F, _F),
        (_D, _D, _D, _A, _A, _B, _C, _C, _B, _F, _F, _F),
    ),
    MoeGemmShape("h100", (9, 0), torch.bfloat16, 8, 128, 2048, 768): _stage_table(
        (_D, _A, _A, _A, _A, _A, _A, _A, _F, _F, _F, _F),
        (_D, _A, _A, _A, _A, _A, _A, _A, _A, _A, _C, _C),
    ),
    MoeGemmShape("h100", (9, 0), torch.bfloat16, 8, 64, 2048, 768): _stage_table(
        (_A, _A, _A, _A, _A, _F, _F, _F, _F, _F, _F, _F),
        (_A, _A, _A, _A, _A, _A, _A, _A, _A, _A, _B, _B),
    ),
    MoeGemmShape("h100", (9, 0), torch.bfloat16, 8, 32, 2048, 768): _stage_table(
        (_A, _D, _D, _A, _A, _A, _A, _A, _A, _A, _F, _F),
        (_A, _D, _D, _A, _A, _A, _A, _A, _A, _A, _A, _B),
    ),
    MoeGemmShape("h100", (9, 0), torch.bfloat16, 8, 256, 2048, 512): {
        "w13": {1: _G, 2: _I, 4: _A, 8: _A},
        "w2": {1: _G, 2: _G, 4: _A, 8: _H},
    },
    MoeGemmShape("h100", (9, 0), torch.bfloat16, 8, 256, 2048, 256): {
        "w13": {1: _G, 2: _G, 4: _G, 8: _H},
        "w2": {1: _G, 2: _I, 4: _J, 8: _H},
    },
    MoeGemmShape("h100", (9, 0), torch.bfloat16, 8, 128, 2048, 512): {
        "w13": {1: _G, 2: _G, 4: _G, 8: _A},
        "w2": {1: _G, 2: _G, 4: _H, 8: _H},
    },
}


# The fused BF16 stage has two FP32 accumulators, so reusing a wide unfused W13
# tile can create register pressure.
_TUNED_GATE_UP_SWIGLU_CONFIGS = {
    MoeGemmShape(
        "h20",
        (9, 0),
        torch.bfloat16,
        8,
        256,
        2048,
        512,
    ): {1: _G, 2: _G, 4: _H, 8: _G},
    MoeGemmShape(
        "h20",
        (9, 0),
        torch.bfloat16,
        8,
        256,
        2048,
        256,
    ): {1: _G, 2: _G, 4: _G, 8: _H},
    MoeGemmShape(
        "h20",
        (9, 0),
        torch.bfloat16,
        8,
        128,
        2048,
        512,
    ): {1: _G, 2: _G, 4: _G, 8: _H},
    MoeGemmShape(
        "h100",
        (9, 0),
        torch.bfloat16,
        8,
        256,
        2048,
        512,
    ): {1: _G, 2: _G, 4: _H, 8: _H},
    MoeGemmShape(
        "h100",
        (9, 0),
        torch.bfloat16,
        8,
        256,
        2048,
        256,
    ): {1: _G, 2: _G, 4: _G, 8: _H},
    MoeGemmShape(
        "h100",
        (9, 0),
        torch.bfloat16,
        8,
        128,
        2048,
        512,
    ): {1: _G, 2: _G, 4: _G, 8: _H},
    MoeGemmShape(
        "h100",
        (9, 0),
        torch.bfloat16,
        8,
        64,
        2048,
        384,
    ): dict(
        zip(
            TUNED_TOKEN_COUNTS,
            (_G, _G, _G, _G, _H, _H, _A, _B, _H, _G, _B, _B),
        )
    ),
}


_FP8_N64_SWAP = MoeGemmConfig(16, 64, 128, 1, 4, 3, True)
_FP8_N64_SWAP_S2 = MoeGemmConfig(16, 64, 128, 1, 4, 2, True)
_FP8_N64_SWAP_S4 = MoeGemmConfig(16, 64, 128, 1, 4, 4, True)
_FP8_N64_SWAP_S5 = MoeGemmConfig(16, 64, 128, 1, 4, 5, True)
_FP8_N128 = MoeGemmConfig(16, 128, 128, 1, 4, 3)
_FP8_N128_SWAP_S2 = MoeGemmConfig(16, 128, 128, 1, 4, 2, True)
_FP8_N128_SWAP_S4 = MoeGemmConfig(16, 128, 128, 1, 4, 4, True)


# Qwen3.6-35B-A3B block-FP8 decode profiles. Unprofiled token buckets retain
# the explicit generic configuration.
_TUNED_FP8_ROUTED_CONFIGS = {
    MoeGemmShape(
        "h20",
        (9, 0),
        torch.float8_e4m3fn,
        8,
        256,
        2048,
        512,
    ): {
        "w13": {
            1: _FP8_N64_SWAP_S4,
            2: _FP8_N128_SWAP_S4,
            4: _FP8_N64_SWAP_S4,
            8: _FP8_N64_SWAP,
        },
        "w2": {
            1: _FP8_N64_SWAP,
            2: _FP8_N64_SWAP_S2,
            4: _FP8_N64_SWAP_S2,
            8: _FP8_N128_SWAP_S2,
        },
    },
    MoeGemmShape(
        "h20",
        (9, 0),
        torch.float8_e4m3fn,
        8,
        256,
        2048,
        256,
    ): {
        "w13": dict.fromkeys((1, 2, 4, 8), _FP8_N64_SWAP_S4),
        "w2": {
            1: _FP8_N64_SWAP,
            2: _FP8_N64_SWAP_S2,
            4: _FP8_N64_SWAP_S2,
            8: _FP8_N128_SWAP_S2,
        },
    },
    MoeGemmShape(
        "h20",
        (9, 0),
        torch.float8_e4m3fn,
        8,
        128,
        2048,
        512,
    ): {
        "w13": dict.fromkeys((1, 2, 4, 8), _FP8_N64_SWAP_S4),
        "w2": {
            1: _FP8_N64_SWAP,
            2: _FP8_N128_SWAP_S4,
            4: _FP8_N64_SWAP_S2,
            8: _FP8_N64_SWAP_S2,
        },
    },
    MoeGemmShape(
        "h100",
        (9, 0),
        torch.float8_e4m3fn,
        8,
        256,
        2048,
        512,
    ): {
        "w13": {
            1: _FP8_N64_SWAP_S5,
            2: _FP8_N64_SWAP_S4,
            4: _FP8_N64_SWAP,
            8: _FP8_N64_SWAP_S4,
        },
        "w2": {
            1: _FP8_N64_SWAP_S4,
            2: _FP8_N128_SWAP_S4,
            4: _FP8_N64_SWAP_S2,
            8: _FP8_N64_SWAP,
        },
    },
    MoeGemmShape(
        "h100",
        (9, 0),
        torch.float8_e4m3fn,
        8,
        256,
        2048,
        256,
    ): {
        "w13": {
            1: _FP8_N64_SWAP_S4,
            2: _FP8_N64_SWAP_S5,
            4: _FP8_N64_SWAP_S4,
            8: _FP8_N64_SWAP,
        },
        "w2": {
            1: _FP8_N64_SWAP,
            2: _FP8_N64_SWAP_S2,
            4: _FP8_N64_SWAP_S2,
            8: _FP8_N64_SWAP_S2,
        },
    },
    MoeGemmShape(
        "h100",
        (9, 0),
        torch.float8_e4m3fn,
        8,
        128,
        2048,
        512,
    ): {
        "w13": dict.fromkeys((1, 2, 4, 8), _FP8_N64_SWAP_S4),
        "w2": {
            1: _FP8_N64_SWAP_S4,
            2: _FP8_N64_SWAP,
            4: _FP8_N64_SWAP_S2,
            8: _FP8_N64_SWAP_S2,
        },
    },
}


@lru_cache(maxsize=None)
def _resolve_moe_gemm_config(
    dtype: torch.dtype,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    stage: str,
    device_name: str,
    capability: tuple[int, int],
) -> MoeGemmConfig:
    shape = MoeGemmShape(
        hardware=_profiled_hardware(device_name),
        capability=capability,
        dtype=dtype,
        top_k=top_k,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    glm_config = _glm_h20_packed_decode_config(
        shape,
        num_tokens=num_tokens,
        stage=stage,
    ) or _glm_sm90_tp2_ep2_config(
        shape,
        num_tokens=num_tokens,
        stage=stage,
    ) or _glm_h100_tp2_config(
        shape,
        num_tokens=num_tokens,
        stage=stage,
    )
    if glm_config is not None:
        return glm_config
    table = _TUNED_CONFIGS.get(shape)
    if stage == "gate_up_swiglu":
        fused_table = _TUNED_GATE_UP_SWIGLU_CONFIGS.get(shape)
        if fused_table is not None:
            tuned = fused_table.get(token_bucket(num_tokens))
            if tuned is not None:
                return tuned
        assignments = num_tokens * top_k
        return MoeGemmConfig(
            block_m=16,
            block_n=32 if assignments <= 256 else 64,
            block_k=64,
            group_m=8,
            num_warps=4,
            num_stages=4 if assignments <= 64 else 3,
        )
    if table is not None:
        tuned = table[stage].get(token_bucket(num_tokens))
        if tuned is not None:
            return tuned

    output_size = 2 * intermediate_size if stage == "w13" else hidden_size
    return _heuristic_config(
        num_tokens=num_tokens,
        top_k=top_k,
        output_size=output_size,
    )


def resolve_moe_gemm_config(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    stage: str,
    device_name: str | None = None,
    device_capability: tuple[int, int] | None = None,
) -> MoeGemmConfig:
    if stage not in {"w13", "gate_up_swiglu", "w2"}:
        raise ValueError(
            "MoE GEMM stage must be 'w13', 'gate_up_swiglu', or 'w2', "
            f"got {stage!r}."
        )
    if device_name is None:
        device_name = torch.cuda.get_device_name()
    if device_capability is None:
        device_capability = torch.cuda.get_device_capability()
    return _resolve_moe_gemm_config(
        dtype,
        int(num_tokens),
        int(top_k),
        int(num_local_experts),
        int(hidden_size),
        int(intermediate_size),
        stage,
        device_name,
        device_capability,
    )


def resolve_fp8_routed_gemm_config(
    *,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    stage: str,
    device_name: str | None = None,
    device_capability: tuple[int, int] | None = None,
) -> MoeGemmConfig:
    if stage not in {"w13", "w2"}:
        raise ValueError(f"FP8 routed GEMM stage must be 'w13' or 'w2', got {stage!r}.")
    if device_name is None:
        device_name = torch.cuda.get_device_name()
    if device_capability is None:
        device_capability = torch.cuda.get_device_capability()
    shape = MoeGemmShape(
        _profiled_hardware(device_name),
        device_capability,
        torch.float8_e4m3fn,
        int(top_k),
        int(num_local_experts),
        int(hidden_size),
        int(intermediate_size),
    )
    tuned = _TUNED_FP8_ROUTED_CONFIGS.get(shape, {}).get(stage, {}).get(
        token_bucket(num_tokens)
    )
    if tuned is not None:
        return tuned

    # Match vLLM's block-FP8 defaults for unprofiled shapes. In particular,
    # large prefill batches need wider M tiles and grouped program ordering;
    # the decode-oriented 16x128 fallback leaves the H100 underoccupied.
    # Source: vllm-project/vllm@ffd46bfab2128bb84146050e98b51a617c6575ab.
    block_m = 16 if num_tokens <= 64 else 64
    block_n = 64 if num_tokens <= 8 else 128
    return MoeGemmConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=128,
        group_m=1 if num_tokens <= 16 else 32,
        num_warps=4,
        num_stages=4 if num_tokens <= 4 else 3,
        swap_ab=device_capability == (9, 0) and block_m < 64,
    )
