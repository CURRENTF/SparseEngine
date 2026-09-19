import torch

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["size_m"])
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    size_m,
    size_n,
    GATE_FIRST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    stride_input_m = stride_input_m.to(tl.int64)
    stride_output_m = stride_output_m.to(tl.int64)

    tid = tl.program_id(0)
    input_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    output_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)

    pid = tl.program_id(1)
    input_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    output_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    first_offsets = (
        input_m_offsets[:, None] * stride_input_m
        + input_n_offsets[None, :] * stride_input_n
    )
    second_offsets = first_offsets + size_n * stride_input_n
    gate_offsets = first_offsets if GATE_FIRST else second_offsets
    up_offsets = second_offsets if GATE_FIRST else first_offsets
    res_offsets = output_m_offsets[:, None] * stride_output_m + output_n_offsets[None, :] * stride_output_n

    up = tl.load(
        input_ptr + up_offsets,
        mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None],
        other=0.0,
    )
    gate = tl.load(
        input_ptr + gate_offsets,
        mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None],
        other=0.0,
    ).to(tl.float32)

    gate = gate / (1 + tl.exp(-gate))
    gate = gate.to(input_ptr.dtype.element_ty)

    tl.store(
        output_ptr + res_offsets,
        up * gate,
        mask=(output_n_offsets < size_n)[None, :] * (output_m_offsets < size_m)[:, None],
    )


def _resolve_silu_launch_config(size_m: int) -> tuple[int, int, int | None]:
    if int(size_m) <= 256:
        return 32, 128, 4
    return 128, 128, None


def silu_and_mul_fwd(
    input,
    *,
    gate_up_order: str = "gate_up",
    output: torch.Tensor | None = None,
):
    if gate_up_order not in {"gate_up", "up_gate"}:
        raise ValueError(
            "gate_up_order must be 'gate_up' or 'up_gate', "
            f"got {gate_up_order!r}."
        )
    stride_input_m = input.stride(0)
    stride_input_n = input.stride(1)
    size_m = input.shape[0]
    size_n = input.shape[-1] // 2
    if output is None:
        output = input[:, :size_n]
    elif tuple(output.shape) != (size_m, size_n):
        raise ValueError(
            f"SwiGLU output must have shape {(size_m, size_n)}, "
            f"got {tuple(output.shape)}."
        )
    stride_output_m = output.stride(0)
    stride_output_n = output.stride(1)
    BLOCK_M, BLOCK_N, num_warps = _resolve_silu_launch_config(size_m)
    grid = (
        triton.cdiv(size_m, BLOCK_M),
        triton.cdiv(size_n, BLOCK_N),
    )
    launch_kwargs = {} if num_warps is None else {"num_warps": num_warps}
    _silu_and_mul_kernel[grid](
        input,
        output,
        stride_input_m,
        stride_input_n,
        stride_output_m,
        stride_output_n,
        size_m,
        size_n,
        GATE_FIRST=gate_up_order == "gate_up",
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        **launch_kwargs,
    )
    return output
