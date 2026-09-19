import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["size_m"])
def _gelu_tanh_and_mul_kernel(
    input_ptr,
    stride_m,
    stride_n,
    size_m,
    size_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * stride_m + cols[None, :] * stride_n
    mask = (rows < size_m)[:, None] & (cols < size_n)[None, :]
    gate = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_ptr + offsets + size_n * stride_n, mask=mask, other=0.0)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    gate = gate * tl.sigmoid(2 * inner)
    tl.store(input_ptr + offsets, gate.to(input_ptr.dtype.element_ty) * up, mask=mask)


def gelu_tanh_and_mul_fwd(input):
    if not input.is_cuda or input.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("Gemma 4 GELU-and-multiply requires CUDA FP16 or BF16 input.")
    if input.ndim != 2 or not input.is_contiguous() or input.shape[1] % 2:
        raise ValueError(
            "Gemma 4 GELU-and-multiply requires contiguous [tokens, 2 * hidden], "
            f"got shape={tuple(input.shape)} contiguous={input.is_contiguous()}."
        )
    size_m, size_n = input.shape[0], input.shape[1] // 2
    block_m, block_n = (32, 128) if size_m <= 256 else (128, 128)
    _gelu_tanh_and_mul_kernel[
        (triton.cdiv(size_m, block_m), triton.cdiv(size_n, block_n))
    ](
        input,
        input.stride(0),
        input.stride(1),
        size_m,
        size_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4 if size_m <= 256 else 8,
    )
    return input[:, :size_n]
