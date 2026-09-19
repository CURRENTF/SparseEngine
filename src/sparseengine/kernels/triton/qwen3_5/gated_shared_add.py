from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _gated_shared_add_kernel(
    routed_ptr,
    shared_ptr,
    gate_ptr,
    output_ptr,
    gate_stride,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < hidden_size
    routed = tl.load(routed_ptr + row * hidden_size + offsets, mask=mask)
    shared = tl.load(shared_ptr + row * hidden_size + offsets, mask=mask)
    gate = tl.load(gate_ptr + row * gate_stride).to(tl.float32)
    gate = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16)
    tl.store(
        output_ptr + row * hidden_size + offsets,
        routed + gate * shared,
        mask=mask,
    )


def triton_gated_shared_add(
    routed: torch.Tensor,
    shared: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(routed)
    hidden_size = int(routed.shape[1])
    block_size = 512
    _gated_shared_add_kernel[(routed.shape[0], triton.cdiv(hidden_size, block_size))](
        routed,
        shared,
        gate_logits,
        output,
        gate_logits.stride(0),
        hidden_size=hidden_size,
        block_size=block_size,
        num_warps=4,
    )
    return output
