"""Assemble MLA keys from projected non-RoPE keys and shared RoPE keys."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_keys(
    K, R, O, size,
    HEADS: tl.constexpr, NOPE: tl.constexpr, ROPE: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    RS0: tl.constexpr, RS1: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = NOPE + ROPE
    dim = offsets % width
    head = offsets // width % HEADS
    token = offsets // (width * HEADS)
    valid = offsets < size
    kn = tl.load(
        K + token * KS0 + head * KS1 + dim * KS2,
        mask=valid & (dim < NOPE), other=0,
    )
    kr = tl.load(
        R + token * RS0 + (dim - NOPE) * RS1,
        mask=valid & (dim >= NOPE), other=0,
    )
    tl.store(O + offsets, tl.where(dim < NOPE, kn, kr), mask=valid)


def pack_mla_keys(non_rope: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Copy strided projected keys and broadcast RoPE in a single launch.

    Inputs remain unmodified. Values may continue to alias the joint KV
    projection; only the required contiguous key tensor is materialized.
    """
    if non_rope.ndim != 3 or rope.ndim != 2 or non_rope.shape[0] != rope.shape[0]:
        raise ValueError("MLA key packing requires [tokens, heads, nope] and [tokens, rope]")
    if non_rope.device != rope.device:
        raise ValueError("MLA key packing inputs must share a device")
    tokens, heads, nope = non_rope.shape
    width = nope + rope.shape[1]
    output = torch.empty((tokens, heads, width), device=non_rope.device, dtype=non_rope.dtype)
    if output.numel():
        _pack_keys[(triton.cdiv(output.numel(), 1024),)](
            non_rope, rope, output, output.numel(), heads, nope, rope.shape[1],
            *non_rope.stride(), *rope.stride(), 1024,
        )
    return output
