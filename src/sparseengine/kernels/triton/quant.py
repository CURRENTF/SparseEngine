import triton
import triton.language as tl
import torch
import numpy as np

# --- 下面抄自 baselines/kivi/quant/new_pack.py 的 Triton Kernel ---

SUPPORTED_PACK_BITS = (2, 4, 8)


def _features_per_int(bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_PACK_BITS:
        raise ValueError(f"Packed quantization supports bits={SUPPORTED_PACK_BITS}, got {bits}.")
    return 32 // bits


@triton.jit
def _round_half_to_even(x):
    floored = tl.floor(x)
    frac = x - floored
    up = floored + 1.0
    floor_i = floored.to(tl.int32)
    floor_is_even = (floor_i & 1) == 0
    return tl.where(frac > 0.5, up, tl.where(frac < 0.5, floored, tl.where(floor_is_even, floored, up)))


@triton.jit
def _quantize_pack_2d_int4_grouped_kernel(
    data_ptr,
    code_ptr,
    scale_ptr,
    mn_ptr,
    stride_data_n,
    stride_data_d,
    stride_code_n,
    stride_code_p,
    stride_scale_n,
    stride_scale_g,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACKS_PER_GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offs = tl.arange(0, BLOCK_G)
    group_offsets = group * GROUP_SIZE + offs
    mask = offs < GROUP_SIZE
    values = tl.load(
        data_ptr + row * stride_data_n + group_offsets * stride_data_d,
        mask=mask,
        other=0.0,
    )
    mx = tl.max(values, axis=0)
    mn = tl.min(values, axis=0)
    scale = (mx - mn) / 15.0
    scale_addr = scale_ptr + row * stride_scale_n + group * stride_scale_g
    mn_addr = mn_ptr + row * stride_scale_n + group * stride_scale_g
    tl.store(scale_addr, scale)
    tl.store(mn_addr, mn)
    scale = tl.load(scale_addr)
    mn = tl.load(mn_addr)

    normalized = (values - mn) / (scale + 1.0e-6)
    rounded = _round_half_to_even(tl.minimum(tl.maximum(normalized, 0.0), 15.0)).to(tl.int32)
    for pack_id in tl.static_range(0, PACKS_PER_GROUP):
        base = pack_id * 8
        packed = tl.zeros((), dtype=tl.int32)
        for j in tl.static_range(0, 8):
            q = tl.max(tl.where(offs == base + j, rounded, 0), axis=0)
            packed = packed | (q << (j * 4))
        tl.store(
            code_ptr + row * stride_code_n + (group * PACKS_PER_GROUP + pack_id) * stride_code_p,
            packed,
        )


def triton_quantize_and_pack_2d_int4_grouped(data: torch.Tensor, group_size: int):
    if data.dim() != 2:
        raise ValueError(f"2D int4 quantization expects rank-2 input, got shape={tuple(data.shape)}.")
    if not data.is_cuda:
        raise ValueError("2D int4 quantization expects a CUDA tensor.")
    group_size = int(group_size)
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"2D int4 quantization requires group_size to be a positive multiple of 8, got {group_size}.")
    n, d = data.shape
    if d % group_size != 0:
        raise ValueError(f"2D int4 quantization requires D divisible by group_size, got D={d}, group={group_size}.")
    if d % 8 != 0:
        raise ValueError(f"2D int4 quantization requires D divisible by 8, got D={d}.")
    data = data.contiguous()
    num_groups = d // group_size
    packs_per_group = group_size // 8
    code = torch.empty((n, d // 8), device=data.device, dtype=torch.int32)
    scale = torch.empty((n, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((n, num_groups), device=data.device, dtype=data.dtype)
    block_g = triton.next_power_of_2(group_size)
    with torch.cuda.device(data.device):
        _quantize_pack_2d_int4_grouped_kernel[(n, num_groups)](
            data,
            code,
            scale,
            mn,
            data.stride(0),
            data.stride(1),
            code.stride(0),
            code.stride(1),
            scale.stride(0),
            scale.stride(1),
            D=d,
            GROUP_SIZE=group_size,
            PACKS_PER_GROUP=packs_per_group,
            BLOCK_G=block_g,
            num_warps=1,
        )
    return code, scale, mn


@triton.jit
def _dequantize_2d_int4_grouped_kernel(
    packed_ptr,
    scale_ptr,
    mn_ptr,
    out_ptr,
    stride_packed_n,
    stride_packed_p,
    stride_scale_n,
    stride_scale_g,
    stride_out_n,
    stride_out_d,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    packed = tl.load(
        packed_ptr + row * stride_packed_n + (offs // 8) * stride_packed_p,
        mask=mask,
        other=0,
    )
    q = ((packed >> ((offs % 8) * 4)) & 15).to(tl.float32)
    groups = offs // GROUP_SIZE
    scale = tl.load(
        scale_ptr + row * stride_scale_n + groups * stride_scale_g,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mn = tl.load(
        mn_ptr + row * stride_scale_n + groups * stride_scale_g,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = q * scale + mn
    tl.store(out_ptr + row * stride_out_n + offs * stride_out_d, out, mask=mask)


def triton_dequantize_2d_int4_grouped(
    packed: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    group_size: int,
    output_dim: int,
):
    if packed.dim() != 2 or scale.dim() != 2 or mn.dim() != 2:
        raise ValueError(
            "2D int4 dequantization expects rank-2 packed/scale/min tensors, "
            f"got packed={tuple(packed.shape)}, scale={tuple(scale.shape)}, mn={tuple(mn.shape)}."
        )
    if not (packed.is_cuda and scale.is_cuda and mn.is_cuda):
        raise ValueError("2D int4 dequantization expects CUDA tensors.")
    n = int(packed.shape[0])
    output_dim = int(output_dim)
    group_size = int(group_size)
    if output_dim <= 0 or output_dim % 8 != 0:
        raise ValueError(f"2D int4 dequantization requires output_dim divisible by 8, got {output_dim}.")
    if group_size <= 0 or output_dim % group_size != 0:
        raise ValueError(
            "2D int4 dequantization requires output_dim divisible by group_size, "
            f"got output_dim={output_dim}, group_size={group_size}."
        )
    if int(packed.shape[1]) != output_dim // 8:
        raise ValueError(
            "2D int4 dequantization packed width mismatch: "
            f"packed={packed.shape[1]}, expected={output_dim // 8}."
        )
    if tuple(scale.shape) != (n, output_dim // group_size) or tuple(mn.shape) != tuple(scale.shape):
        raise ValueError(
            "2D int4 dequantization scale/min shape mismatch: "
            f"scale={tuple(scale.shape)}, mn={tuple(mn.shape)}, expected={(n, output_dim // group_size)}."
        )
    packed = packed.contiguous()
    scale = scale.contiguous()
    mn = mn.contiguous()
    out = torch.empty((n, output_dim), device=packed.device, dtype=scale.dtype)
    block_d = triton.next_power_of_2(output_dim)
    with torch.cuda.device(packed.device):
        _dequantize_2d_int4_grouped_kernel[(n,)](
            packed,
            scale,
            mn,
            out,
            packed.stride(0),
            packed.stride(1),
            scale.stride(0),
            scale.stride(1),
            out.stride(0),
            out.stride(1),
            D=output_dim,
            GROUP_SIZE=group_size,
            BLOCK_D=block_d,
            num_warps=8,
        )
    return out

@triton.jit
def _pack_along_last_dim(
    bits: tl.constexpr,
    intensor_ptr,
    code_ptr,
    N,
    num_feats: tl.constexpr,
    feat_per_int: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    num_int_per_y_dim = num_feats // feat_per_int
    bid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
    packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    for i in range(feat_per_int):
        ptr = block_start + i
        element = tl.load(ptr, mask=offs_N<N, other=0.)
        element = element << (i * bits)
        # Combine the value using bitwise OR
        packed = packed | element
    tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)

@triton.jit(do_not_specialize=["total_elements", "N", "num_groups"])
def _minmax_along_last_dim(
    x_ptr,
    mn_ptr, mx_ptr,
    total_elements,
    N,
    num_groups,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    bid = tl.program_id(axis=0)
    offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
    # Each row is a complete group; use one predicate for the whole row.
    mask = offsets_b[:, None] < total_elements // group_size
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    mx_val = tl.max(x, axis=1)
    mn_val = tl.min(x, axis=1)
    tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
    tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)

# --- 下面是封装逻辑，也保持与 KIVI 一致 ---

def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
    assert len(data.shape) == 4
    feat_per_int = _features_per_int(bit)
    shape = data.shape
    B, nh, D, T = shape
    if T % feat_per_int != 0:
        raise ValueError(
            f"Packed int{bit} quantization requires the last dimension to be divisible by "
            f"{feat_per_int}, got {T}."
        )
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B * nh * D, num_groups, group_size)
    scale_mn_shape = B, nh, D, num_groups
    # Quantize
    data = data.reshape(new_shape)
    mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
    with torch.cuda.device(data.device):
        _minmax_along_last_dim[grid](data, mn, mx,
                             data.numel(), data.shape[0], num_groups, group_size,
                             BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
    scale = (mx - mn) / (2 ** bit - 1)
    data = data - mn.unsqueeze(-1)
    data.div_(scale.unsqueeze(-1) + 1e-6)
    data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
    data = data.view(-1, T)
    packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
    code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
    with torch.cuda.device(data.device):
        _pack_along_last_dim[grid](bit, data, code, data.shape[0],
                                data.shape[1], feat_per_int,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                num_warps=8)
    return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

def unpack_tensor(v_code: torch.Tensor, bits: int, pack_dim: int):
    """
    KIVI 原版解包逻辑 (基于 PyTorch 向量化索引)
    """
    feat_per_int = _features_per_int(bits)
    shape = v_code.shape
    new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
    unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
    i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
    j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
    num = 0xFF >> (8 - bits)
    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i
    packed_indices = tuple(packed_indices)
    if pack_dim == 2:
        unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
    elif pack_dim == 3:
        unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
    else:
        raise NotImplementedError
    return unpacked_v_code

def unpack_quantized_to_16bit(packed, scale, mn, group_size, bits: int):
    """
    适配 ClusterCompressedKVCache 的包装函数。
    """
    bits = int(bits)
    feat_per_int = _features_per_int(bits)
    # 这里的 packed 形状通常是 (B, nh, num_imp, D//feat_per_int)
    # 我们调用 KIVI 的 unpack_tensor，它在 dim=3 上打包
    unpacked = unpack_tensor(packed, bits=bits, pack_dim=3)
    
    # 反量化
    # scale/mn 形状是 (B, nh, num_imp, 1) 或者 (B, nh, num_imp, num_groups)
    # 如果是 Per-token 量化，num_groups = 1
    if scale.shape[-1] == 1:
        return unpacked.to(scale.dtype) * scale + mn
    else:
        # Group-wise 逻辑
        B, nh, num_imp, D = unpacked.shape
        if D % group_size != 0:
            raise ValueError(f"Unpacked dimension {D} must be divisible by group_size {group_size}.")
        num_groups = D // group_size
        res = unpacked.view(B, nh, num_imp, num_groups, group_size)
        res = res * scale.unsqueeze(-1) + mn.unsqueeze(-1)
        return res.view(B, nh, num_imp, D)
