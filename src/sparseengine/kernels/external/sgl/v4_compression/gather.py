"""V4 paged gather from SGLang 3ed56a3; see NOTICE and LICENSE."""
import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)



def gather_packed_kv(storage, slots, page_size, output):
    if slots.numel() == 0:
        return output
    raw = storage.flatten()
    _dequantize_k_cache_paged_kernel[(slots.numel(),)](
        output, raw.view(torch.float8_e4m3fn), raw.view(torch.bfloat16), raw,
        slots, output.stride(0), storage.stride(0), page_size, 448, 64, 64, 7,
        576, 8, page_size * 576,
    )
    return output
