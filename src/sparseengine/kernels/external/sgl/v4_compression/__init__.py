"""Pinned SGL computation only; no SGLang engine/runtime dependency."""

from functools import lru_cache
from pathlib import Path

import torch


_ROOT = Path(__file__).parent
_TYPES = {torch.float32: "float", torch.bfloat16: "__nv_bfloat16"}


@lru_cache(maxsize=None)
def _load(kind: str, head_dim: int, input_dtype: torch.dtype,
          output_dtype: torch.dtype, architecture: tuple[int, int]):
    from tvm_ffi.cpp import load_inline

    if kind == "plan":
        header = "common.cuh"
        exports = "TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, sglang::plan_compress_prefill);"
    elif kind == "norm_rope":
        header = "fused_norm_rope.cuh"
        kernel = f"sglang::FusedNormRopeKernel<{_TYPES[input_dtype]}, {head_dim}, 64, false>"
        exports = f"using Kernel = {kernel};\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, Kernel::forward);"
    elif kind == "query":
        header = "main_norm_rope.cuh"
        kernel = "sglang::FusedQNormRopeKernel<__nv_bfloat16, 512, 64, false>"
        exports = f"using Kernel = {kernel};\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, Kernel::forward);"
    elif kind.startswith("store_"):
        header = "store.cuh"
        page_size = int(kind.split("_")[1])
        kernel = (f"sglang::FusedStoreCacheFlashMLAKernel<{_TYPES[input_dtype]}, int32_t, "
                  f"{page_size}, sglang::deepseek_v4::KVLayout::V4, false>")
        exports = f"using Kernel = {kernel};\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, Kernel::run);"
    else:
        ratio = int(kind)
        header = f"c{ratio}.cuh"
        kernel = f"sglang::FlashCompress{ratio}Kernel<{head_dim}, {_TYPES[input_dtype]}, {_TYPES[output_dtype]}, false>"
        exports = (f"using Kernel = {kernel};\n"
                   "TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode, Kernel::run_decode);\n"
                   "TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill, Kernel::run_prefill);")
    source = f'#include <tvm/ffi/function.h>\n#include "deepseek_v4/{header}"\n{exports}\n'
    arch = f"{architecture[0]}{architecture[1]}"
    return load_inline(
        name=f"sparseengine_sgl_v4_{kind}_{head_dim}_{_TYPES[input_dtype]}_{_TYPES[output_dtype]}_sm{arch}",
        cuda_sources=source,
        extra_cuda_cflags=["-std=c++20", "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", f"-arch=sm_{arch}",
                           f"-DSGL_CUDA_ARCH={architecture[0] * 100 + architecture[1] * 10}"],
        extra_include_paths=[str(_ROOT / "include"), str(_ROOT / "csrc")],
    )


class SGLCompressionKernels:
    """Prepared upstream kernels; caller owns carry, outputs and plan buffers."""

    def __init__(self, *, ratio: int, head_dim: int,
                 architecture: tuple[int, int]):
        if ratio not in (4, 128) or head_dim not in (128, 512):
            raise ValueError("SGL V4 compression requires ratio 4/128 and D128/D512")
        self.ratio = ratio
        self.head_dim = head_dim
        self.module = _load(str(ratio), head_dim, torch.float32, torch.float32, architecture)
        self.planner = _load("plan", 0, torch.float32, torch.float32, architecture)
        self.norm_rope = _load("norm_rope", head_dim, torch.float32, torch.float32, architecture)

    def prefill_plan(self, seq_lens, extend_lens, *, num_tokens, device):
        if seq_lens.device.type != "cpu" or extend_lens.device.type != "cpu":
            raise ValueError("Prefill compression planning consumes scheduler CPU lengths")
        plans = torch.empty(2, num_tokens, 16, dtype=torch.uint8, pin_memory=True)
        n_compress, n_write = self.planner.plan(
            extend_lens.to(torch.int64), seq_lens.to(torch.int64), plans[0], plans[1],
            self.ratio, self.ratio == 4, False,
        )
        return (plans[0, :n_compress].to(device, non_blocking=True),
                plans[1, :n_write].to(device, non_blocking=True))

    def prefill(self, carry, projected, output, ape, rows, plans):
        self.module.prefill(carry, projected, output, ape, rows, *plans, None)

    def decode(self, carry, projected, output, ape, rows, seq_lens):
        self.module.decode(carry, projected, output, ape, rows, seq_lens, None)

    def normalize_rotate(self, output, weight, handle, freqs, eps, *, is_decode):
        self.norm_rope.forward(output, weight, handle, freqs,
                               int(is_decode), eps, self.ratio)
