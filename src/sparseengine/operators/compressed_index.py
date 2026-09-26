from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import inspect

import torch

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class CompressedIndexOpSpec:
    num_heads: int
    head_dim: int
    rope_dim: int
    selection_capacity: int
    page_size: int
    cuda_graph: bool

    def __post_init__(self):
        if (self.num_heads, self.head_dim, self.rope_dim) != (64, 128, 64):
            raise ValueError("V4 indexer requires 64 heads, D128 and 64 rotary dimensions")
        if self.selection_capacity != 512:
            raise ValueError("V4 indexer requires Top-512 selection")
        if self.page_size <= 0 or self.page_size & (self.page_size - 1):
            raise ValueError("Index page size must be a positive power of two")


def _rotate_query(query, freqs, positions):
    # Complex interleaved tail RoPE, including the checkpoint's BF16 rounding.
    tail = query[..., -64:].float().unflatten(-1, (32, 2))
    freq = freqs[positions.long()].unflatten(-1, (32, 2))[:, None]
    real, imag = tail[..., 0], tail[..., 1]
    rotated = torch.stack((real * freq[..., 0] - imag * freq[..., 1],
                           real * freq[..., 1] + imag * freq[..., 0]), dim=-1)
    return torch.cat((query[..., :-64], rotated.flatten(-2).to(query.dtype)), dim=-1)


COMPRESSED_INDEX_REGISTRY = OpRegistry(
    "compressed KV index", portfolio=PortfolioPolicy(repo_nonstandard=("sgl_v4",)),
)


@COMPRESSED_INDEX_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class SGLCompressedIndexProvider:
    name = "sgl_v4"

    @classmethod
    def supports(cls, spec: CompressedIndexOpSpec, caps: DeviceCaps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported("V4 indexer is validated on SM90 CUDA")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph")
        for module in ("flashinfer", "fast_hadamard_transform", "triton"):
            if importlib.util.find_spec(module) is None:
                return SupportResult.dependency_absent(f"V4 indexer requires {module}")
        import flashinfer
        if not hasattr(flashinfer, "top_k_page_table_transform"):
            return SupportResult.unsupported("FlashInfer lacks sparse Top-K page-table transform")
        params = inspect.signature(flashinfer.top_k_page_table_transform).parameters
        if not {"out", "page_size"} <= params.keys():
            return SupportResult.unsupported("V4 indexer requires FlashInfer >= 0.6.18.post1 Top-K buffers")
        return SupportResult.yes("pinned SGL FP4 scorer, Fast Hadamard and FlashInfer Top-K")

    def __init__(self, *, op_spec):
        from fast_hadamard_transform import hadamard_transform
        from sparseengine.kernels.external.sgl.v4_indexer.quantization import fake_quant_fp4
        from sparseengine.kernels.external.sgl.v4_indexer import compute
        import flashinfer

        self.spec = op_spec
        self._hadamard = hadamard_transform
        self._rotate = torch.compile(_rotate_query, fullgraph=True, dynamic=True)
        self._fake_quant = torch.compile(fake_quant_fp4, fullgraph=True, dynamic=True)
        self._compute = compute
        self._topk = flashinfer.top_k_page_table_transform

    def prepare_query(self, query, freqs, positions):
        if query.dtype != torch.bfloat16 or query.shape[1:] != (64, 128):
            raise ValueError("Indexer query must be BF16 [tokens, 64, 128]")
        if freqs.dtype != torch.float32 or freqs.shape[1:] != (64,):
            raise ValueError("Indexer RoPE requires FP32 interleaved frequencies")
        if positions.dtype != torch.int32 or positions.shape != (query.shape[0],):
            raise ValueError("Indexer positions must be int32 [tokens]")
        if not query.shape[0]:
            return query
        rotated = self._rotate(query, freqs, positions)
        return self._fake_quant(self._hadamard(rotated, scale=128 ** -0.5))

    def store_keys(self, normalized_rotated_keys, pages, slots):
        """Write compressed keys already normalized/rotated by the compressor.

        Slots are cache-owned physical addresses. Graph padding uses reserved
        scratch slots; the upstream store has no negative-address mask.
        """
        if normalized_rotated_keys.dtype != torch.bfloat16 or normalized_rotated_keys.shape[1:] != (128,):
            raise ValueError("Indexer keys must be BF16 [tokens, 128]")
        self._validate_pages(pages)
        rows = normalized_rotated_keys.shape[0]
        if slots.dtype != torch.int32 or slots.shape != (rows,):
            raise ValueError("Index write slots must be int32 [tokens]")
        if not rows:
            return
        rotated = self._hadamard(normalized_rotated_keys, scale=128 ** -0.5)
        payload = torch.empty((rows, 64), dtype=torch.uint8, device=pages.device)
        scales = torch.empty((rows,), dtype=torch.int32, device=pages.device)
        if rows:
            self._compute._quantize_fp4_indexer_rows[( (rows + 7) // 8, )](
                rotated, payload, scales, rows, 8, BLOCK_N=128, GROUP_N=32,
                RNE=True, num_warps=4,
            )
            self._compute._store_fp4_index_k_cache_kernel[(rows,)](
                payload, scales, pages, slots, self.spec.page_size,
                pages.stride(0), BLOCK=64,
            )

    def _validate_pages(self, pages):
        if (pages.dtype != torch.uint8 or pages.ndim != 2 or
                pages.shape[1] != self.spec.page_size * 68 or not pages.is_contiguous()):
            raise ValueError("Index cache requires contiguous uint8 pages with 68 bytes per slot")

    def score(self, query, scaled_head_weights, pages, slots, lengths, *, out):
        """Score cache-selected candidates; caller supplies visibility and mapping."""
        self._validate_pages(pages)
        rows, heads, dim = query.shape
        width = slots.shape[1]
        if query.dtype != torch.bfloat16 or (heads, dim) != (64, 128) or not query.is_contiguous():
            raise ValueError("Index scoring requires contiguous prepared BF16 queries")
        if scaled_head_weights.shape != (rows, heads) or scaled_head_weights.dtype != torch.bfloat16:
            raise ValueError("Head weights must be BF16 [tokens, 64] with the semantic scale applied")
        if slots.shape[0] != rows or slots.dtype != torch.int32 or not slots.is_contiguous():
            raise ValueError("Candidate mapping must be contiguous int32 [tokens, capacity]")
        if lengths.shape != (rows,) or lengths.dtype != torch.int32:
            raise ValueError("Candidate lengths must be int32 [tokens]")
        if out.shape != (rows, width) or out.dtype != torch.float32 or not out.is_contiguous():
            raise ValueError("Index score workspace must be contiguous FP32 [tokens, capacity]")
        if rows and width:
            self._compute._fp4_index_logits_kernel[(rows, (width + 63) // 64)](
                query, scaled_head_weights, slots, lengths, pages, out, width,
                self.spec.page_size, pages.stride(0), query.stride(0), query.stride(1),
                scaled_head_weights.stride(0), H=64, HALF_D=64, BLOCK_L=64, num_warps=4,
            )
        return out

    def select(self, scores, attention_page_table, lengths, *, out, page_size=1,
               row_to_batch=None):
        """Map selected logical candidates into the attention cache's physical slots.

        Index-cache slots and attention-cache slots are separate namespaces.
        """
        if not scores.shape[0]:
            return out
        if not scores.shape[1]:
            return out.fill_(-1)
        return self._topk(scores, attention_page_table, lengths,
                          self.spec.selection_capacity, row_to_batch=row_to_batch,
                          deterministic=True, tie_break=1, dsa_graph_safe=True,
                          page_size=page_size, out=out)


def resolve_compressed_index_provider(spec: CompressedIndexOpSpec, *, device_index: int):
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(COMPRESSED_INDEX_REGISTRY).resolve(spec, caps, op_spec=spec).provider
