import os

import torch

from sparseengine.engine.cache_manager import (
    DecodeComputeView,
    ExplicitKVPayload,
    PrefillComputeView,
)
from sparseengine.operators.registry import record_operator_binding
from sparseengine.utils.context import get_context
from sparseengine.kernels.triton.context_flashattention_nopad import context_attention_fwd
from sparseengine.kernels.triton.flash_decoding_stage1 import flash_decode_stage1 as mha_flash_decode_stage1
from sparseengine.kernels.triton.flash_decoding_stage1 import flash_decode_stage1_with_score as mha_flash_decode_stage1_with_score
from sparseengine.kernels.triton.flash_decoding_stage2 import flash_decode_stage2
from sparseengine.kernels.triton.gqa_flash_decoding_stage1 import flash_decode_stage1 as gqa_flash_decode_stage1
from sparseengine.kernels.triton.gqa_flash_decoding_stage1 import flash_decode_stage1_with_score as gqa_flash_decode_stage1_with_score
from sparseengine.utils.log import log_once
from sparseengine.utils.profiler import profiler


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _fake_attention_enabled() -> bool:
    return _env_truthy("SPARSEENGINE_FAKE_ATTENTION")


def _allow_fake_attention() -> None:
    if not _env_truthy("SPARSEENGINE_ALLOW_FAKE_ATTENTION"):
        raise RuntimeError(
            "Sparse-Engine fake attention was requested, but it is disabled by default because it "
            "invalidates correctness and benchmark results. Set SPARSEENGINE_ALLOW_FAKE_ATTENTION=1 "
            "only for explicit fake-attention tests or profiling."
        )
    log_once(
        "Sparse-Engine fake attention is enabled; outputs are not valid for correctness or benchmark results.",
        level="WARNING",
    )


def _fake_prefill_attention_enabled() -> bool:
    enabled = _env_truthy("SPARSEENGINE_FAKE_PREFILL_ATTENTION") or _fake_attention_enabled()
    if enabled:
        _allow_fake_attention()
    return enabled


def _fake_decode_attention_enabled() -> bool:
    enabled = _env_truthy("SPARSEENGINE_FAKE_DECODE_ATTENTION") or _fake_attention_enabled()
    if enabled:
        _allow_fake_attention()
    return enabled


def _fake_attention_output(q: torch.Tensor) -> torch.Tensor:
    mode = os.environ.get("SPARSEENGINE_FAKE_ATTENTION_MODE", "zero").strip().lower()
    if mode in {"zero", "zeros"}:
        return torch.zeros_like(q)
    if mode == "copy":
        return q.clone()
    if mode == "empty":
        return torch.empty_like(q)
    raise ValueError(
        "SPARSEENGINE_FAKE_ATTENTION_MODE must be one of 'zero', 'copy', or 'empty', "
        f"got {mode!r}."
    )


def _fill_fake_attention_score(attn_score: torch.Tensor | None) -> None:
    if attn_score is not None:
        attn_score.zero_()


def _require_explicit_payload(
    view: PrefillComputeView | DecodeComputeView,
    *,
    operation: str,
) -> ExplicitKVPayload:
    payload = view.payload
    if not isinstance(payload, ExplicitKVPayload):
        raise TypeError(
            f"{operation} requires ExplicitKVPayload, got "
            f"{type(payload).__name__}."
        )
    return payload


class TritonAttentionBackend:
    """Thin backend wrapper around the existing Sparse-Engine Triton attention kernels."""

    name = "triton"

    def __init__(self) -> None:
        record_operator_binding("Attention", self)

    def maybe_run_fake_prefill(
        self,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        chunk_lens: torch.Tensor,
        max_input_len: int,
    ) -> torch.Tensor | None:
        if not _fake_prefill_attention_enabled():
            return None
        meta = view.meta
        _fill_fake_attention_score(meta.attn_score)
        return _fake_attention_output(q)

    def run_prefill(
        self,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_input_len: int,
    ) -> torch.Tensor:
        payload = _require_explicit_payload(view, operation="Triton prefill")
        meta = view.meta
        b_seq_len = meta.context_lens
        if b_seq_len.numel() != chunk_lens.numel():
            layer_idx = getattr(get_context(), "now_layer_idx", None)
            raise RuntimeError(
                "prefill context_lens/chunk_lens batch mismatch: "
                f"layer={layer_idx} context_lens_shape={tuple(b_seq_len.shape)} "
                f"chunk_lens_shape={tuple(chunk_lens.shape)} q_shape={tuple(q.shape)} "
                f"req_indices_shape={tuple(meta.req_indices.shape)} "
                f"active_slots_shape={tuple(meta.active_slots.shape)}"
            )
        b_prompt_cache_len = b_seq_len - chunk_lens
        self.debug_check_prefill_bounds(q, view, chunk_lens=chunk_lens)
        if _fake_prefill_attention_enabled():
            _fill_fake_attention_score(meta.attn_score)
            return _fake_attention_output(q)
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            payload.k_cache,
            payload.v_cache,
            o,
            meta.req_indices,
            b_start_loc,
            b_seq_len,
            b_prompt_cache_len,
            max_input_len,
            meta.active_slots,
            attn_score=meta.attn_score,
        )
        return o

    def debug_check_prefill_bounds(
        self,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        chunk_lens: torch.Tensor,
    ):
        if os.environ.get("SENGINE_DEBUG_PREFILL_BOUNDS", "0") != "1":
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        payload = _require_explicit_payload(view, operation="Prefill bounds check")
        meta = view.meta
        if meta.active_slots.dim() != 2:
            raise RuntimeError(
                f"prefill bounds check expects 2D active_slots, got shape={tuple(meta.active_slots.shape)}"
            )
        rows = meta.req_indices.to(torch.long)
        row_min = int(rows.min().item()) if rows.numel() > 0 else 0
        row_max = int(rows.max().item()) if rows.numel() > 0 else -1
        if row_min < 0 or row_max >= int(meta.active_slots.shape[0]):
            raise RuntimeError(
                "prefill req row index out of bounds: "
                f"row_min={row_min} row_max={row_max} num_rows={int(meta.active_slots.shape[0])}"
            )
        if int(chunk_lens.sum().item()) != int(q.shape[0]):
            raise RuntimeError(
                "prefill q/chunk length mismatch: "
                f"q_tokens={int(q.shape[0])} chunk_tokens={int(chunk_lens.sum().item())}"
            )
        if bool((meta.context_lens < chunk_lens).any().item()):
            raise RuntimeError(
                "prefill context_lens shorter than chunk_lens: "
                f"context_lens={meta.context_lens.detach().cpu().tolist()} "
                f"chunk_lens={chunk_lens.detach().cpu().tolist()}"
            )
        visible_len = int(meta.context_lens.max().item()) if meta.context_lens.numel() > 0 else 0
        if visible_len > int(meta.active_slots.shape[1]):
            raise RuntimeError(
                "prefill visible length exceeds active slot table width: "
                f"visible_len={visible_len} active_slots_width={int(meta.active_slots.shape[1])}"
            )
        visible_slots = meta.active_slots.index_select(0, rows)[:, :visible_len]
        pos = torch.arange(visible_len, device=visible_slots.device)[None, :]
        valid_pos = pos < meta.context_lens[:, None]
        slot_cap = int(payload.k_cache.shape[0])
        bad = ((visible_slots < 0) | (visible_slots >= slot_cap)) & valid_pos
        if bool(bad.any().item()):
            layer_idx = getattr(get_context(), "now_layer_idx", None)
            loc = bad.nonzero(as_tuple=False)[0]
            bad_b = int(loc[0].item())
            bad_pos = int(loc[1].item())
            bad_slot = int(visible_slots[bad_b, bad_pos].item())
            bad_req_row = int(rows[bad_b].item())
            raise RuntimeError(
                "prefill physical slot out of bounds before attention: "
                f"layer={layer_idx} batch={bad_b} req_row={bad_req_row} pos={bad_pos} "
                f"slot={bad_slot} slot_cap={slot_cap} context_len={int(meta.context_lens[bad_b].item())} "
                f"k_shape={tuple(payload.k_cache.shape)} v_shape={tuple(payload.v_cache.shape)} "
                f"active_slots_shape={tuple(meta.active_slots.shape)}"
            )

    def run_decode(
        self,
        q: torch.Tensor,
        view: DecodeComputeView,
        *,
        mid_o: torch.Tensor,
        mid_o_logexpsum: torch.Tensor,
        max_len_in_batch: int,
        block_seq: int,
        num_heads: int,
        num_kv_heads: int,
        gqa_block_n: int = 16,
        gqa_num_warps: int = 2,
    ) -> torch.Tensor:
        payload = _require_explicit_payload(view, operation="Triton decode")
        meta = view.meta
        if _fake_decode_attention_enabled():
            _fill_fake_attention_score(meta.attn_score)
            return _fake_attention_output(q)
        if payload.backend == "full_layer_kivi":
            self._run_full_layer_kivi_decode_stage1(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_o_logexpsum,
                max_len_in_batch=max_len_in_batch,
                block_seq=block_seq,
            )
            o = torch.empty_like(q)
            flash_decode_stage2(mid_o, mid_o_logexpsum, meta.context_lens, o, block_seq)
            return o
        self._debug_check_decode_bounds(view)
        if payload.backend == "flash_attn_contiguous":
            from flash_attn import flash_attn_with_kvcache

            if meta.active_slots.dim() != 2:
                raise RuntimeError("flash_attn_contiguous decode expects a 2D active slot table.")
            batch, width = int(meta.active_slots.shape[0]), int(meta.active_slots.shape[1])
            expected = batch * width
            if int(payload.k_cache.shape[0]) < expected or int(payload.v_cache.shape[0]) < expected:
                raise RuntimeError(
                    "flash_attn_contiguous decode got a cache smaller than the materialized active view: "
                    f"cache={int(payload.k_cache.shape[0])}/{int(payload.v_cache.shape[0])} expected={expected}."
                )
            k_cache = payload.k_cache[:expected].view(
                batch,
                width,
                int(payload.k_cache.shape[1]),
                int(payload.k_cache.shape[2]),
            )
            v_cache = payload.v_cache[:expected].view(
                batch,
                width,
                int(payload.v_cache.shape[1]),
                int(payload.v_cache.shape[2]),
            )
            with profiler.record("decode_attention_flash_attn_sparse"):
                # Decode uses q_len=1 and the materialized KV view contains no future tokens.
                out = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=meta.context_lens.to(torch.int32),
                    causal=False,
                )
            return out.squeeze(1)

        profile_kind = "full" if int(max_len_in_batch) > 8192 else "sparse"
        is_gqa = int(num_heads) > int(num_kv_heads)
        with profiler.record(f"decode_attention_stage1_{profile_kind}"):
            if meta.attn_score is not None:
                if is_gqa:
                    gqa_flash_decode_stage1_with_score(
                        q,
                        payload.k_cache,
                        payload.v_cache,
                        meta.active_slots,
                        meta.req_indices,
                        meta.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        meta.attn_score,
                        block_seq,
                    )
                else:
                    mha_flash_decode_stage1_with_score(
                        q,
                        payload.k_cache,
                        payload.v_cache,
                        meta.active_slots,
                        meta.req_indices,
                        meta.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        meta.attn_score,
                        block_seq,
                    )
            else:
                if is_gqa:
                    gqa_flash_decode_stage1(
                        q,
                        payload.k_cache,
                        payload.v_cache,
                        meta.active_slots,
                        meta.req_indices,
                        meta.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        block_seq,
                        gqa_block_n,
                        gqa_num_warps,
                    )
                else:
                    mha_flash_decode_stage1(
                        q,
                        payload.k_cache,
                        payload.v_cache,
                        meta.active_slots,
                        meta.req_indices,
                        meta.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        block_seq,
                    )

        o = torch.empty_like(q)
        with profiler.record(f"decode_attention_stage2_{profile_kind}"):
            flash_decode_stage2(mid_o, mid_o_logexpsum, meta.context_lens, o, block_seq)
        return o

    def _run_full_layer_kivi_decode_stage1(
        self,
        q: torch.Tensor,
        view: DecodeComputeView,
        *,
        mid_o: torch.Tensor,
        mid_o_logexpsum: torch.Tensor,
        max_len_in_batch: int,
        block_seq: int,
    ):
        payload = _require_explicit_payload(
            view,
            operation="Full-layer KIVI decode",
        )
        view_meta = view.meta
        backend_metadata = payload.metadata
        if backend_metadata is None:
            raise RuntimeError("full_layer_kivi decode view is missing metadata.")
        from sparseengine.kernels.triton.deltakv_kernels import full_layer_kivi_flash_decode_stage1

        full_layer_kivi_flash_decode_stage1(
            q=q,
            raw_k=payload.k_cache,
            raw_v=payload.v_cache,
            raw_slots_map=view_meta.active_slots,
            kivi_block_slots_map=backend_metadata["kivi_block_slots_map"],
            kivi_block_start_pos=backend_metadata["kivi_block_start_pos"],
            key_packed=backend_metadata["key_packed"],
            key_scales=backend_metadata["key_scales"],
            key_mins=backend_metadata["key_mins"],
            value_packed=backend_metadata["value_packed"],
            value_scales=backend_metadata["value_scales"],
            value_mins=backend_metadata["value_mins"],
            req_indices=view_meta.req_indices,
            context_lens=view_meta.context_lens,
            max_len_in_batch=max_len_in_batch,
            mid_out=mid_o,
            mid_out_logsumexp=mid_o_logexpsum,
            group_size=int(backend_metadata["group_size"]),
            block_seq=block_seq,
            block_n=int(backend_metadata.get("block_n", 16)),
            num_warps=int(backend_metadata.get("num_warps", 2)),
            num_stages=int(backend_metadata.get("num_stages", 3)),
            attn_score=view_meta.attn_score,
        )

    def _debug_check_decode_bounds(self, view: DecodeComputeView):
        if os.environ.get("SENGINE_DEBUG_DECODE_BOUNDS", "0") != "1":
            return
        payload = _require_explicit_payload(view, operation="Decode bounds check")
        meta = view.meta
        if payload.backend not in {"dense", "flash_attn_contiguous"}:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if meta.active_slots.dim() != 2:
            raise RuntimeError(
                f"debug slot bounds check expects 2D active_slots, got shape={tuple(meta.active_slots.shape)}"
            )
        rows = meta.req_indices.to(torch.long)
        row_min = int(rows.min().item()) if rows.numel() > 0 else 0
        row_max = int(rows.max().item()) if rows.numel() > 0 else -1
        if row_min < 0 or row_max >= int(meta.active_slots.shape[0]):
            raise RuntimeError(
                "decode req row index out of bounds: "
                f"row_min={row_min} row_max={row_max} num_rows={int(meta.active_slots.shape[0])}"
            )
        visible_len = int(meta.context_lens.max().item()) if meta.context_lens.numel() > 0 else 0
        if visible_len > int(meta.active_slots.shape[1]):
            raise RuntimeError(
                "decode visible length exceeds Req_to_tokens width: "
                f"visible_len={visible_len} req_to_tokens_width={int(meta.active_slots.shape[1])}"
            )
        visible_slots = meta.active_slots.index_select(0, rows)[:, :visible_len]
        pos = torch.arange(visible_len, device=visible_slots.device)[None, :]
        valid_pos = pos < meta.context_lens[:, None]
        slot_cap = int(payload.k_cache.shape[0])
        bad = ((visible_slots < 0) | (visible_slots >= slot_cap)) & valid_pos
        if bool(bad.any().item()):
            loc = bad.nonzero(as_tuple=False)[0]
            bad_b = int(loc[0].item())
            bad_pos = int(loc[1].item())
            bad_slot = int(visible_slots[bad_b, bad_pos].item())
            bad_req_row = int(rows[bad_b].item())
            raise RuntimeError(
                "decode physical slot out of bounds before attention: "
                f"batch={bad_b} req_row={bad_req_row} pos={bad_pos} "
                f"slot={bad_slot} slot_cap={slot_cap} context_len={int(meta.context_lens[bad_b].item())}"
            )
