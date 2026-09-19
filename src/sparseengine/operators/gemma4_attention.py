from __future__ import annotations

from dataclasses import dataclass

import torch

from sparseengine.kernels.external.flashinfer.prefill import (
    make_flashinfer_paged_prefill_wrapper,
)
from sparseengine.layers.attention_backend import (
    TritonAttentionBackend,
    _require_explicit_payload,
)
from sparseengine.platforms import device_runtime


@dataclass
class _FlashInferState:
    wrapper: object
    workspace: torch.Tensor
    plan_key: tuple[object, ...] | None = None


@dataclass
class Gemma4DecodeWorkspace:
    mid_output: torch.Tensor
    mid_lse: torch.Tensor
    num_kv_splits: torch.Tensor


class Gemma4FlashInferPrefill:
    """Shared FlashInfer plans for Gemma 4 text-prefill head shapes."""

    def __init__(self) -> None:
        self._states: dict[tuple[int, int, int, int], _FlashInferState] = {}
        self._available_states: list[_FlashInferState] = []

    def prepare(
        self,
        *,
        device_index: int | None = None,
        max_contracts: int = 2,
    ) -> None:
        if self._states or self._available_states:
            return
        max_contracts = int(max_contracts)
        if max_contracts <= 0:
            raise ValueError(
                "Gemma 4 FlashInfer prefill requires max_contracts > 0, "
                f"got {max_contracts}."
            )
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != int(current_device):
            raise RuntimeError(
                "Gemma 4 FlashInfer prefill must be prepared on the selected "
                f"CUDA device: selected={device_index} current={current_device}."
            )
        device = torch.device("cuda", int(device_index))
        for _ in range(max_contracts):
            workspace = torch.empty(
                128 * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )
            self._available_states.append(
                _FlashInferState(
                    wrapper=make_flashinfer_paged_prefill_wrapper(
                        workspace,
                        backend="fa2",
                    ),
                    workspace=workspace,
                )
            )

    def close(self) -> None:
        self._states.clear()
        self._available_states.clear()

    @staticmethod
    def _page_metadata(view, max_context_len: int):
        meta = view.meta
        rows = meta.active_slots.index_select(0, meta.req_indices.to(torch.long))[
            :, :max_context_len
        ]
        positions = torch.arange(
            max_context_len,
            device=meta.context_lens.device,
            dtype=meta.context_lens.dtype,
        )
        indices = rows.masked_select(
            positions.unsqueeze(0) < meta.context_lens.unsqueeze(1)
        ).to(torch.int32).contiguous()
        indptr = torch.cat(
            (
                torch.zeros(1, device=indices.device, dtype=torch.int32),
                meta.context_lens.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )
        return indices, indptr, torch.ones_like(meta.context_lens, dtype=torch.int32)

    def run(
        self,
        q: torch.Tensor,
        view,
        *,
        q_start: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_context_len: int,
        sliding_window: int | None,
    ) -> torch.Tensor:
        from sparseengine.utils.context import get_context

        payload = _require_explicit_payload(view, operation="Gemma 4 prefill")
        meta = view.meta
        if meta.active_slots.dtype != torch.int32 or meta.active_slots.ndim != 2:
            raise TypeError("Gemma 4 FlashInfer prefill requires an int32 page table.")
        q_heads, kv_heads, head_dim = map(
            int, (q.shape[1], payload.k_cache.shape[1], q.shape[2])
        )
        window_left = -1 if sliding_window is None else int(sliding_window) - 1
        contract = q_heads, kv_heads, head_dim, window_left
        state = self._states.get(contract)
        if state is None:
            if not self._available_states:
                raise RuntimeError(
                    "Gemma 4 FlashInfer prefill exceeded its prepared attention "
                    f"contract capacity: contract={contract} "
                    f"prepared={tuple(self._states)}."
                )
            state = self._available_states.pop()
            self._states[contract] = state
        context = get_context()
        plan_key = (
            context.attention_validation_scope,
            meta.active_slots.data_ptr(),
            meta.req_indices.data_ptr(),
            meta.context_lens.data_ptr(),
        )
        if state.plan_key != plan_key:
            indices, kv_indptr, last_page_len = self._page_metadata(
                view, int(max_context_len)
            )
            qo_indptr = torch.cat((q_start, q_start[-1:] + chunk_lens[-1:]))
            state.wrapper.plan(
                qo_indptr,
                kv_indptr,
                indices,
                last_page_len,
                q_heads,
                kv_heads,
                head_dim,
                1,
                causal=True,
                sm_scale=1.0,
                window_left=window_left,
                q_data_type=q.dtype,
                kv_data_type=payload.k_cache.dtype,
                non_blocking=True,
            )
            state.plan_key = plan_key
        output = torch.empty_like(q)
        state.wrapper.run(
            q,
            (payload.k_cache.unsqueeze(1), payload.v_cache.unsqueeze(1)),
            out=output,
        )
        return output


class Gemma4AttentionBackend(TritonAttentionBackend):
    """Gemma 4 attention semantics isolated from the tuned generic kernels."""

    name = "triton_gemma4"
    supports_decode_graph = True

    def __init__(
        self,
        *,
        sliding_window: int | None,
        flashinfer_prefill: Gemma4FlashInferPrefill | None = None,
        decode_workspace: Gemma4DecodeWorkspace | None = None,
        multi_processor_count: int | None = None,
    ) -> None:
        super().__init__()
        self.sliding_window = None if sliding_window is None else int(sliding_window)
        self.flashinfer_prefill = flashinfer_prefill
        self.decode_workspace = decode_workspace
        self.multi_processor_count = (
            None
            if multi_processor_count is None
            else int(multi_processor_count)
        )
        self._runtime_kernel_path_counts: dict[str, dict[str, int]] = {}

    def get_decode_workspace(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace = self.decode_workspace
        if workspace is None:
            raise RuntimeError("Gemma 4 decode backend has no prepared workspace.")
        if (
            self.multi_processor_count is None
            or self.multi_processor_count <= 0
        ):
            raise RuntimeError(
                "Gemma 4 decode backend requires a positive multi-processor count."
            )
        if (
            batch_size > workspace.mid_output.shape[0]
            or num_heads != workspace.mid_output.shape[1]
            or head_dim != workspace.mid_output.shape[3]
            or device != workspace.mid_output.device
        ):
            raise RuntimeError(
                "Gemma 4 fixed-grid workspace does not match the decode contract: "
                f"actual={(batch_size, num_heads, head_dim, device)} "
                f"workspace={tuple(workspace.mid_output.shape)}/"
                f"{workspace.mid_output.device}."
            )
        return workspace.mid_output[:batch_size], workspace.mid_lse[:batch_size]

    def binding_metadata(self) -> dict[str, object]:
        prefill_routes = ["triton_multimodal_context", "triton_context"]
        if self.flashinfer_prefill is not None:
            prefill_routes.insert(1, "flashinfer_paged_prefill_fa2")
        return {
            "implementation_kind": "dispatch_plan",
            "prefill_routes": prefill_routes,
            "decode_routes": ["sglang_fixed_grid"],
            "sliding_window": self.sliding_window,
        }

    def _record_kernel_path(self, path: str) -> None:
        counts = self._runtime_kernel_path_counts.setdefault(
            path,
            {"eager_dispatches": 0, "cuda_graph_capture_dispatches": 0},
        )
        key = (
            "cuda_graph_capture_dispatches"
            if device_runtime.is_stream_capturing()
            else "eager_dispatches"
        )
        counts[key] += 1

    def runtime_kernel_stats(self) -> dict[str, object]:
        return {
            "kernel_paths": {
                path: dict(sorted(counts.items()))
                for path, counts in sorted(self._runtime_kernel_path_counts.items())
            },
            "fallback_reasons": {},
        }

    def _prefill_route(self, view) -> str:
        from sparseengine.utils.context import get_context

        image_groups = getattr(get_context(), "multimodal_image_groups", None)
        if self.sliding_window is not None and isinstance(image_groups, torch.Tensor):
            return "triton_multimodal_context"
        if self.flashinfer_prefill is not None and view.meta.attn_score is None:
            return "flashinfer_paged_prefill_fa2"
        return "triton_context"

    def run_prefill(
        self,
        q: torch.Tensor,
        view,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_input_len: int,
    ) -> torch.Tensor:
        payload = _require_explicit_payload(view, operation="Gemma 4 prefill")
        route = self._prefill_route(view)
        self._record_kernel_path(route)
        if route == "triton_multimodal_context":
            from sparseengine.utils.context import get_context
            from sparseengine.kernels.triton.gemma4_multimodal_context_attention import (
                gemma4_multimodal_context_attention,
            )

            image_groups = get_context().multimodal_image_groups
            output = torch.empty_like(q)
            gemma4_multimodal_context_attention(
                q,
                payload.k_cache,
                payload.v_cache,
                output,
                view.meta.req_indices,
                b_start_loc,
                view.meta.context_lens,
                view.meta.context_lens - chunk_lens,
                max_input_len,
                view.meta.active_slots,
                image_groups,
                sliding_window=self.sliding_window,
                attn_score=view.meta.attn_score,
            )
            return output
        if route == "flashinfer_paged_prefill_fa2":
            assert self.flashinfer_prefill is not None
            return self.flashinfer_prefill.run(
                q,
                view,
                q_start=b_start_loc,
                chunk_lens=chunk_lens,
                max_context_len=max_input_len,
                sliding_window=self.sliding_window,
            )
        output = torch.empty_like(q)
        from sparseengine.kernels.triton.gemma4_context_attention import (
            gemma4_context_attention,
        )

        gemma4_context_attention(
            q,
            payload.k_cache,
            payload.v_cache,
            output,
            view.meta.req_indices,
            b_start_loc,
            view.meta.context_lens,
            view.meta.context_lens - chunk_lens,
            max_input_len,
            view.meta.active_slots,
            sliding_window=self.sliding_window,
            attn_score=view.meta.attn_score,
        )
        return output

    def run_decode(
        self,
        q: torch.Tensor,
        view,
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
        del (
            max_len_in_batch,
            block_seq,
            num_heads,
            num_kv_heads,
            gqa_block_n,
            gqa_num_warps,
        )
        workspace = self.decode_workspace
        if workspace is None:
            raise RuntimeError("Gemma 4 decode backend has no prepared workspace.")
        multi_processor_count = self.multi_processor_count
        if multi_processor_count is None or multi_processor_count <= 0:
            raise RuntimeError(
                "Gemma 4 decode backend requires a positive multi-processor count."
            )
        payload = _require_explicit_payload(view, operation="Gemma 4 decode")
        if payload.backend != "dense":
            raise RuntimeError("Gemma 4 fixed-grid decode requires dense explicit KV.")
        from sparseengine.kernels.triton.sglang_gemma4_decode_attention import (
            sglang_gemma4_decode,
        )

        batch_size = int(q.shape[0])
        self._record_kernel_path("sglang_fixed_grid")
        return sglang_gemma4_decode(
            q,
            payload.k_cache,
            payload.v_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            workspace.mid_output[:batch_size],
            workspace.mid_lse[:batch_size],
            workspace.num_kv_splits[:batch_size],
            sliding_window=self.sliding_window,
            multi_processor_count=multi_processor_count,
            attn_score=view.meta.attn_score,
        )


__all__ = [
    "Gemma4AttentionBackend",
    "Gemma4DecodeWorkspace",
    "Gemma4FlashInferPrefill",
]
