from __future__ import annotations

import torch

from sparseengine.engine.sequence import Sequence
from sparseengine.method_registry import (
    h2o_uses_fused_prefill_score,
    normalize_sparse_method,
    resolve_prefill_sparse_method,
)
from sparseengine.utils.profiler import profiler

from .base import LayerBatchSparseState, SparseStepContext
from .passthrough import PassThroughRuntime


class H2ORuntime(PassThroughRuntime):
    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        self._mla_reduced_decode_score = (
            getattr(config, "attention_cache_layout", "explicit_kv") == "mla_latent"
        )
        self._fused_decode_score = bool(
            getattr(config, "h2o_decode_eviction", False)
            and getattr(config, "h2o_decode_score_fusion", True)
            and not self._mla_reduced_decode_score
        )
        self._h2o_decode_attn_score_buffers: dict[
            tuple[int, ...],
            torch.Tensor,
        ] = {}

    def clear_decode_attn_score_buffers(self) -> None:
        super().clear_decode_attn_score_buffers()
        self._h2o_decode_attn_score_buffers.clear()

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return list(self._h2o_decode_attn_score_buffers.values())

    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        del layer_idx
        return (
            h2o_uses_fused_prefill_score(self.config)
            if step.is_prefill
            else bool(getattr(self.config, "h2o_decode_eviction", False))
        )

    def _prepare_decode_attention_score(
        self,
        layer_idx: int,
        state: LayerBatchSparseState,
        batch_size: int,
        num_heads: int,
        max_len: int,
    ) -> None:
        # Bind slices once every layer's current physical metadata is available.
        del layer_idx, state, batch_size, num_heads, max_len

    def _end_prepare_step(self, step: SparseStepContext) -> None:
        if not step.is_prefill and getattr(self.config, "h2o_decode_eviction", False):
            self._prepare_h2o_decode_attn_score_buffer(step.seqs)

    def prefill_score_shape(
        self,
        batch_size: int,
        num_heads: int,
        max_len: int,
    ) -> tuple[int, ...]:
        del num_heads
        return batch_size, max_len

    def prefill_score_fill_value(self) -> float:
        return -torch.inf

    def finish_step(self, step: SparseStepContext) -> None:
        if not step.is_prefill:
            if getattr(self.config, "h2o_decode_eviction", False):
                self._h2o_decode_eviction(step.seqs)
            return
        prefill_method = resolve_prefill_sparse_method(
            getattr(self.config, "prefill_sparse_method", None),
            sparse_method=getattr(self.config, "sparse_method", None),
        )
        if prefill_method == "h2o_prefill":
            self.cache_manager.evict_after_intermediate_prefill(step.seqs)
        if normalize_sparse_method(
            getattr(self.config, "sparse_method", None)
        ) == "h2o":
            self.cache_manager.compact_final_prefill_for_decode(step.seqs)

    def _h2o_kv_layer_indices(self) -> list[int]:
        return [
            layer_idx
            for layer_idx in range(self.num_layers)
            if self._is_kv_layer(layer_idx)
        ]

    def _h2o_decode_score_width(self, layer_indices: list[int]) -> int:
        max_len = max(
            self._state_max_context_len(
                self.layer_batch_sparse_states[layer_idx]
            )
            for layer_idx in layer_indices
        )
        required_width = max_len
        if bool(getattr(self.config, "decode_graph", False)):
            graph_capacity = getattr(
                self.cache_manager,
                "_decode_static_max_context_len",
                None,
            )
            if graph_capacity is None or int(graph_capacity) < max_len:
                raise RuntimeError(
                    "H2O decode CUDA graph requires a score capacity covering the "
                    f"current context: graph_capacity={graph_capacity} "
                    f"current={max_len}."
                )
            required_width = int(graph_capacity)
        return int(required_width)

    def _get_h2o_decode_score_buffer(
        self,
        num_kv_layers: int,
        batch_size: int,
        width: int,
    ) -> torch.Tensor:
        if min(num_kv_layers, batch_size, width) <= 0:
            raise RuntimeError(
                "H2O decode score buffer requires positive dimensions: "
                f"shape={(num_kv_layers, batch_size, width)}."
            )
        if bool(getattr(self.config, "decode_graph", False)):
            key = (num_kv_layers, batch_size, width)
        else:
            key = (num_kv_layers,)
        buffer = self._h2o_decode_attn_score_buffers.get(key)
        # Raw per-head logits are provider-private and consumed within each
        # layer. Only reduced scores must survive until finish_step.
        shape = (num_kv_layers, batch_size, width)
        needs_alloc = (
            buffer is None
            or buffer.dtype != self.snapkv_decode_score_dtype
            or buffer.device != self.device
            or int(buffer.shape[0]) < num_kv_layers
            or int(buffer.shape[1]) < batch_size
            or int(buffer.shape[-1]) < width
        )
        if needs_alloc:
            buffer = torch.empty(
                shape,
                dtype=self.snapkv_decode_score_dtype,
                device=self.device,
            )
            self._h2o_decode_attn_score_buffers[key] = buffer
        view = buffer[:num_kv_layers, :batch_size, ..., :width]
        if not self._fused_decode_score and (
            needs_alloc or not bool(getattr(self.config, "decode_graph", False))
        ):
            view.fill_(-1e20)
        return view

    def _prepare_h2o_decode_attn_score_buffer(self, seqs: list[Sequence]):
        del seqs
        layer_indices = self._h2o_kv_layer_indices()
        if not layer_indices:
            return
        batch_sizes = []
        kv_indices = []
        for layer_idx in layer_indices:
            state = self.layer_batch_sparse_states[layer_idx]
            if state.context_lens is None:
                raise RuntimeError(
                    "H2O decode state is missing context lengths: "
                    f"layer={layer_idx}."
                )
            batch_sizes.append(int(state.context_lens.numel()))
            kv_indices.append(self._kv_layer_index(layer_idx))
        if any(batch_size != batch_sizes[0] for batch_size in batch_sizes[1:]):
            raise RuntimeError(
                f"H2O decode KV layers disagree on batch size: {batch_sizes}."
            )
        if sorted(kv_indices) != list(range(len(layer_indices))):
            raise RuntimeError(
                "H2O decode KV-layer indices must densely cover the continuous "
                f"buffer: indices={kv_indices}."
            )
        width = self._h2o_decode_score_width(layer_indices)
        reduced_scores = self._get_h2o_decode_score_buffer(
            len(layer_indices),
            batch_sizes[0],
            width,
        )
        for layer_idx, kv_idx in zip(layer_indices, kv_indices):
            self.layer_batch_sparse_states[layer_idx].attn_score = reduced_scores[
                kv_idx
            ]

    def _resolve_h2o_decode_attn_score_buffer(
        self,
        layer_tensors: dict[int, torch.Tensor],
    ) -> tuple[list[int], torch.Tensor]:
        layer_indices = self._h2o_kv_layer_indices()
        if set(layer_tensors) != set(layer_indices):
            raise RuntimeError(
                "H2O decode score slices do not cover every KV layer: "
                f"expected={layer_indices} got={sorted(layer_tensors)}."
            )
        first = layer_tensors[layer_indices[0]]
        if first.dim() != 2:
            raise RuntimeError(
                "H2O decode score slice must be [B, W], got "
                f"{tuple(first.shape)}."
            )
        batch_size, width = int(first.shape[0]), int(first.shape[-1])
        for buffer in self._h2o_decode_attn_score_buffers.values():
            if (
                buffer.dtype != first.dtype
                or buffer.device != first.device
                or int(buffer.shape[0]) < len(layer_indices)
                or int(buffer.shape[1]) < batch_size
                or int(buffer.shape[-1]) < width
            ):
                continue
            view = buffer[: len(layer_indices), :batch_size, ..., :width]
            matches = True
            for layer_idx in layer_indices:
                kv_idx = self._kv_layer_index(layer_idx)
                layer_tensor = layer_tensors[layer_idx]
                if (
                    tuple(layer_tensor.shape) != tuple(first.shape)
                    or layer_tensor.data_ptr() != view[kv_idx].data_ptr()
                ):
                    matches = False
                    break
            if matches:
                return layer_indices, view
        raise RuntimeError(
            "H2O decode layer score slices do not share a known contiguous buffer."
        )

    def reset_decode_attn_scores_for_graph(
        self,
        refs: dict[int, dict[str, object]],
    ) -> bool:
        layer_tensors = {
            layer_idx: layer_refs["attn_score"]
            for layer_idx, layer_refs in refs.items()
            if self._is_kv_layer(layer_idx)
            and isinstance(layer_refs.get("attn_score"), torch.Tensor)
        }
        if not layer_tensors:
            return False
        _layer_indices, reduced_scores = (
            self._resolve_h2o_decode_attn_score_buffer(layer_tensors)
        )
        # Fused probability output overwrites each row, including padded tails.
        if not self._fused_decode_score:
            reduced_scores.fill_(-1e20)
        return True

    @torch.no_grad()
    def _h2o_decode_eviction(self, seqs: list[Sequence]):
        with profiler.record("h2o_decode_eviction"):
            if self.validate_runtime_invariants:
                layer_tensors = {}
                layer_context_lens = []
                for layer_idx in self._h2o_kv_layer_indices():
                    state = self.layer_batch_sparse_states[layer_idx]
                    if state.attn_score is None or state.context_lens is None:
                        raise RuntimeError(
                            "H2O decode requires reduced probability scores for "
                            f"every KV layer: layer={layer_idx}."
                        )
                    if state.attn_score.dim() != 2:
                        raise RuntimeError(
                            "H2O decode must use the SnapKV-style [B, L] score "
                            f"path, got layer={layer_idx} "
                            f"shape={tuple(state.attn_score.shape)}."
                        )
                    if int(state.context_lens.numel()) != int(
                        state.attn_score.shape[0]
                    ):
                        raise RuntimeError(
                            "H2O decode context lengths do not match score batch: "
                            f"layer={layer_idx} "
                            f"contexts={int(state.context_lens.numel())} "
                            f"score_batch={int(state.attn_score.shape[0])}."
                        )
                    layer_tensors[layer_idx] = state.attn_score
                    layer_context_lens.append(state.context_lens)
                layer_indices, probability_scores = (
                    self._resolve_h2o_decode_attn_score_buffer(layer_tensors)
                )
                context_lens = torch.stack(layer_context_lens, dim=0)
                bounds_ok = (
                    (context_lens >= 0)
                    & (context_lens <= int(probability_scores.shape[-1]))
                ).all()
                if context_lens.is_cuda:
                    torch._assert_async(bounds_ok)
                elif not bool(bounds_ok.item()):
                    raise RuntimeError(
                        "H2O decode context lengths exceed the reduced score width: "
                        f"width={int(probability_scores.shape[2])} "
                        f"contexts={context_lens.tolist()}."
                    )
            else:
                # Replay restores the selected graph's layer slices without
                # rerunning prepare_step. Resolve those slices, not the last
                # batch prepared during startup capture.
                layer_tensors = {
                    layer_idx: self.layer_batch_sparse_states[layer_idx].attn_score
                    for layer_idx in self._h2o_kv_layer_indices()
                }
                layer_indices, probability_scores = (
                    self._resolve_h2o_decode_attn_score_buffer(layer_tensors)
                )
            if int(probability_scores.shape[1]) < len(seqs):
                raise RuntimeError(
                    "H2O decode score batch does not cover current sequences: "
                    f"score_batch={int(probability_scores.shape[1])} "
                    f"seqs={len(seqs)}."
                )
            with profiler.record("h2o_decode_score_update"):
                if self._mla_reduced_decode_score:
                    # TODO(h2o-mla-parity): softmax of head-max QK is an
                    # approximation, not a reduction of per-head probabilities.
                    self.cache_manager.update_decode_attention_scores_all_layers(
                        layer_indices,
                        seqs,
                        probability_scores[:, : len(seqs)],
                        normalize_logits=True,
                        softmax_scale=self.attn_softmax_scale,
                    )
                else:
                    self.cache_manager.update_decode_attention_scores_all_layers(
                        layer_indices,
                        seqs,
                        probability_scores[:, : len(seqs)],
                        normalize_logits=False,
                    )
            with profiler.record("h2o_decode_compact_total"):
                self.cache_manager.evict_after_decode(seqs)
