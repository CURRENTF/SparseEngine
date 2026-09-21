from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch

from sparseengine.engine.cache_manager.base import (
    AttentionKeyComputeView,
    DecodeComputeView,
    MlaLatentPayload,
    MlaLatentWrite,
    PrefillComputeView,
)
from sparseengine.operators.mla_attention import (
    MlaAttentionOpSpec,
    MlaAttentionProvider,
    resolve_mla_attention_provider,
)
from sparseengine.operators.mla_prefill import (
    ChunkedMlaPrefill,
    estimate_mla_compressed_prefill_workspace_bytes,
    estimate_mla_prefill_workspace_bytes,
)
from sparseengine.utils.profiler import profiler
from sparseengine.utils.context import get_context


class MLAAttention:
    """Semantic MLA execution over tagged cache views.

    Model code owns projection weights, query absorption, and V reconstruction.
    This object binds providers and coordinates bounded history attention
    and sparse scoring over cache-manager-owned storage.
    """

    def __init__(
        self,
        *,
        spec: MlaAttentionOpSpec,
        provider: MlaAttentionProvider,
        prefill_workspace_bytes: int,
        hidden_size: int,
        projection_chunk_size: int,
        history_chunk_size: int = 16384,
    ) -> None:
        self.spec = spec
        self.provider = provider
        provider_spec = getattr(provider, "spec", None)
        if provider_spec is not None and provider_spec != spec:
            raise ValueError(
                "MLA semantic layer and provider specs must match: "
                f"layer={spec!r} provider={provider_spec!r}."
            )
        self.max_batch_size = int(getattr(provider, "max_batch_size", 0))
        if self.max_batch_size <= 0:
            raise ValueError("MLA provider must expose a positive max_batch_size.")
        self.prefill_workspace_bytes = int(prefill_workspace_bytes)
        if self.prefill_workspace_bytes <= 0:
            raise ValueError(
                "MLA prefill_workspace_bytes must be positive, got "
                f"{self.prefill_workspace_bytes}."
            )
        self.hidden_size = int(hidden_size)
        self.projection_chunk_size = int(projection_chunk_size)
        if self.hidden_size <= 0 or self.projection_chunk_size <= 0:
            raise ValueError(
                "MLA hidden_size and projection_chunk_size must be positive, "
                f"got {self.hidden_size} and {self.projection_chunk_size}."
            )
        if self.spec.qk_head_dim != self.spec.value_head_dim:
            raise ValueError(
                "The existing prefill backend requires equal QK/value widths, "
                f"got {self.spec.qk_head_dim}/{self.spec.value_head_dim}."
            )
        self.chunked_prefill = ChunkedMlaPrefill(spec, provider, history_chunk_size)
        self._use_compressed_prefill = getattr(provider, "use_compressed_prefill", None)
        self._key_materializer_bindings: dict[
            tuple[int, int], tuple[object, Callable]
        ] = {}

    def release_cache_runtime_bindings(self, cache_manager: object) -> None:
        """Drop layer bindings owned by a retiring cache runtime."""
        self.chunked_prefill.clear()
        for key, binding in tuple(self._key_materializer_bindings.items()):
            if binding[0] is cache_manager:
                del self._key_materializer_bindings[key]

    @classmethod
    def bind(
        cls,
        *,
        spec: MlaAttentionOpSpec,
        device: torch.device | str,
        max_batch_size: int,
        prefill_workspace_bytes: int,
        hidden_size: int,
        projection_chunk_size: int,
        history_chunk_size: int = 16384,
    ) -> MLAAttention:
        provider = resolve_mla_attention_provider(
            spec,
            device=device,
            max_batch_size=max_batch_size,
        )
        return cls(
            spec=spec,
            provider=provider,
            prefill_workspace_bytes=prefill_workspace_bytes,
            hidden_size=hidden_size,
            projection_chunk_size=projection_chunk_size,
            history_chunk_size=history_chunk_size,
        )

    @property
    def device(self) -> torch.device:
        return torch.device(self.provider.device)

    def _require_mla_payload(
        self,
        view: PrefillComputeView | DecodeComputeView | AttentionKeyComputeView,
        *,
        operation: str,
    ) -> MlaLatentPayload:
        payload = view.payload
        if not isinstance(payload, MlaLatentPayload):
            raise TypeError(
                f"{operation} requires MlaLatentPayload, got {type(payload).__name__}."
            )
        for name, tensor, width in (
            ("latent_cache", payload.latent_cache, self.spec.kv_lora_rank),
            ("rope_cache", payload.rope_cache, self.spec.rope_dim),
        ):
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}."
                )
            if tensor.dtype != self.spec.cache_dtype:
                raise TypeError(
                    f"{name} must use {self.spec.cache_dtype}, got {tensor.dtype}."
                )
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (1, width):
                raise ValueError(
                    f"{name} must have shape [slots, 1, {width}], got "
                    f"{tuple(tensor.shape)}."
                )
        if payload.latent_cache.shape[0] != payload.rope_cache.shape[0]:
            raise ValueError("MLA latent and RoPE caches must have equal slots.")
        return payload

    @torch.no_grad()
    def materialize_expanded_keys(
        self,
        view: AttentionKeyComputeView,
        *,
        project_latent: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct the exact post-RoPE per-head keys for selected slots."""

        if not isinstance(view, AttentionKeyComputeView):
            raise TypeError(
                "MLA key materialization requires AttentionKeyComputeView, got "
                f"{type(view).__name__}."
            )
        payload = self._require_mla_payload(
            view,
            operation="MLA key materialization",
        )
        slots = view.active_slots
        if slots.ndim == 0:
            raise ValueError("MLA key materialization slots must not be scalar.")
        if slots.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "MLA key materialization slots must use int32 or int64, got "
                f"{slots.dtype}."
            )
        if slots.device != self.device:
            raise ValueError(
                "MLA key materialization slots are on the wrong device: "
                f"slots={slots.device} expected={self.device}."
            )

        flat_slots = slots.to(torch.long).reshape(-1)
        latent = payload.latent_cache.index_select(0, flat_slots).squeeze(1)
        rope = payload.rope_cache.index_select(0, flat_slots).squeeze(1)
        projected = project_latent(latent)
        qk_nope_head_dim = int(self.spec.qk_head_dim) - int(self.spec.rope_dim)
        projected_width = qk_nope_head_dim + int(self.spec.value_head_dim)
        expected_shape = (
            int(flat_slots.numel()),
            int(self.spec.local_q_heads) * projected_width,
        )
        if tuple(projected.shape) != expected_shape:
            raise RuntimeError(
                "MLA key projection returned an invalid shape: "
                f"got={tuple(projected.shape)} expected={expected_shape}."
            )
        if projected.device != self.device:
            raise RuntimeError(
                "MLA key projection returned the wrong device: "
                f"got={projected.device} expected={self.device}."
            )
        if projected.dtype != self.spec.activation_dtype:
            raise TypeError(
                "MLA key projection returned the wrong dtype: "
                f"got={projected.dtype} expected={self.spec.activation_dtype}."
            )

        expanded = projected.view(
            int(flat_slots.numel()),
            self.spec.local_q_heads,
            projected_width,
        )
        expanded_k_nope = expanded[..., :qk_nope_head_dim]
        expanded_rope = rope[:, None, :].expand(
            -1,
            self.spec.local_q_heads,
            -1,
        )
        keys = torch.cat((expanded_k_nope, expanded_rope), dim=-1)
        return keys.view(
            *slots.shape,
            self.spec.local_q_heads,
            self.spec.qk_head_dim,
        )

    def run_decode(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
    ) -> torch.Tensor:
        if not isinstance(view, DecodeComputeView):
            raise TypeError(
                f"MLA decode requires DecodeComputeView, got {type(view).__name__}."
            )
        self._require_mla_payload(view, operation="MLA decode")
        output = torch.empty(
            q_nope_absorbed.shape,
            dtype=q_nope_absorbed.dtype,
            device=q_nope_absorbed.device,
        )
        context = get_context()
        valid_batch_size = (
            int(q_nope_absorbed.shape[0]) if context.seqs is None else len(context.seqs)
        )
        return self.provider.run(
            q_nope_absorbed,
            q_rope,
            view,
            output,
            validation_scope=context.attention_validation_scope,
            valid_batch_size=valid_batch_size,
        )

    def _ensure_key_materializer(
        self,
        cache_manager,
        layer_idx: int,
        project_latent: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        layer_idx = int(layer_idx)
        key = (id(cache_manager), layer_idx)
        binding = self._key_materializer_bindings.get(key)
        if binding is not None and binding[0] is cache_manager:
            return

        def materialize(view: AttentionKeyComputeView) -> torch.Tensor:
            return self.materialize_expanded_keys(
                view,
                project_latent=project_latent,
            )

        cache_manager.register_attention_key_materializer(layer_idx, materialize)
        self._key_materializer_bindings[key] = (cache_manager, materialize)

    def run_cached_attention(
        self,
        q: torch.Tensor,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        latent: torch.Tensor,
        rope: torch.Tensor,
        *,
        project_latent: Callable[[torch.Tensor], torch.Tensor],
        absorb_query: Callable[[torch.Tensor], torch.Tensor],
        reconstruct_values: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Execute MLA over the active cache view for the current layer."""

        context = get_context()
        cache_manager = context.cache_manager
        sparse_controller = context.sparse_controller
        layer_idx = int(context.now_layer_idx)
        self._ensure_key_materializer(cache_manager, layer_idx, project_latent)
        with profiler.trace("mla.cache_store"):
            slot_mapping = cache_manager.store_attention_payload(
                layer_idx,
                MlaLatentWrite(latent=latent.unsqueeze(1), rope=rope.unsqueeze(1)),
            )
            cache_manager.on_kv_stored(layer_idx, latent, slot_mapping)

        temp_slots = None
        try:
            if context.is_prefill:
                selection = sparse_controller.get_prefill_selection(layer_idx)
                cache_manager.before_prefill_layer_attention(layer_idx, selection)
                view = cache_manager.build_prefill_compute_view(
                    layer_idx,
                    latent,
                    rope,
                    selection,
                )
                temp_slots = view.meta.temp_slots
                if context.cu_seqlens_q is None or context.cu_seqlens_q.numel() <= 1:
                    return torch.empty_like(q)
                self._require_mla_payload(view, operation="chunked prefill")
                plan = self.chunked_prefill.prepare(
                    view,
                    context.cu_seqlens_q,
                    context.attention_validation_scope,
                    prepare_history=False,
                )
                request = self.chunked_prefill.score_request(
                    plan, cache_manager.prefill_score_request(layer_idx, context.seqs)
                )
                if self._use_compressed_prefill is not None and self._use_compressed_prefill(plan, request):
                    # Explicit live tensors, including query absorption, latent
                    # output and value reconstruction. FA3's opaque workspace
                    # is measured by the existing startup memory profile.
                    required = estimate_mla_compressed_prefill_workspace_bytes(
                        plan=plan, spec=self.spec,
                        chunk_size=self.chunked_prefill.chunk_size,
                        hidden_size=self.hidden_size,
                        projection_chunk_size=self.projection_chunk_size,
                        score_request=request,
                    )
                    if required > self.prefill_workspace_bytes:
                        raise MemoryError(
                            f"MLA compressed prefill workspace exceeds budget: required={required} "
                            f"budget={self.prefill_workspace_bytes}. Reduce the token batch."
                        )
                    with profiler.trace("mla.prefill.latent_attention"):
                        latent_output, attention_lse = self.provider.run_compressed_prefill(
                            absorb_query(q_nope), q_rope, view, plan,
                        )
                        output = reconstruct_values(latent_output)
                    scores = self.chunked_prefill.score_compressed(
                        q, view, plan, absorb_query, request, attention_lse,
                    )
                else:
                    self.chunked_prefill.prepare(
                        view, context.cu_seqlens_q, context.attention_validation_scope,
                    )
                    required = estimate_mla_prefill_workspace_bytes(
                        plan=plan,
                        spec=self.spec,
                        chunk_size=self.chunked_prefill.chunk_size,
                        hidden_size=self.hidden_size,
                        projection_chunk_size=self.projection_chunk_size,
                        score_request=request,
                        kernel_workspace_bytes=self.chunked_prefill.kernel_workspace_bytes(plan),
                    )
                    if required > self.prefill_workspace_bytes:
                        raise MemoryError(
                            f"MLA chunked prefill workspace exceeds budget: required={required} "
                            f"budget={self.prefill_workspace_bytes}. Reduce the token batch or history chunk size."
                        )
                    output, attention_lse, scores = self.chunked_prefill.run(
                        q,
                        view,
                        context.cu_seqlens_q,
                        context.attention_validation_scope,
                        project_latent,
                        absorb_query,
                        request,
                    )
                b_start_loc = context.cu_seqlens_q[:-1]
                chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                cache_manager.collect_prefill_attention_score(
                    layer_idx,
                    q,
                    replace(view, token_scores=scores),
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                    attention_lse=attention_lse,
                )
                cache_manager.record_prefill_query(
                    layer_idx,
                    q,
                    view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                )
            else:
                selection = sparse_controller.get_decode_selection(layer_idx, q)
                q_nope_absorbed = absorb_query(q_nope)
                selection_query = cache_manager.build_decode_selection_query(
                    q,
                    mla_latent=q_nope_absorbed,
                    mla_rope=q_rope,
                )
                view = cache_manager.build_decode_compute_view(
                    layer_idx,
                    selection_query,
                    selection,
                    num_heads=self.spec.local_q_heads,
                    num_kv_heads=1,
                )
                output = reconstruct_values(
                    self.run_decode(q_nope_absorbed, q_rope, view)
                )
                cache_manager.record_decode_query(layer_idx, q)

            sparse_controller.on_layer_attention_end(layer_idx)
            cache_manager.on_layer_attention_end(layer_idx)
            return output
        finally:
            if temp_slots is not None and temp_slots.numel() > 0:
                cache_manager.release_layer_temp_slots(layer_idx, temp_slots)


__all__ = [
    "MLAAttention",
    "estimate_mla_prefill_workspace_bytes",
]
