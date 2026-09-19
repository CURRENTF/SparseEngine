from __future__ import annotations

from abc import abstractmethod

import torch

from sparseengine.engine.cache_manager import SparseSelection
from sparseengine.utils.log import logger
from sparseengine.utils.profiler import profiler

from .base import (
    DecodeSelectionRequest,
    LayerBatchSparseState,
    LayerEndEvent,
    PrefillSelectionRequest,
    SparseMethodRuntime,
    SparseStepContext,
)


def build_omnikv_keep_and_slots(*args, **kwargs):
    from sparseengine.kernels.triton.omnikv_fused import (
        build_omnikv_keep_and_slots as build,
    )

    return build(*args, **kwargs)


class DynamicSelectionRuntime(SparseMethodRuntime):
    def _prepare_decode_attention_score(
        self, layer_idx, state, batch_size, num_heads, max_len,
    ) -> None:
        # Even an all-short batch needs a valid (possibly empty) candidate slice.
        super()._prepare_decode_attention_score(
            layer_idx, state, batch_size, num_heads, max(max_len, self.num_sink),
        )

    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        return layer_idx in self.obs_layer_ids and not step.is_prefill

    def on_layer_end(self, event: LayerEndEvent) -> None:
        layer_idx = event.layer_idx
        if not self._is_kv_layer(layer_idx):
            self._debug_record_dynamic_selection(
                "on_layer_end",
                layer_idx,
                skipped="linear_attention",
            )
            return
        if event.layer_context.is_prefill:
            self._debug_record_dynamic_selection(
                "on_layer_end",
                layer_idx,
                skipped="prefill_full_attention",
            )
            return
        if layer_idx not in self.obs_layer_ids:
            self._debug_record_dynamic_selection(
                "on_layer_end",
                layer_idx,
                skipped="not_obs",
            )
            return

        self._debug_record_dynamic_selection(
            "on_layer_end",
            layer_idx,
            skipped="",
            method=str(self.sparse_method),
            is_prefill=bool(event.layer_context.is_prefill),
        )
        with profiler.record("sparse_on_layer_end"):
            state = self.layer_batch_sparse_states[layer_idx]
            if state.attn_score is None:
                raise ValueError("Attn Score hasn't been initialized")
            if state.attn_score.dim() == 3:
                state.attn_score = self._normalize_decode_scores(state)

            target_layers = []
            for target_idx in range(layer_idx + 1, self.num_layers):
                if not self._is_kv_layer(target_idx):
                    continue
                if target_idx in self.full_attention_layers:
                    break
                target_layers.append(target_idx)
            if not target_layers:
                raise RuntimeError(
                    "Dynamic sparse observation layer has no target KV layers: "
                    f"method={self.sparse_method} observation_layer={layer_idx} "
                    f"full_attention_layers={self.full_attention_layers}."
                )
            self._update_dynamic_indices(
                layer_idx,
                target_layers,
                event.forward_context,
            )

    @abstractmethod
    def _normalize_decode_scores(
        self,
        state: LayerBatchSparseState,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _update_dynamic_indices(
        self,
        obs_layer_idx: int,
        target_layers: list[int],
        context,
    ) -> None:
        raise NotImplementedError


class OmniKVRuntime(DynamicSelectionRuntime):
    def on_attention_end(self, event):
        if not getattr(self.cache_manager, "offload_enabled", False):
            return
        if event.forward_context.is_prefill or event.layer_idx not in self.obs_layer_ids:
            return
        with self.cache_manager.selection_stream():
            super().on_layer_end(LayerEndEvent(
                layer_idx=event.layer_idx,
                layer_context=event.forward_context,
                forward_context=event.forward_context,
            ))
            targets = []
            for layer in range(event.layer_idx + 1, self.num_layers):
                if layer in self.full_attention_layers:
                    break
                if self._is_kv_layer(layer):
                    targets.append((layer, self._build_selection(layer)))
            self.cache_manager.prefetch_selections(targets)

    def on_layer_end(self, event):
        if not getattr(self.cache_manager, "offload_enabled", False):
            super().on_layer_end(event)

    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        from sparseengine.operators.omnikv_score import OmniKVScoreSpec, prepare_omnikv_score_provider

        dtype = config.hf_config.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.lower().removeprefix('torch.'), torch.float32)
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        self._omnikv_score_provider = prepare_omnikv_score_provider(
            OmniKVScoreSpec(self.num_sink, self.num_recent, self.attn_softmax_scale, dtype),
            device=self.device,
        )
        from sparseengine.operators.omnikv_selection import OmniKVSelectionSpec, prepare_omnikv_selection
        self._omnikv_selection_provider = prepare_omnikv_selection(
            OmniKVSelectionSpec(self.num_sink, int(self.decode_keep_tokens)), device=self.device,
        )
        logger.info("OmniKV decode score provider: {}", self._omnikv_score_provider.name)
        self._omnikv_decode_attn_score_buffer: torch.Tensor | None = None
        self._omnikv_decode_selection_buffers: dict[
            tuple[int, tuple[int, ...], int, int, str],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._pending_decode_score_specs: list[tuple[int, int, int, int]] = []

    def clear_decode_attn_score_buffers(self) -> None:
        super().clear_decode_attn_score_buffers()
        self._omnikv_decode_attn_score_buffer = None
        self._omnikv_score_provider.clear()

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        tensors = self._omnikv_score_provider.keepalive_tensors()
        if self._omnikv_decode_attn_score_buffer is not None:
            tensors.append(self._omnikv_decode_attn_score_buffer)
        for buffers in self._omnikv_decode_selection_buffers.values():
            tensors.extend(buffers)
        return tensors

    def _begin_prepare_step(self, step: SparseStepContext) -> None:
        del step
        self._pending_decode_score_specs = []
        if getattr(self.cache_manager, "offload_enabled", False):
            self.cache_manager.begin_selection_step()

    def _prepare_decode_attention_score(
        self,
        layer_idx: int,
        state: LayerBatchSparseState,
        batch_size: int,
        num_heads: int,
        max_len: int,
    ) -> None:
        del state
        self._pending_decode_score_specs.append(
            (layer_idx, batch_size, num_heads, max(max_len, self.num_sink))
        )

    def _end_prepare_step(self, step: SparseStepContext) -> None:
        del step
        if self._pending_decode_score_specs:
            self._prepare_omnikv_decode_attn_score_buffer(
                self._pending_decode_score_specs,
                fill_value=-1e20,
            )
            for layer_idx, *_ in self._pending_decode_score_specs:
                state = self.layer_batch_sparse_states[layer_idx]
                self._omnikv_score_provider.prepare(state.attn_score, slot=id(state))

    def _prepare_omnikv_decode_attn_score_buffer(
        self,
        score_specs: list[tuple[int, int, int, int]],
        *,
        fill_value: float,
    ) -> None:
        if not score_specs:
            return
        if any(
            min(batch_size, num_heads, max_len) <= 0
            for _, batch_size, num_heads, max_len in score_specs
        ):
            raise RuntimeError(
                "OmniKV decode score workspace requires positive layer shapes: "
                f"specs={score_specs}."
            )
        max_batch_size = max(
            batch_size for _, batch_size, _, _ in score_specs
        )
        max_num_heads = max(num_heads for _, _, num_heads, _ in score_specs)
        max_len = max(length for _, _, _, length in score_specs)
        buffer = self._omnikv_decode_attn_score_buffer
        if (
            buffer is None
            or buffer.dtype != self.attn_score_dtype
            or buffer.device != self.device
            or int(buffer.shape[0]) < int(max_batch_size)
            or int(buffer.shape[1]) < int(max_num_heads)
            or int(buffer.shape[2]) < int(max_len)
        ):
            buffer = torch.empty(
                (int(max_batch_size), int(max_num_heads), int(max_len)),
                dtype=self.attn_score_dtype,
                device=self.device,
            )
            self._omnikv_decode_attn_score_buffer = buffer
        buffer[:max_batch_size, :max_num_heads, :max_len].fill_(fill_value)
        for layer_idx, batch_size, num_heads, layer_max_len in score_specs:
            self.layer_batch_sparse_states[int(layer_idx)].attn_score = buffer[
                :batch_size,
                :num_heads,
                :layer_max_len,
            ]

    def _get_omnikv_decode_selection_buffers(
        self,
        *,
        obs_layer_idx: int,
        target_layers: list[int],
        batch_size: int,
        max_context_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            int(obs_layer_idx),
            tuple(int(layer_idx) for layer_idx in target_layers),
            int(batch_size),
            int(max_context_len),
            str(device),
        )
        buffers = self._omnikv_decode_selection_buffers.get(key)
        if buffers is not None:
            return buffers
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "OmniKV decode CUDA graph capture requires selection buffers "
                "to be allocated by the eager capture warmup."
            )
        selection_shape = (int(batch_size), int(max_context_len))
        buffers = (
            torch.empty(selection_shape, dtype=torch.int32, device=device),
            torch.empty(selection_shape, dtype=torch.int32, device=device),
            torch.empty((int(batch_size),), dtype=torch.int32, device=device),
            torch.arange(int(batch_size), dtype=torch.int32, device=device),
        )
        self._omnikv_decode_selection_buffers[key] = buffers
        return buffers

    def build_prefill_selection(
        self,
        request: PrefillSelectionRequest,
    ) -> SparseSelection:
        return self._build_selection(request.layer_idx)

    def build_decode_selection(
        self,
        request: DecodeSelectionRequest,
    ) -> SparseSelection:
        return self._build_selection(request.layer_idx)

    def _build_selection(self, layer_idx: int) -> SparseSelection:
        if not self._is_kv_layer(layer_idx):
            raise RuntimeError(
                f"layer_idx={layer_idx} is linear_attention and has no KV sparse selection"
            )
        if layer_idx in self.full_attention_layers:
            return self._full_selection(layer_idx)
        state = self.layer_batch_sparse_states[layer_idx]
        if state.active_slots is not None:
            active_slots = state.active_slots
            logger.debug("active_slots 是被 omnikv 选到的 slots")
        else:
            active_slots = None
            logger.debug("active_slots is None")
        return SparseSelection(
            kind="slots",
            req_indices=state.req_indices,
            context_lens=state.context_lens,
            max_context_len=state.max_context_len,
            attn_score=state.attn_score,
            active_indices=state.active_indices,
            active_slots=active_slots,
            global_req_indices=state.global_req_indices,
        )

    def _normalize_decode_scores(
        self,
        state: LayerBatchSparseState,
    ) -> torch.Tensor:
        return self._omnikv_score_provider.run(
            state.attn_score, state.context_lens, slot=id(state),
            selection_keep=int(self.decode_keep_tokens),
        )

    def _update_dynamic_indices(
        self,
        obs_layer_idx: int,
        target_layers: list[int],
        context,
    ) -> None:
        with profiler.record("sparse_update_dynamic_indices"):
            self._debug_record_dynamic_selection(
                "update_dynamic",
                obs_layer_idx,
                method=str(self.sparse_method),
                is_prefill=bool(context.is_prefill),
                is_dynamic_deltakv=False,
                target_layers=[int(layer_idx) for layer_idx in target_layers],
            )
            obs_state = self.layer_batch_sparse_states[obs_layer_idx]
            token_scores = obs_state.attn_score
            batch_size, _max_len = token_scores.shape
            if context.is_prefill:
                chunk_lens = (
                    context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                )
                hist_lens = (
                    obs_state.context_lens - chunk_lens - self.num_recent
                )
            else:
                hist_lens = obs_state.context_lens - self.num_recent
            hist_lens = hist_lens.clamp_min(self.num_sink)

            rel_hist_lens = hist_lens - self.num_sink
            if context.is_prefill:
                search_scores = token_scores[:, self.num_sink:]
                mask = (
                    torch.arange(search_scores.size(1), device=self.device)
                    >= rel_hist_lens.unsqueeze(1)
                )
                search_scores.masked_fill_(mask, -1e10)
            decode_keep = int(self.decode_keep_tokens)
            candidate_capacity = int(token_scores.shape[1]) - self.num_sink
            k_max = min(decode_keep, candidate_capacity)
            if k_max > 0:
                topk_lens = rel_hist_lens.clamp(
                    min=0,
                    max=k_max,
                ).to(torch.int32)
                if context.is_prefill:
                    topk_indices = (
                        search_scores.topk(k_max, dim=1, sorted=True)
                        .indices.to(torch.int32) + self.num_sink
                    )
                else:
                    topk_indices = self._omnikv_selection_provider.select(
                        token_scores, rel_hist_lens.clamp(max=candidate_capacity).to(torch.int32), k_max,
                    )
            else:
                topk_lens = torch.zeros(
                    (batch_size,),
                    dtype=torch.int32,
                    device=self.device,
                )
                topk_indices = torch.empty(
                    (batch_size, 0),
                    device=self.device,
                    dtype=torch.int32,
                )

            if context.is_prefill:
                chunk_lens = (
                    context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                )
                max_recent_or_chunk = int(chunk_lens.max().item()) + int(
                    self.num_recent
                )
            else:
                max_recent_or_chunk = int(self.num_recent)
            max_sparse_context_len = (
                int(self.num_sink) + int(k_max) + max_recent_or_chunk
            )
            slot_source_layer = int(target_layers[0])
            graph_outputs = None
            if not context.is_prefill and bool(
                getattr(self.config, "decode_graph", False)
            ):
                graph_outputs = self._get_omnikv_decode_selection_buffers(
                    obs_layer_idx=obs_layer_idx,
                    target_layers=target_layers,
                    batch_size=batch_size,
                    max_context_len=max_sparse_context_len,
                    device=token_scores.device,
                )
                (
                    keep_indices_out,
                    active_slots_out,
                    new_context_lens_out,
                    local_req_indices,
                ) = graph_outputs
            else:
                keep_indices_out = None
                active_slots_out = None
                new_context_lens_out = None
                local_req_indices = torch.arange(
                    batch_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            output_kwargs = {}
            if graph_outputs is not None:
                output_kwargs = {
                    "keep_indices_out": keep_indices_out,
                    "active_slots_out": active_slots_out,
                    "new_context_lens_out": new_context_lens_out,
                }
            keep_indices, active_slots, new_context_lens = (
                build_omnikv_keep_and_slots(
                    topk_indices,
                    topk_lens,
                    hist_lens,
                    obs_state.context_lens - hist_lens,
                    self.cache_manager.get_layer_buffer_req_to_token_slots(
                        slot_source_layer
                    ),
                    obs_state.req_indices,
                    self.num_sink,
                    context_lens=obs_state.context_lens,
                    max_s=max_sparse_context_len,
                    **output_kwargs,
                )
            )
            if graph_outputs is not None and (
                keep_indices is not keep_indices_out
                or active_slots is not active_slots_out
                or new_context_lens is not new_context_lens_out
            ):
                raise RuntimeError(
                    "OmniKV decode CUDA graph selection builder did not preserve "
                    "caller-owned output buffers."
                )

            for layer_idx in target_layers:
                target_state = self.layer_batch_sparse_states[layer_idx]
                target_state.active_indices = keep_indices
                target_state.active_slots = active_slots
                target_state.context_lens = new_context_lens
                target_state.max_context_len = max_sparse_context_len
                target_state.req_indices = local_req_indices

    def reset_decode_attn_scores_for_graph(
        self,
        refs: dict[int, dict[str, object]],
    ) -> bool:
        score_views = []
        for layer_idx in self.obs_layer_ids:
            score = refs.get(int(layer_idx), {}).get("attn_score")
            if score is None:
                continue
            if not isinstance(score, torch.Tensor) or score.dim() != 3:
                raise RuntimeError(
                    "OmniKV decode CUDA graph requires raw [B, H, L] score "
                    f"views at graph input: layer={layer_idx} score={score!r}."
                )
            score_views.append(score)
        if not score_views:
            return False
        storage_ptr = score_views[0].untyped_storage().data_ptr()
        if any(
            score.untyped_storage().data_ptr() != storage_ptr
            for score in score_views[1:]
        ):
            raise RuntimeError(
                "OmniKV observation layers do not share one decode score workspace."
            )
        reset_views = {}
        for score in score_views:
            key = (score.data_ptr(), tuple(score.shape), tuple(score.stride()))
            reset_views.setdefault(key, score)
        for score in reset_views.values():
            score.fill_(-1e20)
        return True


class DeltaKVRuntime(DynamicSelectionRuntime):
    def build_prefill_selection(
        self,
        request: PrefillSelectionRequest,
    ) -> SparseSelection:
        return self._build_selection(
            request.layer_idx,
            is_prefill=True,
            context=request.forward_context,
        )

    def build_decode_selection(
        self,
        request: DecodeSelectionRequest,
    ) -> SparseSelection:
        return self._build_selection(
            request.layer_idx,
            is_prefill=False,
            context=request.forward_context,
        )

    def _build_selection(
        self,
        layer_idx: int,
        *,
        is_prefill: bool,
        context,
    ) -> SparseSelection:
        if not self._is_kv_layer(layer_idx):
            raise RuntimeError(
                f"layer_idx={layer_idx} is linear_attention and has no KV sparse selection"
            )
        state = self.layer_batch_sparse_states[layer_idx]
        if layer_idx in self.full_attention_layers:
            return SparseSelection(
                kind="full",
                req_indices=state.global_req_indices,
                context_lens=state.context_lens,
                max_context_len=state.max_context_len,
                attn_score=state.attn_score,
                global_req_indices=state.global_req_indices,
            )
        chunk_lens = None
        if is_prefill and (
            context.cu_seqlens_q is not None
            and context.cu_seqlens_q.numel() > 1
        ):
            chunk_lens = (
                context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
            ).to(torch.int32)
        return SparseSelection(
            kind="deltakv",
            req_indices=state.global_req_indices,
            context_lens=state.context_lens,
            max_context_len=state.max_context_len,
            attn_score=state.attn_score,
            active_compressed_indices=state.active_compressed_indices,
            global_req_indices=state.global_req_indices,
            chunk_lens=chunk_lens,
            release_temp_slots=state.deltakv_free_temp_slots,
        )

    def _normalize_decode_scores(
        self,
        state: LayerBatchSparseState,
    ) -> torch.Tensor:
        compressed_lens = self.cache_manager.get_compressed_lens(
            state.req_indices
        )
        return self._decode_softmax_token_scores(
            state.attn_score,
            candidate_start=self.num_sink,
            candidate_lens=compressed_lens,
        )

    def _update_dynamic_indices(
        self,
        obs_layer_idx: int,
        target_layers: list[int],
        context,
    ) -> None:
        with profiler.record("sparse_update_dynamic_indices"):
            self._debug_record_dynamic_selection(
                "update_dynamic",
                obs_layer_idx,
                method=str(self.sparse_method),
                is_prefill=bool(context.is_prefill),
                is_dynamic_deltakv=True,
                target_layers=[int(layer_idx) for layer_idx in target_layers],
            )
            obs_state = self.layer_batch_sparse_states[obs_layer_idx]
            token_scores = obs_state.attn_score
            batch_size, _max_len = token_scores.shape
            if context.is_prefill:
                chunk_lens = (
                    context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                )
                hist_lens = (
                    obs_state.context_lens - chunk_lens - self.num_recent
                )
            else:
                hist_lens = obs_state.context_lens - self.num_recent

            search_scores = token_scores[:, self.num_sink:]
            rel_hist_lens = self.cache_manager.get_compressed_lens(
                obs_state.req_indices
            )
            mask = (
                torch.arange(search_scores.size(1), device=self.device)
                >= rel_hist_lens.unsqueeze(1)
            )
            search_scores.masked_fill_(mask, -1e10)
            if (
                self.dynamic_deltakv_topk_tiebreak
                and not context.is_prefill
                and search_scores.numel() > 0
            ):
                pos_key = torch.arange(
                    search_scores.size(1),
                    device=search_scores.device,
                    dtype=torch.float32,
                )
                pos_key = pos_key / max(1, int(search_scores.size(1)))
                score_scale = search_scores.detach().abs().float().clamp_min(1.0)
                search_scores = search_scores.float() + score_scale * (
                    pos_key.unsqueeze(0) * 1.0e-6
                )
                search_scores.masked_fill_(mask, -1e10)

            if not context.is_prefill:
                k_max = min(
                    int(self.decode_keep_tokens),
                    int(search_scores.size(1)),
                )
                if k_max > 0:
                    topk_indices = search_scores.topk(
                        k_max,
                        dim=1,
                        sorted=True,
                    ).indices.to(torch.int32)
                else:
                    topk_indices = torch.empty(
                        (batch_size, 0),
                        device=self.device,
                        dtype=torch.int32,
                    )
            else:
                topk_list = []
                k_list = []
                for batch_idx in range(batch_size):
                    available = int(rel_hist_lens[batch_idx].item())
                    k = min(
                        int(self.decode_keep_tokens),
                        int(search_scores.size(1)),
                        max(0, available),
                    )
                    k_list.append(k)
                    if k <= 0:
                        topk_list.append(
                            torch.empty(
                                (0,),
                                device=self.device,
                                dtype=torch.int32,
                            )
                        )
                    else:
                        topk_list.append(
                            search_scores[batch_idx]
                            .topk(k, dim=0)
                            .indices.to(torch.int32)
                        )
                k_max = max(k_list) if k_list else 0
                if k_max > 0:
                    topk_indices = torch.full(
                        (batch_size, k_max),
                        -1,
                        device=self.device,
                        dtype=torch.int32,
                    )
                    for batch_idx, k in enumerate(k_list):
                        if k > 0:
                            topk_indices[batch_idx, :k] = topk_list[batch_idx]
                else:
                    topk_indices = torch.empty(
                        (batch_size, 0),
                        device=self.device,
                        dtype=torch.int32,
                    )

            if self.debug_dynamic_selection_detail:
                debug_k = min(16, int(search_scores.shape[1]))
                detail = {
                    "rel_hist_lens_preview": self._debug_tensor_preview(
                        rel_hist_lens,
                        16,
                    ),
                    "search_scores_shape": tuple(
                        int(dim) for dim in search_scores.shape
                    ),
                    "topk_shape": tuple(int(dim) for dim in topk_indices.shape),
                    "topk_rel_preview": self._debug_tensor_preview(
                        topk_indices,
                        32,
                    ),
                    "topk_abs_preview": self._debug_tensor_preview(
                        topk_indices + int(self.num_sink),
                        32,
                    ),
                }
                if debug_k > 0:
                    debug_scores, debug_rel = search_scores.topk(
                        debug_k,
                        dim=1,
                        sorted=True,
                    )
                    detail.update(
                        search_top_rel_preview=self._debug_tensor_preview(
                            debug_rel,
                            32,
                        ),
                        search_top_abs_preview=self._debug_tensor_preview(
                            debug_rel + int(self.num_sink),
                            32,
                        ),
                        search_top_score_preview=self._debug_tensor_preview(
                            debug_scores,
                            32,
                        ),
                    )
                self._debug_record_dynamic_selection(
                    "dynamic_topk_detail",
                    obs_layer_idx,
                    **detail,
                )

            for layer_idx in target_layers:
                target_state = self.layer_batch_sparse_states[layer_idx]
                target_state.active_compressed_indices = topk_indices
                target_state.context_lens = obs_state.context_lens
                target_state.req_indices = obs_state.req_indices
                target_state.global_req_indices = obs_state.req_indices
                target_state.deltakv_free_temp_slots = (
                    layer_idx == target_layers[-1]
                )

    def finish_step(self, step: SparseStepContext) -> None:
        if step.is_prefill:
            if getattr(
                self.cache_manager,
                "defer_prefill_eviction",
                lambda: False,
            )():
                return
            self._deltakv_eviction(step.seqs, step.forward_context)
            return
        self._deltakv_eviction(step.seqs, step.forward_context)

    @torch.no_grad()
    def _deltakv_eviction(self, seqs, context) -> None:
        self.cache_manager.deltakv_evict(seqs)
