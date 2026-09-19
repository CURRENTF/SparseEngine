from __future__ import annotations

import os

import torch

from sparseengine.engine.sequence import Sequence

from .deltakv_less_memory import DeltaKVLessMemoryCacheManager


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


class DeltaKVLessMemoryCudaGraphCacheManager(DeltaKVLessMemoryCacheManager):
    """Unified DeltaKV runtime with graph-safe static workspaces.

    Public DeltaKV now has one manager. CUDA Graph is an execution mode on top
    of the same static decode path, and context capacity is selected by the
    shared runner's bucket plan instead of DeltaKV-specific per-request policy.
    """

    def select_decode_cuda_graph_batch_size(self, real_batch_size: int, capture_sizes: list[int]) -> int | None:
        # Use the shared batch buckets (1, 2, 4, 8, ...) instead of capturing one
        # graph per exact DeltaKV batch size. The runner still checks coverage.
        del real_batch_size, capture_sizes
        return None

    def decode_graph_force_eager(self) -> bool:
        return _env_bool("SPARSEENGINE_DELTAKV_CUDAGRAPH_FORCE_EAGER", False)

    def _decode_cuda_graph_memory_reserve_bytes(self) -> int:
        if not bool(getattr(self.config, "decode_graph", False)):
            return 0

        env_value = os.getenv("SPARSEENGINE_DELTAKV_CUDAGRAPH_RESERVE_BYTES")
        if env_value is None:
            return 0

        try:
            reserve_bytes = int(env_value)
        except ValueError as exc:
            raise ValueError(
                "SPARSEENGINE_DELTAKV_CUDAGRAPH_RESERVE_BYTES must be a non-negative integer, "
                f"got {env_value!r}."
            ) from exc
        if reserve_bytes < 0:
            raise ValueError(
                "SPARSEENGINE_DELTAKV_CUDAGRAPH_RESERVE_BYTES must be a non-negative integer, "
                f"got {env_value!r}."
            )
        return reserve_bytes

    def _extra_workspace_reserve_bytes(self) -> int:
        return self._decode_cuda_graph_memory_reserve_bytes()

    def _is_cuda_graph_capturing(self) -> bool:
        platform = getattr(self, "platform", None)
        if platform is not None:
            return platform.is_stream_capturing()
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    def _raise_if_capture_allocation(self, workspace: str, shape: object) -> None:
        if self._is_cuda_graph_capturing():
            raise RuntimeError(
                "DeltaKV less-memory CUDA Graph capture tried to allocate an uncached "
                f"{workspace} workspace with shape={shape!r}. Warmup must create every "
                "shape before capture; check decode_keep_tokens, capture_sizes, and context-capacity policy."
            )

    def _decode_graph_capture_size_capacity(self, requested_batch_size: int) -> int:
        capacity = max(1, int(requested_batch_size))
        # Graph decode uses exact real batch sizes, bounded by max_decoding_seqs
        # and the configured capture sizes.  Do not include the prefill/admission
        # batch cap here: doing so would reserve unnecessary DeltaKV raw slots for
        # every captured graph.
        config = getattr(self, "config", None)
        value = getattr(config, "max_decoding_seqs", None) if config is not None else None
        if value is not None:
            capacity = max(capacity, int(value))

        capture_sizes = getattr(config, "decode_graph_capture_sizes", None) if config is not None else []
        capture_sizes = capture_sizes or []
        if isinstance(capture_sizes, str):
            raw = capture_sizes.strip().lower()
            if raw in {"", "auto"}:
                capture_sizes = []
            else:
                capture_sizes = [part.strip() for part in capture_sizes.split(",") if part.strip()]
        for size in capture_sizes:
            capacity = max(capacity, int(size))
        return capacity

    def _decode_graph_topk_width_capacity(self, requested_k_max: int) -> int:
        requested_k_max = max(0, int(requested_k_max))
        decode_keep = max(0, int(getattr(self.config, "decode_keep_tokens", requested_k_max) or 0))
        sink = max(0, int(getattr(self.config, "sink_keep_tokens", 0) or 0))
        static_cap = getattr(self, "_decode_static_max_context_len", None)
        if static_cap is None:
            graph_k_max = decode_keep
        else:
            graph_k_max = min(decode_keep, max(0, int(static_cap) - sink))
        return max(requested_k_max, graph_k_max)

    def _prewarm_decode_graph_static_workspaces(self, graph_batch_size: int, device: torch.device) -> None:
        if self._is_cuda_graph_capturing():
            return

        graph_batch_size = max(1, int(graph_batch_size))
        sink = max(0, int(getattr(self.config, "sink_keep_tokens", 0) or 0))
        top_k = self._decode_graph_topk_width_capacity(0)
        max_buffer = self._deltakv_decode_static_max_buffer()

        k_values = {int(top_k)}
        obs_layers = [int(layer) for layer in getattr(self.config, "obs_layer_ids", []) or []]
        if not obs_layers:
            k_values.add(0)
        else:
            first_obs_layer = min(obs_layers)
            if any(int(layer_idx) < first_obs_layer for layer_idx in getattr(self, "deltakv_layer_ids", [])):
                k_values.add(0)

        for k_max in sorted(k_values):
            max_s = sink + int(k_max) + int(max_buffer)
            self._ensure_decode_static_temp_slots(graph_batch_size, k_max)
            self._ensure_decode_static_plan_buffers(graph_batch_size, k_max, max_s, device)
            self._ensure_materialized_sparse_view_capacity(graph_batch_size, max_s, device)

        if self._full_layer_quant_enabled():
            static_cap = getattr(self, "_decode_static_max_context_len", None)
            if static_cap is None:
                static_cap = getattr(self.full_layer_batch_states, "max_context_len", None)
            if static_cap is not None and int(static_cap) > 0:
                self._ensure_full_layer_quant_decode_workspace(graph_batch_size, int(static_cap))

    def _ensure_materialized_sparse_view_capacity(self, batch_size: int, width: int, device: torch.device) -> None:
        batch_size = int(batch_size)
        width = int(width)
        total = batch_size * width
        if total > int(self.deltakv_materialized_compute_num_slots):
            raise RuntimeError(
                "DeltaKV materialized sparse workspace is too small: "
                f"need={total} capacity={int(self.deltakv_materialized_compute_num_slots)} "
                f"batch={batch_size} width={width}. Increase max_num_batched_tokens or reduce decode keep tokens."
            )

        cap_batch = self._decode_graph_capture_size_capacity(batch_size)
        alloc_width = max(1, width)
        cache = getattr(self, "_deltakv_graph_materialized_sparse_view_by_capacity", None)
        if cache is None:
            cache = {}
            self._deltakv_graph_materialized_sparse_view_by_capacity = cache
        key = (cap_batch, alloc_width, str(device))
        if cache.get(key) is not None:
            return
        self._raise_if_capture_allocation("materialized sparse view", (cap_batch, alloc_width, str(device)))
        active = torch.empty((cap_batch, alloc_width), dtype=torch.int32, device=device)
        flat = torch.arange(max(1, cap_batch * alloc_width), dtype=torch.int32, device=device)
        local_req = torch.arange(cap_batch, dtype=torch.int32, device=device)
        cache[key] = (active, flat, local_req)

    def _ensure_decode_static_temp_slots(self, batch_size: int, k_max: int) -> torch.Tensor:
        batch_size = int(batch_size)
        k_max = int(k_max)
        if batch_size < 0 or k_max < 0:
            raise ValueError(f"Invalid static DeltaKV temp shape: batch={batch_size}, k={k_max}.")

        if k_max == 0:
            cache = getattr(self, "_deltakv_graph_static_empty_temp_slots_by_shape", None)
            if cache is None:
                cache = {}
                self._deltakv_graph_static_empty_temp_slots_by_shape = cache
            key = (batch_size, str(torch.device("cuda")))
            slots = cache.get(key)
            if slots is None:
                self._raise_if_capture_allocation("empty decode temp-slot", (batch_size, k_max))
                slots = torch.empty((batch_size, 0), device=self.device, dtype=torch.int32)
                cache[key] = slots
            return slots

        cap_batch = self._decode_graph_capture_size_capacity(batch_size)
        cap_k = self._decode_graph_topk_width_capacity(k_max)
        cache = getattr(self, "_deltakv_graph_static_temp_slots_by_capacity", None)
        if cache is None:
            cache = {}
            self._deltakv_graph_static_temp_slots_by_capacity = cache
        key = (cap_batch, cap_k, str(torch.device("cuda")))
        base_slots = cache.get(key)
        if base_slots is None:
            self._raise_if_capture_allocation("decode temp-slot", (cap_batch, cap_k))
            base_slots = self._allocate_temp_deltakv_full(cap_batch * cap_k).to(torch.int32).view(cap_batch, cap_k)
            self._deltakv_static_temp_slots_reserved_total = int(
                getattr(self, "_deltakv_static_temp_slots_reserved_total", 0) or 0
            ) + int(cap_batch * cap_k)
            cache[key] = base_slots
        if batch_size > int(base_slots.shape[0]) or k_max > int(base_slots.shape[1]):
            raise RuntimeError(
                "DeltaKV graph temp-slot capacity is smaller than requested shape: "
                f"capacity={tuple(base_slots.shape)}, requested=({batch_size}, {k_max})."
            )
        return base_slots[:batch_size, :k_max]

    def _ensure_decode_static_plan_buffers(self, batch_size: int, k_max: int, max_s: int, device: torch.device):
        batch_size = int(batch_size)
        k_max = int(k_max)
        max_s = int(max_s)
        if batch_size < 0 or k_max < 0 or max_s < 0:
            raise ValueError(f"Invalid static DeltaKV plan shape: batch={batch_size}, k={k_max}, max_s={max_s}.")

        cap_batch = self._decode_graph_capture_size_capacity(batch_size)
        cache = getattr(self, "_deltakv_graph_static_plan_buffers_by_capacity", None)
        if cache is None:
            cache = {}
            self._deltakv_graph_static_plan_buffers_by_capacity = cache
        key = (cap_batch, k_max, max_s, str(device))
        buffers = cache.get(key)
        if buffers is None:
            self._raise_if_capture_allocation("decode plan", (cap_batch, k_max, max_s, str(device)))
            active_slots = torch.empty((cap_batch, max_s), device=device, dtype=torch.int32)
            active_pos = torch.empty((cap_batch, max_s), device=device, dtype=torch.int32)
            local_req = torch.arange(cap_batch, device=device, dtype=torch.int32)
            new_context_lens = torch.empty((cap_batch,), device=device, dtype=torch.int32)
            recon_size = cap_batch * k_max
            recon_pos = torch.empty((recon_size,), device=device, dtype=torch.int32)
            recon_latent = torch.empty((recon_size,), device=device, dtype=torch.int32)
            recon_out_slot = torch.empty((recon_size,), device=device, dtype=torch.int32)
            empty = torch.empty((0,), device=device, dtype=torch.int32)
            buffers = (active_slots, active_pos, local_req, new_context_lens, empty, recon_pos, recon_latent, recon_out_slot)
            cache[key] = buffers

        active_slots, active_pos, local_req, new_context_lens, empty, recon_pos, recon_latent, recon_out_slot = buffers
        if batch_size > int(active_slots.shape[0]) or max_s > int(active_slots.shape[1]):
            raise RuntimeError(
                "DeltaKV graph plan capacity is smaller than requested shape: "
                f"capacity=({int(active_slots.shape[0])}, {int(active_slots.shape[1])}), "
                f"requested=({batch_size}, {max_s})."
            )
        recon_n = batch_size * k_max
        return (
            active_slots[:batch_size, :max_s],
            active_pos[:batch_size, :max_s],
            local_req[:batch_size],
            new_context_lens[:batch_size],
            empty,
            recon_pos[:recon_n],
            recon_latent[:recon_n],
            recon_out_slot[:recon_n],
        )

    def _ensure_materialized_sparse_view(self, batch_size: int, width: int, device: torch.device):
        batch_size = int(batch_size)
        width = int(width)
        total = batch_size * width
        if total > int(self.deltakv_materialized_compute_num_slots):
            raise RuntimeError(
                "DeltaKV materialized sparse workspace is too small: "
                f"need={total} capacity={int(self.deltakv_materialized_compute_num_slots)} "
                f"batch={batch_size} width={width}. Increase max_num_batched_tokens or reduce decode keep tokens."
            )

        cap_batch = self._decode_graph_capture_size_capacity(batch_size)
        alloc_width = max(1, width)
        cache = getattr(self, "_deltakv_graph_materialized_sparse_view_by_capacity", None)
        if cache is None:
            cache = {}
            self._deltakv_graph_materialized_sparse_view_by_capacity = cache
        key = (cap_batch, alloc_width, str(device))
        buffers = cache.get(key)
        if buffers is None:
            self._raise_if_capture_allocation("materialized sparse view", (cap_batch, alloc_width, str(device)))
            active = torch.empty((cap_batch, alloc_width), dtype=torch.int32, device=device)
            flat = torch.arange(max(1, cap_batch * alloc_width), dtype=torch.int32, device=device)
            local_req = torch.arange(cap_batch, dtype=torch.int32, device=device)
            buffers = (active, flat, local_req)
            cache[key] = buffers
        active, flat, local_req = buffers
        if batch_size > int(active.shape[0]) or width > int(active.shape[1]):
            raise RuntimeError(
                "DeltaKV graph materialized sparse view capacity is smaller than requested shape: "
                f"capacity={tuple(active.shape)}, requested=({batch_size}, {width})."
            )
        active_view = active[:batch_size, :width]
        if total > 0:
            active_view.copy_(flat[:total].view(batch_size, width))
        self._deltakv_materialized_active_slots = active
        self._deltakv_materialized_flat_slots = flat
        self._deltakv_materialized_local_req = local_req
        return active_view, local_req[:batch_size]

    def _ensure_decode_static_materialized_slots(self, active_slots: torch.Tensor) -> torch.Tensor:
        total = int(active_slots.numel())
        if total == 0:
            cache = getattr(self, "_deltakv_graph_static_materialized_empty_by_shape", None)
            if cache is None:
                cache = {}
                self._deltakv_graph_static_materialized_empty_by_shape = cache
            key = (tuple(active_slots.shape), str(active_slots.device))
            slots = cache.get(key)
            if slots is None:
                self._raise_if_capture_allocation("empty post-RoPE materialized slot", key)
                slots = torch.empty((0,), device=active_slots.device, dtype=torch.int32)
                cache[key] = slots
            return slots

        if active_slots.dim() == 2:
            batch_size = int(active_slots.shape[0])
            width = int(active_slots.shape[1])
            cap_batch = self._decode_graph_capture_size_capacity(batch_size)
            cap_total = cap_batch * width
            key = (cap_batch, width, str(active_slots.device))
        else:
            cap_total = total
            key = (tuple(active_slots.shape), str(active_slots.device))

        cache = getattr(self, "_deltakv_graph_static_materialized_slots_by_capacity", None)
        if cache is None:
            cache = {}
            self._deltakv_graph_static_materialized_slots_by_capacity = cache
        base_slots = cache.get(key)
        if base_slots is None:
            self._raise_if_capture_allocation("post-RoPE materialized slot", key)
            base_slots = self._allocate_temp_deltakv_full(cap_total).to(torch.int32)
            self._deltakv_static_temp_slots_reserved_total = int(
                getattr(self, "_deltakv_static_temp_slots_reserved_total", 0) or 0
            ) + int(cap_total)
            cache[key] = base_slots
        if total > int(base_slots.numel()):
            raise RuntimeError(
                "DeltaKV graph post-RoPE materialized slot capacity is smaller than requested shape: "
                f"capacity={int(base_slots.numel())}, requested={total}."
            )
        return base_slots[:total]

    def _full_layer_quant_temp_cache_needs_alloc(self, num_slots: int) -> bool:
        num_slots = max(1, int(num_slots))
        k_cache = getattr(self, "_full_layer_quant_k_cache", None)
        v_cache = getattr(self, "_full_layer_quant_v_cache", None)
        return (
            k_cache is None
            or v_cache is None
            or k_cache.device.type != "cuda"
            or v_cache.device.type != "cuda"
            or int(k_cache.shape[0]) < num_slots
            or int(v_cache.shape[0]) < num_slots
        )

    def _ensure_full_layer_quant_temp_cache(self, num_slots: int):
        num_slots = max(1, int(num_slots))
        if self._full_layer_quant_temp_cache_needs_alloc(num_slots):
            self._raise_if_capture_allocation("full-layer quant temp cache", (num_slots, self.num_kv_heads, self.head_dim))
            self._full_layer_quant_k_cache = torch.empty(
                num_slots,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.hf_config.dtype,
                device=self.device,
            )
            self._full_layer_quant_v_cache = torch.empty_like(self._full_layer_quant_k_cache)
            return
        return super()._ensure_full_layer_quant_temp_cache(num_slots)

    def _ensure_full_layer_quant_decode_workspace(self, batch_size: int, max_len: int):
        batch_size = max(1, int(batch_size))
        max_len = max(1, int(max_len))
        if self._is_cuda_graph_capturing():
            total_slots = batch_size * max_len
            if self._full_layer_quant_temp_cache_needs_alloc(total_slots):
                self._raise_if_capture_allocation("full-layer quant decode KV", (total_slots, self.num_kv_heads, self.head_dim))
            active = getattr(self, "_full_layer_quant_active_slots", None)
            if active is None or int(active.shape[0]) < batch_size or int(active.shape[1]) < max_len:
                self._raise_if_capture_allocation("full-layer quant active slots", (batch_size, max_len))
            local_req = getattr(self, "_full_layer_quant_local_req", None)
            if local_req is None or int(local_req.numel()) < batch_size:
                self._raise_if_capture_allocation("full-layer quant local req", (batch_size,))
            positions = getattr(self, "_full_layer_quant_positions", None)
            if positions is None or int(positions.numel()) < max_len:
                self._raise_if_capture_allocation("full-layer quant positions", (max_len,))
            out_slots = getattr(self, "_full_layer_quant_out_slots", None)
            if out_slots is None or int(out_slots.numel()) < total_slots:
                self._raise_if_capture_allocation("full-layer quant out slots", (total_slots,))
        return super()._ensure_full_layer_quant_decode_workspace(batch_size, max_len)

    def _ensure_deltakv_postrope_dummy_slot(self) -> torch.Tensor:
        """Reserve one stable scratch slot used to mask invalid padded entries."""
        slot = getattr(self, "_deltakv_postrope_dummy_slot", None)
        if slot is not None:
            return slot
        if self._is_cuda_graph_capturing():
            raise RuntimeError(
                "DeltaKV less-memory CUDA Graph capture reached _set_postrope_slots before "
                "the dummy post-RoPE slot was reserved. Call prepare_decode_static/warmup first."
            )
        slot = self._allocate_temp_deltakv_full(1).to(torch.int32)
        self._deltakv_static_temp_slots_reserved_total = int(
            getattr(self, "_deltakv_static_temp_slots_reserved_total", 0) or 0
        ) + 1
        self._deltakv_postrope_dummy_slot = slot
        return slot

    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        if hasattr(self, "_deltakv_reset_view_cache"):
            self._deltakv_reset_view_cache()
        if len(seqs) <= 0:
            raise ValueError("Static DeltaKV decode requires a non-empty real decode batch.")
        self._ensure_deltakv_postrope_dummy_slot()
        self._prewarm_decode_graph_static_workspaces(int(input_ids.numel()), input_ids.device)
        return super().prepare_decode_static(seqs, input_ids, positions, slot_mapping, context_lens, req_indices)

    def _set_postrope_slots(self, layer_idx: int, slots: torch.Tensor, *, validate: bool = True) -> None:
        masks = getattr(self, "_deltakv_postrope_slot_mask", None)
        if masks is None or slots is None:
            return
        l_idx = self.deltakv_layer_to_idx[int(layer_idx)]
        layer_mask = masks[l_idx]
        layer_mask.zero_()
        if slots.numel() == 0:
            return

        slots_i32 = slots.to(layer_mask.device).to(torch.int32).flatten()
        if not validate:
            layer_mask.index_fill_(0, slots_i32.to(torch.long), True)
            return

        valid = (slots_i32 >= 0) & (slots_i32 < int(layer_mask.shape[0]))
        if self._is_cuda_graph_capturing():
            dummy_slot = self._ensure_deltakv_postrope_dummy_slot().to(layer_mask.device).to(torch.int32)
            safe_slots = torch.where(valid, slots_i32, dummy_slot[0].expand_as(slots_i32)).to(torch.long)
            layer_mask.index_fill_(0, safe_slots, True)
            return

        if not bool(valid.any()):
            return
        layer_mask.index_fill_(0, slots_i32[valid].to(torch.long), True)

    def get_decode_block_seq(self, layer_idx: int, default: int) -> int:
        if self._full_layer_kivi_enabled() and layer_idx in getattr(self, "full_layer_to_idx", {}):
            return int(getattr(self.config, "full_layer_kivi_decode_block_seq", default) or default)
        return super().get_decode_block_seq(layer_idx, default)

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        refs = super().decode_graph_keepalive_tensors()
        for attr_name in (
            "_deltakv_postrope_dummy_slot",
            "_deltakv_graph_static_empty_temp_slots_by_shape",
            "_deltakv_graph_static_temp_slots_by_capacity",
            "_deltakv_graph_static_plan_buffers_by_capacity",
            "_deltakv_graph_materialized_sparse_view_by_capacity",
            "_deltakv_graph_static_materialized_empty_by_shape",
            "_deltakv_graph_static_materialized_slots_by_capacity",
        ):
            self._append_tensor_refs(refs, getattr(self, attr_name, None))
        return refs
