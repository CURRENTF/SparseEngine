from __future__ import annotations

import torch

from sparseengine.kernels.triton.mla.copy_latent import (
    copy_latent_to_cache,
    copy_latent_to_cache_with_quest_metadata,
    validate_copy_slot_mappings,
)
from sparseengine.kernels.triton.mla.decode_stage1 import MLA_LATENT_DIM, MLA_ROPE_DIM

from ..base import AttentionCacheWrite, MlaLatentPayload, MlaLatentWrite
from .base import CacheLayout
from .components import CacheComponentSpec


class MlaLatentStorage:
    """Persistent latent and RoPE caches for GLM-style MLA."""

    layout = CacheLayout.MLA_LATENT

    def __init__(
        self,
        *,
        kv_lora_rank: int,
        rope_dim: int,
        dtype: torch.dtype,
        validate_runtime_invariants: bool = False,
    ) -> None:
        self.kv_lora_rank = int(kv_lora_rank)
        self.rope_dim = int(rope_dim)
        self.dtype = dtype
        self.validate_runtime_invariants = bool(validate_runtime_invariants)
        if self.kv_lora_rank != MLA_LATENT_DIM or self.rope_dim != MLA_ROPE_DIM:
            raise ValueError(
                "The vendored MLA storage kernel requires "
                f"kv_lora_rank={MLA_LATENT_DIM} and rope_dim={MLA_ROPE_DIM}, got "
                f"kv_lora_rank={self.kv_lora_rank} rope_dim={self.rope_dim}."
            )
        if self.dtype != torch.bfloat16:
            raise TypeError(
                f"MLA latent storage v1 requires torch.bfloat16, got {self.dtype}."
            )
        self.latent_cache: torch.Tensor | None = None
        self.rope_cache: torch.Tensor | None = None
        self._validated_store_calls_remaining: dict[
            tuple[str, int, int], int
        ] = {}

    def allocate(
        self,
        *,
        num_layers: int,
        num_slots: int,
        device: torch.device,
    ) -> None:
        num_layers = int(num_layers)
        num_slots = int(num_slots)
        if num_layers <= 0 or num_slots <= 0:
            raise ValueError(
                "MLA allocation requires positive layers and slots, got "
                f"num_layers={num_layers} num_slots={num_slots}."
            )
        self.latent_cache = torch.empty(
            num_layers,
            num_slots,
            1,
            self.kv_lora_rank,
            dtype=self.dtype,
            device=device,
        )
        self.rope_cache = torch.empty(
            num_layers,
            num_slots,
            1,
            self.rope_dim,
            dtype=self.dtype,
            device=device,
        )
        self._validated_store_calls_remaining.clear()

    def _require_caches(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.latent_cache is None or self.rope_cache is None:
            raise RuntimeError("MLA latent storage has not been allocated.")
        return self.latent_cache, self.rope_cache

    def component_specs(self, layer_idx: int) -> tuple[CacheComponentSpec, ...]:
        return (
            CacheComponentSpec("latent", (1, self.kv_lora_rank), self.dtype),
            CacheComponentSpec("rope", (1, self.rope_dim), self.dtype),
        )

    def component_tensors(self, layer_idx: int) -> tuple[torch.Tensor, ...]:
        payload = self.layer_payload(layer_idx)
        return payload.latent_cache, payload.rope_cache

    def layer_payload(self, layer_idx: int) -> MlaLatentPayload:
        latent_cache, rope_cache = self._require_caches()
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < int(latent_cache.shape[0]):
            raise IndexError(
                f"MLA layer index {layer_idx} is outside [0, {int(latent_cache.shape[0])})."
            )
        return MlaLatentPayload(
            latent_cache=latent_cache[layer_idx],
            rope_cache=rope_cache[layer_idx],
        )

    @staticmethod
    def _slot_mapping_key(slot_mapping: torch.Tensor) -> tuple[str, int, int]:
        return (
            str(slot_mapping.device),
            int(slot_mapping.data_ptr()),
            int(slot_mapping.numel()),
        )

    def _validate_slot_mapping_contract(self, slot_mapping: torch.Tensor) -> None:
        latent_cache, _ = self._require_caches()
        if slot_mapping.ndim != 1:
            raise ValueError(
                f"MLA slot_mapping must be 1D, got {tuple(slot_mapping.shape)}."
            )
        if slot_mapping.dtype != torch.int32:
            raise TypeError(
                f"MLA slot_mapping must use torch.int32, got {slot_mapping.dtype}."
            )
        if slot_mapping.device != latent_cache.device:
            raise ValueError(
                "MLA slot_mapping must share the cache device, got "
                f"slots={slot_mapping.device} cache={latent_cache.device}."
            )

    def validate_slot_mappings(
        self,
        slot_mappings: tuple[torch.Tensor, ...],
    ) -> None:
        if not slot_mappings:
            raise ValueError("MLA slot mapping validation requires at least one layer.")
        remaining: dict[tuple[str, int, int], int] = {}
        unique_mappings: dict[tuple[str, int, int], torch.Tensor] = {}
        for slot_mapping in slot_mappings:
            key = self._slot_mapping_key(slot_mapping)
            if key not in unique_mappings:
                self._validate_slot_mapping_contract(slot_mapping)
                unique_mappings[key] = slot_mapping
            remaining[key] = remaining.get(key, 0) + 1
        latent_cache, _ = self._require_caches()
        mappings_by_width: dict[int, list[torch.Tensor]] = {}
        for mapping in unique_mappings.values():
            mappings_by_width.setdefault(int(mapping.numel()), []).append(mapping)
        for mappings in mappings_by_width.values():
            validate_copy_slot_mappings(
                torch.stack(mappings),
                cache_slot_count=int(latent_cache.shape[1]),
            )
        self._validated_store_calls_remaining = remaining

    def validate_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        latent_cache, _ = self._require_caches()
        self.validate_slot_mappings(
            (slot_mapping,) * int(latent_cache.shape[0])
        )

    def store(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
    ) -> None:
        if not isinstance(payload, MlaLatentWrite):
            raise TypeError(
                "MlaLatentStorage.store requires MlaLatentWrite, got "
                f"{type(payload).__name__}."
            )
        destination = self.layer_payload(layer_idx)
        use_prevalidated_mapping = self._consume_prevalidated_mapping(
            slot_mapping
        )
        copy_latent_to_cache(
            payload.latent,
            payload.rope,
            slot_mapping,
            destination.latent_cache,
            destination.rope_cache,
            validate_slots=(
                self.validate_runtime_invariants
                and not use_prevalidated_mapping
            ),
        )

    def _consume_prevalidated_mapping(
        self,
        slot_mapping: torch.Tensor,
    ) -> bool:
        slot_mapping_key = self._slot_mapping_key(slot_mapping)
        remaining = self._validated_store_calls_remaining.get(
            slot_mapping_key,
            0,
        )
        use_prevalidated_mapping = remaining > 0
        if use_prevalidated_mapping:
            if remaining == 1:
                del self._validated_store_calls_remaining[slot_mapping_key]
            else:
                self._validated_store_calls_remaining[slot_mapping_key] = (
                    remaining - 1
                )
        return use_prevalidated_mapping

    def store_with_quest_metadata(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
        page_max: torch.Tensor,
        page_min: torch.Tensor,
        *,
        page_size: int,
    ) -> None:
        if not isinstance(payload, MlaLatentWrite):
            raise TypeError(
                "MlaLatentStorage.store_with_quest_metadata requires "
                f"MlaLatentWrite, got {type(payload).__name__}."
            )
        destination = self.layer_payload(layer_idx)
        use_prevalidated_mapping = self._consume_prevalidated_mapping(
            slot_mapping
        )
        copy_latent_to_cache_with_quest_metadata(
            payload.latent,
            payload.rope,
            slot_mapping,
            destination.latent_cache,
            destination.rope_cache,
            page_max,
            page_min,
            page_size=page_size,
            validate_slots=(
                self.validate_runtime_invariants
                and not use_prevalidated_mapping
            ),
        )

    def bytes_per_slot_per_layer(self) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return int((self.kv_lora_rank + self.rope_dim) * element_size)

    def slot_capacity(self) -> int:
        latent_cache, _ = self._require_caches()
        return int(latent_cache.shape[1])

    @torch.no_grad()
    def copy_slots(
        self,
        layer_idx: int,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None:
        payload = self.layer_payload(layer_idx)
        source_slots = source_slots.to(
            device=payload.latent_cache.device,
            dtype=torch.long,
        ).reshape(-1)
        destination_slots = destination_slots.to(
            device=payload.latent_cache.device,
            dtype=torch.long,
        ).reshape(-1)
        if source_slots.shape != destination_slots.shape:
            raise ValueError(
                "MLA latent slot copy requires equal source/destination shapes, "
                f"got {tuple(source_slots.shape)} and "
                f"{tuple(destination_slots.shape)}."
            )
        if source_slots.numel() == 0:
            return
        slot_count = int(payload.latent_cache.shape[0])
        in_bounds = (
            (source_slots >= 0)
            & (source_slots < slot_count)
            & (destination_slots >= 0)
            & (destination_slots < slot_count)
        ).all()
        if in_bounds.is_cuda:
            torch._assert_async(in_bounds)
        elif not bool(in_bounds.item()):
            raise ValueError(
                f"MLA latent slot copy indices must be in [0, {slot_count})."
            )
        latent_selected = payload.latent_cache.index_select(0, source_slots)
        rope_selected = payload.rope_cache.index_select(0, source_slots)
        payload.latent_cache.index_copy_(
            0,
            destination_slots,
            latent_selected,
        )
        payload.rope_cache.index_copy_(
            0,
            destination_slots,
            rope_selected,
        )

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]:
        return self._require_caches()
