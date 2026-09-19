"""Shared physical-pool budget for startup profiling and production admission."""

from dataclasses import dataclass

from ...offload.allocation import plan_host_allocation
from .lru import OmniKVLRU


def omnikv_host_pool_bytes(
    slots, part_bytes, sparse_layers, full_layers, prefix_slots=0
):
    history = plan_host_allocation(
        [(slots + prefix_slots) * width for width in part_bytes] * sparse_layers,
        allow_packing=True,
    )
    prefix = plan_host_allocation(
        [prefix_slots * width for width in part_bytes] * full_layers
    )
    return history.estimated_bytes + prefix.estimated_bytes


def fit_omnikv_host_slots(
    max_slots, budget, part_bytes, sparse_layers, full_layers, prefix_slots=0
):
    low, high = 0, max(0, max_slots)
    while low < high:
        middle = (low + high + 1) // 2
        needed = omnikv_host_pool_bytes(
            middle, part_bytes, sparse_layers, full_layers, prefix_slots
        )
        if needed <= budget:
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True)
class OmniKVPoolPlan:
    full_layers: tuple[int, ...]
    layer_groups: dict[int, int]
    selected_capacity: int
    cache_capacity: int
    fixed_bytes: int
    slot_bytes: int

    def budget(self, slots):
        return self.fixed_bytes + slots * self.slot_bytes


def plan_omnikv_pools(config, full_layers, num_layers, rows, per_layer):
    full = set(full_layers)
    # GPU-only OmniKV consumes full history before its first observer too.
    if full:
        full.update(range(min(full)))
    sparse = num_layers - len(full)
    if not full or sparse <= 0:
        raise ValueError(
            "OmniKV offload requires both full and sparse attention layers."
        )
    selected = min(
        config.max_model_len,
        config.sink_keep_tokens + config.decode_keep_tokens + config.recent_keep_tokens,
    )
    if selected == 0:
        raise ValueError(
            "OmniKV offload requires a positive total selected-token budget."
        )
    cache = getattr(config, "omnikv_offload_cache_tokens", None)
    if cache is None:
        cache = 1 << (selected - 1).bit_length()
    cache = min(cache, config.max_model_len)
    if cache and cache < selected:
        raise ValueError(
            "omnikv_offload_cache_tokens must cover the full selected-token budget."
        )
    layer_groups = {}
    for layer in range(num_layers):
        if layer in full:
            group = layer
        else:
            layer_groups[layer] = group
    groups = len(set(layer_groups.values()))
    # Logical slot table and stable compute-view table; full-layer storage plus
    # one shared full-history prefill pool and the allocator/host-map vectors.
    fixed = sparse * rows * selected * per_layer + 2 * rows * config.max_model_len * 4
    fixed += rows * 4 + num_layers * 16
    slot_bytes = (len(full) + 1) * per_layer + 8
    if cache:
        fixed += sparse * (rows * (cache - selected) + 1) * per_layer
        fixed += OmniKVLRU.metadata_bytes(rows, 0, cache, selected, groups)
        slot_bytes += groups * rows * 4
    return OmniKVPoolPlan(
        tuple(sorted(full)), layer_groups, selected, cache, fixed, slot_bytes
    )
