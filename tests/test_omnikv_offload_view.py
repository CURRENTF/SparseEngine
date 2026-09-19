"""Physical KV staging must preserve the provider's logical planning envelope."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.engine.cache_manager.base import MlaLatentPayload, MlaLatentWrite
from sparseengine.engine.cache_manager.methods.omnikv.manager import OmniKVCacheManager


@pytest.mark.parametrize("input_capacity", [8, 32])
def test_compute_view_preserves_attention_planning_capacity(input_capacity):
    # FA3 derives its split plan from table width. Shrinking a full-history
    # table to selected capacity changed MLA short-path logits in a real model.
    # This CPU test checks metadata only; CUDA tests establish copy correctness.
    manager = object.__new__(OmniKVCacheManager)
    manager.offload_enabled = True
    manager.lru = None
    manager.device = torch.device("cuda")
    manager.kv_layer_index = int
    manager.config = SimpleNamespace(recent_keep_tokens=2)
    manager.attention_cache_storage = SimpleNamespace(
        full_layers={0}, make_payload=lambda parts: MlaLatentPayload(*parts)
    )
    manager._prefetched = set()
    manager._gather_decode = Mock()
    parts = (torch.empty(8, 1, 512), torch.empty(8, 1, 64))
    manager.selected_staging = {1: parts}
    manager.selected_capacity = 8
    manager.selected_slots = torch.zeros(1, 32, dtype=torch.int32)
    manager.selected_slots[:, :8] = torch.arange(8)
    manager.selected_rows = torch.zeros(1, dtype=torch.int32)
    manager.layer_batch_state = SimpleNamespace(slot_mapping=torch.tensor([3]))
    manager._current_writes = {1: MlaLatentWrite(parts[0][:1], parts[1][:1])}
    table = torch.zeros(1, input_capacity, dtype=torch.int32)
    rows = torch.zeros(1, dtype=torch.int32)
    lengths = torch.tensor([5], dtype=torch.int32)
    with (
        patch("torch.cuda.current_stream"),
        patch("sparseengine.engine.cache_manager.methods.omnikv.manager.append_rows"),
    ):
        payload, staging_table, _, actual_lengths = manager.get_layer_compute_payload(
            1, table, rows, lengths
        )
    assert staging_table.shape == table.shape
    assert staging_table.data_ptr() == manager.selected_slots.data_ptr()
    assert payload.latent_cache is parts[0] and payload.rope_cache is parts[1]
    assert actual_lengths is lengths
