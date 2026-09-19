"""Early prefill reads must wait for prefix remapping and preserve buffer reuse."""

from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.cache_manager.methods.omnikv.manager import OmniKVCacheManager
from sparseengine.operators.indexed_host_copy import gather_prefill_history
from sparseengine.utils.context import get_context


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_prefetch_waits_for_prefix_restore_and_reuses_buffer(monkeypatch):
    # Full-model prefix copies may finish before the reader even without its
    # wait. Deliberately delay a remap to expose that missing-dependency bug.
    device = torch.device("cuda")
    hosts = [
        torch.arange(6 * 16, dtype=torch.float32)
        .reshape(6, 1, 16)
        .to(torch.bfloat16)
        .add(layer * 100)
        .pin_memory()
        for layer in range(4)
    ]
    pointers = torch.tensor([[h.data_ptr(), h.data_ptr()] for h in hosts],
                            dtype=torch.uint64, device=device)
    table = torch.arange(4, dtype=torch.int32, device=device).view(1, 4)
    rows = torch.zeros(1, dtype=torch.int32, device=device)
    lengths = torch.tensor([4], dtype=torch.int32, device=device)
    cu_query = torch.tensor([0, 2], dtype=torch.int32, device=device)
    slot_map = torch.zeros(4, dtype=torch.int32, device=device)
    restored_map = torch.tensor([4, 5, 2, 3], dtype=torch.int32, device=device)
    staging = tuple(torch.zeros(4, 1, 16, dtype=torch.bfloat16, device=device)
                    for _ in range(2))
    current = tuple(torch.full((2, 16), 700 + c, dtype=torch.bfloat16, device=device)
                    for c in range(2))
    # Warm the actual reader so JIT compilation cannot hide the delayed remap.
    for component, destination in enumerate(staging):
        gather_prefill_history(pointers[2], destination, table, rows, lengths,
                               cu_query, slot_map, component=component)
    torch.cuda.synchronize()

    manager = object.__new__(OmniKVCacheManager)
    manager.device = device
    manager.offload_enabled = True
    manager._prefill_next_layer = {0: 2, 1: 2, 2: 3, 3: None}
    manager._prefill_prefetched_layer = None
    manager.kv_layer_index = lambda layer: layer
    manager.prefill_staging = staging
    manager.prefetch_stream = torch.cuda.Stream()
    manager.layer_ready = {layer: torch.cuda.Event() for layer in (2, 3)}
    manager.buffer_req_to_token_slots = table
    manager.layer_batch_state = SimpleNamespace(
        req_indices=rows, context_lens=lengths, slot_mapping=table[0, 2:])
    manager.attention_cache_storage = SimpleNamespace(
        pointers=pointers, host_slot_map=slot_map, full_layers={0, 1},
        make_payload=lambda parts: parts)
    restore_stream = torch.cuda.Stream()
    restored = torch.cuda.Event()
    restore_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(restore_stream):
        torch.cuda._sleep(50_000_000)
        slot_map.copy_(restored_map)
        restored.record()
    waits = []

    def wait_for_prefix(layer, selection):
        waits.append(layer)
        torch.cuda.current_stream().wait_event(restored)

    manager.before_prefill_layer_attention = wait_for_prefix
    monkeypatch.setattr(get_context(), "is_prefill", True)
    monkeypatch.setattr(get_context(), "cu_seqlens_q", cu_query)
    manager.on_layer_attention_end(0)
    manager.on_layer_attention_end(1)
    for layer in (2, 3):
        payload, _, _, _ = manager.get_prefill_compute_payload(
            layer, *current, None, table, rows, lengths)
        for component, result in enumerate(payload):
            expected = torch.cat(
                (hosts[layer][4:6], current[component].cpu().view(2, 1, 16))
            )
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
            assert result.data_ptr() == staging[component].data_ptr()
        if layer == 2:
            assert waits == [2]  # Full layers must not schedule another overwrite.
        manager.on_layer_attention_end(layer)
    assert waits == [2, 3]
    assert manager._prefill_prefetched_layer is None
