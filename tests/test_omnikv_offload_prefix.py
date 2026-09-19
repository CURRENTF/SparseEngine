"""Split prefix backup must survive device-slot reuse and restore both MLA parts."""

import pytest
import torch

from sparseengine.engine.cache_manager.methods.omnikv.prefix import (
    OmniKVPrefixOffloadController,
    OmniKVPrefixPool,
)
from sparseengine.engine.cache_manager.methods.omnikv.storage import OmniKVStorage
from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
from sparseengine.engine.cache_manager.storage import ExplicitKVStorage, MlaLatentStorage
from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
from sparseengine.operators.indexed_host_copy import gather_rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mla,num_layers", [(False, 3), (True, 3), (False, 4), (True, 5)])
def test_split_prefix_restore_after_slot_reuse(mla, num_layers, monkeypatch):
    original = (
        MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16)
        if mla
        else ExplicitKVStorage(num_kv_heads=2, head_dim=128, dtype=torch.bfloat16)
    )
    storage = OmniKVStorage(
        original,
        num_layers=num_layers,
        num_slots=12,
        full_layers=[0],
        prefix_slots=8,
        device="cuda",
    )
    for layer, parts in enumerate(storage.layers):
        for part, tensor in enumerate(parts):
            tensor.copy_(torch.randn_like(tensor))
    source_slots = torch.tensor([1, 5], dtype=torch.int32, device="cuda")
    expected = [
        [x[source_slots.cpu().long()].clone().cpu() for x in parts]
        for parts in storage.layers
    ]
    tree = RadixPrefixIndex(block_size=2, fingerprint=b"split")
    block = PrefixCacheBlock(
        stable_block_id=tree.stable_block_id([7, 8], None),
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=source_slots),
        token_ids=(7, 8),
    )
    tree.insert_block(block)
    pool = OmniKVPrefixPool(storage, 4, 2, "cuda")
    controller = OmniKVPrefixOffloadController(
        prefix_cache=tree,
        storage=storage,
        host_pool=pool,
        block_size=2,
        device=torch.device("cuda"),
    )
    controller.submit_d2h([block])
    assert not block.residency.host_present
    controller.synchronize_all()
    torch.cuda.synchronize()
    assert block.residency.host_present
    assert tree.demote_device_until_freeable(1) == [block]
    for parts in storage.layers:
        for x in parts:
            x[:12].zero_()
    new_slots = torch.tensor([2, 9], dtype=torch.int32, device="cuda")
    block.payload.token_slots = new_slots
    from sparseengine.engine.cache_manager.methods.omnikv import prefix as omnikv_prefix

    transfer = omnikv_prefix.transfer_rows

    def delayed_second_component(*args, **kwargs):
        if kwargs["component"] == 1:
            torch.cuda._sleep(1_000_000)
        transfer(*args, **kwargs)

    monkeypatch.setattr(omnikv_prefix, "transfer_rows", delayed_second_component)
    operation = controller.submit_h2d([block])
    rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    lengths = torch.full((1,), 2, dtype=torch.int32, device="cuda")
    for layer, parts in enumerate(storage.layers):
        controller.wait_for_layer(operation, layer)
        for component, part in enumerate(parts):
            output = torch.empty(2, *part.shape[1:], dtype=part.dtype, device="cuda")
            gather_rows(
                storage.pointers[layer],
                output,
                new_slots.view(1, 2),
                rows,
                lengths,
                capacity=2,
                component=component,
                slot_map=storage.host_slot_map if layer else None,
            )
            torch.testing.assert_close(
                output.cpu(), expected[layer][component], rtol=0, atol=0
            )
    controller.synchronize_all()
    stats = controller.stats()
    per_layer = source_slots.numel() * storage.bytes_per_slot_per_layer()
    assert stats["prefix_cache_h2d_bytes"] == per_layer * len(storage.full_layers)
    assert stats["prefix_cache_d2h_bytes"] == per_layer * len(storage.layers)
    assert stats["prefix_cache_sparse_rehome_h2d_bytes"] == per_layer * (
        len(storage.layers) - len(storage.full_layers)
    )
    controller.free_host_payloads([block])
    assert pool.free_blocks == pool.capacity_blocks
