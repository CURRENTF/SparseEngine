from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "sgl_kernel.kvcacheio",
    reason="prefix offload transfer tests require the prefix-offload extra",
)

from sparseengine.engine.cache_manager.prefix_offload import (
    PinnedPrefixKVPool,
    PinnedQuestPrefixPool,
    QuestPrefixOffloadController,
    StandardPrefixOffloadController,
)
from sparseengine.engine.cache_manager.methods.quest import QuestPrefixBlockPayload
from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
from sparseengine.engine.mixed_prefix_offload import (
    MixedPrefixOffloadController,
    MixedQuestPrefixOffloadController,
    PinnedMixedPrefixPool,
    PinnedMixedQuestPrefixPool,
)
from sparseengine.engine.prefix_cache_coordinator import MixedPrefixBlockPayload
from sparseengine.engine.recurrent_state_manager import (
    RecurrentPrefixPayload,
    RecurrentStateSpec,
    RecurrentTensorSpec,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="prefix offload transfer kernel requires CUDA",
)


def test_prefix_offload_kernel_round_trip_and_layer_events():
    device = torch.device("cuda:0")
    num_layers, num_slots, num_heads, head_dim, block_size = 3, 12, 2, 4, 2
    kv_cache = torch.arange(
        2 * num_layers * num_slots * num_heads * head_dim,
        dtype=torch.float16,
        device=device,
    ).reshape(2, num_layers, num_slots, num_heads, head_dim).contiguous()
    prefix_cache = RadixPrefixIndex(block_size=block_size, fingerprint=b"offload-test")
    blocks = []
    parent_block_id = None
    source_slots = ([1, 4], [7, 9])
    for logical_block_idx, slots in enumerate(source_slots):
        tokens = [logical_block_idx * 2 + 1, logical_block_idx * 2 + 2]
        block_id = prefix_cache.stable_block_id(tokens, parent_block_id)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent_block_id,
            block_size=block_size,
            logical_block_idx=logical_block_idx,
            payload=StandardPrefixBlockPayload(
                token_slots=torch.tensor(slots, dtype=torch.int32, device=device)
            ),
            token_ids=tuple(tokens),
        )
        prefix_cache.insert_block(block)
        blocks.append(block)
        parent_block_id = block_id

    host_pool = PinnedPrefixKVPool(
        capacity_blocks=4,
        num_layers=num_layers,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16,
    )
    controller = StandardPrefixOffloadController(
        prefix_cache=prefix_cache,
        kv_cache=kv_cache,
        host_pool=host_pool,
        block_size=block_size,
        device=device,
    )
    source_indices = torch.tensor(
        [slot for block_slots in source_slots for slot in block_slots],
        dtype=torch.long,
        device=device,
    )
    expected = kv_cache[:, :, source_indices].cpu()

    controller.submit_d2h(blocks)
    assert all(block.residency.host_present is False for block in blocks)
    controller.synchronize_all()
    assert all(block.residency.host_present is True for block in blocks)
    host_indices = host_pool.token_indices(
        [int(block.payload.host_block_index) for block in blocks],
        device,
    ).cpu()
    host_tokens = host_pool.cache.reshape(
        2,
        num_layers,
        -1,
        num_heads,
        head_dim,
    )[:, :, host_indices]
    assert torch.equal(host_tokens, expected)

    assert len(prefix_cache.demote_device_until_freeable(2)) == 2
    destination_slots = ([0, 2], [6, 8])
    for block, slots in zip(blocks, destination_slots):
        block.payload.token_slots = torch.tensor(
            slots,
            dtype=torch.int32,
            device=device,
        )
    destination_indices = torch.tensor(
        [slot for block_slots in destination_slots for slot in block_slots],
        dtype=torch.long,
        device=device,
    )
    kv_cache[:, :, destination_indices].zero_()

    operation = controller.submit_h2d(blocks)
    for layer_index in range(num_layers):
        controller.wait_for_layer(operation, layer_index)
    controller.synchronize_all()

    assert torch.equal(kv_cache[:, :, destination_indices].cpu(), expected)
    assert all(block.residency.device_present for block in blocks)
    assert all(block.residency.host_present for block in blocks)
    stats = controller.stats()
    assert stats["prefix_cache_d2h_merged_blocks"] == 2
    assert stats["prefix_cache_h2d_merged_blocks"] == 2
    assert stats["prefix_cache_h2d_layer_waits"] == num_layers


def test_quest_prefix_offload_round_trips_kv_and_metadata_per_layer():
    device = torch.device("cuda:0")
    num_layers, num_pages, num_heads, head_dim, page_size = 3, 6, 2, 4, 2
    num_slots = num_pages * page_size
    kv_cache = torch.arange(
        2 * num_layers * num_slots * num_heads * head_dim,
        dtype=torch.float16,
        device=device,
    ).reshape(2, num_layers, num_slots, num_heads, head_dim).contiguous()
    metadata_cache = (
        torch.arange(
            2 * num_layers * num_pages * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        )
        .reshape(2, num_layers, num_pages, num_heads, head_dim)
        .contiguous()
        + 4096
    )
    prefix_cache = RadixPrefixIndex(block_size=page_size, fingerprint=b"quest-offload")
    blocks = []
    parent_block_id = None
    source_pages = [1, 4]
    for logical_block_idx, page_slot in enumerate(source_pages):
        tokens = [logical_block_idx * 2 + 1, logical_block_idx * 2 + 2]
        block_id = prefix_cache.stable_block_id(tokens, parent_block_id)
        token_slots = torch.arange(
            page_slot * page_size,
            (page_slot + 1) * page_size,
            dtype=torch.int32,
            device=device,
        )
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent_block_id,
            block_size=page_size,
            logical_block_idx=logical_block_idx,
            payload=QuestPrefixBlockPayload(
                block_slot=page_slot,
                token_slots=token_slots,
            ),
            token_ids=tuple(tokens),
        )
        prefix_cache.insert_block(block)
        blocks.append(block)
        parent_block_id = block_id

    source_token_indices = torch.cat(
        [block.payload.token_slots.to(torch.long) for block in blocks]
    )
    expected_kv = kv_cache[:, :, source_token_indices].cpu()
    expected_metadata = metadata_cache[:, :, source_pages].cpu()
    host_pool = PinnedQuestPrefixPool(
        capacity_blocks=4,
        num_layers=num_layers,
        block_size=page_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16,
    )
    controller = QuestPrefixOffloadController(
        prefix_cache=prefix_cache,
        kv_cache=kv_cache,
        device_metadata_cache=metadata_cache,
        host_pool=host_pool,
        block_size=page_size,
        device=device,
    )

    controller.submit_d2h(blocks)
    controller.synchronize_all()
    host_blocks = [int(block.payload.host_block_index) for block in blocks]
    host_token_indices = host_pool.token_indices(host_blocks, device).cpu()
    assert torch.equal(
        host_pool.cache.reshape(2, num_layers, -1, num_heads, head_dim)[
            :, :, host_token_indices
        ],
        expected_kv,
    )
    assert torch.equal(host_pool.metadata_cache[:, :, host_blocks], expected_metadata)
    transferred_bytes = expected_kv.nbytes + expected_metadata.nbytes
    assert controller.d2h_bytes == transferred_bytes

    assert len(prefix_cache.demote_device_until_freeable(2)) == 2
    destination_pages = [0, 3]
    for block, page_slot in zip(blocks, destination_pages):
        block.payload.block_slot = page_slot
        block.payload.token_slots = torch.arange(
            page_slot * page_size,
            (page_slot + 1) * page_size,
            dtype=torch.int32,
            device=device,
        )
    destination_token_indices = torch.cat(
        [block.payload.token_slots.to(torch.long) for block in blocks]
    )
    kv_cache[:, :, destination_token_indices].zero_()
    metadata_cache[:, :, destination_pages].zero_()

    operation = controller.submit_h2d(blocks)
    for layer_index in range(num_layers):
        controller.wait_for_layer(operation, layer_index)
        torch.cuda.current_stream().synchronize()
        assert torch.equal(
            kv_cache[:, layer_index, destination_token_indices].cpu(),
            expected_kv[:, layer_index],
        )
        assert torch.equal(
            metadata_cache[:, layer_index, destination_pages].cpu(),
            expected_metadata[:, layer_index],
        )
    controller.synchronize_all()
    assert controller.h2d_bytes == transferred_bytes


def _mixed_state_spec():
    return RecurrentStateSpec(
        name="mixed-test",
        tensor_specs=(
            RecurrentTensorSpec("conv", (2, 3), torch.float16),
            RecurrentTensorSpec("counter", (2,), torch.int32),
        ),
    )


def _mixed_recurrent_payload(device, seed):
    return RecurrentPrefixPayload(
        token_count=2,
        layer_states={
            layer_idx: {
                "conv": torch.full((2, 3), seed + layer_idx, dtype=torch.float16, device=device),
                "counter": torch.full((2,), seed + layer_idx, dtype=torch.int32, device=device),
            }
            for layer_idx in (1, 3)
        },
    )


def test_mixed_prefix_offload_round_trips_kv_and_recurrent_atomically():
    device = torch.device("cuda:0")
    num_layers, num_slots, num_heads, head_dim, block_size = 2, 12, 1, 4, 2
    kv_cache = torch.arange(
        2 * num_layers * num_slots * num_heads * head_dim,
        dtype=torch.float16,
        device=device,
    ).reshape(2, num_layers, num_slots, num_heads, head_dim).contiguous()
    prefix_cache = RadixPrefixIndex(block_size=block_size, fingerprint=b"mixed-offload")
    blocks = []
    parent = None
    for logical_idx, slots in enumerate(([1, 2], [7, 8])):
        tokens = [logical_idx * 2 + 1, logical_idx * 2 + 2]
        block_id = prefix_cache.stable_block_id(tokens, parent)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent,
            block_size=block_size,
            logical_block_idx=logical_idx,
            payload=MixedPrefixBlockPayload(
                kv_payload=StandardPrefixBlockPayload(
                    token_slots=torch.tensor(slots, dtype=torch.int32, device=device),
                    block_start=logical_idx * block_size,
                    block_end=(logical_idx + 1) * block_size,
                ),
                recurrent_payload=_mixed_recurrent_payload(device, 10 * (logical_idx + 1)),
                token_count=block_size,
                accounting_bytes=0,
                recurrent_bytes=0,
            ),
            token_ids=tuple(tokens),
        )
        prefix_cache.insert_block(block)
        blocks.append(block)
        parent = block_id
    source_slots = torch.cat([block.payload.kv_payload.token_slots.long() for block in blocks])
    expected_kv = kv_cache[:, :, source_slots].cpu()
    expected_recurrent = [
        {
            layer: {name: tensor.cpu().clone() for name, tensor in states.items()}
            for layer, states in block.payload.recurrent_payload.layer_states.items()
        }
        for block in blocks
    ]
    host_pool = PinnedMixedPrefixPool(
        capacity_blocks=4,
        num_layers=num_layers,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16,
        state_spec=_mixed_state_spec(),
        recurrent_layer_indices=(1, 3),
    )
    controller = MixedPrefixOffloadController(
        prefix_cache=prefix_cache,
        kv_cache=kv_cache,
        host_pool=host_pool,
        block_size=block_size,
        device=device,
        kv_transformer_layer_indices=(0, 2),
    )
    assert controller._h2d_transfer_schedule() == [
        ("kv", 0),
        ("auxiliary", 1),
        ("kv", 1),
        ("auxiliary", 3),
    ]

    controller.submit_d2h(blocks)
    controller.synchronize_all()
    assert all(block.residency.host_present for block in blocks)
    for block_idx, block in enumerate(blocks):
        host_idx = int(block.payload.host_block_index)
        for layer, states in expected_recurrent[block_idx].items():
            for name, expected in states.items():
                assert torch.equal(host_pool.recurrent_cache[layer][name][host_idx], expected)

    assert len(prefix_cache.demote_device_until_freeable(2)) == 2
    destination_slots = ([3, 4], [9, 10])
    for block, slots in zip(blocks, destination_slots):
        block.payload.kv_payload.token_slots = torch.tensor(
            slots, dtype=torch.int32, device=device
        )
        controller.free_device_recurrent(block)
    controller.allocate_device_recurrent(blocks)
    destination = torch.cat([block.payload.kv_payload.token_slots.long() for block in blocks])
    kv_cache[:, :, destination].zero_()

    operation = controller.submit_h2d(blocks)
    assert set(operation.auxiliary_layer_events) == {1, 3}
    controller.wait_for_auxiliary(operation)
    for layer_idx in range(num_layers):
        controller.wait_for_layer(operation, layer_idx)
    controller.synchronize_all()
    assert torch.equal(kv_cache[:, :, destination].cpu(), expected_kv)
    for block_idx, block in enumerate(blocks):
        for layer, states in expected_recurrent[block_idx].items():
            for name, expected in states.items():
                assert torch.equal(
                    block.payload.recurrent_payload.layer_states[layer][name].cpu(), expected
                )


def test_mixed_prefix_d2h_bounds_recurrent_staging_and_drains_all_batches(
    monkeypatch,
):
    device = torch.device("cuda:0")
    num_layers, num_slots, num_heads, head_dim, block_size = 2, 12, 1, 4, 2
    kv_cache = torch.arange(
        2 * num_layers * num_slots * num_heads * head_dim,
        dtype=torch.float16,
        device=device,
    ).reshape(2, num_layers, num_slots, num_heads, head_dim).contiguous()
    prefix_cache = RadixPrefixIndex(
        block_size=block_size, fingerprint=b"mixed-offload-bounded"
    )
    blocks = []
    parent = None
    for logical_idx in range(5):
        tokens = [logical_idx * 2 + 1, logical_idx * 2 + 2]
        block_id = prefix_cache.stable_block_id(tokens, parent)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent,
            block_size=block_size,
            logical_block_idx=logical_idx,
            payload=MixedPrefixBlockPayload(
                kv_payload=StandardPrefixBlockPayload(
                    token_slots=torch.tensor(
                        [logical_idx * 2, logical_idx * 2 + 1],
                        dtype=torch.int32,
                        device=device,
                    ),
                    block_start=logical_idx * block_size,
                    block_end=(logical_idx + 1) * block_size,
                ),
                recurrent_payload=_mixed_recurrent_payload(
                    device, 10 * (logical_idx + 1)
                ),
                token_count=block_size,
                accounting_bytes=0,
                recurrent_bytes=0,
            ),
            token_ids=tuple(tokens),
        )
        prefix_cache.insert_block(block)
        blocks.append(block)
        parent = block_id

    host_pool = PinnedMixedPrefixPool(
        capacity_blocks=len(blocks),
        num_layers=num_layers,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16,
        state_spec=_mixed_state_spec(),
        recurrent_layer_indices=(1, 3),
    )
    recurrent_bytes_per_block = host_pool.recurrent_bytes_per_block
    controller = MixedPrefixOffloadController(
        prefix_cache=prefix_cache,
        kv_cache=kv_cache,
        host_pool=host_pool,
        block_size=block_size,
        device=device,
        kv_transformer_layer_indices=(0, 2),
        d2h_recurrent_staging_byte_budget=2 * recurrent_bytes_per_block,
    )

    stack_block_counts = []
    original_stack = torch.stack

    def _recording_stack(tensors, *args, **kwargs):
        tensors = list(tensors)
        stack_block_counts.append(len(tensors))
        return original_stack(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", _recording_stack)
    controller.submit_d2h(blocks)

    tensors_per_block = len(controller._ordered_recurrent_tensors(blocks[0]))
    assert controller.d2h_staging_blocks == 2
    assert len(controller.d2h_operations) == 1
    assert len(controller.d2h_operations[0].blocks) == 2
    assert controller.pending_d2h_blocks == 3
    assert stack_block_counts == [2] * tensors_per_block

    controller.synchronize_all()

    assert all(block.residency.host_present for block in blocks)
    assert controller.pending_d2h_blocks == 0
    assert controller.d2h_operations == []
    assert controller.d2h_submitted_operations == 3
    assert controller.d2h_completed_operations == 3
    assert max(stack_block_counts) == 2
    assert stack_block_counts.count(1) == tensors_per_block
    assert len(stack_block_counts) == 3 * tensors_per_block


def test_mixed_quest_offload_round_trips_kv_metadata_and_recurrent():
    device = torch.device("cuda:0")
    num_layers, num_pages, num_heads, head_dim = 2, 6, 1, 4
    page_size, block_size = 2, 4
    kv_cache = torch.arange(
        2 * num_layers * num_pages * page_size * num_heads * head_dim,
        dtype=torch.float16,
        device=device,
    ).reshape(2, num_layers, num_pages * page_size, num_heads, head_dim).contiguous()
    metadata = (
        torch.arange(
            2 * num_layers * num_pages * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        ).reshape(2, num_layers, num_pages, num_heads, head_dim).contiguous()
        + 2048
    )
    prefix_cache = RadixPrefixIndex(block_size=block_size, fingerprint=b"mixed-quest")
    block_id = prefix_cache.stable_block_id([1, 2, 3, 4], None)
    pages = torch.tensor([1, 4], dtype=torch.int32, device=device)
    slots = (pages[:, None] * page_size + torch.arange(page_size, device=device)).reshape(-1)
    recurrent = _mixed_recurrent_payload(device, 30)
    recurrent.token_count = block_size
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=block_size,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload=QuestPrefixBlockPayload(
                block_slot=None,
                token_slots=slots.to(torch.int32),
                block_start=0,
                block_end=block_size,
                block_slots=pages,
            ),
            recurrent_payload=recurrent,
            token_count=block_size,
            accounting_bytes=0,
            recurrent_bytes=0,
        ),
        token_ids=(1, 2, 3, 4),
    )
    prefix_cache.insert_block(block)
    expected_kv = kv_cache[:, :, slots.long()].cpu()
    expected_metadata = metadata[:, :, pages.long()].cpu()
    expected_recurrent = {
        layer: {name: tensor.cpu().clone() for name, tensor in states.items()}
        for layer, states in recurrent.layer_states.items()
    }
    host_pool = PinnedMixedQuestPrefixPool(
        capacity_blocks=2,
        num_layers=num_layers,
        block_size=block_size,
        page_size=page_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16,
        state_spec=_mixed_state_spec(),
        recurrent_layer_indices=(1, 3),
    )
    controller = MixedQuestPrefixOffloadController(
        prefix_cache=prefix_cache,
        kv_cache=kv_cache,
        device_metadata_cache=metadata,
        host_pool=host_pool,
        block_size=block_size,
        device=device,
        kv_transformer_layer_indices=(0, 2),
    )
    assert controller._h2d_transfer_schedule() == [
        ("kv", 0),
        ("auxiliary", 1),
        ("kv", 1),
        ("auxiliary", 3),
    ]

    controller.submit_d2h([block])
    controller.synchronize_all()
    assert prefix_cache.demote_device_until_freeable(1) == [block]
    destination_pages = torch.tensor([0, 3], dtype=torch.int32, device=device)
    destination_slots = (
        destination_pages[:, None] * page_size + torch.arange(page_size, device=device)
    ).reshape(-1)
    block.payload.kv_payload.block_slots = destination_pages
    block.payload.kv_payload.token_slots = destination_slots.to(torch.int32)
    controller.free_device_recurrent(block)
    controller.allocate_device_recurrent([block])
    kv_cache[:, :, destination_slots.long()].zero_()
    metadata[:, :, destination_pages.long()].zero_()

    operation = controller.submit_h2d([block])
    assert set(operation.auxiliary_layer_events) == {1, 3}
    controller.wait_for_auxiliary(operation)
    for layer_idx in range(num_layers):
        controller.wait_for_layer(operation, layer_idx)
    controller.synchronize_all()
    assert torch.equal(kv_cache[:, :, destination_slots.long()].cpu(), expected_kv)
    assert torch.equal(metadata[:, :, destination_pages.long()].cpu(), expected_metadata)
    for layer, states in expected_recurrent.items():
        for name, expected in states.items():
            assert torch.equal(block.payload.recurrent_payload.layer_states[layer][name].cpu(), expected)
