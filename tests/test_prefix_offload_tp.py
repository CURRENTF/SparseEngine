from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

pytest.importorskip(
    "sgl_kernel.kvcacheio",
    reason="prefix offload TP tests require the prefix-offload extra",
)

from sparseengine.engine.cache_manager.prefix_offload import (
    PinnedPrefixKVPool,
    PinnedQuestPrefixPool,
    QuestPrefixOffloadController,
    StandardPrefixOffloadController,
)
from sparseengine.engine.cache_manager.methods.quest import QuestPrefixBlockPayload
from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
from sparseengine.engine.mixed_prefix_offload import (
    MixedPrefixOffloadController,
    MixedQuestPrefixOffloadController,
    PinnedMixedPrefixPool,
    PinnedMixedQuestPrefixPool,
)
from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
from sparseengine.engine.prefix_cache_coordinator import MixedPrefixBlockPayload
from sparseengine.engine.recurrent_state_manager import (
    RecurrentPrefixPayload,
    RecurrentStateSpec,
    RecurrentTensorSpec,
)


def _logical_signature(blocks: list[PrefixCacheBlock]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            block.stable_block_id.hex(),
            block.parent_block_id.hex() if block.parent_block_id is not None else None,
            int(block.ref_count),
            bool(block.residency.device_present),
            bool(block.residency.host_present),
            block.residency.transfer.value if block.residency.transfer is not None else None,
        )
        for block in blocks
    )


def _all_gather_object(value, world_size: int):
    values = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _build_standard_blocks(
    *,
    device: torch.device,
    source_slots: tuple[tuple[int, ...], ...],
    block_size: int,
) -> tuple[RadixPrefixIndex, list[PrefixCacheBlock]]:
    prefix_cache = RadixPrefixIndex(block_size=block_size, fingerprint=b"tp2-standard")
    blocks = []
    parent_id = None
    for logical_idx, slots in enumerate(source_slots):
        tokens = tuple(range(logical_idx * block_size + 1, (logical_idx + 1) * block_size + 1))
        block_id = prefix_cache.stable_block_id(tokens, parent_id)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent_id,
            block_size=block_size,
            logical_block_idx=logical_idx,
            payload=StandardPrefixBlockPayload(
                token_slots=torch.tensor(slots, dtype=torch.int32, device=device)
            ),
            token_ids=tokens,
        )
        prefix_cache.insert_block(block)
        blocks.append(block)
        parent_id = block_id
    return prefix_cache, blocks


def _run_standard_round_trip(rank: int, world_size: int, device: torch.device) -> None:
    num_layers, num_slots, num_heads, head_dim, block_size = 2, 12, 1, 4, 2
    kv_cache = (
        torch.arange(
            2 * num_layers * num_slots * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        )
        .reshape(2, num_layers, num_slots, num_heads, head_dim)
        .contiguous()
        .add_(rank * 4096)
    )
    rank_source_slots = (
        ((1, 3), (6, 8)),
        ((2, 4), (7, 9)),
    )
    source_slots = rank_source_slots[rank]
    prefix_cache, blocks = _build_standard_blocks(
        device=device,
        source_slots=source_slots,
        block_size=block_size,
    )
    source_indices = torch.tensor(
        [slot for slots in source_slots for slot in slots],
        dtype=torch.long,
        device=device,
    )
    expected = kv_cache[:, :, source_indices].cpu()
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

    controller.submit_d2h(blocks)
    if rank == 0:
        controller.synchronize_all()
    transient = _all_gather_object(_logical_signature(blocks), world_size)
    assert all(item[4] for item in transient[0])
    assert all(not item[4] for item in transient[1])

    if rank == 1:
        controller.synchronize_all()
    dist.barrier()
    ready = _all_gather_object(_logical_signature(blocks), world_size)
    assert ready[0] == ready[1]
    assert all(item[2:] == (0, True, True, None) for item in ready[rank])
    gathered_source_slots = _all_gather_object(source_slots, world_size)
    assert gathered_source_slots[0] != gathered_source_slots[1]

    host_indices = host_pool.token_indices(
        [int(block.payload.host_block_index) for block in blocks],
        device,
    ).cpu()
    host_values = host_pool.cache.reshape(
        2,
        num_layers,
        -1,
        num_heads,
        head_dim,
    )[:, :, host_indices]
    assert torch.equal(host_values, expected)

    assert len(prefix_cache.demote_device_until_freeable(len(blocks))) == len(blocks)
    rank_destination_slots = (
        ((0, 2), (5, 10)),
        ((1, 3), (6, 11)),
    )
    destination_slots = rank_destination_slots[rank]
    for block, slots in zip(blocks, destination_slots):
        block.payload.token_slots = torch.tensor(
            slots,
            dtype=torch.int32,
            device=device,
        )
    destination_indices = torch.tensor(
        [slot for slots in destination_slots for slot in slots],
        dtype=torch.long,
        device=device,
    )
    kv_cache[:, :, destination_indices].zero_()
    operation = controller.submit_h2d(blocks)
    for layer_idx in range(num_layers):
        controller.wait_for_layer(operation, layer_idx)
    controller.synchronize_all()
    assert torch.equal(kv_cache[:, :, destination_indices].cpu(), expected)

    final = _all_gather_object(_logical_signature(blocks), world_size)
    assert final[0] == final[1]
    assert all(item[2:] == (0, True, True, None) for item in final[rank])


def _run_quest_round_trip(rank: int, world_size: int, device: torch.device) -> None:
    num_layers, num_pages, num_heads, head_dim, page_size = 2, 7, 1, 4, 2
    num_slots = num_pages * page_size
    kv_cache = (
        torch.arange(
            2 * num_layers * num_slots * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        )
        .reshape(2, num_layers, num_slots, num_heads, head_dim)
        .contiguous()
        .add_(rank * 8192)
    )
    metadata_cache = (
        torch.arange(
            2 * num_layers * num_pages * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        )
        .reshape(2, num_layers, num_pages, num_heads, head_dim)
        .contiguous()
        .add_(16384 + rank * 8192)
    )
    rank_source_pages = ((1, 4), (2, 5))
    source_pages = rank_source_pages[rank]
    prefix_cache = RadixPrefixIndex(block_size=page_size, fingerprint=b"tp2-quest")
    blocks = []
    parent_id = None
    for logical_idx, page_slot in enumerate(source_pages):
        tokens = tuple(range(logical_idx * page_size + 1, (logical_idx + 1) * page_size + 1))
        block_id = prefix_cache.stable_block_id(tokens, parent_id)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent_id,
            block_size=page_size,
            logical_block_idx=logical_idx,
            payload=QuestPrefixBlockPayload(
                block_slot=page_slot,
                token_slots=torch.arange(
                    page_slot * page_size,
                    (page_slot + 1) * page_size,
                    dtype=torch.int32,
                    device=device,
                ),
            ),
            token_ids=tokens,
        )
        prefix_cache.insert_block(block)
        blocks.append(block)
        parent_id = block_id

    source_token_indices = torch.cat([block.payload.token_slots.long() for block in blocks])
    expected_kv = kv_cache[:, :, source_token_indices].cpu()
    expected_metadata = metadata_cache[:, :, list(source_pages)].cpu()
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
    ready = _all_gather_object(_logical_signature(blocks), world_size)
    assert ready[0] == ready[1]
    assert all(item[2:] == (0, True, True, None) for item in ready[rank])
    gathered_source_pages = _all_gather_object(source_pages, world_size)
    assert gathered_source_pages[0] != gathered_source_pages[1]

    host_blocks = [int(block.payload.host_block_index) for block in blocks]
    host_token_indices = host_pool.token_indices(host_blocks, device).cpu()
    host_kv = host_pool.cache.reshape(
        2,
        num_layers,
        -1,
        num_heads,
        head_dim,
    )[:, :, host_token_indices]
    assert torch.equal(host_kv, expected_kv)
    assert torch.equal(host_pool.metadata_cache[:, :, host_blocks], expected_metadata)

    assert len(prefix_cache.demote_device_until_freeable(len(blocks))) == len(blocks)
    rank_destination_pages = ((0, 3), (1, 4))
    destination_pages = rank_destination_pages[rank]
    for block, page_slot in zip(blocks, destination_pages):
        block.payload.block_slot = page_slot
        block.payload.token_slots = torch.arange(
            page_slot * page_size,
            (page_slot + 1) * page_size,
            dtype=torch.int32,
            device=device,
        )
    destination_token_indices = torch.cat([block.payload.token_slots.long() for block in blocks])
    kv_cache[:, :, destination_token_indices].zero_()
    metadata_cache[:, :, list(destination_pages)].zero_()
    operation = controller.submit_h2d(blocks)
    for layer_idx in range(num_layers):
        controller.wait_for_layer(operation, layer_idx)
    controller.synchronize_all()
    assert torch.equal(kv_cache[:, :, destination_token_indices].cpu(), expected_kv)
    assert torch.equal(metadata_cache[:, :, list(destination_pages)].cpu(), expected_metadata)

    final = _all_gather_object(_logical_signature(blocks), world_size)
    assert final[0] == final[1]
    assert all(item[2:] == (0, True, True, None) for item in final[rank])


def _mixed_state_spec() -> RecurrentStateSpec:
    return RecurrentStateSpec(
        name="tp2-mixed",
        tensor_specs=(
            RecurrentTensorSpec("conv", (2, 3), torch.float16),
            RecurrentTensorSpec("counter", (2,), torch.int32),
        ),
    )


def _mixed_recurrent_payload(
    device: torch.device,
    *,
    seed: int,
    token_count: int,
) -> RecurrentPrefixPayload:
    return RecurrentPrefixPayload(
        token_count=token_count,
        layer_states={
            layer_idx: {
                "conv": torch.full(
                    (2, 3),
                    seed + layer_idx,
                    dtype=torch.float16,
                    device=device,
                ),
                "counter": torch.full(
                    (2,),
                    seed + layer_idx,
                    dtype=torch.int32,
                    device=device,
                ),
            }
            for layer_idx in (1, 3)
        },
    )


def _recurrent_snapshot(block: PrefixCacheBlock):
    return {
        layer_idx: {
            name: tensor.cpu().clone()
            for name, tensor in states.items()
        }
        for layer_idx, states in block.payload.recurrent_payload.layer_states.items()
    }


def _assert_recurrent_snapshot(block: PrefixCacheBlock, expected) -> None:
    for layer_idx, states in expected.items():
        for name, tensor in states.items():
            actual = block.payload.recurrent_payload.layer_states[layer_idx][name]
            assert torch.equal(actual.cpu(), tensor)


def _run_mixed_round_trip(rank: int, world_size: int, device: torch.device) -> None:
    num_layers, num_slots, num_heads, head_dim, block_size = 2, 12, 1, 4, 2
    kv_cache = (
        torch.arange(
            2 * num_layers * num_slots * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        )
        .reshape(2, num_layers, num_slots, num_heads, head_dim)
        .contiguous()
        .add_(rank * 12288)
    )
    rank_source_slots = (
        ((1, 2), (7, 8)),
        ((3, 4), (9, 10)),
    )
    source_slots = rank_source_slots[rank]
    prefix_cache = RadixPrefixIndex(block_size=block_size, fingerprint=b"tp2-mixed")
    blocks = []
    parent_id = None
    for logical_idx, slots in enumerate(source_slots):
        tokens = tuple(
            range(logical_idx * block_size + 1, (logical_idx + 1) * block_size + 1)
        )
        block_id = prefix_cache.stable_block_id(tokens, parent_id)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent_id,
            block_size=block_size,
            logical_block_idx=logical_idx,
            payload=MixedPrefixBlockPayload(
                kv_payload=StandardPrefixBlockPayload(
                    token_slots=torch.tensor(slots, dtype=torch.int32, device=device),
                    block_start=logical_idx * block_size,
                    block_end=(logical_idx + 1) * block_size,
                ),
                recurrent_payload=_mixed_recurrent_payload(
                    device,
                    seed=rank * 100 + 10 * (logical_idx + 1),
                    token_count=block_size,
                ),
                token_count=block_size,
                accounting_bytes=0,
                recurrent_bytes=0,
            ),
            token_ids=tokens,
        )
        prefix_cache.insert_block(block)
        blocks.append(block)
        parent_id = block_id

    source_indices = torch.cat(
        [block.payload.kv_payload.token_slots.long() for block in blocks]
    )
    expected_kv = kv_cache[:, :, source_indices].cpu()
    expected_recurrent = [_recurrent_snapshot(block) for block in blocks]
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

    controller.submit_d2h(blocks)
    controller.synchronize_all()
    ready = _all_gather_object(_logical_signature(blocks), world_size)
    assert ready[0] == ready[1]
    assert all(item[2:] == (0, True, True, None) for item in ready[rank])
    gathered_slots = _all_gather_object(source_slots, world_size)
    assert gathered_slots[0] != gathered_slots[1]
    rank_local_kv_value = float(expected_kv[0, 0, 0, 0, 0])
    gathered_kv_values = _all_gather_object(rank_local_kv_value, world_size)
    assert gathered_kv_values[0] != gathered_kv_values[1]
    rank_local_recurrent_value = int(expected_recurrent[0][1]["counter"][0])
    gathered_recurrent_values = _all_gather_object(rank_local_recurrent_value, world_size)
    assert gathered_recurrent_values[0] != gathered_recurrent_values[1]

    host_indices = [int(block.payload.host_block_index) for block in blocks]
    host_token_indices = host_pool.token_indices(host_indices, device).cpu()
    host_kv = host_pool.cache.reshape(
        2,
        num_layers,
        -1,
        num_heads,
        head_dim,
    )[:, :, host_token_indices]
    assert torch.equal(host_kv, expected_kv)
    for block_idx, host_idx in enumerate(host_indices):
        for layer_idx, states in expected_recurrent[block_idx].items():
            for name, tensor in states.items():
                assert torch.equal(host_pool.recurrent_cache[layer_idx][name][host_idx], tensor)

    assert len(prefix_cache.demote_device_until_freeable(len(blocks))) == len(blocks)
    rank_destination_slots = (
        ((0, 3), (6, 11)),
        ((1, 5), (7, 11)),
    )
    destination_slots = rank_destination_slots[rank]
    for block, slots in zip(blocks, destination_slots):
        block.payload.kv_payload.token_slots = torch.tensor(
            slots,
            dtype=torch.int32,
            device=device,
        )
        controller.free_device_recurrent(block)
    controller.allocate_device_recurrent(blocks)
    destination_indices = torch.cat(
        [block.payload.kv_payload.token_slots.long() for block in blocks]
    )
    kv_cache[:, :, destination_indices].zero_()

    operation = controller.submit_h2d(blocks)
    controller.wait_for_auxiliary(operation)
    for layer_idx in range(num_layers):
        controller.wait_for_layer(operation, layer_idx)
    controller.synchronize_all()
    assert torch.equal(kv_cache[:, :, destination_indices].cpu(), expected_kv)
    for block, expected in zip(blocks, expected_recurrent):
        _assert_recurrent_snapshot(block, expected)

    final = _all_gather_object(_logical_signature(blocks), world_size)
    assert final[0] == final[1]
    assert all(item[2:] == (0, True, True, None) for item in final[rank])


def _run_mixed_quest_round_trip(
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    num_layers, num_pages, num_heads, head_dim = 2, 7, 1, 4
    page_size, block_size = 2, 4
    num_slots = num_pages * page_size
    kv_cache = (
        torch.arange(
            2 * num_layers * num_slots * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        )
        .reshape(2, num_layers, num_slots, num_heads, head_dim)
        .contiguous()
        .add_(rank * 16384)
    )
    metadata_cache = (
        torch.arange(
            2 * num_layers * num_pages * num_heads * head_dim,
            dtype=torch.float16,
            device=device,
        )
        .reshape(2, num_layers, num_pages, num_heads, head_dim)
        .contiguous()
        .add_(32768 + rank * 8192)
    )
    rank_source_pages = ((1, 4), (2, 5))
    source_pages = rank_source_pages[rank]
    pages = torch.tensor(source_pages, dtype=torch.int32, device=device)
    slots = (
        pages[:, None] * page_size
        + torch.arange(page_size, dtype=torch.int32, device=device)
    ).reshape(-1)
    recurrent = _mixed_recurrent_payload(
        device,
        seed=rank * 100 + 30,
        token_count=block_size,
    )
    prefix_cache = RadixPrefixIndex(
        block_size=block_size,
        fingerprint=b"tp2-mixed-quest",
    )
    block_id = prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=block_size,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload=QuestPrefixBlockPayload(
                block_slot=None,
                token_slots=slots,
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
    expected_metadata = metadata_cache[:, :, pages.long()].cpu()
    expected_recurrent = _recurrent_snapshot(block)
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
        device_metadata_cache=metadata_cache,
        host_pool=host_pool,
        block_size=block_size,
        device=device,
        kv_transformer_layer_indices=(0, 2),
    )

    controller.submit_d2h([block])
    controller.synchronize_all()
    ready = _all_gather_object(_logical_signature([block]), world_size)
    assert ready[0] == ready[1]
    assert ready[rank][0][2:] == (0, True, True, None)
    gathered_pages = _all_gather_object(source_pages, world_size)
    assert gathered_pages[0] != gathered_pages[1]
    rank_local_kv_value = float(expected_kv[0, 0, 0, 0, 0])
    gathered_kv_values = _all_gather_object(rank_local_kv_value, world_size)
    assert gathered_kv_values[0] != gathered_kv_values[1]
    rank_local_metadata_value = float(expected_metadata[0, 0, 0, 0, 0])
    gathered_metadata_values = _all_gather_object(rank_local_metadata_value, world_size)
    assert gathered_metadata_values[0] != gathered_metadata_values[1]
    rank_local_recurrent_value = int(expected_recurrent[1]["counter"][0])
    gathered_recurrent_values = _all_gather_object(rank_local_recurrent_value, world_size)
    assert gathered_recurrent_values[0] != gathered_recurrent_values[1]

    host_idx = int(block.payload.host_block_index)
    host_token_indices = host_pool.token_indices([host_idx], device).cpu()
    host_kv = host_pool.cache.reshape(
        2,
        num_layers,
        -1,
        num_heads,
        head_dim,
    )[:, :, host_token_indices]
    assert torch.equal(host_kv, expected_kv)
    host_pages = [host_idx * host_pool.pages_per_block + offset for offset in range(2)]
    assert torch.equal(host_pool.metadata_cache[:, :, host_pages], expected_metadata)
    for layer_idx, states in expected_recurrent.items():
        for name, tensor in states.items():
            assert torch.equal(host_pool.recurrent_cache[layer_idx][name][host_idx], tensor)

    assert prefix_cache.demote_device_until_freeable(1) == [block]
    rank_destination_pages = ((0, 3), (1, 4))
    destination_pages = torch.tensor(
        rank_destination_pages[rank],
        dtype=torch.int32,
        device=device,
    )
    destination_slots = (
        destination_pages[:, None] * page_size
        + torch.arange(page_size, dtype=torch.int32, device=device)
    ).reshape(-1)
    block.payload.kv_payload.block_slots = destination_pages
    block.payload.kv_payload.token_slots = destination_slots
    controller.free_device_recurrent(block)
    controller.allocate_device_recurrent([block])
    kv_cache[:, :, destination_slots.long()].zero_()
    metadata_cache[:, :, destination_pages.long()].zero_()

    operation = controller.submit_h2d([block])
    controller.wait_for_auxiliary(operation)
    for layer_idx in range(num_layers):
        controller.wait_for_layer(operation, layer_idx)
    controller.synchronize_all()
    assert torch.equal(kv_cache[:, :, destination_slots.long()].cpu(), expected_kv)
    assert torch.equal(
        metadata_cache[:, :, destination_pages.long()].cpu(),
        expected_metadata,
    )
    _assert_recurrent_snapshot(block, expected_recurrent)

    final = _all_gather_object(_logical_signature([block]), world_size)
    assert final[0] == final[1]
    assert final[rank][0][2:] == (0, True, True, None)


def _tp2_offload_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        _run_standard_round_trip(rank, world_size, device)
        dist.barrier()
        _run_quest_round_trip(rank, world_size, device)
        dist.barrier()
        _run_mixed_round_trip(rank, world_size, device)
        dist.barrier()
        _run_mixed_quest_round_trip(rank, world_size, device)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_tp2_prefix_offload_round_trip_and_logical_consistency(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("TP2 prefix offload test requires two visible CUDA devices")
    rendezvous = tmp_path / "tp2-prefix-offload-rendezvous"
    mp.spawn(
        _tp2_offload_worker,
        args=(2, f"file://{rendezvous}"),
        nprocs=2,
        join=True,
    )
