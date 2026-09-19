"""Physical components must survive asynchronous prefix migration without reinterpretation."""

import pytest
import torch

from sparseengine.engine.cache_manager.methods.quest import QuestPrefixBlockPayload
from sparseengine.engine.cache_manager.offload.prefix_components import (
    ComponentPrefixOffloadController,
    ComponentPrefixPool,
    prefix_block_bytes,
    storage_prefix_components,
)
from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
from sparseengine.engine.cache_manager.storage import (
    ExplicitKVStorage,
    HeterogeneousExplicitKVStorage,
    MlaLatentStorage,
)
from sparseengine.engine.cache_manager.storage.components import CacheComponentSpec
from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex


def _storage(kind):
    if kind == "mla":
        return MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16)
    if kind == "heterogeneous":
        return HeterogeneousExplicitKVStorage(
            layer_shapes=((2, 64), (1, 128)), dtype=torch.bfloat16
        )
    return ExplicitKVStorage(num_kv_heads=2, head_dim=64, dtype=torch.bfloat16)


@pytest.mark.parametrize("kind", ["explicit", "mla", "heterogeneous"])
def test_component_description_matches_physical_bytes(kind):
    """Capacity must use real component widths, including heterogeneous layers."""
    storage = _storage(kind)
    storage.allocate(num_layers=2, num_slots=16, device=torch.device("cpu"))
    components = storage_prefix_components(storage, 2)
    assert prefix_block_bytes(components, 16) == sum(
        tensor.numel() * tensor.element_size()
        for layer in components
        for _, tensor in layer
    )
    for layer in components:
        for spec, tensor in layer:
            assert spec.shape == tuple(tensor.shape[1:])
            assert spec.dtype == tensor.dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU indexed host copy"
)
@pytest.mark.parametrize(
    "kind,summaries", [("mla", False), ("mla", True), ("heterogeneous", False)]
)
def test_component_prefix_roundtrip_layer_readiness_and_rollback(
    kind, summaries, monkeypatch
):
    device = torch.device("cuda:0")
    page_size, num_layers, slots = 16, 2, 64
    storage = _storage(kind)
    storage.allocate(num_layers=num_layers, num_slots=slots, device=device)
    components = storage_prefix_components(storage, num_layers)
    if summaries:
        components = tuple(
            layer
            + tuple(
                (
                    CacheComponentSpec(name, (1, 576), torch.bfloat16, "page"),
                    torch.randn(
                        slots // page_size, 1, 576, dtype=torch.bfloat16, device=device
                    ),
                )
                for name in ("page_max", "page_min")
            )
            for layer in components
        )
    for layer in components:
        for _, tensor in layer:
            tensor.normal_()
    cache = RadixPrefixIndex(block_size=page_size, fingerprint=b"component-transfer")
    source_slots = torch.arange(
        page_size, 2 * page_size, device=device, dtype=torch.int32
    )
    payload = (
        QuestPrefixBlockPayload(block_slot=1, token_slots=source_slots)
        if summaries
        else StandardPrefixBlockPayload(token_slots=source_slots)
    )
    block = PrefixCacheBlock(
        stable_block_id=cache.stable_block_id(list(range(page_size)), None),
        parent_block_id=None,
        block_size=page_size,
        logical_block_idx=0,
        payload=payload,
        token_ids=tuple(range(page_size)),
    )
    cache.insert_block(block)
    pool = ComponentPrefixPool(
        components=components, capacity_blocks=2, block_size=page_size
    )
    controller = ComponentPrefixOffloadController(
        components=components,
        prefix_cache=cache,
        host_pool=pool,
        block_size=page_size,
        device=device,
    )
    expected = tuple(
        tuple(
            tensor[1:2].cpu()
            if spec.index_unit == "page"
            else tensor[source_slots.long()].cpu()
            for spec, tensor in layer
        )
        for layer in components
    )
    original = controller._submit_d2h_payload

    def fail_after_submit(*args):
        original(*args)
        raise RuntimeError("injected submission failure")

    monkeypatch.setattr(controller, "_submit_d2h_payload", fail_after_submit)
    with pytest.raises(RuntimeError, match="injected"):
        controller.submit_d2h([block])
    assert pool.used_blocks == 0
    assert not block.residency.host_present
    assert block.residency.transfer is None
    monkeypatch.setattr(controller, "_submit_d2h_payload", original)
    controller.submit_d2h([block])
    assert not block.residency.host_present
    controller.synchronize_all()
    assert block.residency.host_present
    assert controller.d2h_bytes == prefix_block_bytes(components, page_size)
    assert cache.demote_device_until_freeable(1) == [block]
    for layer in components:
        for _, tensor in layer:
            tensor.zero_()
    destination_slots = torch.arange(
        2 * page_size, 3 * page_size, device=device, dtype=torch.int32
    )
    payload.token_slots = destination_slots
    if summaries:
        payload.block_slot = 2
    restore_layer = controller._submit_h2d_layer

    def fail_partial_restore(*args):
        restore_layer(*args)
        raise RuntimeError("injected partial restore failure")

    monkeypatch.setattr(controller, "_submit_h2d_layer", fail_partial_restore)
    with pytest.raises(RuntimeError, match="partial restore"):
        controller.submit_h2d([block])
    assert block.residency.host_present and not block.residency.device_present
    assert block.residency.transfer is None and pool.used_blocks == 1
    monkeypatch.setattr(controller, "_submit_h2d_layer", restore_layer)
    operation = controller.submit_h2d([block])
    snapshots = []
    for layer_idx, layer in enumerate(components):
        controller.wait_for_layer(operation, layer_idx)
        snapshots.append(
            tuple(
                tensor[2:3].clone()
                if spec.index_unit == "page"
                else tensor[destination_slots.long()].clone()
                for spec, tensor in layer
            )
        )
    torch.cuda.synchronize()
    for restored, reference in zip(snapshots, expected):
        for actual, wanted in zip(restored, reference):
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
    controller.poll()
    assert block.residency.device_present and block.residency.transfer is None
    assert controller.h2d_bytes == prefix_block_bytes(components, page_size)
    controller.reset()
    assert pool.used_blocks == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU indexed host copy"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,components", [(64, 1), (513, 3)])
def test_batched_component_transfer_preserves_rows_under_graph(
    dtype, width, components
):
    """Batching must preserve component identity, masked tails, and replay indices."""
    from sparseengine.operators.indexed_host_copy import (
        make_pointer_table,
        transfer_components,
    )

    source = [
        torch.randn(40, width, device="cuda", dtype=dtype) for _ in range(components)
    ]
    target = [torch.empty(40, width, dtype=dtype, pin_memory=True) for _ in source]
    src_ptrs = make_pointer_table(source, device="cuda")
    dst_ptrs = make_pointer_table(target, device="cuda")
    src_rows = torch.randperm(40, device="cuda")[:33].contiguous()
    dst_rows = torch.randperm(40, device="cuda")[:33].contiguous()

    def copy():
        transfer_components(
            src_ptrs, dst_ptrs, src_rows, dst_rows, width=width, dtype=dtype
        )

    copy()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy()
    torch.cuda.synchronize()
    for _ in range(2):
        src_rows.copy_(torch.randperm(40, device="cuda")[:33])
        for tensor in target:
            tensor.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        for src, dst in zip(source, target):
            expected = torch.full_like(dst, -7)
            expected[dst_rows.cpu()] = src.cpu()[src_rows.cpu()]
            torch.testing.assert_close(dst, expected, rtol=0, atol=0)
