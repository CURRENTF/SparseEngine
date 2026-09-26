from types import SimpleNamespace

import torch

from sparseengine.engine.cache_manager.methods.deepseek_v4 import (
    DeepSeekV4CacheManager, NativeSharedKVRequest,
)
from sparseengine.engine.cache_manager.storage.compressed_family import CompressedKVFamily
from sparseengine.engine.cache_manager.storage.shared_kv_state import SharedKVStateRows
from sparseengine.engine.prefix_cache import RadixPrefixIndex


def native_prefix_owner():
    # Exercise physical ownership and retirement independently of CUDA operators.
    cache = object.__new__(DeepSeekV4CacheManager)
    cache.config = SimpleNamespace(hf_config=SimpleNamespace(dtype=torch.bfloat16),
                                   sparse_method="deepseek_v4")
    cache.state_rows = SharedKVStateRows(num_rows=4, reserved_rows=1,
        compress_ratios=(4,), device=torch.device("cpu"))
    family = CompressedKVFamily(layer_ids=(0,), with_index=True, num_pages=5,
        reserved_pages=1, page_size=64, device=torch.device("cpu"))
    lease = family.new_lease()
    family.reserve(lease, 4)
    family.mark_materialized(lease, 4)
    row = cache.state_rows.allocate()
    cache.requests = {1: NativeSharedKVRequest(row, 16, {4: lease})}
    cache.families = {4: family}
    cache.max_buffer_rows = 1
    cache.max_model_len = 256
    cache.prefix_rows = 2
    cache.prefix_cache_block_size = 16
    cache.prefix_cache = RadixPrefixIndex(block_size=16, fingerprint=b"native", max_blocks=2)
    cache.pending_prefix = {}
    cache.private_prefix_records = {}
    cache._async_prefix_records = []
    seq = SimpleNamespace(seq_id=1, num_prompt_tokens=32, token_ids=list(range(32)))
    return cache, seq


def test_prefix_snapshot_is_private_until_retirement_and_survives_append():
    cache, seq = native_prefix_owner()
    request = cache.requests[1]
    for tensor in cache.state_rows.accounting_tensors():
        tensor[request.row].fill_(7)
    family, lease = cache.families[4], request.leases[4]
    for tensor in family.accounting_tensors():
        tensor[lease.pages[0]].fill_(9)
    cache._freeze_prefix_snapshot(seq)
    assert len(cache.prefix_cache) == 0
    frozen_seq, tokens, record = cache._async_prefix_records[0]
    for tensor in cache.state_rows.accounting_tensors():
        tensor[request.row].zero_()
        assert bool((tensor[record.payload.row] == 7).all())
    family.reserve(lease, 5)
    for tensor in family.accounting_tensors():
        tensor[lease.pages[0]].zero_()
        assert bool((tensor[record.payload.leases[4].pages[0]] == 9).all())
    cache._record_frozen_prefix_materialization(frozen_seq, tokens, record)
    cache.publish_pending_prefix_blocks([seq])
    assert len(cache.prefix_cache) == 1
    assert cache.prefix_cache.match_longest_prefix(seq.token_ids, max_usable_tokens=16)[0] == 16
    cache.free_seq(1)
    cache.reset_prefix_cache()
    assert cache.state_rows.num_free_rows == 3
    assert family.allocator.num_free_pages == 4


def test_discarded_async_prefix_releases_unpublished_physical_state():
    cache, seq = native_prefix_owner()
    cache._freeze_prefix_snapshot(seq)
    cache.free_seq(seq.seq_id)
    assert len(cache.prefix_cache) == 0
    assert cache.state_rows.num_free_rows == 3
    assert cache.families[4].allocator.num_free_pages == 4


def test_decode_reservation_counts_shared_tail_copy_only_when_group_publishes():
    cache, seq = native_prefix_owner()
    seq.num_prefilled_tokens, seq.prefix_cache_hit_len = 16, 0
    assert cache.decode_window_costs(seq, 4) == {"ratio_4": 0}
    cache._freeze_prefix_snapshot(seq)
    assert cache.decode_window_costs(seq, 3) == {"ratio_4": 0}
    assert cache.decode_window_costs(seq, 4) == {"ratio_4": 1}
    assert cache.prefill_private_slots_for(seq) == 3
    # Shared budgets use physical pages; scalar scheduling projects their
    # original-token capacity without mixing page units and token units.
    free = cache.num_free_slots
    assert cache.prefill_capacity_after_decode_reservations(
        free, {"ratio_4": 1}, admission=False) == free-cache.page_size*4
    cache.free_seq(seq.seq_id)
