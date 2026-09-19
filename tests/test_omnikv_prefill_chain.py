"""Shared-pool chain admission must reserve physical slots exactly once."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.chain_cache import (
    ChainCacheCoordinator, ChainCapacityError, ChainFingerprintMismatchError,
    build_chain_cache_fingerprint,
)
from test_omnikv_prefill import _runtime_config
from test_prefix_cache import _make_standard_manager_for_prefix


@pytest.mark.parametrize("method,offload", [("", False), ("omnikv", False), ("omnikv", True)])
def test_chain_rejects_changed_prefill_semantics_before_resume(tmp_path, method, offload):
    config = _runtime_config(tmp_path, sparse_method=method,
                             full_attention_layers=[0, 2],
                             enable_omnikv_offload=offload, enable_prefix_caching=True)
    assert config.resolved_prefix_cache_mode == "chain"
    manager = _manager()
    coordinator = ChainCacheCoordinator(config, manager)
    plan = coordinator.plan_admission(chain_id="a", seq_id=1, token_ids=[1, 2, 3])
    coordinator.apply_admission(plan)
    coordinator.index.finish("a", token_ids=[1, 2, 3], processed_token_count=3,
                             physical_slots_by_layer=(3, 3))
    for field, value in (
        ("omnikv_prefill_keep_tokens", 1),
        ("omnikv_prefill_full_attention_layers", [0]),
        ("engine_prefill_chunk_size", 64),
        ("decode_keep_tokens", 7),
    ):
        if field == "decode_keep_tokens" and not method:
            continue
        changed = deepcopy(config)
        setattr(changed, field, value)
        with pytest.raises(ChainFingerprintMismatchError):
            coordinator.index.plan_admission(
                chain_id="a", seq_id=1, token_ids=[1, 2, 3, 4],
                fingerprint=build_chain_cache_fingerprint(changed),
            )
        assert coordinator.index.lookup("a").processed_token_count == 3


def _manager():
    manager = object.__new__(StandardCacheManager)
    manager.num_kv_layers = 2
    manager._num_free_slots = 10
    manager.free_rows = deque([0, 1])
    manager.seq_id_to_row = {}
    manager.row_seq_lens = np.zeros(2, dtype=np.int32)
    manager.kv_transformer_layer_indices = lambda: (0, 1)
    return manager


def test_chain_warmup_reclaims_worker_local_rows_without_scheduler_ledger():
    # Worker ranks retain warmup KV but do not run the rank-0 scheduler. The
    # physical allocator must be sufficient to restore capacity on every rank.
    manager = _make_standard_manager_for_prefix()
    manager.config.resolved_prefix_cache_mode = "chain"
    manager.enable_prefix_caching = False
    manager.prefix_cache = None
    manager._allocate(1, 3)
    manager._allocate(2, 5)
    manager.reset_after_warmup()
    assert manager.seq_id_to_row == {}
    assert manager._num_free_slots == manager.config.num_kvcache_slots
    assert not manager.row_seq_lens.any()
    assert set(manager.free_rows) == {0, 1}


def test_shared_chain_reservations_account_for_active_suffix_and_decode(tmp_path):
    # The original chain managers own separate layer pools. Repeating a shared
    # pool along the layer axis must neither double-charge nor omit reservations.
    config = _runtime_config(tmp_path, enable_prefix_caching=True)
    manager = _manager()
    coordinator = ChainCacheCoordinator(config, manager)
    plan = coordinator.plan_admission(chain_id="a", seq_id=1, token_ids=[1] * 7)
    coordinator.apply_admission(plan)
    before = deepcopy(coordinator.index.records)
    with pytest.raises(ChainCapacityError):
        coordinator.plan_admission(chain_id="b", seq_id=2, token_ids=[2] * 4)
    assert coordinator.index.records == before
    # Realizing three reserved tokens reduces both free space and outstanding
    # reservations; the second request still has exactly three available slots.
    manager.seq_id_to_row[1] = manager.free_rows.popleft()
    manager.row_seq_lens[0] = 3
    manager._num_free_slots -= 3
    assert manager.chain_physical_residency(1) == (3, 3)
    admitted = coordinator.plan_admission(chain_id="b", seq_id=2, token_ids=[2] * 3)
    assert admitted.victim_chain_ids == ()
    coordinator.decode_reservations = SimpleNamespace(outstanding=lambda: {"slots": 1})
    with pytest.raises(ChainCapacityError):
        coordinator.plan_admission(chain_id="b", seq_id=2, token_ids=[2] * 3)
    admitted = coordinator.plan_admission(chain_id="b", seq_id=2, token_ids=[2] * 2)
    assert admitted.decode_reserved_slots_by_layer == (1, 1)
    coordinator.validate_admission_plan(
        admitted, input_token_count=2, input_prefix_digest=b""
    )
