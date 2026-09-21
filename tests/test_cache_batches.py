"""Batch lifetime and atomic alias ownership regressions in prefix control paths."""
import pickle
from types import SimpleNamespace

import pytest
import torch

from cache_contracts.cases import make_radix
from sparseengine.engine.cache_manager.prefix_cache_mixin import PrefixLookupCache
from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
from sparseengine.engine.cache_manager.methods.quest import QuestPrefixBlockPayload
from sparseengine.engine.prefix_cache import PrefixCacheBlock
from sparseengine.engine.prefix_cache_coordinator import PrefixCacheCoordinator, MixedPrefixBlockPayload
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence


@pytest.mark.parametrize('mixed', (False, True))
def test_lookup_batch_failure_keeps_previous_memo_and_next_pass_retires_it(mixed):
    """Aborted RPC fragments cannot retain partial passes or discard reusable hashes."""
    manager = make_radix('')
    owner = manager
    if mixed:
        owner = object.__new__(PrefixCacheCoordinator)
        owner.prefix_cache = manager.prefix_cache
        owner.block_size = 2
        owner.offload_controller = None
    owner.prefix_lookup_cache = PrefixLookupCache(max_entries=1)
    runtime = RuntimeState(config=manager.config, cache_manager=manager,
                           prefix_cache_coordinator=owner if mixed else None)
    seqs = [Sequence([i, i + 1, i + 2]) for i in range(4)]

    def lookup(seq):
        runtime.refresh_prefix_cache_hit(pickle.loads(pickle.dumps(seq)))

    with runtime.prefix_cache_lookup_batch(first=True, last=True):
        for seq in seqs:
            lookup(seq)
    generations = owner.prefix_cache.block_id_generation_requests
    with runtime.prefix_cache_lookup_batch(first=True, last=False):
        lookup(seqs[0])
    with pytest.raises(ValueError, match='injected'):
        with runtime.prefix_cache_lookup_batch(first=False, last=True):
            lookup(seqs[1])
            raise ValueError('injected lookup failure')
    assert owner.prefix_lookup_cache._batch_entries is None
    assert set(owner.prefix_lookup_cache.entries) == {seq.seq_id for seq in seqs}
    with runtime.prefix_cache_lookup_batch(first=True, last=True):
        lookup(seqs[-1])
    assert owner.prefix_cache.block_id_generation_requests == generations
    assert set(owner.prefix_lookup_cache.entries) == {seqs[-1].seq_id}
    with pytest.raises(RuntimeError, match='no first fragment'):
        with runtime.prefix_cache_lookup_batch(first=False, last=True):
            pass


def mixed_payloads(method):
    manager = make_radix(method)
    payloads = []
    for i, page in enumerate((3, 1, 5)):
        slots = torch.tensor([2 * page, 2 * page + 1], dtype=torch.int32)
        kwargs = dict(token_slots=slots, block_start=2 * i, block_end=2 * i + 2)
        payloads.append(QuestPrefixBlockPayload(block_slot=None, block_slots=torch.tensor([page], dtype=torch.int32), **kwargs)
                        if method == 'quest' else StandardPrefixBlockPayload(**kwargs))
    return manager, payloads


@pytest.mark.parametrize('method', ('', 'quest'), ids=('standard', 'quest'))
@pytest.mark.parametrize('row_preexisted', (False, True))
def test_mixed_attach_preserves_alias_order_and_can_rollback(method, row_preexisted):
    """Disjoint physical pages must keep logical order and original row ownership."""
    manager, payloads = mixed_payloads(method)
    seq = SimpleNamespace(seq_id=77)
    if row_preexisted:
        manager._get_free_row(seq.seq_id)
    rows_before = dict(manager.seq_id_to_row)
    free_rows_before = list(manager.free_rows)
    free_before = manager.num_free_slots
    manager.attach_prefix_kv_payloads(seq, payloads)
    row = manager.seq_id_to_row[seq.seq_id]
    assert int(manager.row_seq_lens[row]) == 6
    assert manager.buffer_req_to_token_slots[row, :6].tolist() == [6, 7, 2, 3, 10, 11]
    if method == 'quest':
        assert manager.buffer_req_to_page_slots[row, :3].tolist() == [3, 1, 5]
    else:
        assert manager.seq_id_to_cached_ranges[seq.seq_id] == [(0, 2), (2, 4), (4, 6)]
    manager.rollback_prefix_kv_attach(seq, payloads, row_preexisted=row_preexisted)
    assert manager.seq_id_to_row == rows_before
    assert list(manager.free_rows) == free_rows_before
    assert manager.num_free_slots == free_before
    assert manager.buffer_req_to_token_slots[row, :6].tolist() == [0] * 6
    assert torch.cat([payload.token_slots for payload in payloads]).tolist() == [6, 7, 2, 3, 10, 11]


@pytest.mark.parametrize('method', ('', 'quest'), ids=('standard', 'quest'))
@pytest.mark.parametrize('failure', ('late-range', 'overflow', 'copy'))
def test_mixed_attach_failure_leaves_no_new_row_or_aliases(method, failure, monkeypatch):
    """A bad later block or failed final copy cannot leak an earlier alias or row."""
    manager, payloads = mixed_payloads(method)
    seq = SimpleNamespace(seq_id=77)
    free_rows_before = list(manager.free_rows)
    tokens_before = manager.buffer_req_to_token_slots.clone()
    pages_before = manager.buffer_req_to_page_slots.clone() if method == 'quest' else None
    if failure == 'late-range':
        payloads[-1].block_start += 1
    elif failure == 'overflow':
        manager.max_model_len = 4
        manager.buffer_req_to_token_slots = manager.buffer_req_to_token_slots[:, :4]
        tokens_before = tokens_before[:, :4]
    else:
        original_copy = torch.Tensor.copy_
        failed = False

        def fail_row_copy(target, source, *args, **kwargs):
            nonlocal failed
            if not failed and target.untyped_storage().data_ptr() == manager.buffer_req_to_token_slots.untyped_storage().data_ptr():
                failed = True
                original_copy(target[:2], source[:2])
                raise RuntimeError('injected partial row copy failure')
            return original_copy(target, source, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, 'copy_', fail_row_copy)
    with pytest.raises(RuntimeError):
        manager.attach_prefix_kv_payloads(seq, payloads)
    assert seq.seq_id not in manager.seq_id_to_row
    assert list(manager.free_rows) == free_rows_before
    assert all(length == 0 for length in manager.row_seq_lens)
    assert torch.equal(manager.buffer_req_to_token_slots, tokens_before)
    if method == 'quest':
        assert torch.equal(manager.buffer_req_to_page_slots, pages_before)
        assert seq.seq_id not in manager.seq_id_to_cached_pages
    else:
        assert seq.seq_id not in manager.seq_id_to_cached_ranges


def test_mixed_quest_rejects_corrupt_page_metadata_before_attaching():
    """Batched validation must reject one corrupt page among valid blocks."""
    manager, payloads = mixed_payloads('quest')
    payloads[-1].block_slots[0] += 1
    with pytest.raises(RuntimeError, match='mismatched'):
        manager.attach_prefix_kv_payloads(SimpleNamespace(seq_id=77), payloads)
    assert 77 not in manager.seq_id_to_row
    assert torch.count_nonzero(manager.buffer_req_to_token_slots).item() == 0


@pytest.mark.parametrize('method', ('', 'quest'), ids=('standard', 'quest'))
def test_recurrent_failure_rolls_back_all_batched_aliases_and_refs(method):
    """Recurrent attachment can fail after the entire KV batch was published."""
    manager, payloads = mixed_payloads(method)
    index = manager.prefix_cache
    parent = None
    blocks = []
    for i, payload in enumerate(payloads):
        tokens = (2 * i, 2 * i + 1)
        block_id = index.stable_block_id(tokens, parent)
        block = PrefixCacheBlock(block_id, parent, 2, i,
            MixedPrefixBlockPayload(kv_payload=payload, recurrent_payload=object(), token_count=2,
                                    accounting_bytes=8, recurrent_bytes=0), token_ids=tokens)
        index.insert_block(block)
        blocks.append(block)
        parent = block_id
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.prefix_cache = index
    coordinator.block_size = 2
    coordinator.cache_manager = manager
    coordinator.offload_controller = None
    coordinator.seq_id_to_prefix_blocks = {}
    coordinator._step_h2d_operations = {}

    def fail_recurrent(*args, **kwargs):
        raise RuntimeError('injected recurrent failure')

    coordinator.recurrent_state_manager = SimpleNamespace(attach_prefix_recurrent_payload=fail_recurrent)
    seq = SimpleNamespace(seq_id=77, prefix_cache_hit_len=6, prefix_cache_hit_block_count=3,
                          prefix_cache_hit_last_block_id=parent)
    with pytest.raises(RuntimeError, match='injected recurrent failure'):
        coordinator._attach_seq(seq)
    assert all(block.ref_count == 0 for block in blocks)
    assert not coordinator.seq_id_to_prefix_blocks
    assert seq.seq_id not in manager.seq_id_to_row
    assert torch.count_nonzero(manager.buffer_req_to_token_slots).item() == 0
    assert torch.cat([payload.token_slots for payload in payloads]).tolist() == [6, 7, 2, 3, 10, 11]
