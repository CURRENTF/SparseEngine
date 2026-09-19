from __future__ import annotations

import pytest
import torch
from glm_test_helpers import _glm_config, _single_rank_parallel_context

from sparseengine.engine.cache_manager import MlaLatentWrite, SparseSelection
from sparseengine.engine.cache_manager.methods.quest import QuestCacheManager
from sparseengine.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.utils.context import reset_context, set_context


def _manager():
    config = _glm_config(
        hf_overrides={"num_hidden_layers": 1, "mlp_layer_types": ["dense"]},
        sparse_method="quest",
        enable_prefix_caching=True,
        quest_chunk_size=16,
        quest_skip_layers=0,
        sink_keep_tokens=16,
        decode_keep_tokens=16,
        recent_keep_tokens=16,
        max_num_seqs_in_gpu=4,
        max_num_seqs_in_batch=4,
    )
    return QuestCacheManager(
        config,
        _single_rank_parallel_context(),
        allocation_budget_bytes=256 * 1024,
    )


def _prefill(manager, tokens):
    seq = Sequence(tokens)
    manager.refresh_prefix_cache_hit(seq)
    seq.num_prefilled_tokens = seq.prefix_cache_hit_len
    seq.current_chunk_size = len(tokens) - seq.num_prefilled_tokens
    ids, _, _ = manager.prepare_step([seq], is_prefill=True)
    slots = manager.layer_batch_state.slot_mapping
    payload = manager.attention_cache_storage.layer_payload(0)
    # Deterministic physical payload, independent of any attention kernel.
    payload.latent_cache[slots.long()] = 0
    payload.rope_cache[slots.long()] = 0
    payload.latent_cache[slots.long(), 0, 0] = ids.to(torch.bfloat16)
    set_context(True, cache_manager=manager)
    try:
        manager.on_kv_stored(0, torch.empty(0), slots)
        manager.on_forward_end([seq], is_prefill=True)
    finally:
        reset_context()
    seq.num_prefilled_tokens = len(tokens)
    return seq


def _assert_bounds(manager, seq):
    row = manager.seq_id_to_row[seq.seq_id]
    length = int(manager.row_seq_lens[row])
    payload = manager.attention_cache_storage.layer_payload(0)
    for start in range(0, length, manager.page_size):
        slots = manager.buffer_req_to_token_slots[
            row, start : min(start + manager.page_size, length)
        ].long()
        keys = torch.cat(
            (payload.latent_cache[slots], payload.rope_cache[slots]), dim=-1
        )
        page = int(manager.buffer_req_to_page_slots[row, start // manager.page_size])
        torch.testing.assert_close(manager.metadata_cache[0, 0, page], keys.amax(0))
        torch.testing.assert_close(manager.metadata_cache[1, 0, page], keys.amin(0))


def test_mla_quest_prefix_forks_preserve_bounds_and_release_pages():
    """Shared latent pages must survive divergent suffixes and release only once."""
    manager = _manager()
    initial_free = manager.num_free_slots
    source = _prefill(manager, list(range(49)))
    _assert_bounds(manager, source)
    manager.free_seq(source.seq_id)
    left = _prefill(manager, list(range(48)) + [101, 102])
    right = _prefill(manager, list(range(48)) + [201, 202, 203])
    assert left.prefix_cache_hit_len == right.prefix_cache_hit_len == 48
    left_row, right_row = (manager.seq_id_to_row[s.seq_id] for s in (left, right))
    torch.testing.assert_close(
        manager.buffer_req_to_page_slots[left_row, :3],
        manager.buffer_req_to_page_slots[right_row, :3],
    )
    assert (
        manager.buffer_req_to_page_slots[left_row, 3]
        != manager.buffer_req_to_page_slots[right_row, 3]
    )
    _assert_bounds(manager, left)
    _assert_bounds(manager, right)
    manager.free_seq(left.seq_id)
    _assert_bounds(manager, right)
    manager.free_seq(right.seq_id)
    manager.prefix_cache_delete_subtree(list(range(16)))
    assert manager.num_free_slots == initial_free
    reused = _prefill(manager, [7] * 19)
    assert reused.prefix_cache_hit_len == 0
    _assert_bounds(manager, reused)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA capture/replay"
)
@pytest.mark.parametrize("long_only", [False, True])
def test_mla_quest_prefix_graph_replay_updates_pages_rows_and_padding(long_only):
    """Catches captured selection reading stale pages after growth or prefix row reuse."""
    manager = _manager()
    first = _prefill(manager, list(range(111)))
    second = _prefill(manager, list(range(48)) + [101, 102])
    assert second.prefix_cache_hit_len == 48
    seqs = [first, second]
    for seq in seqs:
        seq.append_token(110)
    contract = DecodeGraphContract(
        method="quest",
        topology_path_id="unified",
        batch_capacity=3,
        context_capacity=128,
    )
    inputs = DecodeGraphInputs.allocate(
        contract, device=manager.device, pin_memory=True
    )
    state = manager.init_decode_graph_state(contract, inputs)
    manager.set_decode_static_max_context_len(128)
    manager.prepare_decode_graph_step(seqs, state)
    manager.set_decode_static_max_context_len(128)
    writes = MlaLatentWrite(
        latent=torch.zeros(3, 1, 512, dtype=torch.bfloat16, device=manager.device),
        rope=torch.zeros(3, 1, 64, dtype=torch.bfloat16, device=manager.device),
    )
    writes.latent[:, 0, 0] = 110
    q_latent = torch.zeros(3, 2, 512, dtype=torch.bfloat16, device=manager.device)
    q_latent[:, :, 0] = 1
    q_rope = torch.zeros(3, 2, 64, dtype=torch.bfloat16, device=manager.device)
    query = manager.build_decode_selection_query(
        torch.empty(0), mla_latent=q_latent, mla_rope=q_rope
    )
    selection = SparseSelection(
        kind="full",
        req_indices=inputs.request_indices,
        context_lens=inputs.context_lens,
        max_context_len=128,
    )

    def forward():
        manager.prepare_decode_graph_in(state)
        manager.store_attention_payload(0, writes)
        return manager.build_decode_compute_view(
            0, query, selection, num_heads=2, num_kv_heads=1
        )

    # Exercise the dynamic short-row fallback inside the fixed long-capacity view.
    set_context(False, cache_manager=manager)
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = forward()
        pointers = inputs.data_ptrs()
        output_pointer = result.meta.active_slots.data_ptr()
        for step in range(20):
            if step:
                if step == 17:
                    for seq in seqs:
                        manager.free_seq(seq.seq_id)
                    prefix_len = 64 if long_only else 32
                    seqs = [_prefill(manager, list(range(prefix_len)) + [211])]
                    assert seqs[0].prefix_cache_hit_len == prefix_len
                for seq in seqs:
                    seq.append_token(110 + step)
                manager.prepare_decode_graph_step(seqs, state)
                manager.set_decode_static_max_context_len(128)
            writes.latent[:, 0, 0] = 110 + step
            sign = -1 if step % 2 else 1
            q_latent[:, :, 0] = sign
            set_context(False, cache_manager=manager)
            graph.replay()
            torch.cuda.synchronize()
            manager.on_forward_end(seqs, is_prefill=False)
            assert inputs.data_ptrs() == pointers
            assert result.meta.active_slots.data_ptr() == output_pointer
            assert inputs.write_slot_mapping[len(seqs) :].eq(-1).all()
            for i, seq in enumerate(seqs):
                _assert_bounds(manager, seq)
                row = manager.seq_id_to_row[seq.seq_id]
                length = int(manager.row_seq_lens[row])
                slots = manager.buffer_req_to_token_slots[row, :length].long()
                if length <= 48:
                    expected = slots.tolist()
                else:
                    payload = manager.attention_cache_storage.layer_payload(0)
                    page_keys = (
                        payload.latent_cache[slots[: ((length - 1) // 16) * 16], 0, 0]
                        .float()
                        .view(-1, 16)
                    )
                    bounds = (page_keys * sign).amax(-1)
                    chosen = bounds.topk(2).indices.tolist()
                    expected = [
                        int(slot)
                        for page in chosen
                        for slot in slots[page * 16 : (page + 1) * 16]
                    ]
                    expected += slots[((length - 1) // 16) * 16 :].tolist()
                actual_length = int(result.meta.context_lens[i])
                assert actual_length == len(expected)
                assert sorted(
                    result.meta.active_slots[i, :actual_length].tolist()
                ) == sorted(expected)
    finally:
        reset_context()
