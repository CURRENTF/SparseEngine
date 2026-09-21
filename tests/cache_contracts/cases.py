"""Closed, reviewed configuration matrix, not a runtime extension registry.

Fixtures initialize actual manager classes on small CPU tensors. No allocator,
capacity hook, method-state hook, or index operation is replaced. CUDA tests use
these same classes; only the CPU offload transport fixture simulates DMA.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class ChainCase:
    method: str
    prefill: str = ''

    @property
    def id(self):
        return (self.method or 'vanilla') + ('+' + self.prefill if self.prefill else '')

    @property
    def shared(self):
        return self.prefill == 'omnikv_prefill'


CHAIN_CASES = tuple(ChainCase(name) for name in (
    'streamingllm', 'snapkv', 'h2o', 'pyramidkv', 'rkv', 'skipkv',
)) + (ChainCase('', 'omnikv_prefill'), ChainCase('omnikv', 'omnikv_prefill'))
RADIX_CASES = ('', 'omnikv', 'quest')


def make_chain(case: ChainCase, *, capacity=16, rows=2, device='cpu'):
    if case.shared:
        if device != 'cpu':
            raise ValueError('Shared-pool device construction uses the model integration suite.')
        m = make_radix(case.method)
        import torch
        m.enable_prefix_caching = False
        m.prefix_cache = None
        m.free_slots_stack = torch.arange(capacity, dtype=torch.int32)
        m._num_free_slots = capacity
        m.config.num_kvcache_slots = capacity
    else:
        from test_chain_offload import make_manager
        from sparseengine.engine.cache_manager.methods.streamingllm import StreamingLLMCacheManager
        from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
        from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
        from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
        from sparseengine.engine.cache_manager.methods.skipkv import SkipKVCacheManager
        cls = {
            'streamingllm': StreamingLLMCacheManager, 'snapkv': SnapKVCacheManager,
            'pyramidkv': SnapKVCacheManager, 'h2o': H2OCacheManager,
            'rkv': RKVCacheManager, 'skipkv': SkipKVCacheManager,
        }[case.method]
        m = make_manager(cls, device=device, rows=rows, capacity=capacity)
    cfg = m.config
    cfg.sparse_method = cfg.resolved_cache_sparse_method = case.method
    cfg.prefill_sparse_method = case.prefill
    cfg.enable_prefix_caching = True
    cfg.resolved_prefix_cache_mode = 'chain'
    cfg.enable_prefix_cache_offload = False
    cfg.engine_prefill_chunk_size = 4
    cfg.sink_keep_tokens = 2
    cfg.recent_keep_tokens = 4
    cfg.decode_reservation_window = 1
    cfg.chain_cache_max_tombstones = 64
    cfg.pyramid_layer_ratios = [1.0, 0.5] if case.method == 'pyramidkv' else None
    cfg.prefill_schedule_policy = 'chunked'
    cfg.h2o_prefill_budget = 8
    cfg.h2o_decode_budget = 8
    cfg.h2o_recent_ratio = 0.5
    cfg.h2o_decode_eviction_interval = 1
    cfg.rkv_compression_interval = 4
    cfg.skipkv_compression_interval = 4
    return m


def make_radix(method: str):
    import torch
    from test_prefix_cache import (
        _make_standard_manager_for_prefix, _make_quest_manager_for_prefix,
    )
    if method == 'quest':
        m = _make_quest_manager_for_prefix(page_size=2)
    else:
        m = _make_standard_manager_for_prefix(block_size=2, method=method)
        # The legacy fixture has a 100-entry stack but only 90 initial free
        # entries. Give the independent oracle a fully owned 90-slot universe.
        m.free_slots_stack = torch.arange(90, dtype=torch.int32)
        m.lru = None
        m.row_logical_lens = m.row_seq_lens.copy()
    m.num_layers = m.num_kv_layers = 2
    m.max_model_len = int(m.buffer_req_to_token_slots.shape[1])
    m.runtime_layout = SimpleNamespace(
        kv_idx_to_layer_idx=(0, 1), kv_layer_index=lambda i: int(i),
    )
    m.config.resolved_prefix_cache_mode = 'radix'
    m.config.enable_prefix_caching = True
    m.config.enable_prefix_cache_offload = False
    return m


def allocate_chain(m, case: ChainCase, seq_id: int, lengths: tuple[int, ...]):
    if case.shared:
        if len(set(lengths)) != 1:
            raise ValueError('A shared physical pool cannot have different row lengths.')
        return (m._allocate(seq_id, lengths[0]),)
    return tuple(m._allocate(layer, seq_id, size)
                 for layer, size in zip(m.kv_transformer_layer_indices(), lengths, strict=True))


def make_inflight_radix(method: str, block_count: int, blocks_per_operation: int):
    """Resident prefix with unfinished transfers, including shared operations."""
    import torch
    from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
    from sparseengine.engine.cache_manager.methods.quest import QuestPrefixBlockPayload
    from sparseengine.engine.prefix_cache import PrefixCacheBlock
    from sparseengine.engine.sequence import Sequence
    from test_prefix_cache import _FakePrefixOffloadController

    m = make_radix(method)
    length = 2 * block_count
    m.max_model_len = length + 2
    m.buffer_req_to_token_slots = torch.zeros((2, length + 2), dtype=torch.int32)
    if method == 'quest':
        m.num_pages = block_count + 4
        m.buffer_req_to_page_slots = torch.full((2, block_count + 1), -1, dtype=torch.int32)
        m.free_pages_stack = torch.cat((torch.arange(block_count, block_count + 4),
                                      torch.arange(block_count))).to(torch.int32)
        m._num_free_pages = 4
    else:
        m.free_slots_stack = torch.cat((torch.arange(length, length + 8),
                                      torch.arange(length))).to(torch.int32)
        m._num_free_slots = 8
        m.config.num_kvcache_slots = length + 8
    blocks = []
    parent = None
    for i in range(block_count):
        tokens = (2 * i, 2 * i + 1)
        block_id = m.prefix_cache.stable_block_id(list(tokens), parent)
        slots = torch.tensor(tokens, dtype=torch.int32)
        payload = (QuestPrefixBlockPayload(block_slot=i, token_slots=slots, host_block_index=i)
                   if method == 'quest' else
                   StandardPrefixBlockPayload(token_slots=slots, host_block_index=i))
        block = PrefixCacheBlock(block_id, parent, 2, i, payload, token_ids=tokens)
        m.prefix_cache.insert_block(block)
        m.prefix_cache.begin_d2h(block)
        m.prefix_cache.finish_d2h(block)
        blocks.append(block)
        parent = block_id
    m.prefix_cache.demote_device_until_freeable(block_count)
    controller = _FakePrefixOffloadController(m.prefix_cache)
    for start in range(0, block_count, blocks_per_operation):
        controller.submit_h2d(blocks[start:start + blocks_per_operation])
    m.prefix_offload_controller = controller
    seq = Sequence(list(range(length + 1)))
    seq.seq_id = 71
    seq.prefix_cache_hit_len = length
    seq.prefix_cache_hit_block_count = block_count
    seq.prefix_cache_hit_last_block_id = parent
    return m, seq
