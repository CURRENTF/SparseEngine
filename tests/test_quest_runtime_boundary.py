"""QuEST logical selection must preserve the cache manager's physical view."""

from types import SimpleNamespace

import torch

from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager import LayerBatchStates
from sparseengine.engine.cache_manager.methods.quest import QuestCacheManager
from sparseengine.engine.cache_manager.storage import ExplicitKVStorage
from sparseengine.engine.sparse_methods.base import (
    DecodeSelectionRequest,
    LayerBatchSparseState,
)
from sparseengine.engine.sparse_methods.quest import QuestRuntime
from sparseengine.operators.quest_selection import (
    QuestPageSelectionOpSpec,
    TorchQuestPageSelectionProvider,
)


def test_explicit_quest_runtime_preserves_ranked_pages_and_short_row_fallback():
    page_size = 2
    manager = object.__new__(QuestCacheManager)
    manager.config = SimpleNamespace(quest_skip_layers=0, quest_token_budget=6)
    manager.runtime_layout = RuntimeLayout.dense(1)
    manager.page_size = page_size
    manager.max_pages_per_row = 4
    manager.metadata_num_heads = 1
    manager.metadata_head_dim = 1
    manager.device = torch.device("cpu")
    manager.platform = SimpleNamespace(is_cuda_alike=lambda: False)
    manager.attention_cache_storage = ExplicitKVStorage(
        num_kv_heads=1, head_dim=1, dtype=torch.float32,
    )
    manager.attention_cache_storage.allocate(
        num_layers=1, num_slots=12, device=manager.device,
    )
    manager.quest_page_selector = TorchQuestPageSelectionProvider(
        op_spec=QuestPageSelectionOpSpec(score_dtype=torch.float32, cuda_graph=False)
    )
    # Logical pages deliberately differ from physical page numbers.
    manager.buffer_req_to_page_slots = torch.tensor(
        [[2, 0, 3, 1], [4, 5, -1, -1]], dtype=torch.int32,
    )
    manager.metadata_cache = torch.zeros(2, 1, 6, 1, 1)
    for logical_page, score in enumerate((-5.0, 10.0, 2.0, -3.0)):
        physical_page = int(manager.buffer_req_to_page_slots[0, logical_page])
        manager.metadata_cache[:, 0, physical_page, 0, 0] = score
    manager.layer_batch_state = LayerBatchStates(max_context_len=8)
    req_indices = torch.tensor([0, 1], dtype=torch.int32)
    context_lens = torch.tensor([8, 4], dtype=torch.int32)

    runtime = object.__new__(QuestRuntime)
    runtime.config = manager.config
    runtime.quest_cache = manager
    runtime.quest_page_scorer = None
    runtime.layer_batch_sparse_states = {0: LayerBatchSparseState(
        req_indices=req_indices, context_lens=context_lens, max_context_len=8,
    )}
    query = torch.ones(2, 2, 1)
    selection = runtime.build_decode_selection(
        DecodeSelectionRequest(0, query, None)
    )
    assert selection.page_selection is not None
    view = manager.build_decode_compute_view(
        0, query, selection, num_heads=2, num_kv_heads=1,
    )
    assert view.meta.is_sparse
    assert view.meta.page_table[0, :3].tolist() == [0, 3, 1]
    assert view.meta.context_lens.tolist() == [6, 4]
    assert view.meta.page_table[1, :2].tolist() == [4, 5]
    assert view.meta.page_counts.tolist() == [3, 2]


def test_quest_runtime_dense_policy_does_not_score_pages():
    manager = object.__new__(QuestCacheManager)
    manager.config = SimpleNamespace(quest_skip_layers=1, quest_token_budget=6)
    manager.runtime_layout = RuntimeLayout.dense(1)
    manager.page_size = 2
    runtime = object.__new__(QuestRuntime)
    runtime.config = manager.config
    runtime.quest_cache = manager
    runtime.layer_batch_sparse_states = {0: LayerBatchSparseState(
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([8], dtype=torch.int32),
        max_context_len=8,
    )}
    selection = runtime.build_decode_selection(
        DecodeSelectionRequest(0, torch.empty(1, 2, 1), None)
    )
    assert selection.page_selection is None
