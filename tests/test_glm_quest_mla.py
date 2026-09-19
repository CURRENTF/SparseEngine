from __future__ import annotations

from types import SimpleNamespace

import torch

from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager import (
    LayerBatchStates,
    MlaLatentPayload,
    MlaLatentSelectionQuery,
    MlaLatentWrite,
    SparseSelection,
)
from sparseengine.engine.cache_manager.methods.quest import QuestCacheManager
from sparseengine.engine.cache_manager.storage import MlaLatentStorage
from sparseengine.operators.quest_selection import (
    QuestPageSelectionOpSpec,
    TorchQuestPageSelectionProvider,
)
from sparseengine.utils.context import reset_context, set_context


def test_quest_mla_fused_query_matches_expanded_logits_and_bounds_each_page():
    """Catches score-space drift between absorbed MLA decode and QuEST routing."""

    torch.manual_seed(431)
    page_size, num_pages, num_heads = 4, 3, 3
    latent_dim, nope_dim, rope_dim = 5, 4, 2
    latent = torch.randn(num_pages * page_size, latent_dim)
    rope = torch.randn(num_pages * page_size, rope_dim)
    key_projection = torch.randn(num_heads, nope_dim, latent_dim)
    q_nope = torch.randn(num_heads, nope_dim)
    q_rope = torch.randn(num_heads, rope_dim)

    expanded_nope = torch.einsum("hdc,tc->thd", key_projection, latent)
    expanded_keys = torch.cat(
        (expanded_nope, rope[:, None, :].expand(-1, num_heads, -1)),
        dim=-1,
    )
    expanded_query = torch.cat((q_nope, q_rope), dim=-1)
    expanded_logits = torch.einsum("hd,thd->ht", expanded_query, expanded_keys)

    absorbed_nope = torch.einsum("hd,hdc->hc", q_nope, key_projection)
    fused_query = torch.cat((absorbed_nope, q_rope), dim=-1)
    fused_keys = torch.cat((latent, rope), dim=-1)
    fused_logits = torch.einsum("hd,td->ht", fused_query, fused_keys)
    torch.testing.assert_close(fused_logits, expanded_logits)

    page_keys = fused_keys.view(num_pages, page_size, -1)
    page_min = page_keys.amin(dim=1)
    page_max = page_keys.amax(dim=1)
    manager = object.__new__(QuestCacheManager)
    manager.attention_cache_storage = object.__new__(MlaLatentStorage)
    manager.metadata_head_dim = latent_dim + rope_dim
    selection_query = MlaLatentSelectionQuery(
        latent=absorbed_nope.unsqueeze(0),
        rope=q_rope.unsqueeze(0),
    )
    shared_query = manager._selection_query_tensor(selection_query)[0]
    torch.testing.assert_close(
        shared_query,
        fused_query.mean(dim=0, keepdim=True),
    )
    production_bounds = QuestCacheManager._score_pages_batched(
        shared_query.unsqueeze(0),
        page_max[None, None, :, :],
        page_min[None, None, :, :],
        1,
    )[0]
    oracle_bounds = (
        shared_query[:, None, :]
        * torch.where(
            shared_query[:, None, :] >= 0,
            page_max[None, :, :],
            page_min[None, :, :],
        )
    ).sum(dim=-1).squeeze(0)
    torch.testing.assert_close(production_bounds, oracle_bounds)
    actual_page_max = expanded_logits.mean(dim=0).view(
        num_pages, page_size
    ).amax(dim=1)
    assert torch.all(production_bounds >= actual_page_max - 1e-5)


def _latent_quest_manager(*, page_size: int, num_pages: int):
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(
        num_layers=1,
        num_slots=page_size * num_pages,
        device=torch.device("cpu"),
    )
    manager = object.__new__(QuestCacheManager)
    manager.config = SimpleNamespace(
        quest_skip_layers=0,
        quest_token_budget=3 * page_size,
    )
    manager.hf_config = SimpleNamespace(dtype=torch.bfloat16)
    manager.device = torch.device("cpu")
    manager.validate_runtime_invariants = False
    manager.platform = SimpleNamespace(
        is_cuda_alike=lambda: False,
        is_stream_capturing=lambda: False,
    )
    manager.runtime_layout = RuntimeLayout.dense(1)
    manager.num_kv_layers = 1
    manager.page_size = page_size
    manager.num_pages = num_pages
    manager.max_pages_per_row = num_pages
    manager.metadata_num_heads = 1
    manager.metadata_head_dim = 576
    manager.attention_cache_storage = storage
    manager.quest_page_selector = TorchQuestPageSelectionProvider(
        op_spec=QuestPageSelectionOpSpec(
            score_dtype=torch.bfloat16,
            cuda_graph=False,
        )
    )
    manager.kv_cache = None
    manager.metadata_cache = torch.empty(
        2, 1, num_pages, 1, 576, dtype=torch.bfloat16
    )
    manager.page_offsets_i32 = torch.arange(page_size, dtype=torch.int32)
    manager.page_offsets_i64 = manager.page_offsets_i32.to(torch.int64)
    manager.layer_batch_state = LayerBatchStates(
        max_context_len=page_size * num_pages
    )
    manager._prefill_metadata_full_pages = True
    manager.buffer_req_to_page_slots = torch.empty(
        (1, num_pages), dtype=torch.int32
    )
    manager.buffer_req_to_token_slots = torch.empty(
        (1, page_size * num_pages), dtype=torch.int32
    )
    return manager, storage


def test_quest_mla_metadata_and_decode_view_keep_latent_payload_typed():
    """Catches regressions that reinterpret latent cache as explicit K/V."""

    page_size, num_pages = 2, 4
    manager, storage = _latent_quest_manager(
        page_size=page_size,
        num_pages=num_pages,
    )
    logical_to_physical_pages = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    manager.buffer_req_to_page_slots[0].copy_(logical_to_physical_pages)
    slots = (
        logical_to_physical_pages[:, None] * page_size
        + manager.page_offsets_i32[None, :]
    ).reshape(-1)
    manager.buffer_req_to_token_slots[0].copy_(slots)

    payload = storage.layer_payload(0)
    logical_scores = (-5.0, 10.0, 2.0, -3.0)
    for logical_page, physical_page in enumerate(
        logical_to_physical_pages.tolist()
    ):
        page_slots = torch.arange(
            physical_page * page_size,
            (physical_page + 1) * page_size,
        )
        payload.latent_cache[page_slots] = 0
        payload.rope_cache[page_slots] = 0
        payload.latent_cache[page_slots, 0, 0] = logical_scores[logical_page]

    set_context(True, cache_manager=manager)
    try:
        manager.on_kv_stored(
            0,
            torch.empty(page_size * num_pages, 512, dtype=torch.bfloat16),
            slots,
        )
    finally:
        reset_context()

    for logical_page, physical_page in enumerate(
        logical_to_physical_pages.tolist()
    ):
        expected = logical_scores[logical_page]
        assert float(manager.metadata_cache[0, 0, physical_page, 0, 0]) == expected
        assert float(manager.metadata_cache[1, 0, physical_page, 0, 0]) == expected

    selection = SparseSelection(
        kind="full",
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([page_size * num_pages], dtype=torch.int32),
        max_context_len=page_size * num_pages,
    )
    prefill_payload, prefill_slots, prefill_rows, prefill_lens = (
        manager.get_prefill_compute_payload(
            0,
            torch.empty(0),
            torch.empty(0),
            selection,
            slots.view(1, -1),
            selection.req_indices,
            selection.context_lens,
        )
    )
    assert isinstance(prefill_payload, MlaLatentPayload)
    assert prefill_payload.latent_cache.data_ptr() == payload.latent_cache.data_ptr()
    assert prefill_payload.rope_cache.data_ptr() == payload.rope_cache.data_ptr()
    assert prefill_slots.data_ptr() == slots.data_ptr()
    assert prefill_rows.data_ptr() == selection.req_indices.data_ptr()
    assert prefill_lens.data_ptr() == selection.context_lens.data_ptr()

    latent_query = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
    latent_query[..., 0] = 1
    rope_query = torch.zeros((1, 2, 64), dtype=torch.bfloat16)
    fused_query = manager.build_decode_selection_query(
        torch.zeros((1, 2, 256), dtype=torch.bfloat16),
        mla_latent=latent_query,
        mla_rope=rope_query,
    )
    assert isinstance(fused_query, MlaLatentSelectionQuery)
    selected_metadata = manager.metadata_cache[
        :, 0, logical_to_physical_pages[:3].long()
    ]
    page_scores = manager._score_pages_batched(
        fused_query.fused(),
        selected_metadata[0].permute(1, 0, 2).unsqueeze(0),
        selected_metadata[1].permute(1, 0, 2).unsqueeze(0),
        1,
    )
    torch.testing.assert_close(
        page_scores,
        torch.tensor([[-5.0, 10.0, 2.0]], dtype=torch.bfloat16),
    )
    set_context(False, cache_manager=manager)
    try:
        view = manager.build_decode_compute_view(
            0,
            fused_query,
            selection,
            num_heads=2,
            num_kv_heads=1,
        )
    finally:
        reset_context()

    assert isinstance(view.payload, MlaLatentPayload)
    assert view.payload.latent_cache.data_ptr() == payload.latent_cache.data_ptr()
    assert view.payload.rope_cache.data_ptr() == payload.rope_cache.data_ptr()
    selected_physical_pages = set(
        torch.div(
            view.meta.active_slots[0, ::page_size],
            page_size,
            rounding_mode="floor",
        ).tolist()
    )
    # Match explicit-KV QuEST: select the best previous pages, then append the
    # partially filled/latest page without scoring it.
    assert selected_physical_pages == {0, 1, 3}
    assert view.meta.context_lens.tolist() == [3 * page_size]
    assert view.meta.max_context_len == 3 * page_size


def test_quest_mla_partial_prefill_initializes_page_metadata():
    """Catches incremental decode reading uninitialized partial-page bounds."""

    page_size, num_pages = 4, 2
    manager, storage = _latent_quest_manager(
        page_size=page_size,
        num_pages=num_pages,
    )
    manager._prefill_metadata_full_pages = False
    payload = storage.layer_payload(0)
    payload.latent_cache.zero_()
    payload.rope_cache.zero_()
    payload.latent_cache[:2, 0, 0] = torch.tensor(
        [-3.0, 7.0],
        dtype=torch.bfloat16,
    )
    manager.metadata_cache.fill_(99)
    slots = torch.tensor([0, 1], dtype=torch.int32)

    set_context(True, cache_manager=manager)
    try:
        manager.on_kv_stored(
            0,
            torch.empty(2, 512, dtype=torch.bfloat16),
            slots,
        )
    finally:
        reset_context()

    assert float(manager.metadata_cache[0, 0, 0, 0, 0]) == 7.0
    assert float(manager.metadata_cache[1, 0, 0, 0, 0]) == -3.0


def test_quest_decode_store_dispatches_fused_metadata_update(monkeypatch):
    """Catches production decode bypassing the fused MLA metadata store."""

    manager, storage = _latent_quest_manager(page_size=4, num_pages=2)
    slots = torch.tensor([1], dtype=torch.int32)
    manager.layer_batch_state.slot_mapping = slots
    payload = MlaLatentWrite(
        latent=torch.zeros(1, 1, 512, dtype=torch.bfloat16),
        rope=torch.zeros(1, 1, 64, dtype=torch.bfloat16),
    )
    calls = []

    def record_fused_store(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        storage,
        "store_with_quest_metadata",
        record_fused_store,
    )
    set_context(False, cache_manager=manager)
    try:
        returned_slots = manager.store_attention_payload(0, payload)
    finally:
        reset_context()

    assert returned_slots is slots
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == 0
    assert args[1] is slots
    assert args[2] is payload
    assert args[3].data_ptr() == manager.metadata_cache[0, 0].data_ptr()
    assert args[4].data_ptr() == manager.metadata_cache[1, 0].data_ptr()
    assert kwargs == {"page_size": 4}


def test_quest_mla_capacity_accounts_two_fused_summaries_per_page():
    """Catches over-allocation from reusing explicit-KV metadata accounting."""

    page_size, num_layers, expected_pages = 16, 2, 3
    manager, storage = _latent_quest_manager(
        page_size=page_size,
        num_pages=expected_pages,
    )
    manager.num_kv_layers = num_layers
    manager.config = SimpleNamespace(num_kvcache_slots=None)
    token_bytes = storage.bytes_per_slot_per_layer()
    metadata_bytes = 2 * (512 + 64) * torch.empty(
        (), dtype=torch.bfloat16
    ).element_size()
    page_bytes = page_size * token_bytes + metadata_bytes
    manager._get_available_slots_info = lambda: (
        num_layers * page_bytes * expected_pages + page_bytes - 1,
        token_bytes,
    )

    manager.allocate_kv_cache()

    assert manager.num_pages == expected_pages
    assert manager.config.num_kvcache_slots == page_size * expected_pages
    assert storage.slot_capacity() == page_size * expected_pages
