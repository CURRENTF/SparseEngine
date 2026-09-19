"""Independent selection, phase ownership, and fused prefill score contracts."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.config import Config
from sparseengine.engine.cache_manager.base import AttentionViewMeta, ExplicitKVPayload, PrefillComputeView
from sparseengine.engine.sparse_methods import create_sparse_method_runtime
from sparseengine.engine.sparse_methods.base import (
    DecodeSelectionRequest, LayerEndEvent, PrefillSelectionRequest, SparseStepContext,
)
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.prefill_attention import (
    PrefillAttentionOpSpec, TritonPagedPrefillAttentionProvider,
    prepare_prefill_attention_op,
)
from sparseengine.utils.context import get_context


def _config(method=""):
    return SimpleNamespace(
        sparse_method=method, prefill_sparse_method="omnikv_prefill",
        obs_layer_ids=[0, 2], full_attention_layers=[0, 2], runtime_layout=None,
        hf_config=SimpleNamespace(num_hidden_layers=4, num_attention_heads=2,
                                  head_dim=16, dtype=torch.bfloat16),
        tensor_parallel_size=1, sink_keep_tokens=0, recent_keep_tokens=1,
        decode_keep_tokens=2, sparse_attn_score_dtype="float32",
        omnikv_prefill_full_attention_layers=[0, 2], omnikv_prefill_keep_tokens=2,
        omnikv_prefill_sink_keep_tokens=2, omnikv_prefill_recent_keep_tokens=1,
    )


def _manager(device, lengths):
    width = max(lengths)
    table = torch.arange(len(lengths) * width, device=device, dtype=torch.int32).reshape(len(lengths), width).flip(1)
    rows = torch.arange(len(lengths), device=device, dtype=torch.int32).flip(0)
    state = SimpleNamespace(context_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
                            max_context_len=width, req_indices=rows)
    return SimpleNamespace(device=torch.device(device), get_layer_batch_states=lambda _: state,
                           get_layer_buffer_req_to_token_slots=lambda _: table), table, rows


def _build_slots_reference(topk, counts, hist, tail, table, rows, sink, max_s, *, context_lens):
    # Stand-in for GPU gathering only; runtime scoring and selection are real.
    keep = torch.zeros((len(rows), max_s), dtype=torch.int32)
    lengths = torch.minimum(sink + counts + tail, context_lens)
    for b in range(len(rows)):
        values = list(range(min(sink, int(context_lens[b])))) + topk[b, :counts[b]].tolist() + list(range(int(hist[b]), int(hist[b] + tail[b])))
        keep[b, :min(len(values), max_s)] = torch.tensor(values[:max_s], dtype=torch.int32)
    return keep, table[rows.long()[:, None], keep.long()], lengths


@pytest.mark.parametrize("method", ["", "omnikv"])
def test_prefill_selection_preserves_chunk_and_decode_ownership(method):
    config = _config(method)
    manager, table, rows = _manager("cpu", [10, 4, 1])
    runtime = create_sparse_method_runtime(config, manager)
    context = SimpleNamespace(is_prefill=True, cu_seqlens_q=torch.tensor([0, 3, 5, 6]))
    step = SparseStepContext([None] * 3, True, context)
    original_table = table.clone()
    with patch("sparseengine.engine.sparse_methods.omnikv_prefill.build_omnikv_keep_and_slots", side_effect=_build_slots_reference):
        # Repeated chunks reset both the selection and the shared atomic scores.
        for _ in range(2):
            runtime.prepare_step(step)
            assert runtime.layer_batch_sparse_states[1].active_slots is None
            for anchor, target, preferred in [(0, 1, [5, 3]), (2, 3, [4, 2])]:
                state = runtime.layer_batch_sparse_states[anchor]
                assert torch.count_nonzero(state.attn_score) == 0
                state.attn_score.fill_(-1_000_000_000_000)
                for pos in preferred:
                    state.attn_score[0, 0, pos] = -pos
                runtime.on_layer_end(LayerEndEvent(anchor, context, context))
                selection = runtime.build_prefill_selection(PrefillSelectionRequest(target, context))
                chosen = selection.active_indices[0, :selection.context_lens[0]].tolist()
                assert set(chosen) == {0, 1, *preferred, 6, 7, 8, 9}
                assert chosen[-3:] == [7, 8, 9]
                for b in (1, 2):
                    n = int(manager.get_layer_batch_states(0).context_lens[b])
                    assert selection.active_indices[b, :selection.context_lens[b]].tolist() == list(range(n))
                torch.testing.assert_close(selection.active_slots, table[rows.long()[:, None], selection.active_indices.long()])
            runtime.finish_step(step)
            assert runtime.prefill._score_buffer is None
            assert runtime.layer_batch_sparse_states is runtime.decode.layer_batch_sparse_states
    torch.testing.assert_close(table, original_table)
    context.is_prefill = False
    runtime.prepare_step(SparseStepContext([None] * 3, False, context))
    assert runtime.active is runtime.decode
    selection = runtime.build_decode_selection(DecodeSelectionRequest(1, torch.empty(0), context))
    assert selection.active_slots is None
    torch.testing.assert_close(selection.context_lens, torch.tensor([10, 4, 1], dtype=torch.int32))
    sentinel = torch.empty(1)
    runtime.decode.decode_graph_keepalive_tensors = Mock(return_value=[sentinel])
    assert runtime.decode_graph_keepalive_tensors()[0] is sentinel


def _runtime_config(tmp_path, **overrides):
    hf = SimpleNamespace(model_type="qwen2", torch_dtype=torch.bfloat16,
                         max_position_embeddings=32768, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2)
    params = dict(model=str(tmp_path), prefill_sparse_method="omnikv_prefill",
                  omnikv_prefill_full_attention_layers=[0, 2])
    params.update(overrides)
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=hf):
        return Config(**params)


def test_zero_history_budget_preserves_only_causal_current_chunk():
    config = _config()
    config.omnikv_prefill_keep_tokens = 0
    config.omnikv_prefill_sink_keep_tokens = 0
    config.omnikv_prefill_recent_keep_tokens = 0
    manager, _, _ = _manager("cpu", [5])
    runtime = create_sparse_method_runtime(config, manager)
    context = SimpleNamespace(is_prefill=True, cu_seqlens_q=torch.tensor([0, 2]))
    runtime.prepare_step(SparseStepContext([None], True, context))
    with patch("sparseengine.engine.sparse_methods.omnikv_prefill.build_omnikv_keep_and_slots", side_effect=_build_slots_reference):
        runtime.on_layer_end(LayerEndEvent(0, context, context))
    selection = runtime.build_prefill_selection(PrefillSelectionRequest(1, context))
    assert selection.active_indices.tolist() == [[3, 4]]
    assert selection.context_lens.tolist() == [2]


def test_prefill_selection_reduces_all_tp_heads_before_topk(monkeypatch):
    from sparseengine.distributed import ParallelGroup

    config = _config()
    config.tensor_parallel_size = 2
    manager, _, _ = _manager("cpu", [10])
    remote_scores = torch.zeros(1, 10)
    remote_scores[0, 4] = 100

    process_group = object()

    def reduce(scores, op, group):
        assert group is process_group
        assert op == torch.distributed.ReduceOp.MAX
        scores.copy_(torch.maximum(scores, remote_scores))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    manager.parallel_context = SimpleNamespace(
        attn_tp=ParallelGroup(process_group, (2, 3), 0, 2),
    )
    runtime = create_sparse_method_runtime(config, manager)
    context = SimpleNamespace(is_prefill=True, cu_seqlens_q=torch.tensor([0, 3]))
    runtime.prepare_step(SparseStepContext([None], True, context))
    runtime.layer_batch_sparse_states[0].attn_score[0, 0, 3] = 30
    with patch("sparseengine.engine.sparse_methods.omnikv_prefill.build_omnikv_keep_and_slots", side_effect=_build_slots_reference):
        runtime.on_layer_end(LayerEndEvent(0, context, context))
    selection = runtime.build_prefill_selection(PrefillSelectionRequest(1, context))
    assert set(selection.active_indices[0, 2:4].tolist()) == {3, 4}


def test_global_selection_crosses_sliding_layers_and_reaches_shared_kv_consumers():
    # A shared global layer reads its source's KV but has its own query and
    # selection. Iterating only physical owners previously dropped this layer.
    from sparseengine.models.layout import RuntimeLayout
    from transformers import Gemma4TextConfig

    config = _config()
    config.hf_config = Gemma4TextConfig(
        num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
        hidden_size=32, head_dim=16, global_head_dim=32,
        num_kv_shared_layers=2, dtype=torch.bfloat16,
        layer_types=["sliding_attention", "full_attention"] * 3,
    )
    config.runtime_layout = RuntimeLayout.from_config(config.hf_config)
    config.omnikv_prefill_full_attention_layers = [1]
    manager, table, _ = _manager("cpu", [10])
    runtime = create_sparse_method_runtime(config, manager)
    context = SimpleNamespace(is_prefill=True, cu_seqlens_q=torch.tensor([0, 3]))
    runtime.prepare_step(SparseStepContext([None], True, context))
    state = runtime.layer_batch_sparse_states[1]
    state.attn_score[0, 0, 3:5] = 100
    with patch("sparseengine.engine.sparse_methods.omnikv_prefill.build_omnikv_keep_and_slots", side_effect=_build_slots_reference):
        runtime.on_layer_end(LayerEndEvent(1, context, context))
    for layer in (0, 2, 4):
        runtime.on_layer_end(LayerEndEvent(layer, context, context))
        selection = runtime.build_prefill_selection(PrefillSelectionRequest(layer, context))
        assert selection.active_slots is None
        assert selection.attn_score is None
        assert selection.context_lens.tolist() == [10]
    for layer in (3, 5):
        selection = runtime.build_prefill_selection(PrefillSelectionRequest(layer, context))
        assert set(selection.active_indices[0].tolist()) == {0, 1, 3, 4, 6, 7, 8, 9}
        torch.testing.assert_close(selection.active_slots, table[:, selection.active_indices[0].long()])


def test_mla_max_workspace_is_consumed_then_reset_between_observers():
    # Negative raw maxima must not be overwritten by a zero-initialized atomic
    # buffer or treated as per-head query sums.
    config = _config()
    config.attention_cache_layout = "mla_latent"
    manager, _, _ = _manager("cpu", [10])
    runtime = create_sparse_method_runtime(config, manager)
    context = SimpleNamespace(is_prefill=True, cu_seqlens_q=torch.tensor([0, 3]))
    runtime.prepare_step(SparseStepContext([None], True, context))
    with patch("sparseengine.engine.sparse_methods.omnikv_prefill.build_omnikv_keep_and_slots", side_effect=_build_slots_reference):
        for anchor, target, preferred in [(0, 1, [3, 4]), (2, 3, [2, 5])]:
            score = runtime.layer_batch_sparse_states[anchor].attn_score
            assert score.shape == (1, 10)
            assert torch.isneginf(score).all()
            score[0, preferred] = torch.tensor([-2., -3.])
            runtime.on_layer_end(LayerEndEvent(anchor, context, context))
            selected = runtime.build_prefill_selection(PrefillSelectionRequest(target, context))
            assert set(selected.active_indices[0, 2:4].tolist()) == set(preferred)
            assert torch.isneginf(score).all()


@pytest.mark.parametrize("override,reason", [
    ({"omnikv_prefill_full_attention_layers": [2]}, "first KV layer"),
    ({"omnikv_prefill_full_attention_layers": [0, 4]}, "KV layer indices"),
    ({"omnikv_prefill_keep_tokens": -1}, "non-negative integer"),
    ({"enable_prefix_caching": True, "prefix_cache_mode": "radix"}, "chain"),
    ({"enable_prefix_caching": True, "enable_prefix_cache_offload": True,
      "prefix_cache_host_size_gb": 0.25}, "shared slot"),
    ({"sparse_method": "h2o"}, "incompatible"),
])
def test_prefill_rejects_invalid_state_and_storage_contracts(tmp_path, override, reason):
    with pytest.raises((ValueError, NotImplementedError), match=reason):
        _runtime_config(tmp_path, **override)


def test_prefill_only_config_does_not_select_decode_cache_owner(tmp_path):
    config = _runtime_config(tmp_path, omnikv_prefill_full_attention_layers="0,2")
    assert config.resolved_cache_sparse_method == ""
    assert config.omnikv_prefill_full_attention_layers == [0, 2]


def _spec(dtype=torch.bfloat16, **kwargs):
    return PrefillAttentionOpSpec(num_query_heads=2, num_kv_heads=1, head_dim=16,
                                  activation_dtype=dtype, softmax_scale=16**-0.5,
                                  score_output=AttentionScoreKind.RAW_QK_PER_HEAD,
                                  optional_score_output=True, layer_varying_page_table=True,
                                  prefill_sparse_method="omnikv_prefill", **kwargs)


def test_prepared_score_dispatch_never_reselects_on_failure():
    scored, plain = Mock(), Mock()
    with patch("sparseengine.operators.prefill_attention._resolve_prefill_attention_provider",
               side_effect=lambda spec, **_: (scored if spec.optional_score_output else plain, spec)) as resolve:
        op = prepare_prefill_attention_op(_spec())
    assert resolve.call_count == 2
    view = SimpleNamespace(attn_score=None)
    op.run(None, view)
    plain.run.assert_called_once()
    view.attn_score = torch.empty(1, 2, 3)
    scored.run.side_effect = RuntimeError("kernel failure")
    with pytest.raises(RuntimeError, match="kernel failure"):
        op.run(None, view)
    assert plain.run.call_count == 1
    op.close()
    op.close()
    scored.close.assert_called_once()
    plain.close.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_context_scores_and_sparse_prefill_match_independent_causal_reference(dtype):
    # Ragged chunks cross a query tile and include context shorter than sink.
    torch.manual_seed(57)
    config = _config()
    manager, table, rows = _manager("cuda", [277, 7, 1])
    runtime = create_sparse_method_runtime(config, manager)
    chunks = torch.tensor([131, 3, 1], device="cuda", dtype=torch.int32)
    cu = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), chunks.cumsum(0, dtype=torch.int32)))
    context = SimpleNamespace(is_prefill=True, cu_seqlens_q=cu)
    runtime.prepare_step(SparseStepContext([None] * 3, True, context))
    q = torch.randn(135, 2, 16, dtype=dtype, device="cuda")
    k, v = [torch.randn(table.numel(), 1, 16, dtype=dtype, device="cuda") for _ in range(2)]
    provider = TritonPagedPrefillAttentionProvider()
    get_context().attention_validation_scope = object()
    for layer in range(4):
        selection = runtime.build_prefill_selection(PrefillSelectionRequest(layer, context))
        slots = table if selection.active_slots is None else selection.active_slots
        view = PrefillComputeView(payload=ExplicitKVPayload(k_cache=k, v_cache=v), meta=AttentionViewMeta(
            active_slots=slots, req_indices=selection.req_indices, context_lens=selection.context_lens,
            attn_score=selection.attn_score))
        out = provider.run(_spec(dtype=dtype), q, view, qo_indptr=cu, chunk_lens=chunks, max_context_len=slots.shape[1], layer_idx=layer)
        for b in range(3):
            count = int(selection.context_lens[b])
            selected = slots[selection.req_indices[b], :count].long()
            qb = q[cu[b]:cu[b + 1]].float().transpose(0, 1)
            kb = k[selected, 0].float()
            vb = v[selected, 0].float()
            raw = qb @ kb.T
            mask = torch.arange(count, device="cuda")[None, :] <= torch.arange(int(chunks[b]), device="cuda")[:, None] + count - chunks[b]
            logits = (raw * 0.25).masked_fill(~mask, -torch.inf)
            expected = (logits.softmax(-1) @ vb).transpose(0, 1)
            torch.testing.assert_close(out[cu[b]:cu[b + 1]].float(), expected, rtol=0.025, atol=0.025)
            if selection.attn_score is not None:
                expected_score = raw.masked_fill(~mask, 0).sum(dim=1)
                torch.testing.assert_close(selection.attn_score[b, :, :count], expected_score, rtol=0.003, atol=0.003)
        runtime.on_layer_end(LayerEndEvent(layer, context, context))
