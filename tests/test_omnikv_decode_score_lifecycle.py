from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import pytest

from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.utils.context import get_context, reset_context, set_context


class _Manager:
    device = torch.device("cpu")

    def __init__(self, context_len: int = 6):
        self.context_len = int(context_len)

    def get_layer_batch_states(self, layer_idx):
        del layer_idx
        return SimpleNamespace(
            context_lens=torch.tensor([self.context_len], dtype=torch.int32),
            max_context_len=self.context_len,
            req_indices=torch.tensor([0], dtype=torch.int32),
        )

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        del layer_idx
        return torch.arange(self.context_len, dtype=torch.int32).reshape(1, -1)


def _make_controller():
    layers = 4
    config = SimpleNamespace(
        sparse_method="omnikv",
        obs_layer_ids=[0, 2],
        full_attention_layers=[0, 2],
        runtime_layout=SimpleNamespace(
            is_full_attention=lambda layer_idx: 0 <= int(layer_idx) < layers,
            kv_layer_index=lambda layer_idx: int(layer_idx),
        ),
        hf_config=SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=8,
            num_attention_heads=2,
            head_dim=4,
            dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=1,
        decode_keep_tokens=2,
        sparse_attn_score_dtype="float32",
        decode_graph=True,
    )
    manager = _Manager()
    controller = SparseController(config, manager)
    seqs = [Sequence([1])]
    set_context(
        False,
        cache_manager=manager,
        seqs=seqs,
    )
    controller.prepare_forward(seqs, is_prefill=False)
    return controller


def teardown_function():
    reset_context()


def test_omnikv_observation_layers_share_one_decode_score_workspace():
    controller = _make_controller()
    score0 = controller.layer_batch_sparse_states[0].attn_score
    score2 = controller.layer_batch_sparse_states[2].attn_score

    assert score0 is not None and score2 is not None
    assert score0.untyped_storage().data_ptr() == score2.untyped_storage().data_ptr()
    assert score0.data_ptr() == score2.data_ptr()
    assert controller.layer_batch_sparse_states[1].attn_score is None
    assert controller.layer_batch_sparse_states[3].attn_score is None
    assert controller.runtime._decode_attn_score_buffers == {}
    assert controller.runtime._omnikv_decode_attn_score_buffer is not None


def test_omnikv_consumes_shared_raw_scores_before_the_next_observation_layer():
    controller = _make_controller()
    controller.runtime._update_dynamic_indices = MagicMock()
    states = controller.layer_batch_sparse_states
    raw0 = states[0].attn_score
    raw2 = states[2].attn_score
    assert raw0 is not None and raw2 is not None

    layer0_logits = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, -7.0], [2.0, 1.0, 4.0, 3.0, 6.0, -8.0]]]
    )
    raw0.copy_(layer0_logits)
    controller.on_layer_end(0, SimpleNamespace(is_prefill=False))

    layer0_scores = states[0].attn_score
    expected = torch.full((1, 6), torch.finfo(torch.float32).min)
    expected[:, :5] = torch.softmax(layer0_logits[:, :, :5] * 0.5, dim=-1).amax(dim=1)
    torch.testing.assert_close(layer0_scores, expected)
    assert layer0_scores is not None and layer0_scores.dim() == 2
    assert raw2.dim() == 3

    raw2.fill_(9.0)
    torch.testing.assert_close(states[0].attn_score, expected)
    controller.on_layer_end(2, SimpleNamespace(is_prefill=False))
    assert states[2].attn_score is not None and states[2].attn_score.dim() == 2
    assert controller.runtime._update_dynamic_indices.call_count == 2


def test_omnikv_graph_reset_and_keepalive_cover_the_shared_workspace():
    controller = _make_controller()
    shared = controller.runtime._omnikv_decode_attn_score_buffer
    assert shared is not None
    refs = {
        layer_idx: {"attn_score": state.attn_score}
        for layer_idx, state in controller.layer_batch_sparse_states.items()
    }
    shared.fill_(1.0)

    assert controller.reset_decode_attn_scores_for_graph(refs)
    assert torch.all(shared == -1e20)
    assert any(
        tensor.untyped_storage().data_ptr() == shared.untyped_storage().data_ptr()
        for tensor in controller.decode_graph_keepalive_tensors()
    )

    controller.clear_decode_attn_score_buffers()
    assert controller.runtime._omnikv_decode_attn_score_buffer is None


def test_omnikv_decode_graph_reuses_selection_output_buffers():
    controller = _make_controller()
    states = controller.layer_batch_sparse_states

    def fake_build(*args, **kwargs):
        del args
        keep = kwargs["keep_indices_out"]
        slots = kwargs["active_slots_out"]
        context_lens = kwargs["new_context_lens_out"]
        keep.copy_(torch.arange(keep.shape[1], dtype=torch.int32).expand_as(keep))
        slots.copy_(keep)
        context_lens.fill_(keep.shape[1])
        return keep, slots, context_lens

    pointers = []
    with patch(
        "sparseengine.engine.sparse_methods.dynamic.build_omnikv_keep_and_slots",
        side_effect=fake_build,
    ):
        for _ in range(2):
            states[0].attn_score = torch.arange(6, dtype=torch.float32).reshape(1, 6)
            controller.runtime._update_dynamic_indices(0, [1], get_context())
            pointers.append(
                tuple(
                    int(tensor.data_ptr())
                    for tensor in (
                        states[1].active_indices,
                        states[1].active_slots,
                        states[1].context_lens,
                        states[1].req_indices,
                    )
                )
            )

    assert pointers[0] == pointers[1]
    keepalive_ptrs = {
        int(tensor.data_ptr())
        for tensor in controller.decode_graph_keepalive_tensors()
    }
    assert set(pointers[0]).issubset(keepalive_ptrs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_runtime_scores_feed_real_selection_without_observer_alias(monkeypatch):
    # Exercise runtime preparation -> bound score provider -> actual fused slot
    # selection, not just an isolated kernel or a mocked selection callback.
    original_states = _Manager.get_layer_batch_states
    original_slots = _Manager.get_layer_buffer_req_to_token_slots

    def gpu_states(self, layer):
        state = original_states(self, layer)
        state.context_lens = state.context_lens.cuda()
        state.req_indices = state.req_indices.cuda()
        return state

    monkeypatch.setattr(_Manager, "device", torch.device("cuda", 0))
    monkeypatch.setattr(_Manager, "get_layer_batch_states", gpu_states)
    monkeypatch.setattr(
        _Manager, "get_layer_buffer_req_to_token_slots",
        lambda self, layer: original_slots(self, layer).cuda(),
    )
    controller = _make_controller()
    states = controller.layer_batch_sparse_states
    raw = states[0].attn_score
    raw.copy_(torch.tensor([[[1., 2., 3., 4., 5., -7.], [2., 1., 4., 3., 6., -8.]]], device="cuda"))
    expected = torch.softmax(raw[:, :, :5].double() * .5, dim=-1).amax(1)
    controller.on_layer_end(0, SimpleNamespace(is_prefill=False))
    first = states[0].attn_score.clone()
    torch.testing.assert_close(first[:, :5].double(), expected, rtol=2e-5, atol=1e-8)
    chosen = states[1].active_indices[0, :2].sort().values
    torch.testing.assert_close(chosen, expected.topk(2).indices[0].sort().values.to(torch.int32))
    raw.fill_(9.)
    controller.on_layer_end(2, SimpleNamespace(is_prefill=False))
    torch.testing.assert_close(states[0].attn_score, first, rtol=0, atol=0)
    assert states[0].attn_score.data_ptr() != states[2].attn_score.data_ptr()
