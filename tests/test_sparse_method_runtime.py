from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.sparse_methods import (
    PrefillSelectionRequest,
    SparseMethodRuntime,
    SparseStepContext,
    create_sparse_method_runtime,
)
from sparseengine.engine.sparse_methods.dynamic import DeltaKVRuntime
from sparseengine.engine.sparse_methods.h2o import H2ORuntime
from sparseengine.engine.sparse_methods.passthrough import PassThroughRuntime
from sparseengine.engine.sparse_methods.snapkv import PyramidKVRuntime
from sparseengine.engine.sparse_methods.streamingllm import StreamingLLMRuntime


def _config(method: str):
    return SimpleNamespace(
        sparse_method=method,
        prefill_sparse_method=None,
        obs_layer_ids=[],
        full_attention_layers=[],
        runtime_layout=None,
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            head_dim=4,
            dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=1,
        sparse_attn_score_dtype="float32",
    )


@pytest.mark.parametrize(
    ("method", "runtime_type", "canonical"),
    [
        ("vanilla", PassThroughRuntime, ""),
        ("attention-sink", StreamingLLMRuntime, "streamingllm"),
        ("pyramidkv", PyramidKVRuntime, "pyramidkv"),
        ("deltakv-less-memory", DeltaKVRuntime, "deltakv"),
    ],
)
def test_sparse_runtime_factory_normalizes_before_binding(
    method,
    runtime_type,
    canonical,
):
    runtime = create_sparse_method_runtime(
        _config(method),
        SimpleNamespace(device=torch.device("cpu")),
    )

    assert isinstance(runtime, SparseMethodRuntime)
    assert isinstance(runtime, runtime_type)
    assert runtime.sparse_method == canonical


def test_sparse_runtime_factory_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported sparse_method"):
        create_sparse_method_runtime(
            _config("unknown-method"),
            SimpleNamespace(device=torch.device("cpu")),
        )


def test_sparse_runtime_factory_binds_h2o_for_prefill_only():
    config = _config("")
    config.prefill_sparse_method = "h2o_prefill"

    runtime = create_sparse_method_runtime(
        config,
        SimpleNamespace(device=torch.device("cpu")),
    )

    assert isinstance(runtime, H2ORuntime)
    assert runtime.sparse_method == ""


def test_passthrough_runtime_preserves_cache_manager_batch_tensor_identity():
    context_lens = torch.tensor([3], dtype=torch.int32)
    req_indices = torch.tensor([0], dtype=torch.int32)
    manager = SimpleNamespace(
        device=torch.device("cpu"),
        get_layer_batch_states=lambda _layer_idx: SimpleNamespace(
            context_lens=context_lens,
            max_context_len=3,
            req_indices=req_indices,
        ),
    )
    runtime = create_sparse_method_runtime(_config(""), manager)
    forward_context = SimpleNamespace(is_prefill=True)

    runtime.prepare_step(
        SparseStepContext(
            seqs=[],
            is_prefill=True,
            forward_context=forward_context,
        )
    )
    selection = runtime.build_prefill_selection(
        PrefillSelectionRequest(
            layer_idx=0,
            forward_context=forward_context,
        )
    )

    assert selection.kind == "full"
    assert selection.context_lens is context_lens
    assert selection.req_indices is req_indices


def test_controller_forwards_prefill_selection_query_without_interpreting_it(monkeypatch):
    from sparseengine.engine import sparse_controller as facade
    controller = object.__new__(facade.SparseController)
    context, query, selection = object(), object(), object()
    received = []
    def build(request):
        received.append(request)
        return selection
    controller.runtime = SimpleNamespace(build_prefill_selection=build)
    monkeypatch.setattr(facade, "get_context", lambda: context)
    assert controller.get_prefill_selection(3, selection_query=query) is selection
    assert received[0].layer_idx == 3
    assert received[0].forward_context is context
    assert received[0].selection_query is query
