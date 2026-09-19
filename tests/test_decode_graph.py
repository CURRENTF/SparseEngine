from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sparseengine.configs.cuda_graph import (
    build_decode_cuda_graph_startup_plan,
)
from sparseengine.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparseengine.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
    DecodeGraphState,
)
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.kernels.triton.paged_flash_decoding import (
    paged_flash_decode,
)
from sparseengine.kernels.triton.sglang_gemma4_decode_attention import (
    sglang_gemma4_decode,
)
from sparseengine.operators.decode_attention import (
    FixedGridTritonPagedDecodeAttentionProvider,
    DECODE_ATTENTION_REGISTRY,
    DecodeAttentionOpSpec,
    TritonPagedDecodeAttentionProvider,
    build_graph_stable_decode_launch_plan,
)
from sparseengine.operators.registry import OpResolver
from sparseengine.operators.gemma4 import Gemma4OpSpec, TritonGemma4OperatorProvider
from sparseengine.operators.mla_attention import (
    MlaAttentionOpSpec,
    MlaTritonProvider,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


def test_decode_graph_runner_blocks_replay_until_collectives_are_ready() -> None:
    runner = object.__new__(DecodeCudaGraphRunner)
    graph = SimpleNamespace(replay=Mock())
    state = SimpleNamespace(key=object(), graph=graph, logits=None, token_ids=None)
    runner.cache_manager = SimpleNamespace()
    runner.dp_batch_capacity = None
    runner.method = ""
    runner._select_graph_batch_size = lambda batch_size: batch_size
    runner._graph_path_id = lambda: ""
    runner._graph_path_capacity = lambda seqs: 128
    runner._select_state = lambda **kwargs: state
    runner._prepare_static_step = lambda state, seqs: (None, None)
    runner._restore_sparse_state_refs = lambda state: None
    runner.collective_runtime = SimpleNamespace(
        assert_cuda_graph_replayable=Mock(
            side_effect=RuntimeError("collectives are not replayable")
        )
    )

    with pytest.raises(RuntimeError, match="not replayable"):
        runner.run([object()])

    graph.replay.assert_not_called()


def _cuda_caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="test sm90",
        compute_capability=(9, 0),
        runtime_version="12.8",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        multi_processor_count=120,
    )


def test_decode_graph_startup_plan_has_one_graph_per_batch() -> None:
    config = SimpleNamespace(
        decode_graph_capture_sizes=[1, 4],
        decode_graph_startup_capture_limit=8,
        sparse_method="quest",
        max_model_len=32768,
        sink_keep_tokens=64,
        decode_keep_tokens=4096,
        recent_keep_tokens=512,
    )
    assert build_decode_cuda_graph_startup_plan(config) == [
        (4, 32768),
        (1, 32768),
    ]


@pytest.mark.parametrize("legacy_path", ["short", "long"])
def test_decode_graph_contract_rejects_removed_length_paths(legacy_path) -> None:
    with pytest.raises(ValueError, match="Length-specific decode paths"):
        DecodeGraphContract(
            method="quest",
            topology_path_id=legacy_path,
            batch_capacity=4,
            context_capacity=32768,
        )


def test_decode_graph_state_identity_omits_context_capacity() -> None:
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.startup_plan_sealed = False
    runner._graphs = {}
    runner.cache_manager = SimpleNamespace(device=torch.device("cpu"))

    state = runner._select_state(
        method="quest",
        batch_size=4,
        context_capacity=32768,
        capture_sampling=False,
        graph_path_id="unified",
    )
    reused = runner._select_state(
        method="quest",
        batch_size=4,
        context_capacity=8192,
        capture_sampling=False,
        graph_path_id="unified",
    )
    assert reused is state
    assert state.capture_context_capacity == 32768
    assert state.decode_state is not None
    assert state.decode_state.contract == DecodeGraphContract(
        method="quest",
        topology_path_id="unified",
        batch_capacity=4,
        context_capacity=32768,
    )
    assert state.decode_state.inputs.batch_capacity == 4
    assert state.decode_state.contract.capability_level == "strict"

    reused_with_default_path = runner._select_state(
        method="quest",
        batch_size=4,
        context_capacity=16384,
        capture_sampling=False,
    )
    assert reused_with_default_path is state

    with pytest.raises(RuntimeError, match="exceeded captured topology-path capacity"):
        runner._select_state(
            method="quest",
            batch_size=4,
            context_capacity=65536,
            capture_sampling=False,
            graph_path_id="unified",
        )


def test_typed_decode_graph_participant_delegates_to_cache_owner() -> None:
    calls = []
    private_keepalive = torch.empty(1)
    operator_keepalive = torch.empty(1)

    class CacheOwner:
        num_free_slots = 16

        def init_decode_graph_state(self, contract, inputs):
            calls.append(("init", contract.topology_path_id))
            return SimpleNamespace(contract=contract, inputs=inputs)

        def prepare_decode_graph_step(self, seqs, state):
            calls.append(("prepare_out", len(seqs)))
            state.inputs.input_ids.fill_(7)

        def prepare_decode_graph_in(self, state):
            calls.append(("prepare_in", state.contract.topology_path_id))

        def decode_graph_state_keepalive_tensors(self, state):
            calls.append(("keepalive", state.contract.topology_path_id))
            return [private_keepalive]

    class OperatorOwner:
        def init_decode_graph_state(self, contract, inputs):
            calls.append(("operator_init", contract.batch_capacity))
            return SimpleNamespace(contract=contract, inputs=inputs)

        def prepare_decode_graph_out(self, state):
            calls.append(("operator_out", state.contract.context_capacity))

        def prepare_decode_graph_in(self, state):
            calls.append(("operator_in", state.contract.topology_path_id))

        def decode_graph_keepalive_tensors(self, state):
            calls.append(("operator_keepalive", state.contract.topology_path_id))
            return [operator_keepalive]

        def close_decode_graph_state(self, state):
            calls.append(("operator_close", state.contract.batch_capacity))

    contract = DecodeGraphContract(
        method="",
        topology_path_id="dense",
        batch_capacity=2,
        context_capacity=32,
    )
    graph_state = DecodeGraphState(
        contract=contract,
        inputs=DecodeGraphInputs.allocate(
            contract,
            device=torch.device("cpu"),
            pin_memory=False,
        ),
    )
    runtime = RuntimeState(
        SimpleNamespace(),
        CacheOwner(),
        decode_graph_participants=(OperatorOwner(),),
    )

    participant = runtime.init_decode_graph_state(graph_state)
    runtime.prepare_decode_graph_step([object()], graph_state)
    participant.prepare_in_graph()
    keepalive = graph_state.keepalive_tensors()

    assert graph_state.runtime_state is participant
    assert graph_state.inputs.input_ids.tolist() == [7, 7]
    assert any(tensor is private_keepalive for tensor in keepalive)
    assert any(tensor is operator_keepalive for tensor in keepalive)
    assert calls == [
        ("init", "dense"),
        ("operator_init", 2),
        ("prepare_out", 1),
        ("operator_out", 32),
        ("prepare_in", "dense"),
        ("operator_in", "dense"),
        ("keepalive", "dense"),
        ("operator_keepalive", "dense"),
    ]
    graph_state.close()
    assert calls[-1] == ("operator_close", 2)


def test_mha_resolver_prefers_sgl_fa3_for_decode_graph_on_supported_sm90() -> None:
    spec = DecodeAttentionOpSpec(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        activation_dtype=torch.bfloat16,
        softmax_scale=128**-0.5,
        max_batch_size=8,
        context_capacity=32768,
    )
    caps = _cuda_caps()
    assert FixedGridTritonPagedDecodeAttentionProvider.supports(spec, caps).supported
    assert not TritonPagedDecodeAttentionProvider.supports(spec, caps).supported

    h2o_spec = DecodeAttentionOpSpec(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        activation_dtype=torch.bfloat16,
        softmax_scale=128**-0.5,
        max_batch_size=8,
        may_require_attention_scores=True,
        h2o_layerwise_probability_scores=True,
        context_capacity=32768,
    )
    assert FixedGridTritonPagedDecodeAttentionProvider.supports(
        h2o_spec, caps
    ).supported

    unsupported = DecodeAttentionOpSpec(
        **{
            **spec.__dict__,
            "activation_dtype": torch.float32,
        }
    )
    assert not FixedGridTritonPagedDecodeAttentionProvider.supports(
        unsupported,
        caps,
    ).supported

    plan = build_graph_stable_decode_launch_plan(spec, caps)
    assert plan.context_capacity == spec.context_capacity
    assert plan.max_kv_splits > 0
    assert plan.target_tokens_per_split > 0
    assert plan.block_n > 0
    from unittest.mock import patch

    with patch(
        "sparseengine.operators.decode_attention.sgl_fa3_device_support",
        return_value=(True, "available"),
    ):
        resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(spec, caps)
    assert resolved.provider.name == "sgl_fa3_paged_decode_sm90"
    assert resolved.report.selection_basis == "upstream_default"


def test_mha_resolver_falls_back_to_fixed_grid_when_upstream_is_ineligible() -> None:
    spec = DecodeAttentionOpSpec(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        activation_dtype=torch.bfloat16,
        softmax_scale=128**-0.5,
        max_batch_size=8,
        context_capacity=32768,
    )
    caps = DeviceCaps(
        **{
            **_cuda_caps().__dict__,
            "compute_capability": (8, 0),
        }
    )
    from unittest.mock import patch

    with patch(
        "sparseengine.operators.decode_attention.flashinfer_paged_decode_support",
        return_value=(False, "unavailable"),
    ):
        resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(spec, caps)
    assert isinstance(
        resolved.provider,
        FixedGridTritonPagedDecodeAttentionProvider,
    )
    metadata = resolved.report.as_dict()["provider_metadata"]
    assert metadata["launch_plan"]["plan_id"] == "portable_fixed_grid_v1"


def test_mla_resolver_contract_selects_fixed_launch_provider() -> None:
    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=2,
        cuda_graph=True,
        context_capacity=32768,
    )
    caps = _cuda_caps()
    assert MlaTritonProvider.supports(spec, caps).supported


def test_gemma4_resolver_contract_selects_fixed_grid_provider() -> None:
    spec = Gemma4OpSpec(
        activation_dtype=torch.bfloat16,
        head_dims=(256, 512),
        cuda_graph=True,
        attention_contracts=((8, 2, 256, 1023), (8, 1, 512, -1)),
        max_batch_size=8,
        context_capacity=32768,
    )
    caps = _cuda_caps()
    assert TritonGemma4OperatorProvider.supports(spec, caps).supported
    provider = TritonGemma4OperatorProvider.bind(spec, caps)
    assert provider.name == "triton"
    assert provider.binding_metadata()["attention_dispatch"]["decode_routes"] == [
        "sglang_fixed_grid"
    ]


@pytest.mark.parametrize("multi_processor_count", [None, 0, -1])
def test_gemma4_provider_rejects_missing_multi_processor_count(
    multi_processor_count,
) -> None:
    spec = Gemma4OpSpec(
        activation_dtype=torch.bfloat16,
        head_dims=(256,),
        cuda_graph=True,
        attention_contracts=((8, 2, 256, -1),),
        max_batch_size=8,
        context_capacity=32768,
    )
    caps = DeviceCaps(
        **{
            **_cuda_caps().__dict__,
            "multi_processor_count": multi_processor_count,
        }
    )
    result = TritonGemma4OperatorProvider.supports(spec, caps)
    assert not result.supported
    assert "multi-processor count" in result.reason
    with pytest.raises(ValueError, match="multi-processor count"):
        TritonGemma4OperatorProvider.bind(spec, caps)


def _decode_reference(
    q, k, v, slots, req_indices, lengths, window=None, *, scale=True
):
    output = torch.empty_like(q)
    lse = torch.empty(
        q.shape[1], q.shape[0], dtype=torch.float32, device=q.device
    )
    group_size = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths.tolist()):
        start = max(0, length - int(window or length))
        indices = slots[req_indices[batch], start:length].long()
        keys = k[indices].repeat_interleave(group_size, dim=1)
        values = v[indices].repeat_interleave(group_size, dim=1)
        logits = torch.einsum("hd,lhd->hl", q[batch].float(), keys.float())
        if scale:
            logits = logits / q.shape[-1] ** 0.5
        probabilities = logits.softmax(-1)
        output[batch] = torch.einsum(
            "hl,lhd->hd", probabilities, values.float()
        ).to(q.dtype)
        lse[:, batch] = logits.logsumexp(-1)
    return output, lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA GPU")
@pytest.mark.parametrize(
    ("dtype", "heads", "kv_heads", "head_dim"),
    [
        (torch.bfloat16, 4, 4, 64),
        (torch.float16, 8, 2, 64),
        (torch.bfloat16, 8, 2, 128),
        (torch.float16, 4, 4, 128),
        (torch.bfloat16, 8, 2, 256),
        (torch.float16, 4, 4, 256),
    ],
)
def test_decode_graph_mha_matches_reference_and_replays_new_lengths(
    dtype,
    heads,
    kv_heads,
    head_dim,
) -> None:
    torch.manual_seed(11)
    batch, capacity = 2, 257
    device = torch.device("cuda")
    q = torch.randn(batch, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(
        batch * capacity, kv_heads, head_dim, dtype=dtype, device=device
    )
    v = torch.randn_like(k)
    slots = torch.arange(
        batch * capacity, dtype=torch.int32, device=device
    ).view(batch, capacity)
    req_indices = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.tensor([129, 257], dtype=torch.int32, device=device)
    mid_o = torch.empty(
        batch, heads, 8, head_dim, dtype=torch.float32, device=device
    )
    mid_lse = torch.empty(batch, heads, 8, dtype=torch.float32, device=device)
    output_lse = torch.empty(heads, batch, dtype=torch.float32, device=device)

    def run():
        return paged_flash_decode(
            q,
            k,
            v,
            slots,
            req_indices,
            lengths,
            mid_o,
            mid_lse,
            target_tokens_per_split=64,
            return_softmax_lse=True,
            output_lse=output_lse,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_lse = run()
    lengths.copy_(torch.tensor([17, 201], dtype=torch.int32, device=device))
    q.copy_(torch.randn_like(q))
    graph.replay()
    expected_output, expected_lse = _decode_reference(
        q, k, v, slots, req_indices, lengths
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(graph_lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA GPU")
def test_decode_graph_gqa_replays_exact_context_capacity() -> None:
    torch.manual_seed(29)
    batch, heads, kv_heads, head_dim, capacity = 1, 8, 2, 128, 8352
    device = torch.device("cuda")
    q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        capacity,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    slots = torch.arange(capacity, dtype=torch.int32, device=device).view(
        batch,
        capacity,
    )
    req_indices = torch.zeros(batch, dtype=torch.int32, device=device)
    lengths = torch.tensor([4097], dtype=torch.int32, device=device)
    mid_o = torch.empty(
        batch,
        heads,
        16,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    mid_lse = torch.empty(batch, heads, 16, dtype=torch.float32, device=device)
    output_lse = torch.empty(heads, batch, dtype=torch.float32, device=device)

    def run():
        return paged_flash_decode(
            q,
            k,
            v,
            slots,
            req_indices,
            lengths,
            mid_o,
            mid_lse,
            target_tokens_per_split=256,
            return_softmax_lse=True,
            output_lse=output_lse,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_lse = run()
    lengths.fill_(capacity)
    q.copy_(torch.randn_like(q))
    graph.replay()
    expected_output, expected_lse = _decode_reference(
        q,
        k,
        v,
        slots,
        req_indices,
        lengths,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(graph_lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA GPU")
def test_decode_graph_gqa_produces_raw_per_head_scores() -> None:
    torch.manual_seed(23)
    batch, heads, kv_heads, head_dim, capacity = 2, 4, 2, 64, 33
    device = torch.device("cuda")
    q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        batch * capacity,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    slots = torch.arange(
        batch * capacity,
        dtype=torch.int32,
        device=device,
    ).view(batch, capacity)
    req_indices = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.tensor([17, 29], dtype=torch.int32, device=device)
    mid_o = torch.empty(
        batch,
        heads,
        8,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    mid_lse = torch.empty(batch, heads, 8, dtype=torch.float32, device=device)
    scores = torch.full(
        (batch, heads, capacity),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )

    output = paged_flash_decode(
        q,
        k,
        v,
        slots,
        req_indices,
        lengths,
        mid_o,
        mid_lse,
        attn_score=scores,
        target_tokens_per_split=8,
    )
    expected, _ = _decode_reference(q, k, v, slots, req_indices, lengths)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    group_size = heads // kv_heads
    for batch_idx, length in enumerate(lengths.tolist()):
        keys = k[slots[batch_idx, :length].long()].repeat_interleave(
            group_size,
            dim=1,
        )
        expected_scores = torch.einsum(
            "hd,lhd->hl",
            q[batch_idx].float(),
            keys.float(),
        )
        torch.testing.assert_close(
            scores[batch_idx, :, :length],
            expected_scores,
            rtol=2e-2,
            atol=2e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA GPU")
@pytest.mark.parametrize("window", [None, 8])
def test_gemma4_fixed_grid_matches_reference_and_graph(window) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    batch, heads, kv_heads, head_dim, capacity = 2, 4, 2, 256, 33
    q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        batch * capacity, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn_like(k)
    slots = torch.arange(
        batch * capacity, dtype=torch.int32, device=device
    ).view(batch, capacity)
    req_indices = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.tensor([33, 21], dtype=torch.int32, device=device)
    mid_o = torch.empty(
        batch, heads, 8, head_dim, dtype=torch.float32, device=device
    )
    mid_lse = torch.empty(batch, heads, 8, dtype=torch.float32, device=device)
    splits = torch.empty(batch, dtype=torch.int32, device=device)

    def run():
        return sglang_gemma4_decode(
            q,
            k,
            v,
            slots,
            req_indices,
            lengths,
            mid_o,
            mid_lse,
            splits,
            sliding_window=window,
            multi_processor_count=120,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    lengths.copy_(torch.tensor([17, 29], dtype=torch.int32, device=device))
    q.copy_(torch.randn_like(q))
    graph.replay()
    expected, _ = _decode_reference(
        q, k, v, slots, req_indices, lengths, window, scale=False
    )
    torch.cuda.synchronize()
    cosine = torch.nn.functional.cosine_similarity(
        graph_output.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine > 0.999
