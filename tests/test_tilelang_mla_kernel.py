from __future__ import annotations

import pytest
import torch

from sparseengine.kernels.tilelang.mla.runtime import (
    TileMlaDecodeKernel,
    TileMlaLaunchConfig,
    TileMlaLaunchPlan,
)

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the TileLang MLA kernel test",
)


def _torch_oracle(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_keys = latent_cache[slots.long(), 0].float()
    rope_keys = rope_cache[slots.long(), 0].float()
    raw = torch.matmul(q_latent.float(), latent_keys.T) + torch.matmul(
        q_rope.float(), rope_keys.T
    )
    score = raw.max(dim=0).values
    probability = torch.softmax(raw * (256**-0.5), dim=-1)
    output = torch.matmul(probability, latent_keys)
    return output.to(torch.bfloat16), raw, score


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("valid_heads", "num_split", "block_h", "score_mode"),
    [
        (5, 1, 16, "direct"),
        (5, 4, 16, "direct"),
        (10, 1, 16, "direct"),
        (10, 32, 16, "direct"),
        (20, 1, 16, "atomic"),
        (20, 4, 16, "partial"),
        (20, 4, 32, "direct"),
        (5, 4, 16, "per_head"),
        (10, 16, 16, "per_head"),
        (20, 4, 32, "per_head"),
    ],
)
def test_tilelang_mla_score_matches_torch_with_indirect_slots_and_graph(
    valid_heads: int,
    num_split: int,
    block_h: int,
    score_mode: str,
) -> None:
    torch.manual_seed(20260808 + valid_heads + num_split)
    device = torch.device("cuda")
    batch_size = 2
    capacity = 64
    cache_slots = 96
    q_latent = torch.randn(
        valid_heads, batch_size, 512, dtype=torch.bfloat16, device=device
    ).transpose(0, 1)
    q_rope = torch.randn(
        valid_heads, batch_size, 64, dtype=torch.bfloat16, device=device
    ).transpose(0, 1)
    latent_cache = torch.randn(
        cache_slots, 1, 512, dtype=torch.bfloat16, device=device
    )
    rope_cache = torch.randn(
        cache_slots, 1, 64, dtype=torch.bfloat16, device=device
    )
    active_slots = torch.full(
        (3, capacity), -1, dtype=torch.int32, device=device
    )
    active_slots[0, :17] = torch.arange(
        50, 67, dtype=torch.int32, device=device
    )
    active_slots[2, :33] = torch.randperm(
        cache_slots, dtype=torch.int64, device=device
    )[:33].to(torch.int32)
    request_indices = torch.tensor([2, -1], dtype=torch.int32, device=device)
    context_lens = torch.tensor([33, 0], dtype=torch.int32, device=device)
    output = torch.empty(
        q_latent.shape, dtype=q_latent.dtype, device=q_latent.device
    )
    score = torch.full(
        (
            (batch_size, valid_heads, capacity)
            if score_mode == "per_head"
            else (batch_size, capacity)
        ),
        -1e20,
        dtype=torch.float32,
        device=device,
    )
    runner = TileMlaDecodeKernel(
        device=device,
        softmax_scale=256**-0.5,
        valid_heads=valid_heads,
        fixed_config=TileMlaLaunchConfig(
            num_split,
            block_h=block_h,
            score_mode=score_mode,
        ),
    )
    assert not q_latent.is_contiguous()
    assert not q_rope.is_contiguous()
    assert output.is_contiguous()

    def run() -> None:
        score.fill_(-1e20)
        runner(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            attn_score=score,
            max_context_len=capacity,
        )

    run()
    torch.cuda.synchronize()
    expected_output, expected_per_head_score, expected_reduced_score = _torch_oracle(
        q_latent[0],
        q_rope[0],
        latent_cache,
        rope_cache,
        active_slots[2, :33],
    )
    torch.testing.assert_close(
        output[0], expected_output, rtol=3e-2, atol=3e-2
    )
    if score_mode == "per_head":
        torch.testing.assert_close(
            score[0, :, :33],
            expected_per_head_score,
            rtol=3e-2,
            atol=3e-2,
        )
        assert torch.all(score[0, :, 33:] == -1e20)
    else:
        torch.testing.assert_close(
            score[0, :33], expected_reduced_score, rtol=3e-2, atol=3e-2
        )
        assert torch.all(score[0, 33:] == -1e20)
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    assert torch.all(score[1] == -1e20)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    graph_output = output.clone()
    graph_score = score.clone()
    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, output, rtol=0, atol=0)
    torch.testing.assert_close(graph_score, score, rtol=0, atol=0)


@CUDA_REQUIRED
@pytest.mark.parametrize("valid_heads", [5, 20])
def test_tilelang_score_ignores_zero_padded_heads(valid_heads: int) -> None:
    device = torch.device("cuda")
    capacity = 64
    q_latent = torch.ones(
        1, valid_heads, 512, dtype=torch.bfloat16, device=device
    )
    q_rope = torch.ones(
        1, valid_heads, 64, dtype=torch.bfloat16, device=device
    )
    latent_cache = -torch.ones(
        capacity, 1, 512, dtype=torch.bfloat16, device=device
    )
    rope_cache = -torch.ones(
        capacity, 1, 64, dtype=torch.bfloat16, device=device
    )
    active_slots = torch.arange(
        capacity, dtype=torch.int32, device=device
    ).unsqueeze(0)
    request_indices = torch.zeros(1, dtype=torch.int32, device=device)
    context_lens = torch.full((1,), capacity, dtype=torch.int32, device=device)
    output = torch.empty_like(q_latent)
    score = torch.full(
        (1, capacity), -1e20, dtype=torch.float32, device=device
    )
    runner = TileMlaDecodeKernel(
        device=device,
        softmax_scale=256**-0.5,
        valid_heads=valid_heads,
        fixed_config=TileMlaLaunchConfig(
            1,
            block_h=32 if valid_heads == 20 else 16,
        ),
    )

    runner(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        output,
        attn_score=score,
        max_context_len=capacity,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        score,
        torch.full_like(score, -576.0),
        rtol=0,
        atol=0,
    )


@CUDA_REQUIRED
def test_static_plan_replays_across_contexts_with_unaligned_capacity() -> None:
    torch.manual_seed(20260825)
    device = torch.device("cuda")
    valid_heads = 10
    capacity = 127
    cache_slots = 160
    q_latent = torch.randn(
        1, valid_heads, 512, dtype=torch.bfloat16, device=device
    )
    q_rope = torch.randn(
        1, valid_heads, 64, dtype=torch.bfloat16, device=device
    )
    latent_cache = torch.randn(
        cache_slots, 1, 512, dtype=torch.bfloat16, device=device
    )
    rope_cache = torch.randn(
        cache_slots, 1, 64, dtype=torch.bfloat16, device=device
    )
    active_slots = torch.randperm(
        cache_slots, dtype=torch.int64, device=device
    )[:capacity].to(torch.int32).unsqueeze(0)
    request_indices = torch.zeros(1, dtype=torch.int32, device=device)
    context_lens = torch.full((1,), 31, dtype=torch.int32, device=device)
    output = torch.empty_like(q_latent)
    score_storage = torch.empty(
        1, valid_heads, capacity + 1, dtype=torch.float32, device=device
    )
    score = score_storage[:, :, :capacity]
    assert not score.is_contiguous()
    plan = TileMlaLaunchPlan.build(
        context_capacity=8192,
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        local_q_heads=valid_heads,
        max_batch_size=1,
        need_score=True,
        score_mode="per_head",
    )
    runner = TileMlaDecodeKernel(
        device=device,
        softmax_scale=256**-0.5,
        valid_heads=valid_heads,
        launch_plan=plan,
    )

    def run() -> None:
        score.fill_(-1e20)
        runner(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            attn_score=score,
            max_context_len=capacity,
        )

    run()
    torch.cuda.synchronize()
    metadata = runner.runtime_metadata()
    assert metadata["compiled_variant_count"] == 1
    workspace_ptrs = metadata["compiled_variants"][0]["workspace_data_ptrs"]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for context_len in (31, 64, 65, capacity, 31):
        context_lens.fill_(context_len)
        graph.replay()
        torch.cuda.synchronize()
        expected_output, expected_score, _ = _torch_oracle(
            q_latent[0],
            q_rope[0],
            latent_cache,
            rope_cache,
            active_slots[0, :context_len],
        )
        torch.testing.assert_close(
            output[0], expected_output, rtol=3e-2, atol=3e-2
        )
        torch.testing.assert_close(
            score[0, :, :context_len],
            expected_score,
            rtol=3e-2,
            atol=3e-2,
        )
        assert torch.all(score[0, :, context_len:] == -1e20)
        metadata = runner.runtime_metadata()
        assert metadata["compiled_variant_count"] == 1
        assert (
            metadata["compiled_variants"][0]["workspace_data_ptrs"]
            == workspace_ptrs
        )


@CUDA_REQUIRED
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_rule_bound_provider_replays_ragged_128k_without_workspace_changes(tp_size: int) -> None:
    """Exercise real profile binding and the changed BS5 plan, not a fixed override."""
    from sparseengine import platforms
    from sparseengine.engine.cache_manager import AttentionViewMeta, DecodeComputeView, MlaLatentPayload
    from sparseengine.operators.attention_capabilities import AttentionScoreKind
    from sparseengine.operators.mla_attention import (
        MLA_ATTENTION_REGISTRY, MlaAttentionOpSpec, MlaTileLangScoreProvider,
    )
    from sparseengine.operators.registry import OpResolver

    torch.manual_seed(20260909 + tp_size)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    caps = platforms.current_platform.get_device_caps(0)
    if "H100" not in caps.device_name:
        pytest.skip("This integration test exercises the H100 profile binding")
    batch, capacity, heads = 5, 131072, 20 // tp_size
    spec = MlaAttentionOpSpec(
        num_q_heads=20, kv_lora_rank=512, rope_dim=64, qk_head_dim=256,
        value_head_dim=256, activation_dtype=torch.bfloat16, cache_dtype=torch.bfloat16,
        tp_size=tp_size, cuda_graph=True, score_output=AttentionScoreKind.RAW_QK_PER_HEAD,
        context_capacity=capacity, batch_capacity=batch,
    )
    resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
        spec, caps, op_spec=spec, device=device, max_batch_size=batch,
    )
    provider = resolved.provider
    assert isinstance(provider, MlaTileLangScoreProvider)
    assert provider.tilelang_launch_plan.sm_count == caps.multi_processor_count
    packed = torch.randn(batch, heads, 576, device=device, dtype=torch.bfloat16)
    q, qr = packed[..., :512], packed[..., 512:]
    kv = torch.randn(capacity, 1, 512, device=device, dtype=torch.bfloat16)
    kr = torch.randn(capacity, 1, 64, device=device, dtype=torch.bfloat16)
    slots = torch.stack([torch.randperm(capacity, device=device).int() for _ in range(batch)])
    indices = torch.tensor([0, 1, 2, 3, -1], device=device, dtype=torch.int32)
    lengths = torch.tensor([capacity, capacity - 1, 8193, 17, 0], device=device, dtype=torch.int32)
    output = torch.empty(batch, heads, 512, device=device, dtype=torch.bfloat16)
    score = torch.empty(batch, heads, capacity, device=device, dtype=torch.float32)
    view = DecodeComputeView(
        meta=AttentionViewMeta(active_slots=slots, req_indices=indices, context_lens=lengths,
                               max_context_len=capacity, attn_score=score),
        payload=MlaLatentPayload(latent_cache=kv, rope_cache=kr),
    )
    scope = object()

    def run():
        score.fill_(-1e20)
        provider.run(q, qr, view, output, validation_scope=scope, valid_batch_size=4)

    def check():
        for row, length in enumerate(lengths.tolist()):
            if length:
                expected, raw, _ = _torch_oracle(q[row], qr[row], kv, kr, slots[row, :length])
                torch.testing.assert_close(output[row], expected, rtol=0.03, atol=0.03)
                torch.testing.assert_close(score[row, :, :length], raw, rtol=0.03, atol=0.03)
            else:
                torch.testing.assert_close(output[row], torch.zeros_like(output[row]))
            assert torch.all(score[row, :, length:] == -1e20)

    run()
    torch.cuda.synchronize()
    check()
    bound = provider.tilelang_score.runtime_metadata()
    assert bound["compiled_variant_count"] == 1
    plan = provider.tilelang_launch_plan
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for values in ([17, 2048, 65537, capacity, 0], [31, 63, 65, 511, 0],
                   [capacity, capacity - 1, 8193, 17, 0]):
        lengths.copy_(torch.tensor(values, device=device, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        check()
        assert provider.tilelang_score.runtime_metadata() == bound
        assert provider.tilelang_launch_plan is plan


@CUDA_REQUIRED
def test_static_plan_replays_representative_contexts_through_64k() -> None:
    torch.manual_seed(20260825)
    device = torch.device("cuda")
    valid_heads = 5
    capacity = 65536
    q_latent = torch.randn(
        1, valid_heads, 512, dtype=torch.bfloat16, device=device
    )
    q_rope = torch.randn(
        1, valid_heads, 64, dtype=torch.bfloat16, device=device
    )
    latent_cache = torch.randn(
        capacity, 1, 512, dtype=torch.bfloat16, device=device
    )
    rope_cache = torch.randn(
        capacity, 1, 64, dtype=torch.bfloat16, device=device
    )
    active_slots = torch.arange(
        capacity, dtype=torch.int32, device=device
    ).unsqueeze(0)
    request_indices = torch.zeros(1, dtype=torch.int32, device=device)
    context_lens = torch.full((1,), 1024, dtype=torch.int32, device=device)
    output = torch.empty_like(q_latent)
    score = torch.empty(
        1, valid_heads, capacity, dtype=torch.float32, device=device
    )
    plan = TileMlaLaunchPlan.build(
        context_capacity=capacity,
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        local_q_heads=valid_heads,
        max_batch_size=1,
        need_score=True,
        score_mode="per_head",
    )
    runner = TileMlaDecodeKernel(
        device=device,
        softmax_scale=256**-0.5,
        valid_heads=valid_heads,
        launch_plan=plan,
    )

    def run() -> None:
        score.fill_(-1e20)
        runner(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            attn_score=score,
            max_context_len=capacity,
        )

    run()
    torch.cuda.synchronize()
    metadata = runner.runtime_metadata()
    assert metadata["compiled_variant_count"] == 1
    workspace_ptrs = metadata["compiled_variants"][0]["workspace_data_ptrs"]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for context_len in (1024, 4096, 8192, 16384, 32768, capacity):
        context_lens.fill_(context_len)
        graph.replay()
        torch.cuda.synchronize()
        expected_output, expected_score, _ = _torch_oracle(
            q_latent[0],
            q_rope[0],
            latent_cache,
            rope_cache,
            active_slots[0, :context_len],
        )
        torch.testing.assert_close(
            output[0], expected_output, rtol=3e-2, atol=3e-2
        )
        torch.testing.assert_close(
            score[0, :, :context_len],
            expected_score,
            rtol=3e-2,
            atol=3e-2,
        )
        assert torch.all(score[0, :, context_len:] == -1e20)
        metadata = runner.runtime_metadata()
        assert metadata["compiled_variant_count"] == 1
        assert (
            metadata["compiled_variants"][0]["workspace_data_ptrs"]
            == workspace_ptrs
        )
