"""Regressions for last-representative semantics and tiled column reduction."""
import pytest
import torch

from sparseengine.operators.rkv_similarity import (
    RKV_SIMILARITY_REGISTRY,
    RKVSimilaritySpec,
    TorchRKVSimilarityProvider,
    prepare_rkv_similarity_provider,
)


def reference_columns(sim, row_start):
    # Row-wise oracle: retain ALL below-threshold mass and remove only the last
    # matching column (column zero when no match), after removing the diagonal.
    values = sim.clone()
    for unit in range(sim.shape[0]):
        for row in range(sim.shape[1]):
            values[unit, row, row + row_start] = 0
            matches = (values[unit, row] > 0.5).nonzero().flatten()
            representative = int(matches[-1]) if matches.numel() else 0
            values[unit, row, representative] = 0
    return values.double().sum(-2).float()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_representatives_and_masked_tails_match_independent_oracle(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(37)
    sim = (torch.rand(2, 131, 5003, device=device) - 0.45).to(dtype)
    sim[:, 0] = 0.25  # No match: column zero still must be removed.
    sim[:, 1] = 0.125
    sim[:, 1, 9] = 0.5  # Strict threshold, not >=.
    sim[:, 1, 17] = 0.875
    sim[:, 1, 4999] = 0.75  # Last match, not maximum value or first match.
    sim[:, 2] = 0.25
    sim[:, 2, 21] = 1  # Diagonal must not count as a representative.
    sim[:, 3] = 0.125
    sim[:, 3, 9] = 0.5  # Equality alone must not select column nine.
    original = sim.clone()
    provider = prepare_rkv_similarity_provider(dtype, device=torch.device(device))
    expected = reference_columns(sim, 19)
    first = provider.column_sums(sim, 19)
    second = provider.column_sums(sim, 19)
    torch.testing.assert_close(first, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(sim, original, rtol=0, atol=0)


def test_invalid_row_domain_and_layout_fail_before_launch():
    provider = TorchRKVSimilarityProvider(op_spec=RKVSimilaritySpec(torch.float32))
    with pytest.raises(ValueError, match="resident domain"):
        provider.column_sums(torch.ones(1, 3, 5), 3)
    with pytest.raises(ValueError, match="contiguous"):
        provider.column_sums(torch.ones(1, 5, 3).transpose(1, 2), 0)
    with pytest.raises(TypeError, match="dtype"):
        provider.column_sums(torch.ones(1, 3, 5, dtype=torch.float16), 0)


def test_unavailable_gpu_backend_does_not_silently_select_torch():
    # An unsupported GPU must fail at binding instead of unexpectedly doing
    # quadratic Torch work; CPU fixtures remain independently supported.
    from sparseengine.operators.registry import NoProviderError, OpResolver
    from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
    caps = DeviceCaps(platform=PlatformEnum.CUDA, device_type="cuda", device_index=0,
                      device_name="unavailable-backend", supports_triton=False)
    spec = RKVSimilaritySpec(torch.float32)
    with pytest.raises(NoProviderError, match="Triton"):
        OpResolver(RKV_SIMILARITY_REGISTRY).resolve(spec, caps, op_spec=spec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selected_kernel_failure_propagates_without_fallback(monkeypatch):
    provider = prepare_rkv_similarity_provider(torch.float32, device=torch.device("cuda", 0))
    def fail(*args, **kwargs):
        raise RuntimeError("RKV kernel launch failed")
    monkeypatch.setattr(provider, "_run", fail)
    with pytest.raises(RuntimeError, match="RKV kernel launch failed"):
        provider.column_sums(torch.zeros(1, 1, 3, device="cuda"), 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_complete_scores_and_global_selection_match_reference(dtype):
    from sparseengine.engine.cache_manager.methods.rkv_scoring import rkv_head_scores
    from tests.test_rkv_vllm import reference_scores
    torch.manual_seed(93)
    # Batch interpreted as layers here, exercising the serving head-mean /
    # layer-sum selection as well as noncontiguous keys, GQA and tiled scores.
    keys = torch.randn(2, 137, 2, 16, device="cuda", dtype=dtype).transpose(1, 2)
    keys[:, :, 33:41] = keys[:, :, 9:17]
    queries = torch.randn(2, 6, 8, 16, device="cuda", dtype=dtype)
    expected = reference_scores(keys, queries, 8, 7, .1)
    for cap in (256 * 1024, 16 * 1024**2):
        actual = rkv_head_scores(keys, queries, window=8, kernel_size=7,
                                 alpha=.1, workspace_bytes=cap)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=.02)
        assert torch.equal(actual.mean(1).sum(0).topk(32).indices,
                           expected.mean(1).sum(0).topk(32).indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_row_tiles_reconstruct_full_domain_and_preserve_nonfinite_failures():
    # Global diagonal offsets and merge boundaries must survive row splitting.
    torch.manual_seed(51)
    sim = torch.rand(1, 257, 257, device="cuda", dtype=torch.float32)
    provider = prepare_rkv_similarity_provider(sim.dtype, device=sim.device)
    pieces = [provider.column_sums(sim[:, a:b].contiguous(), a)
              for a, b in ((0, 1), (1, 129), (129, 257))]
    torch.testing.assert_close(sum(pieces), reference_columns(sim, 0), atol=3e-5, rtol=1e-5)
    sim[0, 5, 8] = float("nan")
    assert torch.isnan(provider.column_sums(sim, 0)[0, 8])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_changing_resident_lengths_does_not_recompile_after_warmup():
    # Agent contexts vary on every compression; exact lengths must not create
    # new compiler keys or consume the post-startup compilation budget.
    from sparseengine.utils.compilation_guard import RuntimeCompilationGuard
    provider = prepare_rkv_similarity_provider(torch.bfloat16, device=torch.device("cuda", 0))
    provider.column_sums(torch.zeros(1, 131, 5003, device="cuda", dtype=torch.bfloat16), 19)
    torch.cuda.synchronize()
    guard = RuntimeCompilationGuard(0, 0)
    guard.arm()
    try:
        for rows, length, offset in ((1, 513, 0), (257, 8192, 127), (19, 20131, 3)):
            actual = provider.column_sums(torch.zeros(1, rows, length, device="cuda", dtype=torch.bfloat16), offset)
            assert torch.count_nonzero(actual) == 0
        torch.cuda.synchronize()
    finally:
        guard.close()
