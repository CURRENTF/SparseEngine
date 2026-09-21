"""Regressions for independent prefill resolution and paged latent attention."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sparseengine.kernels.external.support import ExternalKernelContractError, KernelFamilyState
from sparseengine.operators import mla_compressed_prefill as prefill
from sparseengine.operators.registry import OpResolver
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from test_mla_attention_operator import _h100_caps, _spec


@pytest.mark.parametrize("available", [True, False])
def test_prefill_resolution_ignores_decode_scores_and_handles_device_rejection(monkeypatch, available):
    # Score-producing decode formerly disabled even supported FA3 prefill.
    monkeypatch.setattr(prefill, "sgl_kernel_metadata_health", lambda: SimpleNamespace(state=KernelFamilyState.READY))
    monkeypatch.setattr(prefill, "sgl_fa3_device_support", lambda _: (available, "device probe"))
    monkeypatch.setattr(prefill.SglLatentPrefill, "bind", classmethod(lambda cls, *a, **kw: object.__new__(cls)))
    monkeypatch.setattr(prefill.TritonLatentPrefill, "bind", classmethod(lambda cls, *a, **kw: object.__new__(cls)))
    result = OpResolver(prefill.MLA_COMPRESSED_PREFILL_REGISTRY).resolve(
        _spec(score_output=AttentionScoreKind.RAW_QK_PER_HEAD), _h100_caps(), max_batch_size=4,
    )
    assert result.provider.name == ("sgl_fa3_latent" if available else "triton_latent")


def test_broken_fa3_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(prefill, "sgl_kernel_metadata_health", lambda: SimpleNamespace(state=KernelFamilyState.READY))
    monkeypatch.setattr(prefill, "sgl_fa3_device_support", Mock(side_effect=ExternalKernelContractError("sglang-kernel", "FA3", "bad ABI")))
    bind = Mock()
    monkeypatch.setattr(prefill.TritonLatentPrefill, "bind", bind)
    with pytest.raises(ExternalKernelContractError, match="bad ABI"):
        prefill.resolve_mla_compressed_prefill(_spec(), _h100_caps(), max_batch_size=1)
    bind.assert_not_called()


def test_absent_fa3_can_bind_compressed_prefill_fallback(monkeypatch):
    monkeypatch.setattr(prefill, "sgl_kernel_metadata_health", lambda: SimpleNamespace(state=KernelFamilyState.ABSENT, reason="absent"))
    probe = Mock(side_effect=AssertionError("must not import absent package"))
    monkeypatch.setattr(prefill, "sgl_fa3_device_support", probe)
    monkeypatch.setattr(prefill.TritonLatentPrefill, "bind", classmethod(lambda cls, *a, **kw: object.__new__(cls)))
    result = OpResolver(prefill.MLA_COMPRESSED_PREFILL_REGISTRY).resolve(_spec(), _h100_caps(), max_batch_size=1)
    assert result.provider.name == "triton_latent"
    assert result.report.selection_basis == "dependency_degraded"
    probe.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits", [1, 8])
def test_paged_latent_prefill_graph_replay_ragged_causal_lse(dtype, splits):
    # Ragged tails and empty split partitions must produce finite outputs;
    # replay must consume changed page tables and lengths, not captured values.
    from sparseengine.kernels.triton.mla.prefill_latent import attention_latent

    torch.manual_seed(318)
    device = "cuda"
    q = (torch.randn(5, 20, 512, dtype=dtype, device=device) * .2).transpose(0, 1)
    qr = torch.randn(20, 5, 128, dtype=dtype, device=device)[..., 64:] * .2
    latent = torch.randn(600, 1, 512, dtype=dtype, device=device)
    rope = torch.randn(600, 1, 64, dtype=dtype, device=device) * .2
    slots = torch.randperm(600, device=device).int().reshape(2, 300)
    rows = torch.tensor([1, 0], device=device, dtype=torch.int32)
    lengths = torch.tensor([299, 5], device=device, dtype=torch.int32)
    cu = torch.tensor([0, 17, 20], device=device, dtype=torch.int32)

    def run():
        return attention_latent(q, qr, latent, rope, slots, rows, lengths, cu,
                                max_q=17, scale=1 / 16, splits=splits)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = run()
    for kn in (299, 131):
        lengths[0] = kn
        slots.copy_(slots.flip(1))
        graph.replay()
        for a, b, row, n in ((0, 17, 1, kn), (17, 20, 0, 5)):
            c, r = latent[slots[row, :n].long(), 0].float(), rope[slots[row, :n].long(), 0].float()
            z = (torch.einsum("qhd,kd->hqk", q[a:b].float(), c)
                 + torch.einsum("qhd,kd->hqk", qr[a:b].float(), r)) / 16
            valid = torch.arange(n, device=device)[None] <= n - (b - a) + torch.arange(b - a, device=device)[:, None]
            z.masked_fill_(~valid[None], -torch.inf)
            expected = torch.einsum("hqk,kd->qhd", z.softmax(-1), c)
            torch.testing.assert_close(output[a:b].float(), expected, atol=.006, rtol=.03)
            torch.testing.assert_close(lse[:, a:b], z.logsumexp(-1), atol=.002, rtol=.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_latent_prefill_new_suffix_lengths_reuse_jit_variants(monkeypatch):
    # Absorbed Q is head-major: changing suffix length changes its head stride.
    # A constexpr stride would compile anew for each conversation turn.
    import triton
    from sparseengine.kernels.triton.mla.prefill_latent import attention_latent

    def run(qn):
        q = (torch.randn(5, qn, 512, device="cuda", dtype=torch.bfloat16) * .2).transpose(0, 1)
        qr = torch.zeros(qn, 5, 64, device="cuda", dtype=q.dtype)
        kn = qn + 240
        c = torch.ones(kn, 1, 512, device="cuda", dtype=q.dtype)
        r = torch.zeros(kn, 1, 64, device="cuda", dtype=q.dtype)
        out, lse = attention_latent(
            q, qr, c, r, torch.arange(kn, device="cuda", dtype=torch.int32)[None],
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([kn], device="cuda", dtype=torch.int32),
            torch.tensor([0, qn], device="cuda", dtype=torch.int32),
            max_q=qn, scale=1 / 16, splits=8,
        )
        expected_lse = q.float().sum(-1).T / 16 + torch.arange(241, kn + 1, device="cuda").float().log()[None]
        torch.testing.assert_close(out, torch.ones_like(out), atol=.005, rtol=.005)
        torch.testing.assert_close(lse, expected_lse, atol=.002, rtol=.001)

    run(16)
    events = []
    previous = triton.knobs.compilation.listener

    def listener(**kwargs):
        if kwargs["src"].name in {"_attention", "_merge"}:
            events.append(kwargs["src"].name)
        if previous is not None:
            previous(**kwargs)

    monkeypatch.setattr(triton.knobs.compilation, "listener", listener)
    for qn in (17, 33, 47):
        run(qn)
    assert not events, f"New query length caused compilation/cache lookup: {events}"
