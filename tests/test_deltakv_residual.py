from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.cache_manager.methods.deltakv_runtime import DeltaKVCacheManager


@pytest.mark.parametrize("bits", [0, 4])
def test_direct_residual_store_load_preserves_live_kv(bits):
    """P1: prevent corrupted centers/KV and wrong reconstruction without a compressor.

    Existing kernel tests bypass the manager's encoding and payload-size choices.
    The CPU case checks the lossless algebra; CUDA checks packed residuals against
    a torch quantization oracle and replay with changed slot indices.
    """
    if bits and not torch.cuda.is_available():
        pytest.skip("CUDA is required for packed residual storage")
    device = "cuda" if bits else "cpu"
    manager = object.__new__(DeltaKVCacheManager)
    manager.config = SimpleNamespace(
        use_compression=False,
        deltakv_checkpoint_path=None,
        deltakv_latent_dim=16,
        deltakv_latent_quant_bits=bits,
        deltakv_latent_quant_group_size=32,
        mlp_chunk_size=3,
    )
    manager._init_compressor_modules(manager.config, 1)
    n, kv_dim, capacity = 7, 64, 11
    payload_dim = manager._sparse_payload_dim(kv_dim)
    manager.deltakv_latent_cache = torch.zeros(
        (1, capacity, payload_dim // 8 if bits else payload_dim),
        device=device, dtype=torch.int32 if bits else torch.float32,
    )
    manager.deltakv_latent_scales = torch.zeros((1, capacity, payload_dim // 32), device=device)
    manager.deltakv_latent_mins = torch.zeros_like(manager.deltakv_latent_scales)
    generator = torch.Generator().manual_seed(17)
    kv = torch.randn((1, n, kv_dim), generator=generator).to(device)
    base = torch.randn((1, n, kv_dim), generator=generator).to(device)
    original_kv, original_base = kv.clone(), base.clone()

    for indices in (torch.arange(n, device=device), torch.tensor([5, 1, 3], device=device)):
        slots = (indices + 2).to(torch.int32)
        store_all = indices.numel() == n
        manager._deltakv_store_layer_latent(
            l_idx=0, latent_slots=slots, kv_block=kv, base_kv=base,
            store_all=store_all, store_indices=None if store_all else indices,
        )
        expected = (original_kv - original_base)[0, indices]
        if bits:
            groups = expected.reshape(-1, kv_dim // 32, 32)
            mins = groups.amin(dim=-1, keepdim=True)
            scales = (groups.amax(dim=-1, keepdim=True) - mins) / 15
            codes = ((groups - mins) / (scales + 1e-6)).clamp(0, 15).round()
            expected = (codes * scales + mins).reshape(-1, kv_dim)
        actual = manager._load_residual(0, slots, kv_dim)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(kv, original_kv, rtol=0, atol=0)
        torch.testing.assert_close(base, original_base, rtol=0, atol=0)
        if not bits:
            torch.testing.assert_close(actual + base[0, indices], original_kv[0, indices])

    if bits:
        # Warm the existing dequantization kernel before capture.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = manager._load_residual(0, slots, kv_dim)
        for _ in range(2):
            slots.copy_(slots.flip(0))
            expected = expected.flip(0)
            graph.replay()
            torch.testing.assert_close(replayed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reconstruction oracle")
def test_model_key_transform_reconstruction_and_graph():
    """P1: full-head RoPE or plain norm weights silently corrupt Qwen3.5/3.6 KV.

    Compare the manager with an independent partial-RoPE/offset-RMSNorm formula,
    including padded graph rows and changed output positions on replay.
    """
    from sparseengine.layers.layernorm import GemmaRMSNorm
    from sparseengine.operators.qwen35_mrope import Qwen35MRotaryEmbedding

    torch.manual_seed(29)
    device, dtype = 'cuda', torch.bfloat16
    n, heads, dim, rotary_dim = 5, 2, 256, 64
    norm = GemmaRMSNorm(dim, eps=1e-6).to(device=device, dtype=dtype)
    norm.weight.data.uniform_(-0.2, 0.2)
    rope = Qwen35MRotaryEmbedding(dim, rotary_dim, 128, 10000000, [11, 11, 10]).to(device)
    manager = object.__new__(DeltaKVCacheManager)
    manager.device = device
    manager.hf_config = SimpleNamespace(dtype=dtype)
    manager.head_dim, manager.num_kv_heads = dim, heads
    manager.deltakv_layer_ids = [0]
    manager.set_model_layers([SimpleNamespace(self_attn=SimpleNamespace(k_norm=norm, rotary_emb=rope))])
    dummy = torch.tensor([11], device=device, dtype=torch.int32)
    manager._ensure_deltakv_postrope_dummy_slot = lambda: dummy
    k = torch.randn(12, heads, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    original_k, original_v = k.clone(), v.clone()
    delta = torch.randn(n, 2 * heads * dim, device=device, dtype=dtype) * 0.1
    fathers = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]], device=device, dtype=torch.int32)
    slots = torch.tensor([5, 6, 7, 8, -1], device=device, dtype=torch.int32)
    positions = torch.tensor([4, 17, 32, 63, -1], device=device, dtype=torch.int32)
    slot_positions = torch.zeros(12, device=device, dtype=torch.int32)
    def run():
        manager._deltakv_reconstruct_writeback(
            kv_delta=delta, father_slots=fathers, slot_to_pos=slot_positions,
            out_slots=slots, out_pos=positions, cos_sin=rope.text_rope.cos_sin_cache,
            k_cache=k, v_cache=v, l_idx=0,
        )
    def check():
        dk, dv = delta.reshape(n, 2, heads, dim).unbind(1)
        raw = (original_k[fathers.long()].float().mean(1) + dk.float()).to(dtype).float()
        normalized = (raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
                      * (1 + norm.weight.float())).to(dtype).float()
        inv_freq = 1 / (10000000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
        angles = positions.clamp_min(0).float()[:, None] * inv_freq[None, :]
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        x1, x2 = normalized[..., :32], normalized[..., 32:64]
        expected = torch.cat((x1*cos-x2*sin, x2*cos+x1*sin, normalized[..., 64:]), -1).to(dtype)
        expected_v = (original_v[fathers.long()].float().mean(1) + dv.float()).to(dtype)
        torch.testing.assert_close(k[5:9], expected[:4], rtol=0.02, atol=0.02)
        torch.testing.assert_close(v[5:9], expected_v[:4], rtol=0, atol=0)
        torch.testing.assert_close(k[:5], original_k[:5], rtol=0, atol=0)
    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        positions[:4].add_(3)
        graph.replay()
        check()


def test_model_key_norm_preserves_heads_with_a_2d_provider():
    """P1: resumed/reconstructed KV must obey the model norm provider's 2D ABI.

    The GPU reconstruction test checks numerics and replay; this CPU boundary
    test catches the provider shape error without depending on an idle GPU.
    """
    class HeadNorm(torch.nn.Module):
        def forward(self, values):
            assert values.ndim == 2, "model norm provider requires 2D input"
            return values / values.square().mean(-1, keepdim=True).sqrt()

    manager = object.__new__(DeltaKVCacheManager)
    manager._model_key_transform = True
    manager._sparse_key_norms = [HeadNorm()]
    key = torch.arange(1, 49, dtype=torch.float32).reshape(3, 2, 8)
    original = key.clone()
    actual = manager._apply_sparse_k_norm_if_needed(0, key)
    assert actual.shape == key.shape
    torch.testing.assert_close(actual.square().mean(-1), torch.ones(3, 2))
    torch.testing.assert_close(actual / key, (actual / key)[..., :1].expand_as(key))
    torch.testing.assert_close(key, original, rtol=0, atol=0)
