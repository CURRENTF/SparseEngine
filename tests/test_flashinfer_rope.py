import pytest
import torch

from sparseengine.layers.rotary_embedding import (
    RotaryEmbedding,
    apply_partial_rotary_emb,
    apply_rotary_emb,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "head_size", "rotary_dim", "base"),
    [
        pytest.param(8, 2, 128, 128, 1_000_000.0, id="qwen3"),
        pytest.param(6, 1, 128, 64, 5_000_000.0, id="minimax-m2.7"),
    ],
)
def test_flashinfer_rope_matches_torch_reference(
    num_q_heads,
    num_kv_heads,
    head_size,
    rotary_dim,
    base,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    positions = torch.tensor([0, 1, 17, 127, 1023], device=device)
    query = torch.randn(
        positions.numel(), num_q_heads, head_size, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        positions.numel(), num_kv_heads, head_size, device=device, dtype=torch.bfloat16
    )
    rotary = RotaryEmbedding(
        head_size=rotary_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=2048,
        base=base,
        backend="flashinfer",
    ).to(device)

    cos, sin = rotary.cos_sin_cache[positions].chunk(2, dim=-1)
    expected_query_prefix = apply_rotary_emb(query[..., :rotary_dim], cos, sin)
    expected_key_prefix = apply_rotary_emb(key[..., :rotary_dim], cos, sin)
    expected_query = torch.cat(
        (expected_query_prefix, query[..., rotary_dim:]), dim=-1
    )
    expected_key = torch.cat((expected_key_prefix, key[..., rotary_dim:]), dim=-1)

    if rotary_dim == head_size:
        actual_query, actual_key = rotary(positions, query, key)
    else:
        actual_query, actual_key = apply_partial_rotary_emb(
            rotary, positions, query, key, rotary_dim
        )

    torch.testing.assert_close(actual_query, expected_query, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_key, expected_key, atol=2e-2, rtol=2e-2)
    if rotary_dim < head_size:
        assert torch.equal(actual_query[..., rotary_dim:], query[..., rotary_dim:])
        assert torch.equal(actual_key[..., rotary_dim:], key[..., rotary_dim:])


def test_torch_rope_backend_remains_available():
    rotary = RotaryEmbedding(
        head_size=4,
        rotary_dim=4,
        max_position_embeddings=8,
        base=10_000.0,
        backend="torch",
    )
    positions = torch.tensor([0, 1])
    query = torch.randn(2, 1, 4)
    key = torch.randn(2, 1, 4)

    query_out, key_out = rotary.compiled_forward.__wrapped__(
        rotary, positions, query, key
    )

    assert query_out.shape == query.shape
    assert key_out.shape == key.shape
