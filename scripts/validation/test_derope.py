#!/usr/bin/env python
"""Validate De-RoPE inversion."""

import torch
import sys
sys.path.insert(0, 'src')

from sparseengine.layers.rotary_embedding import apply_rotary_emb, reverse_rotary_emb


def test_derope_basic():
    """Verify that De-RoPE inverts RoPE."""
    print("=" * 60)
    print("Test 1: basic De-RoPE correctness")
    print("=" * 60)


    batch_size, seq_len, head_dim = 2, 32, 128
    x = torch.randn(batch_size, seq_len, head_dim)


    theta = torch.randn(batch_size, seq_len, head_dim // 2)
    cos = torch.cos(theta)
    sin = torch.sin(theta)


    y = apply_rotary_emb(x, cos, sin)


    x_recovered = reverse_rotary_emb(y, cos, sin)


    max_diff = (x - x_recovered).abs().max().item()
    mean_diff = (x - x_recovered).abs().mean().item()

    print(f"  Input shape: {x.shape}")
    print(f"  Shape after RoPE: {y.shape}")
    print(f"  Recovered shape: {x_recovered.shape}")
    print(f"  Maximum error: {max_diff:.2e}")
    print(f"  Mean error: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✅ Passed!")
        return True
    else:
        print("  ❌ Failed!")
        return False


def test_derope_with_real_rope():
    """Validate inversion with RotaryEmbedding."""
    print("\n" + "=" * 60)
    print("Test 2: RotaryEmbedding integration")
    print("=" * 60)

    from sparseengine.layers.rotary_embedding import get_rope

    head_dim = 128
    max_position = 4096
    rope_base = 10000.0


    rope = get_rope(head_dim, head_dim, max_position, rope_base)


    seq_len = 64
    num_heads = 8
    positions = torch.arange(seq_len)


    k_original = torch.randn(seq_len, num_heads, head_dim)


    cos_sin = rope.cos_sin_cache[positions]  # (seq_len, 1, head_dim)
    cos, sin = cos_sin.chunk(2, dim=-1)


    k_with_rope = []
    for h in range(num_heads):
        k_head = k_original[:, h, :]  # (seq_len, head_dim)
        k_head_roped = apply_rotary_emb(k_head, cos.squeeze(1), sin.squeeze(1))
        k_with_rope.append(k_head_roped)
    k_with_rope = torch.stack(k_with_rope, dim=1)


    k_recovered = []
    for h in range(num_heads):
        k_head = k_with_rope[:, h, :]
        k_head_deroped = reverse_rotary_emb(k_head, cos.squeeze(1), sin.squeeze(1))
        k_recovered.append(k_head_deroped)
    k_recovered = torch.stack(k_recovered, dim=1)


    max_diff = (k_original - k_recovered).abs().max().item()
    mean_diff = (k_original - k_recovered).abs().mean().item()

    print(f"  Original K shape: {k_original.shape}")
    print(f"  K shape after RoPE: {k_with_rope.shape}")
    print(f"  Recovered K shape: {k_recovered.shape}")
    print(f"  Maximum error: {max_diff:.2e}")
    print(f"  Mean error: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✅ Passed!")
        return True
    else:
        print("  ❌ Failed!")
        return False


def test_derope_bf16():
    """Validate inversion in BF16."""
    print("\n" + "=" * 60)
    print("Test 3: BF16 precision")
    print("=" * 60)


    batch_size, seq_len, head_dim = 1, 128, 128
    x = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16)


    theta = torch.randn(batch_size, seq_len, head_dim // 2, dtype=torch.bfloat16)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # RoPE -> De-RoPE
    y = apply_rotary_emb(x, cos, sin)
    x_recovered = reverse_rotary_emb(y, cos, sin)


    max_diff = (x.float() - x_recovered.float()).abs().max().item()

    print(f"  Input dtype: {x.dtype}")
    print(f"  Output dtype: {x_recovered.dtype}")
    print(f"  Maximum error: {max_diff:.2e}")


    if max_diff < 2e-2:
        print("  ✅ Passed within BF16 tolerance.")
        return True
    else:
        print("  ❌ Failed!")
        return False


def main():
    print("\n🔧 De-RoPE (reverse_rotary_emb) validation tests")
    print("=" * 60)

    results = []

    results.append(("Basic correctness", test_derope_basic()))
    results.append(("RotaryEmbedding integration", test_derope_with_real_rope()))
    results.append(("BF16 precision", test_derope_bf16()))

    print("\n" + "=" * 60)
    print("Test summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ Passed" if passed else "❌ Failed"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed. De-RoPE inversion verified.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
