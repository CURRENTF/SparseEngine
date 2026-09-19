#!/usr/bin/env python
"""
测试 De-RoPE (reverse_rotary_emb) 的正确性。

验证：对一个向量先应用 RoPE，再应用 De-RoPE，应该能恢复到原始向量。
"""

import torch
import sys
sys.path.insert(0, 'src')

from sparseengine.layers.rotary_embedding import apply_rotary_emb, reverse_rotary_emb


def test_derope_basic():
    """基本测试：De-RoPE 应该是 RoPE 的逆操作"""
    print("=" * 60)
    print("测试 1: De-RoPE 基本正确性")
    print("=" * 60)
    
    # 创建测试数据
    batch_size, seq_len, head_dim = 2, 32, 128
    x = torch.randn(batch_size, seq_len, head_dim)
    
    # 创建符合 cos² + sin² = 1 的 cos/sin
    # 使用随机角度生成真实的三角函数值
    theta = torch.randn(batch_size, seq_len, head_dim // 2)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    
    # 应用 RoPE
    y = apply_rotary_emb(x, cos, sin)
    
    # 应用 De-RoPE
    x_recovered = reverse_rotary_emb(y, cos, sin)
    
    # 验证
    max_diff = (x - x_recovered).abs().max().item()
    mean_diff = (x - x_recovered).abs().mean().item()
    
    print(f"  输入 shape: {x.shape}")
    print(f"  RoPE 后 shape: {y.shape}")
    print(f"  恢复后 shape: {x_recovered.shape}")
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  平均误差: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("  ✅ 测试通过！")
        return True
    else:
        print("  ❌ 测试失败！")
        return False


def test_derope_with_real_rope():
    """使用真实的 RotaryEmbedding 类测试"""
    print("\n" + "=" * 60)
    print("测试 2: 与 RotaryEmbedding 类配合使用")
    print("=" * 60)
    
    from sparseengine.layers.rotary_embedding import get_rope
    
    head_dim = 128
    max_position = 4096
    rope_base = 10000.0
    
    # 创建 RoPE 实例
    rope = get_rope(head_dim, head_dim, max_position, rope_base)
    
    # 创建测试数据
    seq_len = 64
    num_heads = 8
    positions = torch.arange(seq_len)
    
    # 原始 K 向量 (seq_len, num_heads, head_dim)
    k_original = torch.randn(seq_len, num_heads, head_dim)
    
    # 获取 cos/sin
    cos_sin = rope.cos_sin_cache[positions]  # (seq_len, 1, head_dim)
    cos, sin = cos_sin.chunk(2, dim=-1)  # 各 (seq_len, 1, head_dim//2... 不对)
    
    # 注意：cos_sin_cache 的结构是 (max_pos, 1, head_dim)，其中 head_dim = cos + sin 拼接
    # 所以 cos 和 sin 各占 head_dim // 2
    # 但 apply_rotary_emb 期望 cos/sin 的 shape 和输入 x 的最后一维的一半大小相同
    
    # 对每个 head 应用 RoPE
    k_with_rope = []
    for h in range(num_heads):
        k_head = k_original[:, h, :]  # (seq_len, head_dim)
        k_head_roped = apply_rotary_emb(k_head, cos.squeeze(1), sin.squeeze(1))
        k_with_rope.append(k_head_roped)
    k_with_rope = torch.stack(k_with_rope, dim=1)
    
    # 对每个 head 应用 De-RoPE
    k_recovered = []
    for h in range(num_heads):
        k_head = k_with_rope[:, h, :]
        k_head_deroped = reverse_rotary_emb(k_head, cos.squeeze(1), sin.squeeze(1))
        k_recovered.append(k_head_deroped)
    k_recovered = torch.stack(k_recovered, dim=1)
    
    # 验证
    max_diff = (k_original - k_recovered).abs().max().item()
    mean_diff = (k_original - k_recovered).abs().mean().item()
    
    print(f"  原始 K shape: {k_original.shape}")
    print(f"  RoPE 后 K shape: {k_with_rope.shape}")
    print(f"  恢复后 K shape: {k_recovered.shape}")
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  平均误差: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("  ✅ 测试通过！")
        return True
    else:
        print("  ❌ 测试失败！")
        return False


def test_derope_bf16():
    """测试 BF16 精度下的表现"""
    print("\n" + "=" * 60)
    print("测试 3: BF16 精度测试")
    print("=" * 60)
    
    # 创建 BF16 测试数据
    batch_size, seq_len, head_dim = 1, 128, 128
    x = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    
    # 使用真实的三角函数值（cos² + sin² = 1）
    theta = torch.randn(batch_size, seq_len, head_dim // 2, dtype=torch.bfloat16)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    
    # RoPE -> De-RoPE
    y = apply_rotary_emb(x, cos, sin)
    x_recovered = reverse_rotary_emb(y, cos, sin)
    
    # 验证
    max_diff = (x.float() - x_recovered.float()).abs().max().item()
    
    print(f"  输入 dtype: {x.dtype}")
    print(f"  输出 dtype: {x_recovered.dtype}")
    print(f"  最大误差: {max_diff:.2e}")
    
    # BF16 精度较低，误差阈值放宽到 2e-2
    if max_diff < 2e-2:
        print("  ✅ 测试通过！（BF16 精度范围内）")
        return True
    else:
        print("  ❌ 测试失败！")
        return False



def main():
    print("\n🔧 De-RoPE (reverse_rotary_emb) 验证测试")
    print("=" * 60)
    
    results = []
    
    results.append(("基本正确性", test_derope_basic()))
    results.append(("RotaryEmbedding 集成", test_derope_with_real_rope()))
    results.append(("BF16 精度", test_derope_bf16()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有测试通过！De-RoPE 实现正确。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit(main())
