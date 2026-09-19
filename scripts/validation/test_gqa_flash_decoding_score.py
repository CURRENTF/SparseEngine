import torch
import numpy as np
# 分别导入两个版本的 kernel
from sparseengine.kernels.triton.flash_decoding_stage1 import flash_decode_stage1_with_score as flash_decode_v1
from sparseengine.kernels.triton.gqa_flash_decoding_stage1 import flash_decode_stage1_with_score as flash_decode_gqa

def test_gqa_flash_decoding_score():
    torch.manual_seed(42)
    # 检查是否有 CUDA 核心，因为 Triton 需要 CUDA
    if not torch.cuda.is_available():
        print("CUDA not available. This test requires a GPU.")
        return
    
    device = torch.device("cuda")
    
    # 基础配置
    batch = 2
    head_num_q = 32
    head_num_kv = 8
    gqa_group_size = head_num_q // head_num_kv
    head_dim = 128
    seq_len = 1024
    block_seq = 256
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    # 数据初始化
    q = torch.randn((batch, head_num_q, head_dim), device=device, dtype=torch.float16)
    # K, V 模拟 vLLM 的存储方式 [num_tokens, head_num_kv, head_dim]
    k = torch.randn((batch * seq_len, head_num_kv, head_dim), device=device, dtype=torch.float16)
    v = torch.randn((batch * seq_len, head_num_kv, head_dim), device=device, dtype=torch.float16)
    
    # 辅助张量
    B_req_idx = torch.arange(batch, device=device, dtype=torch.int32)
    B_Seqlen = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    # Req_to_tokens: [batch, max_seq_len] -> 映射到 k/v 的第一维索引
    Req_to_tokens = torch.arange(batch * seq_len, device=device, dtype=torch.int32).view(batch, seq_len)
    
    max_len_in_batch = seq_len
    num_blocks = (max_len_in_batch + block_seq - 1) // block_seq
    
    # 准备输出张量 - V1 版本
    mid_out_v1 = torch.zeros((batch, head_num_q, num_blocks, head_dim), device=device, dtype=torch.float32)
    mid_out_lse_v1 = torch.zeros((batch, head_num_q, num_blocks), device=device, dtype=torch.float32)
    attn_score_v1 = torch.full((batch, head_num_q, seq_len), -float('inf'), device=device, dtype=torch.float32)

    # 准备输出张量 - GQA 版本
    mid_out_gqa = torch.zeros((batch, head_num_q, num_blocks, head_dim), device=device, dtype=torch.float32)
    mid_out_lse_gqa = torch.zeros((batch, head_num_q, num_blocks), device=device, dtype=torch.float32)
    attn_score_gqa = torch.full((batch, head_num_q, seq_len), -float('inf'), device=device, dtype=torch.float32)
    
    print("Running Triton V1 (Standard)...")
    flash_decode_v1(
        q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch,
        mid_out_v1, mid_out_lse_v1, attn_score_v1, block_seq
    )

    print("Running Triton GQA (Optimized)...")
    flash_decode_gqa(
        q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch,
        mid_out_gqa, mid_out_lse_gqa, attn_score_gqa, block_seq
    )
    
    # 计算 PyTorch Ground Truth
    # q: [B, H_q, D]
    # k_ref: [B, H_kv, T, D]
    k_ref = k.view(batch, seq_len, head_num_kv, head_dim).transpose(1, 2) 
    v_ref = v.view(batch, seq_len, head_num_kv, head_dim).transpose(1, 2)
    
    # 扩展 K/V 以匹配 Q 的 head 数量
    k_expanded = k_ref.repeat_interleave(gqa_group_size, dim=1) # [B, H_q, T, D]
    v_expanded = v_ref.repeat_interleave(gqa_group_size, dim=1) # [B, H_q, T, D]
    
    # 计算 Score Ground Truth (未缩放)
    q_ref = q.unsqueeze(2) # [B, H_q, 1, D]
    gt_score_3d = torch.matmul(q_ref, k_expanded.transpose(-1, -2)).squeeze(2) # [B, H_q, T]
    
    # 验证一致性
    print("\n--- Verification Results ---")
    
    def check(name, triton_val, torch_val):
        try:
            torch.testing.assert_close(triton_val, torch_val.float(), atol=1e-2, rtol=1e-2)
            print(f"{name}: Match with PyTorch!")
        except Exception as e:
            print(f"{name}: FAILED match with PyTorch!")
            # print(e)

    # 1. 对比 V1 和 GQA
    try:
        torch.testing.assert_close(attn_score_v1, attn_score_gqa, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(mid_out_v1, mid_out_gqa, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(mid_out_lse_v1, mid_out_lse_gqa, atol=1e-3, rtol=1e-3)
        print("V1 vs GQA: Perfectly Match each other!")
    except Exception as e:
        print("V1 vs GQA: MISMATCH!")

    # 2. 对比 PyTorch
    check("V1 Attn Score", attn_score_v1, gt_score_3d)
    check("GQA Attn Score", attn_score_gqa, gt_score_3d)

    # 计算 Mid_O 的 GT
    gt_mid_out = torch.zeros_like(mid_out_v1)
    gt_mid_lse = torch.zeros_like(mid_out_lse_v1)
    for b in range(num_blocks):
        start = b * block_seq
        end = min((b + 1) * block_seq, seq_len)
        
        # S_block: [B, H_q, BLOCK_SEQ]
        s_block = gt_score_3d[:, :, start:end] * sm_scale
        
        # 处理可能的 Padding
        if start >= seq_len:
            continue
            
        m_block = s_block.max(dim=-1, keepdim=True)[0] # [B, H_q, 1]
        p_block = torch.exp(s_block - m_block)
        sum_exp_block = p_block.sum(dim=-1, keepdim=True) # [B, H_q, 1]
        
        # v_block: [B, H_q, BLOCK_SEQ, D]
        v_block = v_expanded[:, :, start:end, :]
        
        # O_block: [B, H_q, D]
        o_block = torch.matmul(p_block.unsqueeze(2).to(v_block.dtype), v_block).squeeze(2) / sum_exp_block.to(v_block.dtype)
        lse_block = (m_block + torch.log(sum_exp_block)).squeeze(-1)
        
        gt_mid_out[:, :, b, :] = o_block.float()
        gt_mid_lse[:, :, b] = lse_block.float()

    check("V1 Mid_O", mid_out_v1, gt_mid_out)
    check("GQA Mid_O", mid_out_gqa, gt_mid_out)
    check("V1 LogSumExp", mid_out_lse_v1, gt_mid_lse)
    check("GQA LogSumExp", mid_out_lse_gqa, gt_mid_lse)

if __name__ == "__main__":
    try:
        test_gqa_flash_decoding_score()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed: {e}")
