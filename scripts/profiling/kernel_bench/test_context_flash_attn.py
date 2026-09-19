import torch
import triton
import time
import os
import math
import fire
from typing import List

from sparseengine.kernels.triton.context_flashattention_nopad import context_attention_fwd

def get_block_m():
    return 64 if "Tesla" in torch.cuda.get_device_name(0) else 128

def ref_context_attn(q, k, v, b_req_idx, b_seq_len, b_prompt_cache_len, req_to_token_indexs, score_mode):
    # q: (TotalQ, H, D)
    # k, v: (Pool, H, D)
    device = q.device
    total_q, n_heads, dim = q.shape
    pool_size, n_kv_heads, _ = k.shape
    batch_size = len(b_req_idx)
    out = torch.zeros_like(q)
    max_kv_len = req_to_token_indexs.shape[1]
    
    attn_scores_3d = torch.zeros((batch_size, n_heads, max_kv_len), dtype=torch.float32, device=device)
    attn_scores_2d = torch.zeros((batch_size, max_kv_len), dtype=torch.float32, device=device) - float('inf')
    
    start_q = 0
    BLOCK_M = get_block_m()
    sm_scale = 1.0 / (dim ** 0.5) * 1.4426950408889634 
    
    for i in range(batch_size):
        req_idx = b_req_idx[i]
        seq_len_total = b_seq_len[i].item()
        prompt_len = b_prompt_cache_len[i].item()
        seq_len_new = seq_len_total - prompt_len
        
        q_b = q[start_q : start_q + seq_len_new]
        if seq_len_new <= 0:
            continue

        indices = req_to_token_indexs[req_idx][:seq_len_total]
        k_b = k[indices]
        v_b = v[indices]
        
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            k_b_exp = k_b.repeat_interleave(n_rep, dim=1)
            v_b_exp = v_b.repeat_interleave(n_rep, dim=1)
        else:
            k_b_exp, v_b_exp = k_b, v_b
            
        q_t, k_t, v_t = q_b.transpose(0, 1), k_b_exp.transpose(0, 1), v_b_exp.transpose(0, 1)
        scores = torch.matmul(q_t, k_t.transpose(-1, -2))
        
        q_idx = torch.arange(seq_len_new, device=device).unsqueeze(1)
        k_idx = torch.arange(seq_len_total, device=device).unsqueeze(0)
        mask_bool = (q_idx + prompt_len) >= k_idx
        mask_bool = mask_bool.unsqueeze(0).expand(n_heads, -1, -1)
        
        scores_scaled = scores * sm_scale
        scores_masked = torch.where(mask_bool, scores_scaled, torch.tensor(-1.0e8, device=device))
        
        max_val = torch.max(scores_masked, dim=-1, keepdim=True)[0]
        p = torch.exp2(scores_masked - max_val)
        l_i = p.sum(dim=-1, keepdim=True)
        o_b = (torch.matmul(p, v_t) / l_i).transpose(0, 1)
        
        out[start_q : start_q + seq_len_new] = o_b
        
        if score_mode == "3d":
            attn_scores_3d[i, :, :seq_len_total] = scores_masked.sum(dim=1)
        elif score_mode == "2d":
            block_means = []
            for m_start in range(0, seq_len_new, BLOCK_M):
                m_end = min(m_start + BLOCK_M, seq_len_new)
                block_mean = scores_masked[:, m_start:m_end, :].sum(dim=1) / float(seq_len_new)
                block_means.append(block_mean)
            if block_means:
                attn_scores_2d[i, :seq_len_total] = torch.stack(block_means).max(dim=0)[0].max(dim=0)[0]
        
        start_q += seq_len_new
                
    return out, attn_scores_3d if score_mode == "3d" else (attn_scores_2d if score_mode == "2d" else None)

def benchmark_context_attention(
    batch_sizes: List[int] = [1, 4],
    seq_lens: List[int] = [4096, 16384],
    new_token_ratio: float = 0.1, 
    head: int = 28,
    kv_head: int = 4,
    dim: int = 128,
    num_iters: int = 20,
    warmup_iters: int = 5,
    check_correctness: bool = True
):
    device = "cuda"
    dtype = torch.float16
    print(f"\nBenchmarking Context Attention (Prefill) Kernels Comparison")
    print(f"Heads: {head}, KV Heads: {kv_head}, Dim: {dim}, New Ratio: {new_token_ratio}")
    print("-" * 140)
    header = f"{ 'Batch':>5} | { 'TotalSeq':>10} | { 'NewToken':>10} | { 'Base (ms)':>12} | { 'Score2D (ms)':>12} | { 'Score3D (ms)':>12} | {'Status'}"
    print(header)
    print("-" * 140)

    for batch_size in batch_sizes:
        for total_len in seq_lens:
            new_len = max(1, int(total_len * new_token_ratio))
            context_len = total_len - new_len
            
            # Data preparation
            b_prompt_cache_len = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
            b_seq_len = torch.full((batch_size,), total_len, dtype=torch.int32, device=device)
            new_lens = b_seq_len - b_prompt_cache_len
            
            pool_size = batch_size * total_len
            req_to_token_indexs = torch.arange(pool_size, device=device, dtype=torch.int32).reshape(batch_size, total_len)
            
            total_new_tokens = new_lens.sum().item()
            q = torch.randn((total_new_tokens, head, dim), dtype=dtype, device=device)
            k = torch.randn((pool_size, kv_head, dim), dtype=dtype, device=device)
            v = torch.randn((pool_size, kv_head, dim), dtype=dtype, device=device)
            
            b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
            b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device=device)
            cur = 0
            for i in range(batch_size):
                b_start_loc[i] = cur
                cur += new_lens[i].item()
            
            o = torch.zeros_like(q)
            real_max_new = new_lens.max().item()

            def time_func(mode):
                attn_score = None
                if mode == "3d":
                    attn_score = torch.zeros((batch_size, head, total_len), dtype=torch.float32, device=device)
                elif mode == "2d":
                    attn_score = torch.full((batch_size, total_len), -float('inf'), dtype=torch.float32, device=device)
                
                def run():
                    context_attention_fwd(q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, real_max_new, req_to_token_indexs, attn_score)
                
                for _ in range(warmup_iters): run()
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters): run()
                torch.cuda.synchronize()
                return (time.time() - start) / num_iters * 1000

            t_base = time_func("none")
            t_2d = time_func("2d")
            t_3d = time_func("3d")

            status = "N/A"
            if check_correctness:
                ref_o, _ = ref_context_attn(q, k, v, b_req_idx, b_seq_len, b_prompt_cache_len, req_to_token_indexs, "none")
                diff = (o - ref_o).abs().max().item()
                status = "✅" if diff < 1e-2 else f"❌({diff:.1e})"

            print(f"{batch_size:5d} | {total_len:10d} | {new_len:10d} | {t_base:12.4f} | {t_2d:12.4f} | {t_3d:12.4f} | {status}")

if __name__ == "__main__":
    fire.Fire(benchmark_context_attention)