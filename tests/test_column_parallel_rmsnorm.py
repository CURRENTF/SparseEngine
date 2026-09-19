from types import SimpleNamespace

import pytest
import torch

from sparseengine.layers.layernorm import ColumnParallelRMSNorm


class _ReferenceTpContext(SimpleNamespace):
    def __init__(self, rank, remote_square_sums):
        super().__init__(attn_tp_rank=rank, attn_tp_size=2)
        self.attn_tp = SimpleNamespace(all_reduce=self.all_reduce)
        self.remote_square_sums = remote_square_sums
        self.all_reduce_calls = 0

    def all_reduce(self, values):
        values.add_(self.remote_square_sums)
        self.all_reduce_calls += 1
        return values


def test_column_parallel_qk_norm_fuses_global_statistics():
    torch.manual_seed(11)
    query = torch.randn(3, 8, dtype=torch.bfloat16)
    key = torch.randn(3, 4, dtype=torch.bfloat16)
    query_weight = torch.randn(8, dtype=torch.bfloat16)
    key_weight = torch.randn(4, dtype=torch.bfloat16)
    outputs = []

    for rank in range(2):
        local_query = query.chunk(2, dim=-1)[rank]
        local_key = key.chunk(2, dim=-1)[rank]
        other_rank = 1 - rank
        remote_sums = torch.stack(
            (
                query.chunk(2, dim=-1)[other_rank].float().square().sum(-1),
                key.chunk(2, dim=-1)[other_rank].float().square().sum(-1),
            ),
            dim=-1,
        )
        context = _ReferenceTpContext(rank, remote_sums)
        q_norm = ColumnParallelRMSNorm(8, parallel_context=context)
        k_norm = ColumnParallelRMSNorm(4, parallel_context=context)
        q_norm.weight.data.copy_(query_weight.chunk(2)[rank])
        k_norm.weight.data.copy_(key_weight.chunk(2)[rank])

        outputs.append(q_norm.forward_pair(local_query, local_key, k_norm))
        assert context.all_reduce_calls == 1

    actual_query = torch.cat([item[0] for item in outputs], dim=-1)
    actual_key = torch.cat([item[1] for item in outputs], dim=-1)
    expected_query = (
        query.float()
        * torch.rsqrt(query.float().square().mean(-1, keepdim=True) + 1.0e-6)
    ).to(query.dtype) * query_weight
    expected_key = (
        key.float() * torch.rsqrt(key.float().square().mean(-1, keepdim=True) + 1.0e-6)
    ).to(key.dtype) * key_weight

    torch.testing.assert_close(actual_query, expected_query, atol=0, rtol=0)
    torch.testing.assert_close(actual_key, expected_key, atol=0, rtol=0)


def test_column_parallel_rmsnorm_exposes_rank_local_weight_slice():
    context = _ReferenceTpContext(1, torch.empty(0))
    norm = ColumnParallelRMSNorm(8, parallel_context=context)

    assert norm.rank_local_weight_slice((8,)) == (slice(4, 8),)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_column_parallel_qk_norm_cuda_fused_path_matches_reference():
    torch.manual_seed(17)
    query = torch.randn(5, 16, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(5, 8, device="cuda", dtype=torch.bfloat16)
    query_weight = torch.randn(16, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(8, device="cuda", dtype=torch.bfloat16)
    outputs = []

    for rank in range(2):
        local_query = query.chunk(2, dim=-1)[rank]
        local_key = key.chunk(2, dim=-1)[rank]
        other_rank = 1 - rank
        remote_sums = torch.stack(
            (
                query.chunk(2, dim=-1)[other_rank].float().square().sum(-1),
                key.chunk(2, dim=-1)[other_rank].float().square().sum(-1),
            ),
            dim=-1,
        )
        context = _ReferenceTpContext(rank, remote_sums)
        q_norm = ColumnParallelRMSNorm(16, parallel_context=context).to(
            device="cuda", dtype=torch.bfloat16
        )
        k_norm = ColumnParallelRMSNorm(8, parallel_context=context).to(
            device="cuda", dtype=torch.bfloat16
        )
        q_norm.weight.data.copy_(query_weight.chunk(2)[rank])
        k_norm.weight.data.copy_(key_weight.chunk(2)[rank])

        outputs.append(q_norm.forward_pair(local_query, local_key, k_norm))
        assert context.all_reduce_calls == 1

    actual_query = torch.cat([item[0] for item in outputs], dim=-1)
    actual_key = torch.cat([item[1] for item in outputs], dim=-1)
    expected_query = (
        query.float()
        * torch.rsqrt(query.float().square().mean(-1, keepdim=True) + 1.0e-6)
    ).to(query.dtype) * query_weight
    expected_key = (
        key.float() * torch.rsqrt(key.float().square().mean(-1, keepdim=True) + 1.0e-6)
    ).to(key.dtype) * key_weight

    torch.testing.assert_close(actual_query, expected_query, atol=0.015625, rtol=0)
    torch.testing.assert_close(actual_key, expected_key, atol=0.015625, rtol=0)
