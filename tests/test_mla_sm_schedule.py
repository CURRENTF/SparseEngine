"""Protect dynamic partition/workspace safety across one captured MLA graph."""
import pytest
import torch

from sparseengine.kernels.triton.mla import (
    allocate_mla_decode_workspace, run_mla_decode, select_glm_mla_decode_config,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads,per_head", [(20, False), (10, True), (5, False)])
def test_sm_schedule_replay_ragged_padding_and_output_oracle(heads, per_head):
    # The old fixed-config tests cannot expose underallocation or stale dynamic
    # splits when active request counts change under this prepared schedule.
    torch.manual_seed(91)
    batch, capacity = 5, 4160
    packed = torch.randn(batch, heads, 576, device="cuda", dtype=torch.bfloat16)
    q, qr = packed[..., :512], packed[..., 512:]
    k = torch.randn(batch * capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(batch * capacity, 1, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(batch * capacity, device="cuda").view(batch, capacity).int()
    req = torch.arange(batch, device="cuda", dtype=torch.int32)
    lens = torch.full((batch,), capacity, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    score = torch.empty((batch, heads, capacity) if per_head else (batch, capacity), device="cuda")
    cfg = select_glm_mla_decode_config(batch_size=batch, local_q_heads=heads,
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
    ws = allocate_mla_decode_workspace(batch_size=batch, head_count=heads, device="cuda", config=cfg)
    tensors = [ws.block_size, ws.batch_start_indices, ws.mid_output, ws.mid_logsumexp, output, score]
    pointers = [t.data_ptr() for t in tensors]

    def run():
        run_mla_decode(q, qr, k, r, slots, req, lens, output, ws,
            softmax_scale=1/16, attn_score=score, config=cfg, validate_metadata=False)

    run(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths in ([4160]*5, [0, 1, 33, 257, 4160], [0]*5, [4160, 2048, 0, 4095, 64]):
        lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        req.copy_(torch.tensor([i if n else -1 for i, n in enumerate(lengths)], device="cuda", dtype=torch.int32))
        ws.mid_output.fill_(float("nan")); ws.mid_logsumexp.fill_(float("nan")); score.fill_(float("nan"))
        graph.replay(); torch.cuda.synchronize()
        block = int(ws.block_size)
        counts = [(n + block - 1)//block for n in lengths]
        assert sum(counts) <= ws.mid_output.shape[1]
        starts = ws.batch_start_indices.tolist()
        assert starts == [sum(counts[:i]) for i in range(batch)]
        for i, n in enumerate(lengths):
            if n:
                ids = slots[i, :n].long()
                latent, rope = k[ids, 0].float(), r[ids, 0].float()
                logits = q[i].float() @ latent.T + qr[i].float() @ rope.T
                expected = (logits/16).softmax(-1) @ latent
                torch.testing.assert_close(output[i].float(), expected, rtol=.02, atol=.02)
                expected_score = logits if per_head else logits.amax(0)
                torch.testing.assert_close(score[i, ..., :n], expected_score, rtol=.002, atol=.02)
            else:
                assert torch.count_nonzero(output[i]) == 0
            assert torch.all(score[i, ..., n:] == -1e20)
        assert [t.data_ptr() for t in tensors] == pointers
