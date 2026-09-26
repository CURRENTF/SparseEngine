import pytest
import torch
import torch.nn.functional as F

from sparseengine.operators.moe import MoeOpSpec, resolve_moe_provider

pytestmark = pytest.mark.cuda


def _dequantize(weights, scales):
    codes = torch.stack((weights & 15, weights >> 4), -1).flatten(-2)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=weights.device, dtype=torch.float64)
    values = grid[(codes & 7).long()] * torch.where(codes < 8, 1., -1.)
    return values * torch.exp2(scales.view(torch.uint8).double() - 127).repeat_interleave(32, -1)


def _reference(x, ids, weights, w13, w2, limit=10):
    result = torch.zeros_like(x, dtype=torch.float64)
    intermediate = w13.shape[1] // 2
    for row in range(x.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[row, route])
            projected = x[row].double() @ w13[expert].T
            up = projected[:intermediate].clamp(-limit, limit)
            gate = projected[intermediate:].clamp(max=limit)
            result[row] += weights[row, route] * ((F.silu(gate) * up) @ w2[expert].T)
    return result


@pytest.mark.parametrize("ep_size,hidden,intermediate", [(1, 512, 256), (2, 4096, 2048)])
def test_mxfp4_preparation_clipped_swiglu_ep_and_graph(ep_size, hidden, intermediate):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    torch.manual_seed(1701)
    device = torch.device("cuda:0")
    experts, rows, top_k = 4, 3, 2
    # Distinct groups/exponents catch scale byte order and interleave errors.
    raw13 = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2), device=device, dtype=torch.uint8)
    raw2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), device=device, dtype=torch.uint8)
    scale13 = torch.randint(119, 123, (experts, 2 * intermediate, hidden // 32), device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    scale2 = torch.randint(119, 123, (experts, hidden, intermediate // 32), device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    reference13, reference2 = _dequantize(raw13, scale13), _dequantize(raw2, scale2)
    providers = []
    local = experts // ep_size
    for rank in range(ep_size):
        spec = MoeOpSpec(experts, local, hidden, intermediate, top_k,
                         torch.bfloat16, torch.uint8, (1, 32), ep_size, True,
                         routing_method="sqrtsoftplus", scale_dtype=torch.float8_e8m0fnu,
                         activation="clipped_silu", activation_limit=10, max_num_tokens=rows)
        provider = resolve_moe_provider(spec, device_index=0)
        provider.prepare(spec, device=device, tp_rank=0, ep_rank=rank)
        shard = slice(rank * local, (rank + 1) * local)
        tensors = [t[shard].clone() for t in (raw13, raw2, scale13, scale2)]
        addresses = [t.data_ptr() for t in tensors]
        provider.prepare_weights(*tensors)
        assert addresses == [t.data_ptr() for t in tensors]
        with pytest.raises(RuntimeError):
            provider.prepare_weights(*tensors)
        providers.append((provider, spec, rank, tensors))

    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16) * 8
    ids = torch.tensor([[0, 1], [2, 3], [0, 3]], device=device, dtype=torch.int32)
    weights = torch.rand(rows, top_k, device=device)
    weights.mul_(1.5 / weights.sum(-1, keepdim=True))

    def run():
        partials = [p.run(spec, x, ids, weights, *tensors,
                          local_expert_start=rank * local, tp_rank=0, ep_rank=rank)
                    for p, spec, rank, tensors in providers]
        return sum(partials)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for new_ids in ([[0, 1], [2, 3], [0, 3]], [[3, 2], [1, 0], [2, 0]]):
        ids.copy_(torch.tensor(new_ids, device=device, dtype=torch.int32))
        x.normal_().mul_(8)
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(x, ids, weights, reference13, reference2)
        # Upstream W4A16 intermediate/output rounding differs from FP64.
        relative_error = (output.double() - expected).norm() / expected.norm()
        assert relative_error < .012
        unclipped = _reference(x, ids, weights, reference13, reference2, limit=1e9)
        assert (unclipped - expected).norm() > .1 * expected.norm()
