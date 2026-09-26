import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from sparseengine.layers.mxfp4_experts import PackedMxfp4Experts


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
        parallel = SimpleNamespace(moe_tp_rank=0, moe_tp_size=1, moe_ep_rank=rank,
                                   moe_ep_size=ep_size)
        with torch.device(device):
            module = PackedMxfp4Experts(num_experts=experts, hidden_size=hidden,
                                        intermediate_size=intermediate, top_k=top_k,
                                        activation_limit=10, cuda_graph=True, max_num_tokens=rows,
                                        parallel_context=parallel)
        for expert in range(rank*local, (rank+1)*local):
            module.load_expert_weight(expert, "w3", raw13[expert, :intermediate],
                                      scale13[expert, :intermediate])
            module.load_expert_weight(expert, "w1", raw13[expert, intermediate:],
                                      scale13[expert, intermediate:])
            module.load_expert_weight(expert, "w2", raw2[expert], scale2[expert])
        tensors = (module.w13_weight, module.w2_weight, module.w13_scale_inv, module.w2_scale_inv)
        addresses = [t.data_ptr() for t in tensors]
        module.prepare_weights()
        assert addresses == [t.data_ptr() for t in tensors]
        with pytest.raises(RuntimeError):
            module.prepare_weights()
        providers.append(module)

    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16) * 8
    ids = torch.tensor([[0, 1], [2, 3], [0, 3]], device=device, dtype=torch.int32)
    weights = torch.rand(rows, top_k, device=device)
    weights.mul_(1.5 / weights.sum(-1, keepdim=True))

    def run():
        partials = [module(x, ids, weights) for module in providers]
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
