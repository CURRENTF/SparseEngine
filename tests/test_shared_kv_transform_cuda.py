import pytest
import torch

from sparseengine.engine.cache_manager.storage.packed_shared_kv import PackedSharedKVPool
from sparseengine.operators.shared_kv_transform import (
    SharedKVTransformOpSpec, resolve_shared_kv_transform_provider,
)

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA",
)]


def query_reference(q, positions, freqs):
    q = q.double()
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    cs = freqs[positions].double()[:, None]
    tail = q[..., -64:].reshape(*q.shape[:-1], 32, 2)
    cs = cs.reshape(len(q), 1, 32, 2)
    a, b = tail.unbind(-1)
    c, s = cs.unbind(-1)
    q[..., -64:] = torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2)
    return q.bfloat16()


def kv_reference(kv):
    nope = kv[:, :448].float().reshape(-1, 7, 64)
    scale = torch.exp2((nope.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil())
    nope = (nope / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
    return torch.cat((nope.flatten(1), kv[:, 448:].float()), -1).bfloat16()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_normalize_store_gather_changed_graph_inputs(dtype):
    torch.manual_seed(109)
    provider = resolve_shared_kv_transform_provider(
        SharedKVTransformOpSpec(64, torch.bfloat16, 1e-6),
        device_index=torch.cuda.current_device(),
    )
    pool = PackedSharedKVPool(num_pages=4, page_size=64, reserved_pages=1,
                              device=torch.device("cuda"))
    q = torch.randn(3, 64, 512, dtype=torch.bfloat16, device="cuda")
    values = torch.randn(3, 512, dtype=dtype, device="cuda")
    # Covers the quantization floor and BF16 rotary tail independently.
    values[0, :448] *= 1e-6
    slots = torch.tensor([4, 65, 193], dtype=torch.int32, device="cuda")
    positions = torch.tensor([0, 17, 89], device="cuda")
    angle = torch.randn(128, 32, device="cuda")
    freqs = torch.stack((angle.cos(), angle.sin()), -1).flatten(1)

    def run():
        rotated = provider.normalize_rotate_query(q, positions, freqs)
        provider.store(values, pool.byte_storage, slots)
        return rotated, provider.gather(pool.byte_storage, slots)

    for _ in range(2):
        rotated, gathered = run()
    torch.testing.assert_close(rotated, query_reference(q, positions, freqs), rtol=.02, atol=.02)
    torch.testing.assert_close(gathered[:, 0], kv_reference(values), rtol=.005, atol=.005)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rotated, gathered = run()
    q.copy_(torch.randn_like(q))
    values.copy_(torch.randn_like(values))
    slots.copy_(torch.tensor([129, 63, 71], device="cuda", dtype=torch.int32))
    positions.copy_(torch.tensor([127, 2, 30], device="cuda"))
    graph.replay()
    torch.testing.assert_close(rotated, query_reference(q, positions, freqs), rtol=.02, atol=.02)
    torch.testing.assert_close(gathered[:, 0], kv_reference(values), rtol=.005, atol=.005)


def test_inverse_rope_and_prepared_grouped_projection():
    from sparseengine.operators.shared_kv_transform import (
        GroupedSharedKVProjection, inverse_shared_kv_rope,
    )
    torch.manual_seed(119)
    values = torch.randn(3, 4, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([7, 19, 0], device="cuda")
    angles = torch.randn(32, 32, device="cuda")
    freqs = torch.stack((angles.cos(), angles.sin()), -1).flatten(1)
    # Independent complex arithmetic checks the interleaving and inverse sign.
    tail = torch.view_as_complex(values[..., -64:].float().reshape(3, 4, 32, 2))
    rotation = torch.view_as_complex(freqs[positions].reshape(3, 1, 32, 2))
    ref = torch.cat((values[..., :-64], torch.view_as_real(tail * rotation.conj()).flatten(-2).bfloat16()), -1)
    inverse = inverse_shared_kv_rope(values, positions, freqs)
    torch.testing.assert_close(inverse, ref, rtol=.005, atol=.015)
    weight = torch.randn(256, 1024, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-3, 3, (2, 8), device="cuda").float()).to(torch.float8_e8m0fnu)
    projection = GroupedSharedKVProjection(num_groups=2, input_size_per_group=1024,
                                           output_size_per_group=128)
    projection.prepare_weights(weight, scale)
    expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    dequant = (weight.float() * expanded).bfloat16().double().view(2, 128, 1024)
    inputs = inverse.reshape(3, 2, 1024)
    expected = torch.stack([inputs[:, g].double() @ dequant[g].T for g in range(2)], 1).flatten(1).bfloat16()
    actual = projection.forward(inputs)
    torch.testing.assert_close(actual, expected, rtol=.015, atol=.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = projection.forward(inverse_shared_kv_rope(values, positions, freqs).reshape(3, 2, 1024))
    values.copy_(torch.randn_like(values))
    graph.replay()
    expected_graph = projection.forward(inverse_shared_kv_rope(values, positions, freqs).reshape(3, 2, 1024))
    torch.testing.assert_close(result, expected_graph, rtol=0, atol=0)
