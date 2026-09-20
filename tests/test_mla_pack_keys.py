import pytest
import torch

from sparseengine.kernels.triton.mla.pack_keys import pack_mla_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "tokens,heads,nope,rope",
    [(0, 5, 192, 64), (17, 5, 192, 64), (129, 20, 192, 64), (3, 2, 7, 6)],
)
def test_pack_strided_keys_and_shared_rope(dtype, tokens, heads, nope, rope):
    torch.manual_seed(341)
    projected = torch.randn(tokens, heads, nope + 256, device="cuda", dtype=dtype)
    kn = projected[..., :nope]
    # RoPE is also permitted to be a strided view.
    kr = torch.randn(tokens, rope * 2, device="cuda", dtype=dtype)[:, ::2]
    expected = torch.cat((kn, kr[:, None].expand(-1, heads, -1)), -1)
    torch.testing.assert_close(pack_mla_keys(kn, kr), expected, atol=0, rtol=0)
    if not tokens:
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = pack_mla_keys(kn, kr)
    for _ in range(3):
        projected.normal_()
        kr.normal_()
        graph.replay()
        expected = torch.cat((kn, kr[:, None].expand(-1, heads, -1)), -1)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pack_casts_rope_to_projection_dtype():
    kn = torch.randn(13, 5, 192, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(13, 64, device="cuda", dtype=torch.float32)
    expected = torch.cat((kn, rope.to(kn.dtype)[:, None].expand(-1, 5, -1)), -1)
    torch.testing.assert_close(pack_mla_keys(kn, rope), expected, atol=0, rtol=0)
