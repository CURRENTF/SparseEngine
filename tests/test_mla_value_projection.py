import pytest
import torch

from sparseengine.operators.mla_projection import project_mla_values


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('tokens,heads', [(1, 20), (4, 10), (47, 20), (128, 5), (384, 10)])
def test_value_projection_token_major_matches_float32_and_graph(tokens, heads):
    torch.manual_seed(47)
    # Strided cache/projection slices are valid inputs, as in the real model.
    latent = torch.randn(tokens, heads, 1024, device='cuda', dtype=torch.bfloat16)[..., ::2]
    weight = torch.randn(heads, 448, 512, device='cuda', dtype=torch.bfloat16)[:, 192:]
    with torch.inference_mode():
        actual = project_mla_values(latent, weight)
        expected = torch.einsum('thr,hvr->thv', latent.float(), weight.float()).bfloat16()
        torch.testing.assert_close(actual, expected, atol=.125, rtol=.008)
        previous = torch.bmm(latent.transpose(0, 1), weight.transpose(1, 2)).transpose(0, 1)
        torch.testing.assert_close(actual, previous, atol=0, rtol=0)
        assert actual.is_contiguous()
        assert actual.flatten(1).data_ptr() == actual.data_ptr()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                project_mla_values(latent, weight)
        stream.synchronize()
        with torch.cuda.graph(graph):
            replay_output = project_mla_values(latent, weight)
        for _ in range(3):
            latent.mul_(.5)
            graph.replay()
            expected = torch.einsum('thr,hvr->thv', latent.float(), weight.float()).bfloat16()
            torch.testing.assert_close(replay_output, expected, atol=.125, rtol=.008)
