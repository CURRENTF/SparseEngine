import pytest
import torch

from sparseengine.kernels.triton.h2o_score import h2o_headwise_softmax_accumulate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [129, 4225, 32769])
def test_mla_singleton_head_accumulation_masks_ragged_rows(width):
    """Catch long-row reduction errors and reading new-token history or padding.

    MLA supplies one already-reduced logit row, unlike explicit-KV headwise
    scoring. Compare that singleton-head adapter with direct Torch softmax.
    """
    torch.manual_seed(721)
    logits = torch.randn(2, 4, 2 * (width + 7), device="cuda")[:, :3, ::2]
    lengths = [[width, width // 2, 1], [width - 3, 17, 0]]
    history = torch.randn(2, 4, width + 11, device="cuda")[:, :3]
    expected = history.clone()
    scale = 192**-0.5
    for layer in range(2):
        for row, length in enumerate(lengths[layer]):
            if length:
                expected[layer, row, length - 1] = 0
                expected[layer, row, :length] += torch.softmax(
                    logits[layer, row, :length] * scale, -1
                )
                history[layer, row, length - 1] = torch.nan
            logits[layer, row, length:] = torch.nan

    h2o_headwise_softmax_accumulate(
        logits[..., :width].unsqueeze(2),
        history,
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        softmax_scale=scale,
    )
    torch.testing.assert_close(history, expected, atol=1e-6, rtol=1e-5)
