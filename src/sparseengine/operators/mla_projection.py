"""Latent value projection with token-major output for the next linear layer."""

import torch


def project_mla_values(latent_output: torch.Tensor, value_weight: torch.Tensor) -> torch.Tensor:
    """Map [tokens, heads, latent] through [heads, value, latent].

    Write directly to token-major storage. A head-major bmm result would need
    another copy when the output projection flattens the heads and value axes.
    Single-token output already has the desired layout without an out buffer.
    """
    inputs = latent_output.transpose(0, 1)
    weight = value_weight.transpose(1, 2)
    if latent_output.shape[0] == 1:
        return torch.bmm(inputs, weight).transpose(0, 1)
    if torch.is_grad_enabled() and (latent_output.requires_grad or value_weight.requires_grad):
        # The out= form is inference-only; preserve the ordinary Torch contract
        # for callers checking projections with autograd enabled.
        return torch.bmm(inputs, weight).transpose(0, 1).contiguous()
    output = torch.empty(
        (*latent_output.shape[:2], value_weight.shape[1]),
        dtype=latent_output.dtype, device=latent_output.device,
    )
    torch.bmm(inputs, weight, out=output.transpose(0, 1))
    return output
