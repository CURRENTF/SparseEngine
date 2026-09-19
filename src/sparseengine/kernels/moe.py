from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoeAlignment:
    sorted_token_ids: torch.Tensor | None
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    block_size: int
    naive: bool
