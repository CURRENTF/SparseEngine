import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def apply_penalties(
        self,
        logits: torch.Tensor,
        *,
        presence_penalties: list[float],
        repetition_penalties: list[float],
        presence_token_ids: list[torch.Tensor | None],
        repetition_token_ids: list[torch.Tensor | None],
    ) -> torch.Tensor:
        batch_size = int(logits.shape[0])
        inputs = (
            presence_penalties,
            repetition_penalties,
            presence_token_ids,
            repetition_token_ids,
        )
        if any(len(values) != batch_size for values in inputs):
            raise ValueError(
                "Sampling penalties must provide one value/token set per logits row."
            )
        if not any(
            presence_penalty != 0.0 or repetition_penalty != 1.0
            for presence_penalty, repetition_penalty in zip(
                presence_penalties,
                repetition_penalties,
            )
        ):
            return logits

        penalized_logits = logits.to(dtype=torch.float32, copy=True)
        for row, (
            presence_penalty,
            repetition_penalty,
            completion_ids,
            prompt_and_completion_ids,
        ) in enumerate(
            zip(
                presence_penalties,
                repetition_penalties,
                presence_token_ids,
                repetition_token_ids,
            )
        ):
            row_logits = penalized_logits[row]
            if repetition_penalty != 1.0 and prompt_and_completion_ids is not None:
                repeated_logits = row_logits[prompt_and_completion_ids]
                repeated_logits = torch.where(
                    repeated_logits > 0,
                    repeated_logits / repetition_penalty,
                    repeated_logits * repetition_penalty,
                )
                row_logits[prompt_and_completion_ids] = repeated_logits
            if presence_penalty != 0.0 and completion_ids is not None:
                row_logits[completion_ids] -= presence_penalty
        return penalized_logits

    def _sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        *,
        all_unfiltered: bool = False,
        max_top_k: int | None = None,
    ):
        logits = logits.float()
        greedy_mask = temperatures <= 1e-10
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        sampled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))

        if all_unfiltered:
            # The host already owns these sampling parameters; do not read a
            # device tensor to select this path. No sort/cumsum is necessary.
            probs = torch.softmax(sampled_logits, dim=-1)
            sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            return torch.where(greedy_mask, logits.argmax(dim=-1), sample_tokens)

        if max_top_k is not None:
            # Host-selected top-k-only batch; retain per-row truncation.
            values, indices = torch.topk(sampled_logits, k=max_top_k, dim=-1)
            positions = torch.arange(max_top_k, device=logits.device)
            limits = torch.where(greedy_mask, max_top_k, top_ks)
            values = values.masked_fill(positions[None, :] >= limits[:, None], -torch.inf)
            probs = torch.softmax(values, dim=-1)
            chosen = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)
            tokens = indices.gather(1, chosen[:, None]).squeeze(1)
            return torch.where(greedy_mask, logits.argmax(-1), tokens)

        sorted_logits, sorted_indices = torch.sort(sampled_logits, dim=-1, descending=True)
        vocab_size = sorted_logits.shape[-1]
        top_k_limits = torch.where(
            top_ks <= 0,
            torch.full_like(top_ks, vocab_size),
            top_ks.clamp(max=vocab_size),
        )
        top_k_positions = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
        top_k_mask = top_k_positions >= top_k_limits.unsqueeze(1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs > top_ps.unsqueeze(dim=1)
        top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()
        top_p_mask[:, 0] = False
        remove_mask = top_p_mask | top_k_mask
        sampled_logits = sorted_logits.masked_fill(remove_mask, -torch.inf)

        probs = torch.softmax(sampled_logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        sample_tokens = sorted_indices.gather(1, sample_tokens.unsqueeze(1)).squeeze(1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
        all_greedy: bool = False,
        all_unfiltered: bool = False,
        max_top_k: int | None = None,
    ):
        if all_greedy:
            return logits.argmax(dim=-1)
        if temperatures is None:
            raise ValueError("temperatures must be provided when all_greedy=False")
        if top_ps is None:
            raise ValueError("top_ps must be provided when all_greedy=False")
        if top_ks is None:
            top_ks = torch.zeros_like(temperatures, dtype=torch.int64)
        return self._sample(logits, temperatures, top_ps, top_ks, all_unfiltered=all_unfiltered, max_top_k=max_top_k)
