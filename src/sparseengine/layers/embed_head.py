import torch
from torch import nn
import torch.nn.functional as F

from sparseengine.distributed import get_parallel_context
from sparseengine.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        reduce_results: bool = True,
    ):
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.tp_rank = self.parallel_context.attn_tp_rank
        self.tp_size = self.parallel_context.attn_tp_size
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.reduce_results = bool(reduce_results)
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def rank_local_weight_slice(
        self,
        source_shape: tuple[int, ...],
        *,
        loaded_shard_id=None,
        is_scale: bool = False,
    ) -> tuple[slice, ...] | None:
        del loaded_shard_id, is_scale
        if self.tp_size == 1:
            return None
        shard_size = int(source_shape[0]) // self.tp_size
        start = self.tp_rank * shard_size
        return (slice(start, start + shard_size),) + (slice(None),) * (
            len(source_shape) - 1
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        if loaded_weight.size(0) != shard_size:
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
        return self.parallel_context.attn_tp.all_reduce(y) if self.reduce_results else y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = self.parallel_context.attn_tp.gather(logits)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
