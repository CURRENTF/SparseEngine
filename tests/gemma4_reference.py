"""Torch providers used by Gemma 4 correctness tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sparseengine.layers.rotary_embedding import apply_rotary_emb
from sparseengine.operators.gemma4 import Gemma4OperatorProvider
from sparseengine.operators.gemma4_router import Gemma4RouterProvider


class TorchGemma4OperatorProvider(Gemma4OperatorProvider):
    """Explicit correctness oracle; never selected for production inference."""

    name = "torch_oracle"

    def attention_backend(self, *, sliding_window: int | None):
        from sparseengine.operators.gemma4_attention import Gemma4AttentionBackend

        return self._register_attention_backend(
            Gemma4AttentionBackend(sliding_window=sliding_window)
        )

    def rmsnorm(self, x, weight, eps):
        output = x.float()
        output *= torch.rsqrt(output.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            output *= weight.float()
        return output.to(x.dtype)

    def qkv_norm_rope(
        self, q, k, v, q_weight, k_weight, rope_cache, positions, eps
    ):
        q = self.rmsnorm(q, q_weight, eps)
        cos, sin = rope_cache[positions].chunk(2, -1)
        q = apply_rotary_emb(q, cos, sin)
        if k is not None:
            k = apply_rotary_emb(self.rmsnorm(k, k_weight, eps), cos, sin)
            v = self.rmsnorm(v, None, eps)
        return q, k, v

    def router_input(self, hidden_states, scale, root_size, eps):
        return self.rmsnorm(hidden_states, None, eps) * scale * root_size

    def gelu_tanh_and_mul(self, x):
        gate, up = x.chunk(2, -1)
        return F.gelu(gate, approximate="tanh") * up

    def gelu_mul(self, gate, value):
        return F.gelu(gate, approximate="tanh") * value

    def rmsnorm_residual(self, x, weight, residual, eps, scalar=None):
        output = self.rmsnorm(x, weight, eps) + residual
        return output if scalar is None else output * scalar


class TorchGemma4RouterProvider(Gemma4RouterProvider):
    """Explicit correctness oracle; never selected for production inference."""

    name = "torch_oracle"

    def topk(self, logits, per_expert_scale, top_k):
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, ids = probabilities.topk(top_k, dim=-1)
        weights.div_(weights.sum(-1, keepdim=True)).mul_(per_expert_scale[ids])
        return weights, ids
