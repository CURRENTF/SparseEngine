"""Palu projection factorization and model-side binding (TP=1)."""

import hashlib

import torch
from torch import nn

from sparseengine.engine.cache_manager.base import LowRankKVWrite
from sparseengine.layers.attention import Attention
from sparseengine.utils.context import get_context


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def factorize_grouped(weight, group_size, head_dim, rank, scale=None):
    """Return A[groups, rank, hidden], B[heads, rank, dim], W ~= B A."""
    heads, hidden = weight.shape[0] // head_dim, weight.shape[1]
    down, up = [], []
    for block in weight.float().reshape(heads // group_size, group_size * head_dim, hidden):
        scaled = block if scale is None else block @ scale
        u, s, vh = torch.linalg.svd(scaled, full_matrices=False)
        root = s[:rank].sqrt()
        right = vh[:rank]
        if scale is not None:
            # W L ~= U S Vt, hence W ~= U S (Vt L^-1).
            right = torch.linalg.solve_triangular(scale.T, right.T, upper=True).T
        down.append(root[:, None] * right)
        up.append((u[:, :rank] * root).reshape(group_size, head_dim, rank).transpose(1, 2))
    return torch.stack(down).to(weight.dtype).contiguous(), torch.cat(up).to(weight.dtype).contiguous()


def fuse_value_output(output_weight, value_up, query_heads):
    """Fuse each query head's V reconstruction into its O projection block."""
    kv_heads, rank, dim = value_up.shape
    blocks = output_weight.float().reshape(output_weight.shape[0], query_heads, dim)
    up = value_up.float().repeat_interleave(query_heads // kv_heads, dim=0)
    return torch.einsum('oqd,qrd->oqr', blocks, up).reshape(output_weight.shape[0], query_heads * rank).to(output_weight.dtype).contiguous()


class PaluAttention(Attention):
    def forward(self, q, zk, zv, positions, weights):
        context = get_context()
        manager = context.cache_manager
        layer = context.now_layer_idx
        manager.store_attention_payload(layer, LowRankKVWrite(zk, zv, positions))
        controller = context.sparse_controller
        selection = (controller.get_prefill_selection(layer) if context.is_prefill
                     else controller.get_decode_selection(layer, q))
        view = manager.build_decode_compute_view(layer, q, selection,
                    num_heads=self.num_heads, num_kv_heads=self.num_kv_heads)
        if context.is_prefill:
            return self.prefill_op.run(q, view, weights, manager.prefill_plan)
        return self.decode_op.run(q, view, weights)


class PaluSelfAttention(nn.Module):
    def __init__(self, original, owner, rk, rv):
        super().__init__()
        self.qkv_proj, self.o_proj = original.qkv_proj, original.o_proj
        self.rotary_emb = original.rotary_emb
        self.q_norm = getattr(original, 'q_norm', None)
        self.k_norm = getattr(original, 'k_norm', None)
        self.q_heads, self.kv_heads, self.dim = original.num_heads, original.num_kv_heads, original.head_dim
        self.rk, self.rv = rk, rv
        self.groups = self.kv_heads // owner.group_size
        self.proj_chunk_size = original.proj_chunk_size
        self.attn = PaluAttention(self.q_heads, self.dim, original.scaling, self.kv_heads)
        self.attn.full_attention_provider = owner
        self.attn.prefill_op, self.attn.decode_op = owner.phase_ops(rk, rv)
        self.register_buffer('key_up', None)
        self.ready = False

    @torch.no_grad()
    def prepare_weights(self, tensors, layer, fingerprints):
        prefix = f'model.layers.{layer}.self_attn.'
        qsize, ksize = self.q_heads * self.dim, self.kv_heads * self.dim
        original_k, original_v = self.qkv_proj.weight[qsize:].split(ksize)
        for name, tensor in [('k_proj.weight', original_k), ('v_proj.weight', original_v)]:
            if tensor_digest(tensor) != fingerprints.get(prefix + name):
                raise ValueError(f'Palu source weight fingerprint mismatch: {prefix + name}; use the original model and dtype.')
        dtype, device = original_k.dtype, original_k.device
        ak, bk, av, bv = (tensors[f'layers.{layer}.{name}'].to(device=device, dtype=dtype)
                          for name in ('key_down', 'key_up', 'value_down', 'value_up'))
        expected = [(self.groups, self.rk, original_k.shape[1]), (self.kv_heads, self.rk, self.dim),
                    (self.groups, self.rv, original_k.shape[1]), (self.kv_heads, self.rv, self.dim)]
        if any(tuple(t.shape) != shape or not torch.isfinite(t).all() for t, shape in zip((ak, bk, av, bv), expected)):
            raise ValueError(f'Invalid Palu factor tensors at layer {layer}.')
        weight = torch.cat((self.qkv_proj.weight[:qsize], ak.flatten(0, 1), av.flatten(0, 1)))
        fused = fuse_value_output(self.o_proj.weight, bv, self.q_heads)
        # cuBLAS linear operations; no factorization or weight products during inference.
        self.qkv_proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=device, dtype=dtype)
        self.qkv_proj.weight = nn.Parameter(weight, requires_grad=False)
        self.o_proj = nn.Linear(fused.shape[1], fused.shape[0], bias=False, device=device, dtype=dtype)
        self.o_proj.weight = nn.Parameter(fused, requires_grad=False)
        self.key_up = bk.contiguous()
        self.ready = True

    def forward(self, positions, hidden_states):
        if not self.ready:
            raise RuntimeError('Palu factors have not been prepared.')
        q, zk, zv = self.qkv_proj(hidden_states).split(
            (self.q_heads * self.dim, self.groups * self.rk, self.groups * self.rv), dim=-1)
        q = q.view(-1, self.q_heads, self.dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        owner = self.attn.full_attention_provider
        q = owner.rotate_query(q, positions, self.rotary_emb.cos_sin_cache)
        weights = (self.key_up, None if self.k_norm is None else self.k_norm.weight,
                   self.rotary_emb.cos_sin_cache, 0. if self.k_norm is None else self.k_norm.eps)
        out = self.attn(q, zk.view(-1, self.groups, self.rk), zv.view(-1, self.groups, self.rv), positions, weights).flatten(1)
        if out.shape[0] <= self.proj_chunk_size:
            return self.o_proj(out)
        for start in range(0, out.shape[0], self.proj_chunk_size):
            end = min(start + self.proj_chunk_size, out.shape[0])
            hidden_states[start:end].copy_(self.o_proj(out[start:end]))
        return hidden_states
