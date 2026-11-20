"""Bounded RoPE positions for PACE: delta re-rotation and the reference store.

Keys are stored post-RoPE at their physical position. A renumber moves a row's
resident keys from physical positions ``old`` to ``new``. The rotation by
``new - old`` is derived from a cos/sin table with the model's RoPE parameters
(``build_rope_cos_sin_table``), so it matches how each key was first rotated
up to float32 rounding, including YaRN-style magnitude scaling, which the
division by ``cos^2 + sin^2`` cancels.

Rotating a stored BF16/FP16 key again on every renumber compounds rounding.
The reference store keeps a copy of a key and the position it is valid at
from its first renumber on; later renumbers rotate that copy in float32, so a
key's error stays at about two roundings however many renumbers it survives.

Renumbers are rare (a few per 10K tokens). They run in ``finish_step``, outside
graph capture, and use host-synchronizing calls (``nonzero``, ``numel``) once
per layer per renumber. Bounded positions are also rejected with
``decode_graph=True`` at configuration (``configs/sparse.py``, ``_normalize_pace``).
Eager decode still goes through the graph-shaped static buffers
(``run_eager_static`` -> ``prepare_decode_graph_step``), which is why PACE
overrides those hooks.
"""
from __future__ import annotations

import torch

from sparseengine.layers.rotary_embedding import apply_rotary_emb


def delta_cos_sin(
    cos_sin_cache: torch.Tensor,
    old_pos: torch.Tensor,
    new_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-magnitude ``cos/sin(new - old)``, each ``[n, 1, D/2]`` float32.

    The table layout is ``cat(cos, sin)`` along the last dimension, as produced by
    ``build_rope_cos_sin_table`` and ``RotaryEmbedding.cos_sin_cache``; a change to
    that layout must be mirrored here.
    """
    cache = cos_sin_cache.reshape(cos_sin_cache.shape[0], -1).float()
    co, so = cache.index_select(0, old_pos.long()).chunk(2, dim=-1)
    cn, sn = cache.index_select(0, new_pos.long()).chunk(2, dim=-1)
    norm = co.square() + so.square()
    cos_d = (cn * co + sn * so) / norm
    sin_d = (sn * co - cn * so) / norm
    return cos_d.unsqueeze(1), sin_d.unsqueeze(1)


def rotate_keys(
    keys: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    old_pos: torch.Tensor,
    new_pos: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``keys`` ``[n, H, D]`` valid at ``old_pos`` to ``new_pos``.

    Full-width split-half RoPE only. This is a shape guard (table against keys);
    partial rotary is detected from the model config at startup
    (``PACECacheManager._pace_validate_rotary`` and configuration), because a
    table built from the head dimension always matches it.
    """
    table_dim = int(cos_sin_cache.reshape(cos_sin_cache.shape[0], -1).shape[-1])
    if table_dim != int(keys.shape[-1]):
        raise ValueError(
            "PACE re-rotation supports full-width split-half RoPE only: rotary table "
            f"dimension {table_dim} != key head dimension {int(keys.shape[-1])}."
        )
    cos_d, sin_d = delta_cos_sin(cos_sin_cache, old_pos, new_pos)
    return apply_rotary_emb(keys, cos_d, sin_d)


class PACEReferenceStore:
    """Per-KV-layer pool of reference keys, owned by row columns.

    Columns hold an entry index (``-1`` when none) in row-aligned state that
    follows H2O keep indices. Entries are released when their column is
    evicted or its sequence is freed, so a live entry always belongs to exactly
    one resident column.

    One pool serves every sequence on the layer. The default capacity covers one
    session's resident row; concurrent bounded sessions compete for entries, and
    columns that find none are rotated in place (counted as overflow). Size
    ``pace_reference_tokens`` as concurrent sessions x resident tokens to avoid it.
    """

    def __init__(
        self,
        *,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.keys = torch.zeros(
            (self.capacity, int(num_kv_heads), int(head_dim)), dtype=dtype, device=device,
        )
        self.positions = torch.zeros(self.capacity, dtype=torch.int64, device=device)
        self.in_use = torch.zeros(self.capacity, dtype=torch.bool, device=device)

    @staticmethod
    def nbytes(*, capacity: int, num_kv_heads: int, head_dim: int, element_size: int) -> int:
        per_entry = int(num_kv_heads) * int(head_dim) * int(element_size) + 8 + 1
        return int(capacity) * per_entry

    def release(self, entries: torch.Tensor) -> None:
        """Free the entries owned by evicted or freed columns. Columns without a
        reference carry ``-1`` and are ignored."""
        entries = entries[entries >= 0]
        if entries.numel():
            self.in_use[entries.long()] = False

    def reset(self) -> None:
        self.in_use.zero_()

    @torch.no_grad()
    def renumber(
        self,
        k_cache: torch.Tensor,
        slots: torch.Tensor,
        entries: torch.Tensor,
        old_pos: torch.Tensor,
        new_pos: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        priority: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Move ``k_cache[slots]`` from ``old_pos`` to ``new_pos``; return entries, overflow.

        Columns without an entry capture one while free entries remain; the
        rest are rotated in place from the stored key. When entries run short,
        columns with higher ``priority`` capture first (ties keep row order);
        without it, row order decides. Held entries are never preempted.

        A column that overflowed at an earlier renumber and captures an entry
        later copies its key as it is then, including the rounding it already
        accumulated; from then on that error stops growing.
        """
        slots = slots.long()
        entries = entries.clone()
        missing = (entries < 0).nonzero().squeeze(1)
        if priority is not None and missing.numel():
            order = torch.argsort(priority.index_select(0, missing), descending=True,
                                  stable=True)
            missing = missing.index_select(0, order)
        overflow = 0
        if missing.numel():
            free = (~self.in_use).nonzero().squeeze(1)
            take = min(int(free.numel()), int(missing.numel()))
            if take:
                columns, chosen = missing[:take], free[:take]
                self.keys[chosen] = k_cache.index_select(0, slots[columns])
                self.positions[chosen] = old_pos[columns].long()
                self.in_use[chosen] = True
                entries[columns] = chosen
            rest = missing[take:]
            overflow = int(rest.numel())
            if overflow:
                k_cache.index_copy_(
                    0,
                    slots[rest],
                    rotate_keys(
                        k_cache.index_select(0, slots[rest]),
                        cos_sin_cache,
                        old_pos[rest],
                        new_pos[rest],
                    ),
                )
        held = (entries >= 0).nonzero().squeeze(1)
        if held.numel():
            index = entries[held].long()
            k_cache.index_copy_(
                0,
                slots[held],
                rotate_keys(
                    self.keys.index_select(0, index),
                    cos_sin_cache,
                    self.positions.index_select(0, index),
                    new_pos[held],
                ),
            )
        return entries, overflow
