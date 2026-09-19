import math
from functools import lru_cache

import torch
from torch import nn


def yarn_get_mscale(factor: float, multiplier: float = 1.0) -> float:
    factor = float(factor)
    if factor <= 1.0:
        return 1.0
    return 0.1 * float(multiplier) * math.log(factor) + 1.0


def _compute_rope_parameters(
    rotary_dim: int,
    base: float,
    rope_scaling: tuple[tuple[str, object], ...] | None,
) -> tuple[torch.Tensor, float]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    if rope_scaling is None:
        return inv_freq, 1.0

    scaling = dict(rope_scaling)
    rope_type = str(scaling.get("rope_type", scaling.get("type", "default"))).lower()
    factor = float(scaling["factor"])
    if rope_type == "linear":
        return inv_freq / factor, 1.0
    if rope_type == "llama3":
        low_freq_factor = float(scaling["low_freq_factor"])
        high_freq_factor = float(scaling["high_freq_factor"])
        old_context_len = float(scaling["original_max_position_embeddings"])

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / factor,
            inv_freq,
        )
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            (1 - smooth_factor) * inv_freq_llama / factor
            + smooth_factor * inv_freq_llama
        )
        is_medium_freq = ~(
            (wavelen < high_freq_wavelen) | (wavelen > low_freq_wavelen)
        )
        return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama), 1.0
    if rope_type == "yarn":
        beta_fast = float(scaling.get("beta_fast", 32.0))
        beta_slow = float(scaling.get("beta_slow", 1.0))
        old_context_len = float(scaling["original_max_position_embeddings"])
        truncate = bool(scaling.get("truncate", True))

        def correction_dim(num_rotations: float) -> float:
            return (
                rotary_dim
                * math.log(old_context_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = correction_dim(beta_fast)
        high = correction_dim(beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        low = max(low, 0)
        high = min(high, rotary_dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(rotary_dim // 2, dtype=torch.float32) - low)
            / (high - low),
            0,
            1,
        )
        extrapolation_weight = 1 - ramp
        inv_freq = (
            inv_freq / factor * (1 - extrapolation_weight)
            + inv_freq * extrapolation_weight
        )

        attention_factor = scaling.get("attention_factor")
        if attention_factor is None:
            mscale = scaling.get("mscale")
            mscale_all_dim = scaling.get("mscale_all_dim")
            if mscale and mscale_all_dim:
                attention_factor = yarn_get_mscale(factor, float(mscale)) / yarn_get_mscale(
                    factor,
                    float(mscale_all_dim),
                )
            else:
                attention_factor = yarn_get_mscale(factor)
        return inv_freq, float(attention_factor)
    raise NotImplementedError(f"Unsupported rope_scaling={scaling!r}.")


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def apply_interleaved_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate adjacent pairs and return the split-half layout used by GLM."""

    x_even = x.float()[..., 0::2]
    x_odd = x.float()[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_odd * cos + x_even * sin
    return torch.cat((y_even, y_odd), dim=-1).to(x.dtype)


def apply_partial_rotary_emb(
    rotary_emb: "RotaryEmbedding",
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to a leading Q/K slice and preserve the remaining features."""

    rotary_dim = int(rotary_dim)
    query_dim = int(query.shape[-1])
    key_dim = int(key.shape[-1])
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim}.")
    if rotary_dim > query_dim or rotary_dim > key_dim:
        raise ValueError(
            f"rotary_dim={rotary_dim} exceeds Q/K dimensions "
            f"query={query_dim}, key={key_dim}."
        )
    if rotary_dim == query_dim == key_dim:
        return rotary_emb(positions, query, key)

    cache_rotary_dim = int(rotary_emb.cos_sin_cache.shape[-1])
    if cache_rotary_dim != rotary_dim:
        raise RuntimeError(
            "Rotary cache dimension does not match partial rotary_dim: "
            f"cache={cache_rotary_dim}, rotary_dim={rotary_dim}."
        )
    if rotary_emb.backend == "flashinfer":
        return rotary_emb.flashinfer_forward(positions, query, key)

    cos_sin = rotary_emb.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query_rotated = apply_rotary_emb(query[..., :rotary_dim], cos, sin)
    key_rotated = apply_rotary_emb(key[..., :rotary_dim], cos, sin)
    return (
        torch.cat((query_rotated, query[..., rotary_dim:]), dim=-1),
        torch.cat((key_rotated, key[..., rotary_dim:]), dim=-1),
    )


def reverse_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对已应用 RoPE 的向量执行逆操作，恢复到位置无关状态。

    RoPE 公式:     y1 = x1*cos - x2*sin,  y2 = x2*cos + x1*sin
    De-RoPE 分子:  x1 = y1*cos + y2*sin,  x2 = y2*cos - y1*sin

    YaRN 可以给 cos/sin 缓存施加非单位幅值，因此逆变换需要再除以
    cos²+sin²。普通单位 RoPE 下该分母为 1。
    """
    y1, y2 = torch.chunk(x.float(), 2, dim=-1)
    norm = cos.float().square() + sin.float().square()
    x1 = (y1 * cos + y2 * sin) / norm
    x2 = (y2 * cos - y1 * sin) / norm
    return torch.cat((x1, x2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: tuple[tuple[str, object], ...] | None = None,
        backend: str = "flashinfer",
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        if backend not in {"flashinfer", "torch"}:
            raise ValueError(
                f"Unsupported RoPE backend={backend!r}; expected 'flashinfer' or 'torch'."
            )
        self.backend = backend
        self.interleaved = bool(interleaved)
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq, attention_scaling = _compute_rope_parameters(
            rotary_dim,
            base,
            rope_scaling,
        )
        self.attention_scaling = float(attention_scaling)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    @lru_cache(1)
    def _load_flashinfer_op():
        try:
            from flashinfer import apply_rope_with_cos_sin_cache
        except ImportError as exc:
            raise ImportError(
                "FlashInfer RoPE requires flashinfer-python and the JIT cache "
                "matching torch.version.cuda."
            ) from exc
        return apply_rope_with_cos_sin_cache

    @torch.compile(fullgraph=True, dynamic=True)
    def compiled_forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        apply = (
            apply_interleaved_rotary_emb
            if self.interleaved
            else apply_rotary_emb
        )
        query = apply(query, cos, sin)
        key = apply(key, cos, sin)
        return query, key

    def flashinfer_forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin_cache = self.cos_sin_cache.squeeze(1)
        head_size = int(query.shape[-1])
        query_flat = query.view(query.shape[0], -1)
        key_flat = key.view(key.shape[0], -1)
        apply_rope_with_cos_sin_cache = self._load_flashinfer_op()
        query_out, key_out = apply_rope_with_cos_sin_cache(
            positions=positions,
            query=query_flat,
            key=key_flat,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=not self.interleaved,
        )
        query_out = query_out.view_as(query)
        key_out = key_out.view_as(key)
        if self.interleaved:
            query_out = query_out.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
            key_out = key_out.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
        return query_out, key_out

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "flashinfer":
            return self.flashinfer_forward(positions, query, key)
        return self.compiled_forward(positions, query, key)


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: tuple[tuple[str, object], ...] | None = None,
    backend: str = "flashinfer",
    interleaved: bool = False,
):
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_scaling=rope_scaling,
        backend=backend,
        interleaved=interleaved,
    )
    return rotary_emb
