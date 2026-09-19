"""Packed page storage and its explicit, typed decode payload."""

from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import NormalDist

import torch

from ..base import ExplicitKVPayload
from .base import CacheLayout


def quantized_kv_reserved_bytes(config, *, num_layers, num_heads, head_dim):
    """Shared startup/allocation budget for persistent and bounded transient state."""
    g, h, d = int(config.kv_quant_page_size), num_heads, head_dim
    item = torch.empty((), dtype=config.hf_config.dtype).element_size()
    capacity = int(config.max_num_seqs_in_batch) * int(config.max_model_len)
    rows = int(config.max_num_seqs_in_gpu)
    workspace = 2 * capacity * h * d * (item + (8 if config.sparse_method == "turboquant" else 0))
    tail = 2 * num_layers * rows * g * h * d * item
    maps = (rows * int(config.max_model_len) + capacity) * 4
    scratch = 2 * (int(config.max_num_batched_tokens) + g) * h * d * (item + 4)
    return workspace + tail + maps + scratch + d * d * 4 + 64


@lru_cache(maxsize=3)
def gaussian_codebook(bits: int) -> tuple[float, ...]:
    if bits not in {2, 3, 4}:
        raise ValueError("TurboQuant codebook requires 2, 3, or 4 bits.")
    normal = NormalDist()
    count = 1 << bits
    centers = [normal.inv_cdf((i + 0.5) / count) for i in range(count)]
    # Lloyd-Max conditional centroids for a standard normal distribution.
    for _ in range(200):
        bounds = [-math.inf] + [(a + b) / 2 for a, b in zip(centers, centers[1:])] + [math.inf]
        updated = []
        for left, right in zip(bounds, bounds[1:]):
            phi_left = math.exp(-left * left / 2) / math.sqrt(2 * math.pi)
            phi_right = math.exp(-right * right / 2) / math.sqrt(2 * math.pi)
            updated.append((phi_left - phi_right) / (normal.cdf(right) - normal.cdf(left)))
        delta = max(abs(a - b) for a, b in zip(centers, updated))
        centers = updated
        if delta < 1e-9:
            break
    return tuple(centers)


def orthogonal_rotation(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float64, device="cpu")
    q, r = torch.linalg.qr(matrix)
    return (q * torch.where(r.diag() >= 0, 1.0, -1.0)).float()


@dataclass(frozen=True)
class QuantizedKVPayload(ExplicitKVPayload):
    format: str = ""
    bits: int = 0
    page_size: int = 32
    key_scale: torch.Tensor | None = None
    key_min: torch.Tensor | None = None
    value_scale: torch.Tensor | None = None
    value_min: torch.Tensor | None = None
    raw_key: torch.Tensor | None = None
    raw_value: torch.Tensor | None = None
    codebook: torch.Tensor | None = None
    rotation: torch.Tensor | None = None


class QuantizedKVStorage:
    layout = CacheLayout.EXPLICIT_KV

    def __init__(self, *, format: str, bits: int, page_size: int,
                 num_kv_heads: int, head_dim: int, dtype: torch.dtype, seed: int):
        if format not in {"kivi", "turboquant", "fp8_kv"}:
            raise ValueError(f"Unsupported quantized KV format {format!r}.")
        if (format == "kivi" and bits not in {2, 4}) or (format == "turboquant" and bits not in {2, 3, 4}):
            raise ValueError(f"Unsupported {format} bit width {bits}.")
        if page_size <= 0 or head_dim % page_size or num_kv_heads <= 0:
            raise ValueError("Quantized KV requires positive dimensions and head_dim divisible by page_size.")
        self.format, self.bits, self.page_size = format, bits, page_size
        self.num_kv_heads, self.head_dim, self.dtype = num_kv_heads, head_dim, dtype
        self.seed = seed
        self.packed_dim = head_dim if format == "fp8_kv" else math.ceil(head_dim / (32 // bits))
        self.tensors: tuple[torch.Tensor, ...] = ()

    def bytes_per_page_per_layer(self) -> int:
        g, h, d = self.page_size, self.num_kv_heads, self.head_dim
        data = 2 * g * h * self.packed_dim * (1 if self.format == "fp8_kv" else 4)
        # FP32 metadata avoids underflow for small vector magnitudes.
        metadata = 4 * h * d if self.format == "kivi" else 2 * g * h
        return data + metadata * 4

    def bytes_per_slot_per_layer(self) -> int:
        return math.ceil(self.bytes_per_page_per_layer() / self.page_size)

    def allocate(self, *, num_layers: int, num_slots: int, num_rows: int, device: torch.device) -> None:
        if num_slots <= 0 or num_slots % self.page_size:
            raise ValueError("Quantized KV allocation must contain whole pages.")
        self.num_slots = num_slots
        p, g, h, d = num_slots // self.page_size, self.page_size, self.num_kv_heads, self.head_dim
        packed_dtype = torch.float8_e4m3fn if self.format == "fp8_kv" else torch.int32
        self.data = torch.empty(2, num_layers, p, g, h, self.packed_dim, dtype=packed_dtype, device=device)
        self.raw = torch.empty(2, num_layers, num_rows, g, h, d, dtype=self.dtype, device=device)
        if self.format == "kivi":
            self.key_scale = torch.empty(num_layers, p, h, d, dtype=torch.float32, device=device)
            self.value_scale = torch.empty(num_layers, p, g, h, d // g, dtype=torch.float32, device=device)
            self.key_min = torch.empty_like(self.key_scale)
            self.value_min = torch.empty_like(self.value_scale)
        else:
            self.key_scale = torch.empty(num_layers, p, g, h, dtype=torch.float32, device=device)
            self.value_scale = torch.empty_like(self.key_scale)
            self.key_min = self.value_min = torch.empty(0, dtype=torch.float32, device=device)
        self.rotation = (orthogonal_rotation(d, self.seed).to(device)
                         if self.format == "turboquant" else None)
        self.codebook = torch.tensor(gaussian_codebook(self.bits) if self.format == "turboquant" else [0.0],
                                     dtype=torch.float32, device=device)
        self.tensors = (self.data, self.raw, self.key_scale, self.key_min,
                        self.value_scale, self.value_min, self.codebook)
        if self.rotation is not None:
            self.tensors += (self.rotation,)

    def layer_payload(self, layer_idx: int) -> QuantizedKVPayload:
        return QuantizedKVPayload(
            k_cache=self.data[0, layer_idx], v_cache=self.data[1, layer_idx],
            backend="quantized_pages", format=self.format, bits=self.bits, page_size=self.page_size,
            key_scale=self.key_scale[layer_idx], value_scale=self.value_scale[layer_idx],
            key_min=self.key_min[layer_idx] if self.format == "kivi" else self.key_min,
            value_min=self.value_min[layer_idx] if self.format == "kivi" else self.value_min,
            raw_key=self.raw[0, layer_idx], raw_value=self.raw[1, layer_idx],
            codebook=self.codebook, rotation=self.rotation,
        )

    def validate_slot_mapping(self, slots: torch.Tensor) -> None:
        if slots.ndim != 1 or slots.dtype != torch.int32 or slots.device != self.data.device:
            raise ValueError("Quantized KV slots must be 1D int32 on the cache device.")

    def validate_slot_mappings(self, mappings: tuple[torch.Tensor, ...]) -> None:
        for mapping in mappings:
            self.validate_slot_mapping(mapping)

    def slot_capacity(self) -> int:
        return self.num_slots

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.tensors
