"""GPU wave index for RetroInfer's mean-centered segmented clustering.

The training and final assignment follow microsoft/RetrievalAttention at
03f912c6e917c380d9d90c5ec85bb0f161ba53ef (MIT). Assignment matrices
are chunked; the reverse index is a packed sort instead of a padded matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetroInferIndex:
    centroids: torch.Tensor  # [Hkv, C, D]
    value_sums: torch.Tensor  # [Hkv, C, D], float32
    sizes: torch.Tensor  # [Hkv, C], int32
    offsets: torch.Tensor  # [Hkv, C + 1], int32
    sorted_positions: torch.Tensor  # [Hkv, indexed tokens], int32
    indexed_start: int
    indexed_end: int
    retrieval_clusters: int
    estimation_clusters: int


def _assign_and_reduce(
    data: torch.Tensor,
    centers: torch.Tensor,
    values: torch.Tensor | None,
    *,
    chunk_tokens: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch, length, dim = data.shape
    clusters = centers.shape[1]
    assignments = torch.empty((batch, length), dtype=torch.int32, device=data.device)
    sums = torch.zeros((batch, clusters, dim), dtype=torch.float32, device=data.device)
    counts = torch.zeros((batch, clusters), dtype=torch.int32, device=data.device)
    value_sums = torch.zeros_like(sums) if values is not None else None
    for start in range(0, length, chunk_tokens):
        stop = min(start + chunk_tokens, length)
        chunk = data[:, start:stop]
        ids = torch.bmm(chunk, centers.transpose(1, 2)).argmax(dim=-1).to(torch.int32)
        assignments[:, start:stop] = ids
        counts.scatter_add_(1, ids.to(torch.int64), torch.ones_like(ids))
        expanded = ids.to(torch.int64)[..., None].expand(-1, -1, dim)
        sums.scatter_add_(1, expanded, chunk.float())
        if values is not None:
            assert value_sums is not None
            value_sums.scatter_add_(1, expanded, values[:, start:stop].float())
    return assignments, sums, counts, value_sums


def build_retroinfer_segment(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    start_position: int,
    avg_cluster_size: int,
    num_segments: int,
    retrieval_ratio: float,
    estimation_ratio: float,
    iterations: int = 10,
) -> RetroInferIndex:
    """Build an index over a contiguous logical token range on the GPU."""
    if keys.ndim != 3 or keys.shape != values.shape or not keys.is_cuda:
        raise ValueError("RetroInfer index requires matching CUDA [Hkv,L,D] KV tensors.")
    heads, length, dim = keys.shape
    if heads <= 0 or dim <= 0 or length < avg_cluster_size or iterations <= 0:
        raise ValueError("RetroInfer index dimensions and iteration count are invalid.")
    if num_segments <= 0 or num_segments > length:
        raise ValueError("RetroInfer index segment count is invalid.")
    clusters = (length // avg_cluster_size // num_segments) * num_segments
    if clusters <= 0:
        raise ValueError("RetroInfer index has no clusters.")
    mean = keys.float().mean(dim=1, keepdim=True)
    centered = (keys.float() - mean).to(keys.dtype)
    seeds = ((torch.arange(clusters, device=keys.device, dtype=torch.float32) + 0.5)
             * (length / clusters)).to(torch.int64)
    centers = centered.index_select(1, seeds)
    train_length = length // num_segments
    train_clusters = clusters // num_segments
    training = centered[:, :train_length * num_segments].reshape(
        heads * num_segments, train_length, dim
    )
    centers = centers.reshape(heads * num_segments, train_clusters, dim)
    for _ in range(iterations - 1):
        _, sums, counts, _ = _assign_and_reduce(training, centers, None)
        candidate = F.normalize(sums / counts.clamp_min(1)[..., None], dim=-1).to(keys.dtype)
        centers = torch.where(counts[..., None] > 0, candidate, centers)
    centers = centers.reshape(heads, clusters, dim)
    assignments, sums, sizes, value_sums = _assign_and_reduce(centered, centers, values)
    assert value_sums is not None
    centroids = ((sums / sizes.clamp_min(1)[..., None]) + mean).to(keys.dtype)
    offsets = F.pad(torch.cumsum(sizes, dim=1, dtype=torch.int32), (1, 0))
    sorted_positions = torch.argsort(assignments, dim=1, stable=True).to(torch.int32)
    sorted_positions += int(start_position)
    retrieved = min(max(round(clusters * retrieval_ratio), 1), clusters)
    estimated = min(round(clusters * estimation_ratio), clusters - retrieved)
    return RetroInferIndex(
        centroids.contiguous(), value_sums.contiguous(), sizes.contiguous(),
        offsets.contiguous(), sorted_positions.contiguous(),
        int(start_position), int(start_position) + length, retrieved, estimated,
    )


def append_retroinfer_index(old: RetroInferIndex | None, new: RetroInferIndex) -> RetroInferIndex:
    if old is None:
        return new
    if old.indexed_end != new.indexed_start:
        raise ValueError("RetroInfer index segments must be adjacent.")
    sizes = torch.cat((old.sizes, new.sizes), dim=1)
    return RetroInferIndex(
        torch.cat((old.centroids, new.centroids), dim=1).contiguous(),
        torch.cat((old.value_sums, new.value_sums), dim=1).contiguous(),
        sizes.contiguous(),
        F.pad(torch.cumsum(sizes, dim=1, dtype=torch.int32), (1, 0)).contiguous(),
        torch.cat((old.sorted_positions, new.sorted_positions), dim=1).contiguous(),
        old.indexed_start, new.indexed_end,
        old.retrieval_clusters + new.retrieval_clusters,
        old.estimation_clusters + new.estimation_clusters,
    )
