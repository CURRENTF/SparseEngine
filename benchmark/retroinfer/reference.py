"""Small, deliberately unoptimized oracle for RetroInfer's wave index.

This is a numerical reference for kernel and cache-manager validation. It does
not implement the CPU/GPU wave buffer and must not be used in serving.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WaveIndexReference:
    centroids: torch.Tensor  # [KV heads, clusters, head dimension]
    value_sums: torch.Tensor  # same shape as centroids
    sizes: torch.Tensor  # [KV heads, clusters]
    assignments: torch.Tensor  # [KV heads, indexed tokens]


def build_wave_index_reference(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    num_clusters: int,
    num_segments: int,
    iterations: int = 10,
) -> WaveIndexReference:
    """Match the author's mean-centered segmented training and global assignment.

    The dense similarity matrices here are intentional: this routine is only an
    independent oracle for small inputs, not the production index builder.
    """
    if keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("keys and values must have equal [KV heads, tokens, dim] shapes")
    heads, length, dim = keys.shape
    if heads <= 0 or length <= 0 or dim <= 0:
        raise ValueError("wave-index dimensions must be positive")
    if not 0 < num_segments <= length or not 0 < num_clusters <= length:
        raise ValueError("segment and cluster counts must be in [1, tokens]")
    if num_clusters % num_segments:
        raise ValueError("cluster count must be divisible by segment count")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    centered = keys.to(torch.float64) - keys.to(torch.float64).mean(dim=1, keepdim=True)
    values_f = values.to(torch.float64)
    seed = ((torch.arange(num_clusters, device=keys.device, dtype=torch.float64) + 0.5)
            * (length / num_clusters)).to(torch.long)
    centroids = centered[:, seed].clone()
    tokens_per_segment = length // num_segments
    clusters_per_segment = num_clusters // num_segments
    training = centered[:, :tokens_per_segment * num_segments].reshape(
        heads, num_segments, tokens_per_segment, dim
    )
    trained = centroids.reshape(heads, num_segments, clusters_per_segment, dim)

    for _ in range(iterations - 1):
        scores = torch.einsum("hsnd,hscd->hsnc", training, trained)
        assignment = scores.argmax(dim=-1)
        sums = torch.zeros_like(trained)
        sums.scatter_add_(
            2, assignment[..., None].expand(-1, -1, -1, dim), training
        )
        counts = torch.zeros(
            heads, num_segments, clusters_per_segment,
            dtype=torch.float64, device=keys.device,
        )
        counts.scatter_add_(2, assignment, torch.ones_like(assignment, dtype=torch.float64))
        candidate = sums / counts.clamp_min(1)[..., None]
        candidate = torch.nn.functional.normalize(candidate, dim=-1)
        trained = torch.where(counts[..., None] > 0, candidate, trained)

    trained = trained.reshape(heads, num_clusters, dim)
    assignments = torch.einsum("hnd,hcd->hnc", centered, trained).argmax(dim=-1)
    centroid_sums = torch.zeros_like(trained)
    centroid_sums.scatter_add_(1, assignments[..., None].expand(-1, -1, dim), centered)
    value_sums = torch.zeros_like(trained)
    value_sums.scatter_add_(1, assignments[..., None].expand(-1, -1, dim), values_f)
    sizes = torch.zeros(heads, num_clusters, dtype=torch.long, device=keys.device)
    sizes.scatter_add_(1, assignments, torch.ones_like(assignments))
    centroids = centroid_sums / sizes.clamp_min(1)[..., None]
    centroids += keys.to(torch.float64).mean(dim=1, keepdim=True)
    return WaveIndexReference(centroids, value_sums, sizes, assignments)


def wave_attention_reference(
    queries: torch.Tensor,
    indexed_keys: torch.Tensor,
    indexed_values: torch.Tensor,
    steady_keys: torch.Tensor,
    steady_values: torch.Tensor,
    index: WaveIndexReference,
    *,
    retrieval_clusters: int,
    estimation_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention output and ranked cluster IDs for one decode step.

    Inputs use [query heads, D] and [KV heads, tokens, D]. The ranking sums
    per-query-head softmax probabilities within each GQA group, as in the
    pinned author implementation. Estimated clusters contribute their value
    sums and size-weighted normalization; all other clusters are omitted.
    """
    if queries.ndim != 2 or indexed_keys.ndim != 3:
        raise ValueError("queries must be [query heads, dim]; KV must be [KV heads, tokens, dim]")
    if indexed_values.shape != indexed_keys.shape or steady_values.shape != steady_keys.shape:
        raise ValueError("key and value shapes must match within each zone")
    kv_heads, length, dim = indexed_keys.shape
    query_heads = queries.shape[0]
    if steady_keys.shape[0] != kv_heads or queries.shape[1] != dim or query_heads % kv_heads:
        raise ValueError("query and KV head layout mismatch")
    clusters = index.centroids.shape[1]
    if index.centroids.shape != (kv_heads, clusters, dim):
        raise ValueError("centroid layout mismatch")
    if index.value_sums.shape != index.centroids.shape or index.sizes.shape != (kv_heads, clusters):
        raise ValueError("wave-index metadata shape mismatch")
    if index.assignments.shape != (kv_heads, length):
        raise ValueError("wave-index assignment shape mismatch")
    if retrieval_clusters < 0 or estimation_clusters < 0 or retrieval_clusters + estimation_clusters > clusters:
        raise ValueError("retrieval and estimation cluster counts exceed the index")

    group = query_heads // kv_heads
    q = queries.to(torch.float64).reshape(kv_heads, group, dim)
    scale = dim ** -0.5
    scores = torch.einsum("hgd,hcd->hgc", q, index.centroids.to(torch.float64)) * scale
    scores = scores.masked_fill(index.sizes[:, None, :] == 0, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    ranking_scores = probabilities.sum(dim=1)
    ranked = ranking_scores.argsort(dim=-1, descending=True)

    outputs = []
    for head in range(kv_heads):
        retrieved = ranked[head, :retrieval_clusters]
        estimated = ranked[head, retrieval_clusters:retrieval_clusters + estimation_clusters]
        if retrieved.numel():
            selected = torch.isin(index.assignments[head], retrieved)
            exact_keys = torch.cat((steady_keys[head], indexed_keys[head, selected]), dim=0)
            exact_values = torch.cat((steady_values[head], indexed_values[head, selected]), dim=0)
        else:
            exact_keys = steady_keys[head]
            exact_values = steady_values[head]
        for query in q[head]:
            exact_logits = exact_keys.to(torch.float64) @ query * scale
            estimated_logits = (index.centroids[head, estimated].to(torch.float64) @ query * scale)
            estimated_sizes = index.sizes[head, estimated].to(torch.float64)
            valid_estimate = estimated_sizes > 0
            estimated_logits = estimated_logits[valid_estimate]
            estimated_sizes = estimated_sizes[valid_estimate]
            estimated_sums = index.value_sums[head, estimated][valid_estimate].to(torch.float64)
            if not exact_logits.numel() and not estimated_logits.numel():
                raise ValueError("attention needs at least one exact or estimated entry")
            maximum = torch.cat((exact_logits, estimated_logits)).max()
            exact_weights = torch.exp(exact_logits - maximum)
            estimate_weights = torch.exp(estimated_logits - maximum)
            denominator = exact_weights.sum() + (estimate_weights * estimated_sizes).sum()
            numerator = exact_weights @ exact_values.to(torch.float64)
            numerator += estimate_weights @ estimated_sums
            outputs.append(numerator / denominator)
    return torch.stack(outputs).to(queries.dtype), ranked
