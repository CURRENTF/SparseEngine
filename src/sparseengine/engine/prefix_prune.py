from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Literal

import torch


PrefixPrunePolicy = Literal["snapkv_global", "kvzip_global"]
PrefixPruneStatus = Literal["queued", "running", "completed", "blocked", "failed"]


@dataclass(frozen=True)
class PrefixPruneRecord:
    prune_id: str
    policy: PrefixPrunePolicy
    range_start: int
    range_end: int
    original_tokens: int
    retained_tokens: int
    created_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "prune_id": self.prune_id,
            "policy": self.policy,
            "range": [int(self.range_start), int(self.range_end)],
            "original_tokens": int(self.original_tokens),
            "retained_tokens": int(self.retained_tokens),
            "quality_degraded": True,
            "created_at": float(self.created_at),
        }


@dataclass
class PrefixPruneJob:
    prune_id: str
    token_ids: list[int]
    range_start: int
    range_end: int
    keep_tokens: int
    policy: PrefixPrunePolicy
    allow_recompress: bool = False
    observation_tokens: int = 64
    score_chunk_size: int = 2048
    prev_postfix_size: int = 64
    status: PrefixPruneStatus = "queued"
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, object] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time()

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "prune_id": self.prune_id,
            "status": self.status,
            "policy": self.policy,
            "range": [int(self.range_start), int(self.range_end)],
            "keep_tokens": int(self.keep_tokens),
            "created_at": float(self.created_at),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.result is not None:
            value["result"] = self.result
        if self.error is not None:
            value["error"] = self.error
        return value


def validate_prefix_prune_request(
    *,
    token_count: int,
    range_start: int,
    range_end: int,
    keep_tokens: int,
    block_size: int,
    policy: str,
) -> PrefixPrunePolicy:
    if policy not in {"snapkv_global", "kvzip_global"}:
        raise ValueError(
            "prefix prune policy must be 'snapkv_global' or 'kvzip_global', "
            f"got {policy!r}."
        )
    if block_size <= 0:
        raise ValueError(f"prefix prune block_size must be positive, got {block_size}.")
    if range_start < 0 or range_end <= range_start or range_end > token_count:
        raise ValueError(
            "prefix prune range must satisfy 0 <= L < R <= selector length: "
            f"range=[{range_start}, {range_end}) selector_tokens={token_count}."
        )
    if range_start % block_size or range_end % block_size:
        raise ValueError(
            "prefix prune range must be block-aligned: "
            f"range=[{range_start}, {range_end}) block_size={block_size}."
        )
    width = range_end - range_start
    if keep_tokens < 0 or keep_tokens >= width:
        raise ValueError(
            "prefix prune keep_tokens must satisfy 0 <= keep_tokens < R-L: "
            f"keep_tokens={keep_tokens} width={width}."
        )
    return policy  # type: ignore[return-value]


def select_global_keep_indices(
    scores: torch.Tensor,
    *,
    keep_tokens: int,
    protected_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one deterministic, sorted mask shared by every layer/head/rank."""
    scores = scores.detach().float().reshape(-1)
    keep_tokens = int(keep_tokens)
    if keep_tokens < 0 or keep_tokens > int(scores.numel()):
        raise ValueError(
            f"invalid global keep budget: keep={keep_tokens} candidates={scores.numel()}."
        )
    protected: set[int] = set()
    if protected_indices is not None:
        for index in protected_indices.detach().cpu().reshape(-1).tolist():
            index = int(index)
            if index < 0 or index >= int(scores.numel()):
                raise ValueError(f"protected token index is out of range: {index}.")
            protected.add(index)
    if len(protected) > keep_tokens:
        raise ValueError(
            "protected prefix-prune tokens exceed keep budget: "
            f"protected={len(protected)} keep_tokens={keep_tokens}."
        )
    candidates = [index for index in range(int(scores.numel())) if index not in protected]
    # Python ordering makes equal-score selection stable by original token position.
    host_scores = scores.cpu().tolist()
    candidates.sort(key=lambda index: (-host_scores[index], index))
    selected = sorted(protected | set(candidates[: keep_tokens - len(protected)]))
    return torch.tensor(selected, dtype=torch.long, device=scores.device)
