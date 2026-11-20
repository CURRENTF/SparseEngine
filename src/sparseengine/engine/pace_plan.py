"""Per-request foresight plan consumed by ``sparse_method="pace"``.

A plan states which token spans future agent steps will read. Spans are
half-open ``[start, end)`` ranges in logical session coordinates: the token
index in the full (chain) history, independent of eviction or bounded
positions. The engine consumes a plan; it never produces one.

The bonus is added to H2O's float32 score vectors. The precision matters:
in BF16 a 5e4 floor rounds in steps of 256, which would erase the ordering
between near-future and far-future spans (planned tokens would still outrank
unplanned ones).
"""
from __future__ import annotations

import heapq
import math
import operator
from dataclasses import dataclass

import numpy as np
import torch


PACESpan = tuple[int, int, tuple[int, ...]]


def _require_int(name: str, value, *, minimum: int | None = None) -> int:
    """Any integer type (int, NumPy integers, ...) except bool."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"PACEPlan {name} must be an integer, got a bool.")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(
            f"PACEPlan {name} must be an integer, got {type(value).__name__}."
        ) from None
    if minimum is not None and value < minimum:
        raise ValueError(f"PACEPlan {name} must be >= {minimum}, got {value}.")
    return int(value)


@dataclass(frozen=True)
class PACEPlan:
    """Future references of token spans, relative to ``current_step``.

    ``spans`` items are ``(start, end, steps)``. ``start`` is clamped to 0,
    then empty or inverted spans are dropped; ``steps`` are sorted and
    deduplicated, and the spans themselves are sorted and deduplicated. Plans
    with the same spans in any order compare equal, so a caller that reorders
    spans between turns does not force a recompile.
    """

    current_step: int
    spans: tuple[PACESpan, ...] = ()

    def __post_init__(self) -> None:
        step = _require_int("current_step", self.current_step, minimum=0)
        normalized: list[PACESpan] = []
        for item in self.spans:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise TypeError("PACEPlan spans must be (start, end, steps) triples.")
            start = _require_int("span start", item[0])
            end = _require_int("span end", item[1])
            if isinstance(item[2], (str, bytes)):
                raise TypeError("PACEPlan span steps must be a collection of ints.")
            steps = tuple(sorted({
                _require_int("span step", s, minimum=0) for s in item[2]
            }))
            start = max(0, start)
            if end <= start:
                continue
            normalized.append((start, end, steps))
        object.__setattr__(self, "current_step", step)
        object.__setattr__(self, "spans", tuple(sorted(set(normalized))))


def span_weights(plan: PACEPlan, *, decay: float) -> list[tuple[int, int, float]]:
    """Return ``(start, end, weight)`` for spans with a future reference.

    ``weight = 0.5 + 0.5 * exp(-decay * (next_ref - current_step))`` rounded to
    float32, where ``next_ref`` is the first planned step at or after the
    current step. Spans whose last reference has passed contribute nothing.
    """
    t = plan.current_step
    out: list[tuple[int, int, float]] = []
    for start, end, steps in plan.spans:
        next_ref = next((s for s in steps if s >= t), None)
        if next_ref is None:
            continue
        weight = 0.5 + 0.5 * math.exp(-float(decay) * float(next_ref - t))
        out.append((start, end, float(np.float32(weight))))  # round to float32
    return out


@dataclass(frozen=True)
class CompiledPACEBonus:
    """Piecewise-constant bonus: ``values[j]`` on ``[breaks[j], breaks[j+1])``.

    Positions before ``breaks[0]`` and at or after the last break get 0, and
    so does any interval no span covers. Overlapping spans take the maximum
    weight, then the weight is multiplied by ``scale`` in float32.
    """

    breaks: torch.Tensor  # int64 [K], strictly increasing
    values: torch.Tensor  # float32 [K]; values[-1] == 0

    @property
    def empty(self) -> bool:
        return int(self.breaks.numel()) == 0

    def to(self, device: torch.device) -> "CompiledPACEBonus":
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self.breaks.device == device:
            return self
        return CompiledPACEBonus(self.breaks.to(device), self.values.to(device))

    def lookup(self, positions: torch.Tensor) -> torch.Tensor:
        """Return the float32 bonus for int64 logical ``positions``."""
        positions = positions.to(dtype=torch.int64)
        if self.empty:
            return torch.zeros(positions.shape, dtype=torch.float32, device=positions.device)
        index = torch.searchsorted(self.breaks, positions, right=True) - 1
        bonus = self.values.index_select(0, index.clamp(min=0).reshape(-1)).reshape(index.shape)
        return torch.where(index >= 0, bonus, torch.zeros_like(bonus))


def compile_bonus(plan: PACEPlan, *, scale: float, decay: float) -> CompiledPACEBonus:
    """Compile a plan into a piecewise-constant bonus in O(S log S).

    ``scale`` must be finite and positive and ``decay`` finite and non-negative:
    otherwise weights leave ``[0.5, 1]`` or the bonus no longer lifts planned
    tokens above unplanned ones.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"PACE bonus scale must be finite and positive, got {scale}.")
    if not math.isfinite(decay) or decay < 0.0:
        raise ValueError(f"PACE bonus decay must be finite and non-negative, got {decay}.")
    weighted = sorted(span_weights(plan, decay=decay))
    if not weighted:
        empty = torch.empty(0, dtype=torch.int64)
        return CompiledPACEBonus(empty, torch.empty(0, dtype=torch.float32))
    points = sorted({p for start, end, _ in weighted for p in (start, end)})
    active: list[tuple[float, int]] = []  # max-heap of (-weight, end)
    values: list[float] = []
    cursor = 0
    for point in points:
        while cursor < len(weighted) and weighted[cursor][0] <= point:
            _, end, weight = weighted[cursor]
            heapq.heappush(active, (-weight, end))
            cursor += 1
        while active and active[0][1] <= point:
            heapq.heappop(active)
        values.append(-active[0][0] if active else 0.0)
    breaks = torch.tensor(points, dtype=torch.int64)
    scaled = torch.tensor(values, dtype=torch.float32) * float(scale)
    return CompiledPACEBonus(breaks, scaled)
