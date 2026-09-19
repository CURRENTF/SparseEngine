"""Lazy TileLang adapter for GLM TP1/TP2/TP4 MLA decode.

Length and cache dimensions are symbolic. Compilation and split-KV workspaces
are cached by batch, launch configuration, and query layout; changing context
length does not create another compiled kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sparseengine.kernels.tilelang.support import tilelang_dependency_support


_VALID_SPLITS = (1, 2, 4, 8, 16, 32)
_SUPPORTED_VALID_HEADS = (5, 10, 20)
_SCORE_MODES = ("direct", "atomic", "partial", "per_head")
_HEAD_TILE_SIZE = 16
_LATENT_DIM = 512
_ROPE_DIM = 64
_BLOCK_N = 64
_SPLIT_RULE = "sm_parallel_nearest_v1"
_PROFILE_SPLITS = (4, 8, 16, 32)


def _padded_head_count(valid_heads: int) -> int:
    if valid_heads not in _SUPPORTED_VALID_HEADS:
        raise ValueError(
            "TileLang MLA valid_heads must be one of "
            f"{_SUPPORTED_VALID_HEADS}, got {valid_heads}."
        )
    return 32 if valid_heads > _HEAD_TILE_SIZE else _HEAD_TILE_SIZE


def tilelang_mla_support() -> tuple[bool, str]:
    """Check the optional package without importing or initializing TileLang."""

    return tilelang_dependency_support()


@dataclass(frozen=True, slots=True)
class TileMlaLaunchConfig:
    num_split: int
    block_n: int = _BLOCK_N
    block_h: int = _HEAD_TILE_SIZE
    score_mode: str = "direct"

    def __post_init__(self) -> None:
        if self.num_split not in _VALID_SPLITS:
            raise ValueError(
                f"TileLang MLA num_split must be one of {_VALID_SPLITS}, "
                f"got {self.num_split}."
            )
        if self.block_h not in (16, 32):
            raise ValueError(
                f"TileLang MLA block_h must be 16 or 32, got {self.block_h}."
            )
        if self.score_mode not in _SCORE_MODES:
            raise ValueError(
                f"TileLang MLA score_mode must be one of {_SCORE_MODES}, "
                f"got {self.score_mode!r}."
            )


@dataclass(frozen=True, slots=True)
class TileMlaLaunchPlan:
    """Capture-time TileLang variants for one model/device envelope."""

    context_capacity: int
    local_q_heads: int
    max_batch_size: int
    need_score: bool
    sm_count: int
    configs: tuple[TileMlaLaunchConfig, ...]

    @classmethod
    def build(
        cls,
        *,
        context_capacity: int,
        local_q_heads: int,
        max_batch_size: int,
        need_score: bool,
        sm_count: int,
        score_mode: str | None = None,
    ) -> TileMlaLaunchPlan:
        if min(context_capacity, max_batch_size) <= 0:
            raise ValueError(
                "TileLang MLA launch plan requires positive context and batch "
                f"capacities, got context={context_capacity} "
                f"batch={max_batch_size}."
            )
        configs = []
        for batch_size in range(1, int(max_batch_size) + 1):
            config = select_tile_mla_config(
                batch_size=batch_size,
                need_score=bool(need_score),
                local_q_heads=int(local_q_heads),
                sm_count=sm_count,
            )
            if score_mode is not None:
                config = TileMlaLaunchConfig(
                    num_split=config.num_split,
                    block_n=config.block_n,
                    block_h=config.block_h,
                    score_mode=score_mode,
                )
            configs.append(config)
        return cls(
            context_capacity=int(context_capacity),
            local_q_heads=int(local_q_heads),
            max_batch_size=int(max_batch_size),
            need_score=bool(need_score),
            sm_count=int(sm_count),
            configs=tuple(configs),
        )

    def config_for(self, batch_size: int, *, need_score: bool) -> TileMlaLaunchConfig:
        if bool(need_score) != self.need_score:
            raise ValueError(
                "TileLang MLA launch plan score contract changed after binding: "
                f"planned={self.need_score} requested={bool(need_score)}."
            )
        if not 0 < int(batch_size) <= self.max_batch_size:
            raise ValueError(
                "TileLang MLA batch exceeds the static launch plan: "
                f"batch={batch_size} max={self.max_batch_size}."
            )
        return self.configs[int(batch_size) - 1]

    def metadata(self) -> dict[str, object]:
        return {
            "context_capacity": self.context_capacity,
            "local_q_heads": self.local_q_heads,
            "max_batch_size": self.max_batch_size,
            "need_score": self.need_score,
            "split_rule": _SPLIT_RULE,
            "sm_count": self.sm_count,
            "target_ctas_per_sm": 1,
            "split_rounding": "nearest_absolute_ties_lower",
            "batch_configs": [
                {
                    "batch_size": batch_size,
                    "num_split": config.num_split,
                    "block_n": config.block_n,
                    "block_h": config.block_h,
                    "score_mode": config.score_mode,
                }
                for batch_size, config in enumerate(self.configs, start=1)
            ],
        }


def select_tile_mla_config(
    *,
    batch_size: int,
    need_score: bool,
    sm_count: int,
    local_q_heads: int = 10,
) -> TileMlaLaunchConfig:
    """Target one CTA per SM using batch/head parallelism, independent of context.

    Choose the nearest split count in the conservative 4..32 profile range;
    ties prefer fewer splits. This is a hardware-scaled heuristic, not a claim
    of measured optimality on every GPU. Bind before CUDA Graph capture.
    """

    if batch_size <= 0:
        raise ValueError(f"TileLang MLA batch must be positive, got {batch_size}.")
    if sm_count is None or sm_count <= 0:
        raise ValueError(f"TileLang MLA sm_count must be positive, got {sm_count}.")
    padded_heads = _padded_head_count(local_q_heads)
    block_h = 32 if local_q_heads == 20 and batch_size > 1 else 16
    head_tiles = padded_heads // block_h
    # Integer distance is equivalent to abs(split - SM_count / parallelism).
    parallelism = batch_size * head_tiles
    split = min(_PROFILE_SPLITS, key=lambda s: abs(s * parallelism - sm_count))
    score_mode = "direct"
    if need_score and local_q_heads == 20 and block_h == 16:
        score_mode = "partial"
    return TileMlaLaunchConfig(
        num_split=split,
        block_h=block_h,
        score_mode=score_mode,
    )


@dataclass(slots=True)
class TileMlaWorkspace:
    glse: torch.Tensor
    partial_output: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True, slots=True)
class _KernelKey:
    batch_size: int
    num_split: int
    block_h: int
    score_mode: str
    need_score: bool
    q_latent_strides: tuple[int, int, int]
    q_rope_strides: tuple[int, int, int]


@dataclass(slots=True)
class _BoundKernel:
    call: Callable[..., object]
    workspace: TileMlaWorkspace
    retired_scores: list[torch.Tensor] = field(default_factory=list)


class TileMlaDecodeKernel:
    """Batch/layout-bound GLM TP1/TP2/TP4 TileLang MLA runner."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        softmax_scale: float,
        valid_heads: int = 10,
        fixed_config: TileMlaLaunchConfig | None = None,
        launch_plan: TileMlaLaunchPlan | None = None,
        sm_count: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.softmax_scale = float(softmax_scale)
        self.valid_heads = int(valid_heads)
        self.padded_heads = _padded_head_count(self.valid_heads)
        if fixed_config is not None and launch_plan is not None:
            raise ValueError(
                "TileLang MLA accepts either fixed_config or launch_plan, not both."
            )
        if launch_plan is not None and launch_plan.local_q_heads != self.valid_heads:
            raise ValueError(
                "TileLang MLA launch plan head count does not match the runner: "
                f"plan={launch_plan.local_q_heads} runner={self.valid_heads}."
            )
        if fixed_config is not None:
            if self.padded_heads % fixed_config.block_h:
                raise ValueError(
                    "TileLang MLA padded heads must be divisible by block_h: "
                    f"padded_heads={self.padded_heads} "
                    f"block_h={fixed_config.block_h}."
                )
        self.fixed_config = fixed_config
        self.launch_plan = launch_plan
        if fixed_config is None and launch_plan is None and (sm_count is None or sm_count <= 0):
            raise ValueError("TileLang MLA without a fixed config or launch plan requires a positive sm_count.")
        self.sm_count = sm_count
        self._kernels: dict[_KernelKey, _BoundKernel] = {}

    def runtime_metadata(self) -> dict[str, object]:
        variants = []
        for key, bound in self._kernels.items():
            workspace_tensors = (
                bound.workspace.glse,
                bound.workspace.partial_output,
                bound.workspace.score,
            )
            variants.append(
                {
                    "batch_size": key.batch_size,
                    "num_split": key.num_split,
                    "block_h": key.block_h,
                    "score_mode": key.score_mode,
                    "need_score": key.need_score,
                    "q_latent_strides": key.q_latent_strides,
                    "q_rope_strides": key.q_rope_strides,
                    "workspace_score_capacity": bound.workspace.score.shape[-1],
                    "workspace_bytes": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in (*workspace_tensors, *bound.retired_scores)
                    ),
                    "workspace_data_ptrs": [
                        tensor.data_ptr() for tensor in (*workspace_tensors, *bound.retired_scores)
                    ],
                }
            )
        return {
            "compiled_variant_count": len(variants),
            "compiled_variants": variants,
        }

    def _config_for(
        self,
        *,
        batch_size: int,
        context_capacity: int,
        need_score: bool,
    ) -> TileMlaLaunchConfig:
        if self.launch_plan is not None:
            if context_capacity > self.launch_plan.context_capacity:
                raise ValueError(
                    "TileLang MLA runtime context exceeds the static launch plan: "
                    f"runtime={context_capacity} "
                    f"plan={self.launch_plan.context_capacity}."
                )
            return self.launch_plan.config_for(
                batch_size,
                need_score=need_score,
            )
        if self.fixed_config is not None:
            return self.fixed_config
        return select_tile_mla_config(
            batch_size=batch_size,
            need_score=need_score,
            local_q_heads=self.valid_heads,
            sm_count=self.sm_count,
        )

    def _bind(self, key: _KernelKey) -> _BoundKernel:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "TileLang MLA shape was not warmed before CUDA Graph capture: "
                f"{key}."
            )
        supported, reason = tilelang_mla_support()
        if not supported:
            raise RuntimeError(reason)
        # Importing TileLang can initialize its compiler, so keep it behind the
        # selected provider and outside module import/resolver paths.
        from sparseengine.kernels.tilelang.mla.decode import (
            build_glm_mla_decode_kernel,
        )

        config = TileMlaLaunchConfig(
            key.num_split,
            block_h=key.block_h,
            score_mode=key.score_mode,
        )
        kernel = build_glm_mla_decode_kernel(
            batch=key.batch_size,
            h_q=self.padded_heads,
            h_kv=1,
            valid_output_heads=self.valid_heads,
            dv=_LATENT_DIM,
            dpe=_ROPE_DIM,
            block_N=config.block_n,
            block_H=config.block_h,
            num_split=config.num_split,
            block_size=config.block_n,
            softmax_scale=self.softmax_scale,
            need_score=key.need_score,
            score_mode=config.score_mode,
            q_latent_strides=key.q_latent_strides,
            q_rope_strides=key.q_rope_strides,
        )
        dtype = torch.bfloat16
        workspace = TileMlaWorkspace(
            glse=torch.empty(
                key.batch_size,
                self.padded_heads,
                key.num_split,
                dtype=dtype,
                device=self.device,
            ),
            partial_output=torch.empty(
                key.batch_size,
                self.padded_heads,
                key.num_split,
                _LATENT_DIM,
                dtype=dtype,
                device=self.device,
            ),
            score=torch.empty(
                key.batch_size,
                (
                    self.valid_heads
                    if config.score_mode == "per_head"
                    else self.padded_heads // config.block_h
                    if config.score_mode == "partial"
                    else 1
                ),
                1,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        return _BoundKernel(call=kernel, workspace=workspace)

    def _validate(
        self,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        active_slots: torch.Tensor,
        request_indices: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        attn_score: torch.Tensor | None,
        max_context_len: int,
        config: TileMlaLaunchConfig,
    ) -> tuple[int, int]:
        batch_size = int(q_latent.shape[0])
        expected = {
            "q_latent": (batch_size, self.valid_heads, _LATENT_DIM),
            "q_rope": (batch_size, self.valid_heads, _ROPE_DIM),
            "output": (batch_size, self.valid_heads, _LATENT_DIM),
            "latent_cache": (int(latent_cache.shape[0]), 1, _LATENT_DIM),
            "rope_cache": (int(latent_cache.shape[0]), 1, _ROPE_DIM),
            "request_indices": (batch_size,),
            "context_lens": (batch_size,),
        }
        actual = {
            "q_latent": tuple(q_latent.shape),
            "q_rope": tuple(q_rope.shape),
            "output": tuple(output.shape),
            "latent_cache": tuple(latent_cache.shape),
            "rope_cache": tuple(rope_cache.shape),
            "request_indices": tuple(request_indices.shape),
            "context_lens": tuple(context_lens.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(
                    f"TileLang MLA {name} must have shape {shape}, "
                    f"got {actual[name]}."
                )
        if active_slots.ndim != 2:
            raise ValueError(
                "TileLang MLA active_slots must be 2D, got "
                f"{tuple(active_slots.shape)}."
            )
        tensors = {
            "q_latent": (q_latent, torch.bfloat16),
            "q_rope": (q_rope, torch.bfloat16),
            "latent_cache": (latent_cache, torch.bfloat16),
            "rope_cache": (rope_cache, torch.bfloat16),
            "active_slots": (active_slots, torch.int32),
            "request_indices": (request_indices, torch.int32),
            "context_lens": (context_lens, torch.int32),
            "output": (output, torch.bfloat16),
        }
        for name, (tensor, dtype) in tensors.items():
            if tensor.device != q_latent.device or tensor.dtype != dtype:
                raise TypeError(
                    f"TileLang MLA {name} must be {dtype} on "
                    f"{q_latent.device}, got {tensor.dtype} on {tensor.device}."
                )
            if name not in {"q_latent", "q_rope"} and not tensor.is_contiguous():
                raise ValueError(
                    f"TileLang MLA {name} must be contiguous, got stride "
                    f"{tuple(tensor.stride())}."
                )
        score_capacity = int(active_slots.shape[1])
        if attn_score is not None:
            expected_prefix = (
                (batch_size, self.valid_heads)
                if config.score_mode == "per_head"
                else (batch_size,)
            )
            if tuple(attn_score.shape[:-1]) != expected_prefix:
                raise ValueError(
                    "TileLang MLA attn_score shape does not match the static "
                    f"{config.score_mode!r} contract: expected prefix "
                    f"{expected_prefix}, got {tuple(attn_score.shape)}."
                )
            if (
                attn_score.dtype != torch.float32
                or attn_score.device != q_latent.device
            ):
                raise TypeError(
                    "TileLang MLA attn_score must be FP32 on the query device, "
                    f"got {attn_score.dtype} on {attn_score.device}."
                )
            if (
                not attn_score.is_contiguous()
                and config.score_mode != "per_head"
            ):
                raise ValueError(
                    "TileLang MLA attn_score must be contiguous, got stride "
                    f"{tuple(attn_score.stride())}."
                )
            score_capacity = int(attn_score.shape[-1])
        if score_capacity <= 0:
            raise ValueError(
                "TileLang MLA context/score capacity must be positive, got "
                f"{score_capacity}."
            )
        if int(max_context_len) > int(active_slots.shape[1]):
            raise ValueError(
                "TileLang MLA active slot width does not cover max_context_len: "
                f"max={max_context_len} slots={active_slots.shape[1]}."
            )
        if not 0 < int(max_context_len) <= score_capacity:
            raise ValueError(
                "TileLang MLA max_context_len must fit the context/score "
                f"capacity, got max={max_context_len} capacity={score_capacity}."
            )
        return batch_size, score_capacity

    @torch.no_grad()
    def __call__(
        self,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        active_slots: torch.Tensor,
        request_indices: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        *,
        attn_score: torch.Tensor | None,
        max_context_len: int,
    ) -> torch.Tensor:
        batch_size = int(q_latent.shape[0])
        need_score = attn_score is not None
        config = self._config_for(
            batch_size=batch_size,
            context_capacity=int(max_context_len),
            need_score=need_score,
        )
        batch_size, score_capacity = self._validate(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            attn_score,
            max_context_len,
            config,
        )
        key = _KernelKey(
            batch_size=batch_size,
            num_split=config.num_split,
            block_h=config.block_h,
            score_mode=config.score_mode,
            need_score=need_score,
            q_latent_strides=tuple(map(int, q_latent.stride())),
            q_rope_strides=tuple(map(int, q_rope.stride())),
        )
        bound = self._kernels.get(key)
        if bound is None:
            bound = self._bind(key)
            self._kernels[key] = bound

        workspace = bound.workspace
        score_output = workspace.score
        if attn_score is not None:
            if config.score_mode == "partial":
                if workspace.score.shape[-1] < score_capacity:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("TileLang MLA partial score capacity was not warmed before CUDA Graph capture.")
                    capacity = 1 << (score_capacity - 1).bit_length()
                    # Captured graphs may still reference the previous allocation.
                    bound.retired_scores.append(workspace.score)
                    workspace.score = torch.empty(
                        (*workspace.score.shape[:-1], capacity),
                        dtype=torch.float32, device=self.device,
                    )
                score_output = workspace.score[..., :score_capacity]
                score_output.fill_(-1e20)
            elif config.score_mode == "per_head":
                score_output = attn_score
                if not attn_score.is_contiguous():
                    score_output.fill_(-1e20)
            else:
                score_output = attn_score.unsqueeze(1)
        bound.call(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            workspace.glse,
            workspace.partial_output,
            output,
            score_output,
        )
        if attn_score is not None and config.score_mode == "partial":
            torch.amax(score_output, dim=1, out=attn_score)
        return output


__all__ = [
    "TileMlaDecodeKernel",
    "TileMlaLaunchConfig",
    "TileMlaLaunchPlan",
    "TileMlaWorkspace",
    "select_tile_mla_config",
    "tilelang_mla_support",
]
