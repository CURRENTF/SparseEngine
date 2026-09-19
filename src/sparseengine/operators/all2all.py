"""Expert-directed token transport through a prepared external provider.

Providers return token-major rows and global expert IDs, so expert weights and
compute providers are independent of the communication library's wire layout.
Dispatch handles are opaque, per-invocation state, never cached routing plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class AllToAllOpSpec:
    world_size: int
    hidden_size: int
    num_experts: int
    top_k: int
    max_local_tokens: int
    dtype: torch.dtype
    cuda_graph: bool

    def __post_init__(self):
        if (
            min(
                self.world_size,
                self.hidden_size,
                self.num_experts,
                self.top_k,
                self.max_local_tokens,
            )
            <= 0
        ):
            raise ValueError("All-to-all dimensions must be positive.")
        if self.num_experts % self.world_size or self.top_k > self.num_experts:
            raise ValueError(
                "All-to-all requires evenly partitioned experts and valid top-k."
            )


@dataclass(frozen=True)
class ExpertDispatch:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    handle: object


class PreparedAllToAllOp(Protocol):
    name: str

    def dispatch(
        self, hidden_states, topk_ids, topk_weights, *, capacity: int
    ) -> ExpertDispatch: ...

    def combine(
        self, output: torch.Tensor, dispatch: ExpertDispatch
    ) -> torch.Tensor: ...

    def close(self) -> None: ...


def check_all2all_dependency():
    from sparseengine.kernels.external.deepep import load_deepep_v1

    return load_deepep_v1()


def prepare_parallel_all2all(spec, *, group, device_index) -> PreparedAllToAllOp:
    # All2all explicitly selects the external implementation; an incompatible
    # installation is a startup error, never a request to choose AG/RS.
    from sparseengine.kernels.external.deepep import DeepEPV1Normal

    return DeepEPV1Normal(spec, group=group, device_index=device_index)
