"""DeepEP V1 public normal NVLink API adapter; no repository all-to-all kernels."""

from __future__ import annotations

import importlib
import inspect
import socket
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.distributed as dist
from packaging.version import Version

from sparseengine import platforms
from sparseengine.operators.all2all import ExpertDispatch
from sparseengine.platforms.interface import PlatformEnum


def load_deepep_v1():
    message = (
        "all2all requires DeepEP V1 >=1.2.1,<2 built for this Torch/CUDA environment"
    )
    try:
        installed = Version(version("deep_ep"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"{message}; optional package deep_ep is not installed."
        ) from exc
    if installed.major != 1 or installed < Version("1.2.1"):
        raise RuntimeError(f"{message}; found {installed}.")
    try:
        module = importlib.import_module("deep_ep")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{message}; importing the installed extension failed: {exc}"
        ) from exc
    required = {
        "__init__": {
            "num_nvl_bytes",
            "num_rdma_bytes",
            "low_latency_mode",
            "explicitly_destroy",
        },
        "dispatch": {"num_worst_tokens", "topk_idx", "topk_weights", "async_finish"},
        "combine": {"handle", "async_finish"},
        "get_dispatch_layout": {"topk_idx", "num_experts"},
    }
    buffer = getattr(module, "Buffer", None)
    for method, arguments in required.items():
        fn = getattr(buffer, method, None)
        if fn is None or not arguments <= inspect.signature(fn).parameters.keys():
            raise RuntimeError(
                f"{message}; Buffer.{method} lacks the required public API."
            )
    for method in (
        "destroy",
        "get_dispatch_config",
        "get_combine_config",
        "is_sm90_compiled",
    ):
        if not callable(getattr(buffer, method, None)):
            raise RuntimeError(f"{message}; Buffer.{method} is unavailable.")  # noqa: TRY004
    return module, str(installed)


class DeepEPV1Normal:
    name = "deepep_v1_normal_nvlink"

    def __init__(self, spec, *, group, device_index):
        self.spec = spec
        self.group = group
        self.buffer = None
        platform = platforms.current_platform
        caps = platform.get_device_caps(device_index)
        # Agree on startup failures before any peer creates an IPC buffer.
        error = None
        module = None
        try:
            module, self.version = load_deepep_v1()
            if caps.platform != PlatformEnum.CUDA or caps.compute_capability < (8, 0):
                raise ValueError("requires CUDA Ampere or newer with NVLink")
            if spec.dtype != torch.bfloat16 or spec.hidden_size % 256:
                raise ValueError(
                    "requires BF16 activations and hidden size divisible by 256"
                )
            if (
                spec.world_size != group.size
                or group.size not in (2, 4, 8)
                or str(dist.get_backend(group.process_group)) != "nccl"
            ):
                raise ValueError(
                    "requires a matching single-host NCCL EP group of 2, 4 or 8 ranks"
                )
            if module.Buffer.is_sm90_compiled() and caps.compute_capability < (9, 0):
                raise ValueError("installed extension requires SM90; rebuild for SM80")
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
        peers = [None] * group.size
        dist.all_gather_object(
            peers,
            (socket.gethostname(), device_index, error),
            group=group.process_group,
        )
        errors = [f"rank {i}: {p[2]}" for i, p in enumerate(peers) if p[2]]
        if errors:
            raise RuntimeError(
                "DeepEP normal startup validation failed: " + "; ".join(errors)
            )
        if len({p[0] for p in peers}) != 1 or len({p[1] for p in peers}) != group.size:
            raise RuntimeError(
                "DeepEP V1 normal requires distinct devices on one host."
            )
        error = None
        try:
            if not platform.supports_nvlink_group(tuple(p[1] for p in peers)):
                error = "every EP device pair must support NVLink peer access"
        except RuntimeError as exc:
            error = str(exc)
        errors = [None] * group.size
        dist.all_gather_object(errors, error, group=group.process_group)
        if any(errors):
            raise RuntimeError(f"DeepEP normal NVLink validation failed: {errors}")

        self.dispatch_config = module.Buffer.get_dispatch_config(group.size)
        self.combine_config = module.Buffer.get_combine_config(group.size)
        hidden_bytes = (
            spec.hidden_size * torch.tensor([], dtype=spec.dtype).element_size()
        )
        self.buffer = module.Buffer(
            group.process_group,
            num_nvl_bytes=max(
                c.get_nvl_buffer_size_hint(hidden_bytes, group.size)
                for c in (self.dispatch_config, self.combine_config)
            ),
            num_rdma_bytes=0,
            low_latency_mode=False,
            explicitly_destroy=True,
        )
        self.device_index = device_index
        self.expert_offset = group.rank * (spec.num_experts // group.size)

    def dispatch(self, hidden_states, topk_ids, topk_weights, *, capacity):
        spec = self.spec
        if self.buffer is None:
            raise RuntimeError("DeepEP communication buffer is closed.")
        if (
            not 0 < capacity <= spec.max_local_tokens
            or hidden_states.ndim != 2
            or hidden_states.shape[0] > capacity
            or hidden_states.shape[1] != spec.hidden_size
            or hidden_states.dtype != spec.dtype
            or hidden_states.device.type != "cuda"
            or hidden_states.device.index != self.device_index
            or topk_ids.device != hidden_states.device
            or topk_weights.device != hidden_states.device
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or not topk_weights.is_floating_point()
            or topk_ids.shape != (hidden_states.shape[0], spec.top_k)
            or topk_weights.shape != topk_ids.shape
        ):
            raise ValueError(
                "DeepEP dispatch exceeds its prepared tensor/capacity contract."
            )
        ids = topk_ids.to(torch.int64).contiguous()
        weights = topk_weights.to(torch.float32).contiguous()
        per_rank, _, per_expert, in_rank, _ = self.buffer.get_dispatch_layout(
            ids, spec.num_experts
        )
        x, ids, weights, _, handle, _ = self.buffer.dispatch(
            hidden_states.contiguous(),
            topk_idx=ids,
            topk_weights=weights,
            num_tokens_per_rank=per_rank,
            num_tokens_per_expert=per_expert,
            is_token_in_rank=in_rank,
            num_worst_tokens=capacity * spec.world_size,
            config=self.dispatch_config,
            async_finish=False,
        )
        # V1 guarantees -1 IDs for unused rows, but leaves their weights and
        # activations uninitialized. Never pass garbage/NaN weights to an expert.
        valid = ids >= 0
        weights = torch.where(valid, weights, 0)
        ids = torch.where(valid, ids + self.expert_offset, -1)
        return ExpertDispatch(x, ids, weights, handle)

    def combine(self, output, dispatch):
        if self.buffer is None:
            raise RuntimeError("DeepEP communication buffer is closed.")
        if (
            output.shape != dispatch.hidden_states.shape
            or output.dtype != self.spec.dtype
            or output.device != dispatch.hidden_states.device
        ):
            raise ValueError("DeepEP combine requires the dispatched BF16 row layout.")
        # Expert compute already applies route weights exactly once.
        result, _, _ = self.buffer.combine(
            output.contiguous(),
            dispatch.handle,
            config=self.combine_config,
            async_finish=False,
        )
        return result

    def close(self):
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
