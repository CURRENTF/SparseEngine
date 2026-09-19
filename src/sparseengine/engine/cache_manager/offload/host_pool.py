"""Fixed host tensors with optional shared backing and stable physical views."""

from math import prod

import torch

from .allocation import plan_host_allocation


class HostTensorPool:
    """Own same-dtype host tensors without imposing KV or request semantics.

    Packing is opt-in: callers must share the lifetime of all packed views.
    Separate tensors can be retained/released independently by their owners.
    No pointer tables, streams, or device work are created by allocation.
    """

    def __init__(
        self, shapes, *, dtype: torch.dtype, pin_memory: bool = True,
        allow_packing: bool = False,
    ):
        shapes = tuple(tuple(shape) for shape in shapes)
        if any(dim < 0 for shape in shapes for dim in shape):
            raise ValueError("Host tensor dimensions must be non-negative.")
        counts = tuple(prod(shape) for shape in shapes)
        self.plan = plan_host_allocation(
            (count * dtype.itemsize for count in counts),
            allow_packing=allow_packing, pin_memory=pin_memory,
        )
        self.backing = None
        if self.plan.packed:
            self.backing = torch.empty(
                sum(counts), dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            offset = 0
            tensors = []
            for count, shape in zip(counts, shapes):
                tensors.append(self.backing[offset : offset + count].view(shape))
                offset += count
            self.tensors = tuple(tensors)
        else:
            self.tensors = tuple(
                torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
                for shape in shapes
            )
