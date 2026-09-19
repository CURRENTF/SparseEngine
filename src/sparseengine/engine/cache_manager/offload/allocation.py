"""Allocation planning for the PyTorch caching pinned-host allocator."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HostAllocationPlan:
    tensor_bytes: tuple[int, ...]
    packed: bool
    pin_memory: bool
    estimated_bytes: int

    @property
    def logical_bytes(self) -> int:
        return sum(self.tensor_bytes)


def plan_host_allocation(
    tensor_bytes, *, allow_packing: bool = False, pin_memory: bool = True
) -> HostAllocationPlan:
    """Pack only jointly owned tensors, and only when it reduces rounded cost.

    The power-of-two estimate belongs to the current PyTorch caching host
    backend. It is not a measurement of process RSS or a cross-platform rule.
    Unpinned allocations report logical bytes only.
    """
    sizes = tuple(tensor_bytes)
    if any(size < 0 for size in sizes):
        raise ValueError("Host allocation sizes must be non-negative.")

    def cost(size):
        return (1 << (size - 1).bit_length()) if pin_memory and size else size

    separate = sum(cost(size) for size in sizes)
    packed = bool(allow_packing and cost(sum(sizes)) < separate)
    return HostAllocationPlan(
        sizes, packed, pin_memory, cost(sum(sizes)) if packed else separate
    )
