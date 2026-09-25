from __future__ import annotations

import hashlib
import heapq
import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from sparseengine.method_registry import prefill_sparse_method_fingerprint
from sparseengine.utils.profiler import cpu_timing



def usable_prefix_cache_tokens(prompt_len: int, block_size: int) -> int:
    """Return the largest cache-hit prefix that still leaves logits work."""
    prompt_len = int(prompt_len)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"prefix cache block_size must be > 0, got {block_size}.")
    if prompt_len <= 1:
        return 0
    return ((prompt_len - 1) // block_size) * block_size


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return str(value)


def resolve_prefix_cache_block_size(config: Any) -> int:
    configured = getattr(config, "prefix_cache_block_size", None)
    if configured is not None and (isinstance(configured, bool) or not isinstance(configured, int)):
        raise ValueError(f"prefix_cache_block_size must be a positive integer, got {configured!r}.")
    method = str(getattr(config, "sparse_method", "") or "")
    if method == "quest":
        quest_chunk_size = int(getattr(config, "quest_chunk_size"))
        runtime_layout = getattr(config, "runtime_layout", None)
        is_mixed = bool(getattr(runtime_layout, "linear_attention_layer_indices", ()))
        if is_mixed:
            block_size = quest_chunk_size if configured is None else configured
            if block_size <= 0 or block_size % quest_chunk_size != 0:
                raise ValueError(
                    "mixed Quest prefix_cache_block_size must be a positive multiple of quest_chunk_size: "
                    f"prefix_cache_block_size={block_size}, quest_chunk_size={quest_chunk_size}."
                )
            return block_size
        if configured is not None and configured != quest_chunk_size:
            raise ValueError(
                "prefix_cache_block_size must equal quest_chunk_size for quest prefix caching: "
                f"prefix_cache_block_size={configured}, quest_chunk_size={quest_chunk_size}."
            )
        return quest_chunk_size

    block_size = 16 if configured is None else configured
    if block_size <= 0:
        raise ValueError(f"prefix_cache_block_size must be > 0, got {block_size}.")
    return block_size


def build_prefix_cache_fingerprint(config: Any, block_size: int) -> bytes:
    hf_config = getattr(config, "hf_config", None)
    payload = {
        "model": getattr(config, "model", None),
        "model_type": getattr(hf_config, "model_type", None),
        "dtype": str(hf_config.dtype),
        "tp_size": int(getattr(config, "tensor_parallel_size", 1)),
        "method": str(getattr(config, "sparse_method", "") or ""),
        "block_size": int(block_size),
        "salt": str(getattr(config, "prefix_cache_salt", "") or ""),
        "decode_keep_tokens": _jsonable(getattr(config, "decode_keep_tokens", None)),
        "sink_keep_tokens": _jsonable(getattr(config, "sink_keep_tokens", None)),
        "recent_keep_tokens": _jsonable(getattr(config, "recent_keep_tokens", None)),
        "sparse_prefill_score_mode": _jsonable(
            getattr(config, "sparse_prefill_score_mode", None)
        ),
        "full_attention_layers": _jsonable(getattr(config, "full_attention_layers", None)),
        "obs_layer_ids": _jsonable(getattr(config, "obs_layer_ids", None)),
        "quest_chunk_size": _jsonable(getattr(config, "quest_chunk_size", None)),
        "quest_skip_layers": _jsonable(getattr(config, "quest_skip_layers", None)),
    }
    payload.update(
        {
            key: _jsonable(value)
            for key, value in prefill_sparse_method_fingerprint(config).items()
        }
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _pack_token_ids(token_ids: list[int] | tuple[int, ...]) -> bytes:
    # One native pack preserves signed-int64 little-endian IDs without a
    # separate Python call and temporary bytes object for every token.
    return struct.pack(f"<{len(token_ids)}q", *map(int, token_ids))


def _stable_prefix_block_id(
    fingerprint: bytes,
    token_ids: list[int] | tuple[int, ...],
    parent_block_id: bytes | None,
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(fingerprint)
    if parent_block_id is None:
        hasher.update(b"\x00")
    else:
        hasher.update(b"\x01")
        hasher.update(parent_block_id)
    hasher.update(_pack_token_ids(token_ids))
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class _PrefixCacheRoutingMembershipNode:
    block_id: bytes
    left: _PrefixCacheRoutingMembershipNode | None = None
    right: _PrefixCacheRoutingMembershipNode | None = None
    height: int = 1


def _routing_membership_height(
    node: _PrefixCacheRoutingMembershipNode | None,
) -> int:
    return 0 if node is None else node.height


def _make_routing_membership_node(
    block_id: bytes,
    left: _PrefixCacheRoutingMembershipNode | None,
    right: _PrefixCacheRoutingMembershipNode | None,
) -> _PrefixCacheRoutingMembershipNode:
    return _PrefixCacheRoutingMembershipNode(
        block_id=block_id,
        left=left,
        right=right,
        height=1 + max(
            _routing_membership_height(left),
            _routing_membership_height(right),
        ),
    )


def _rotate_routing_membership_left(
    node: _PrefixCacheRoutingMembershipNode,
) -> _PrefixCacheRoutingMembershipNode:
    pivot = node.right
    if pivot is None:
        raise RuntimeError("Cannot rotate prefix-cache routing membership left.")
    moved = _make_routing_membership_node(
        node.block_id,
        node.left,
        pivot.left,
    )
    return _make_routing_membership_node(
        pivot.block_id,
        moved,
        pivot.right,
    )


def _rotate_routing_membership_right(
    node: _PrefixCacheRoutingMembershipNode,
) -> _PrefixCacheRoutingMembershipNode:
    pivot = node.left
    if pivot is None:
        raise RuntimeError("Cannot rotate prefix-cache routing membership right.")
    moved = _make_routing_membership_node(
        node.block_id,
        pivot.right,
        node.right,
    )
    return _make_routing_membership_node(
        pivot.block_id,
        pivot.left,
        moved,
    )


def _balance_routing_membership(
    node: _PrefixCacheRoutingMembershipNode,
) -> _PrefixCacheRoutingMembershipNode:
    balance = (
        _routing_membership_height(node.left)
        - _routing_membership_height(node.right)
    )
    if balance > 1:
        left = node.left
        if left is None:
            raise RuntimeError("Invalid left-heavy prefix-cache routing membership.")
        if _routing_membership_height(left.left) < _routing_membership_height(
            left.right
        ):
            left = _rotate_routing_membership_left(left)
            node = _make_routing_membership_node(
                node.block_id,
                left,
                node.right,
            )
        return _rotate_routing_membership_right(node)
    if balance < -1:
        right = node.right
        if right is None:
            raise RuntimeError("Invalid right-heavy prefix-cache routing membership.")
        if _routing_membership_height(right.right) < _routing_membership_height(
            right.left
        ):
            right = _rotate_routing_membership_right(right)
            node = _make_routing_membership_node(
                node.block_id,
                node.left,
                right,
            )
        return _rotate_routing_membership_left(node)
    return node


def _insert_routing_membership(
    node: _PrefixCacheRoutingMembershipNode | None,
    block_id: bytes,
) -> tuple[_PrefixCacheRoutingMembershipNode, bool]:
    if node is None:
        return _PrefixCacheRoutingMembershipNode(block_id=block_id), True
    if block_id == node.block_id:
        return node, False
    if block_id < node.block_id:
        left, inserted = _insert_routing_membership(node.left, block_id)
        if not inserted:
            return node, False
        updated = _make_routing_membership_node(
            node.block_id,
            left,
            node.right,
        )
    else:
        right, inserted = _insert_routing_membership(node.right, block_id)
        if not inserted:
            return node, False
        updated = _make_routing_membership_node(
            node.block_id,
            node.left,
            right,
        )
    return _balance_routing_membership(updated), True


def _minimum_routing_membership_node(
    node: _PrefixCacheRoutingMembershipNode,
) -> _PrefixCacheRoutingMembershipNode:
    while node.left is not None:
        node = node.left
    return node


def _remove_routing_membership(
    node: _PrefixCacheRoutingMembershipNode | None,
    block_id: bytes,
) -> tuple[_PrefixCacheRoutingMembershipNode | None, bool]:
    if node is None:
        return None, False
    if block_id < node.block_id:
        left, removed = _remove_routing_membership(node.left, block_id)
        if not removed:
            return node, False
        updated = _make_routing_membership_node(
            node.block_id,
            left,
            node.right,
        )
    elif block_id > node.block_id:
        right, removed = _remove_routing_membership(node.right, block_id)
        if not removed:
            return node, False
        updated = _make_routing_membership_node(
            node.block_id,
            node.left,
            right,
        )
    else:
        if node.left is None:
            return node.right, True
        if node.right is None:
            return node.left, True
        successor = _minimum_routing_membership_node(node.right)
        right, removed = _remove_routing_membership(
            node.right,
            successor.block_id,
        )
        if not removed:
            raise RuntimeError(
                "Failed to remove prefix-cache routing membership successor."
            )
        updated = _make_routing_membership_node(
            successor.block_id,
            node.left,
            right,
        )
    return _balance_routing_membership(updated), True


@dataclass(frozen=True, slots=True)
class _PrefixCacheRoutingMembership:
    """Immutable AVL set shared by old and new routing snapshots."""

    root: _PrefixCacheRoutingMembershipNode | None = None
    live_blocks: int = 0

    @property
    def height(self) -> int:
        return _routing_membership_height(self.root)

    def contains(self, block_id: bytes) -> bool:
        node = self.root
        while node is not None:
            if block_id == node.block_id:
                return True
            node = node.left if block_id < node.block_id else node.right
        return False

    def insert(self, block_id: bytes) -> _PrefixCacheRoutingMembership:
        root, inserted = _insert_routing_membership(self.root, block_id)
        if not inserted:
            raise RuntimeError(
                "Prefix-cache routing membership already contains inserted block."
            )
        return _PrefixCacheRoutingMembership(
            root=root,
            live_blocks=self.live_blocks + 1,
        )

    def remove(self, block_id: bytes) -> _PrefixCacheRoutingMembership:
        root, removed = _remove_routing_membership(self.root, block_id)
        if not removed:
            raise RuntimeError(
                "Prefix-cache routing membership is missing removed block."
            )
        return _PrefixCacheRoutingMembership(
            root=root,
            live_blocks=self.live_blocks - 1,
        )


@dataclass(frozen=True)
class PrefixCacheRoutingSnapshot:
    supported: bool
    enabled: bool
    method: str
    block_size: int | None = None
    fingerprint: bytes = b""
    block_ids: frozenset[bytes] = frozenset()
    routing_membership: _PrefixCacheRoutingMembership | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    reason: str | None = None

    def match(self, token_ids: list[int]) -> dict[str, object]:
        token_ids = [int(token_id) for token_id in token_ids]
        if not self.supported or not self.enabled:
            return {
                "supported": bool(self.supported),
                "enabled": bool(self.enabled),
                "method": self.method,
                "matched_tokens": 0,
                "matched_blocks": 0,
                "match_ratio": 0.0,
                "reason": self.reason,
                "snapshot": True,
            }
        if self.block_size is None or self.block_size <= 0:
            raise RuntimeError(
                "Enabled prefix-cache routing snapshot has no valid block size."
            )

        usable_tokens = usable_prefix_cache_tokens(
            len(token_ids),
            self.block_size,
        )
        hit_blocks = 0
        last_block_id: bytes | None = None
        parent_block_id: bytes | None = None
        for start in range(0, usable_tokens, self.block_size):
            block_id = _stable_prefix_block_id(
                self.fingerprint,
                token_ids[start : start + self.block_size],
                parent_block_id,
            )
            if self.routing_membership is None:
                contains_block = block_id in self.block_ids
            else:
                contains_block = self.routing_membership.contains(block_id)
            if not contains_block:
                break
            hit_blocks += 1
            last_block_id = block_id
            parent_block_id = block_id
        hit_len = hit_blocks * self.block_size
        return {
            "supported": True,
            "enabled": True,
            "method": self.method,
            "block_size": int(self.block_size),
            "prompt_tokens": int(len(token_ids)),
            "usable_tokens": int(usable_tokens),
            "matched_tokens": int(hit_len),
            "matched_blocks": int(hit_blocks),
            "match_ratio": (
                0.0
                if usable_tokens <= 0
                else float(hit_len) / float(usable_tokens)
            ),
            "last_block_id": (
                None if last_block_id is None else last_block_id.hex()
            ),
            "live_blocks": int(
                len(self.block_ids)
                if self.routing_membership is None
                else self.routing_membership.live_blocks
            ),
            "snapshot": True,
        }


class PrefixBlockPayload(Protocol):
    """Marker protocol for method-owned prefix block payloads."""


class PrefixTransferKind(str, Enum):
    D2H = "d2h"
    H2D = "h2d"


@dataclass
class PrefixBlockResidency:
    """Physical residency for one immutable logical prefix block.

    ``device_present`` and ``host_present`` describe allocated payloads.  An
    H2D transfer may therefore have both set while later layers are still
    unavailable; the transfer ticket owns that readiness boundary.
    """

    device_present: bool = True
    host_present: bool = False
    transfer: PrefixTransferKind | None = None

    def validate(self) -> None:
        if not self.device_present and not self.host_present:
            raise RuntimeError("Prefix cache block has no resident payload.")
        if self.transfer == PrefixTransferKind.D2H:
            if not self.device_present or self.host_present:
                raise RuntimeError(
                    "D2H prefix transfer requires a device payload and no published host payload."
                )
        elif self.transfer == PrefixTransferKind.H2D:
            if not self.device_present or not self.host_present:
                raise RuntimeError(
                    "H2D prefix transfer requires allocated device and host payloads."
                )


@dataclass
class PrefixCacheBlock:
    stable_block_id: bytes
    parent_block_id: bytes | None
    block_size: int
    logical_block_idx: int
    payload: PrefixBlockPayload
    token_ids: tuple[int, ...] = ()
    ref_count: int = 0
    last_access: int = 0
    eviction_priority: int = 0
    residency: PrefixBlockResidency = field(default_factory=PrefixBlockResidency)
    prune_record: object | None = None


def select_write_through_candidates(
    prefix_cache: RadixPrefixIndex,
    pending: dict[bytes, PrefixCacheBlock],
    newly_unreferenced: list[PrefixCacheBlock] | None = None,
) -> list[PrefixCacheBlock]:
    """Select root-contiguous device blocks without scanning the full radix."""
    if newly_unreferenced is None:
        newly_unreferenced = list(prefix_cache.blocks.values())
    for block in newly_unreferenced:
        if int(block.ref_count) == 0:
            pending[block.stable_block_id] = block

    candidates: list[PrefixCacheBlock] = []
    for block_id, block in list(pending.items()):
        residency = block.residency
        if (
            prefix_cache.get_block(block_id) is not block
            or int(block.ref_count) != 0
            or not residency.device_present
            or residency.host_present
            or residency.transfer is not None
        ):
            pending.pop(block_id, None)
            continue
        candidates.append(block)

    selected: list[PrefixCacheBlock] = []
    selected_ids: set[bytes] = set()
    for block in sorted(
        candidates,
        key=lambda item: (int(item.logical_block_idx), int(item.last_access)),
    ):
        parent_ready = block.parent_block_id is None
        if block.parent_block_id is not None:
            parent = prefix_cache.get_block(block.parent_block_id)
            parent_ready = parent is not None and (
                parent.residency.host_present
                or parent.residency.transfer == PrefixTransferKind.D2H
                or parent.stable_block_id in selected_ids
            )
        if parent_ready:
            selected.append(block)
            selected_ids.add(block.stable_block_id)
    return selected


@dataclass(frozen=True)
class RadixLookupResult:
    hit_block_count: int
    last_block_id: bytes | None


@dataclass
class RadixTreeNode:
    segment: tuple[bytes, ...] = ()
    parent: RadixTreeNode | None = None
    children: dict[bytes, RadixTreeNode] = field(default_factory=dict)


class RadixTreeBackend:
    """Block-level radix backend.

    Edges store one or more stable block ids, and splits only occur between
    block ids. Every block remains directly addressable through the location
    map so cache managers can attach, inspect, delete, and evict by block id.
    """

    def __init__(self):
        self.root = RadixTreeNode()
        self._locations: dict[bytes, tuple[RadixTreeNode, int]] = {}
        self._leaf_block_ids: set[bytes] = set()

    def _index_segment(self, node: RadixTreeNode) -> None:
        for index, block_id in enumerate(node.segment):
            self._locations[block_id] = (node, index)

    def _unindex_segment(self, segment: tuple[bytes, ...]) -> None:
        for block_id in segment:
            self._locations.pop(block_id, None)

    def _discard_leaf(self, node: RadixTreeNode) -> None:
        if node.segment:
            self._leaf_block_ids.discard(node.segment[-1])

    def _add_leaf(self, node: RadixTreeNode) -> None:
        if node.segment and not node.children:
            self._leaf_block_ids.add(node.segment[-1])

    def _split_after(self, node: RadixTreeNode, index: int) -> RadixTreeNode:
        old_segment = node.segment
        prefix = old_segment[: index + 1]
        suffix = old_segment[index + 1:]
        if not suffix:
            return node
        old_children = node.children
        self._unindex_segment(old_segment)

        suffix_node = RadixTreeNode(segment=suffix, parent=node, children=old_children)
        for child in suffix_node.children.values():
            child.parent = suffix_node
        node.segment = prefix
        node.children = {suffix[0]: suffix_node}
        self._index_segment(node)
        self._index_segment(suffix_node)
        return node

    def insert_child(self, parent_block_id: bytes | None, block_id: bytes) -> None:
        if block_id in self._locations:
            return
        if parent_block_id is None:
            if block_id in self.root.children:
                raise RuntimeError("Radix tree root child exists without a block location.")
            child = RadixTreeNode(segment=(block_id,), parent=self.root)
            self.root.children[block_id] = child
            self._index_segment(child)
            self._add_leaf(child)
            return

        location = self._locations.get(parent_block_id)
        if location is None:
            raise KeyError("Prefix cache parent block id is not present in radix tree.")
        parent_node, index = location
        parent_node = self._split_after(parent_node, index)
        if not parent_node.children:
            self._discard_leaf(parent_node)
        if block_id in parent_node.children:
            raise RuntimeError("Radix tree child exists without a block location.")
        child = RadixTreeNode(segment=(block_id,), parent=parent_node)
        parent_node.children[block_id] = child
        self._index_segment(child)
        self._add_leaf(child)

    def lookup(self, block_ids: list[bytes] | tuple[bytes, ...], max_blocks: int) -> RadixLookupResult:
        node = self.root
        hit_count = 0
        last_block_id: bytes | None = None
        limit = min(int(max_blocks), len(block_ids))
        while hit_count < limit:
            child = node.children.get(block_ids[hit_count])
            if child is None:
                break
            for segment_block_id in child.segment:
                if hit_count >= limit or block_ids[hit_count] != segment_block_id:
                    return RadixLookupResult(hit_block_count=hit_count, last_block_id=last_block_id)
                hit_count += 1
                last_block_id = segment_block_id
            node = child
        return RadixLookupResult(hit_block_count=hit_count, last_block_id=last_block_id)

    def insert(self, block_ids: list[bytes] | tuple[bytes, ...]) -> None:
        block_ids = tuple(block_ids)
        if not block_ids:
            return
        node = self.root
        offset = 0
        while offset < len(block_ids):
            child = node.children.get(block_ids[offset])
            if child is None:
                if not node.children:
                    self._discard_leaf(node)
                child = RadixTreeNode(segment=block_ids[offset:], parent=node)
                node.children[block_ids[offset]] = child
                self._index_segment(child)
                self._add_leaf(child)
                return

            common = 0
            while (
                offset + common < len(block_ids)
                and common < len(child.segment)
                and block_ids[offset + common] == child.segment[common]
            ):
                common += 1

            if common == len(child.segment):
                node = child
                offset += common
                continue

            if common <= 0:
                raise RuntimeError("Radix tree child map is inconsistent with edge segment.")

            old_segment = child.segment
            self._unindex_segment(old_segment)
            prefix = child.segment[:common]
            suffix = child.segment[common:]
            split = RadixTreeNode(segment=prefix, parent=node)
            node.children[prefix[0]] = split
            child.segment = suffix
            child.parent = split
            split.children[suffix[0]] = child

            offset += common
            if offset < len(block_ids):
                new_segment = block_ids[offset:]
                new_child = RadixTreeNode(segment=new_segment, parent=split)
                split.children[new_segment[0]] = new_child
                self._index_segment(new_child)
                self._add_leaf(new_child)
            self._index_segment(split)
            self._index_segment(child)
            return

        return

    def remove_block(self, block_id: bytes) -> None:
        location = self._locations.get(block_id)
        if location is None:
            raise KeyError("Prefix cache block id is not present in radix tree.")
        node, index = location
        if index != len(node.segment) - 1 or node.children:
            raise RuntimeError("Cannot remove a prefix cache tree block with live children.")
        if node.parent is None:
            raise RuntimeError("Cannot remove radix tree root.")
        self._locations.pop(block_id, None)
        self._leaf_block_ids.discard(block_id)
        if len(node.segment) == 1:
            parent = node.parent
            del parent.children[node.segment[0]]
            self._add_leaf(parent)
        else:
            node.segment = node.segment[:-1]
            self._add_leaf(node)

    def path_to_block(self, block_id: bytes) -> tuple[bytes, ...]:
        location = self._locations.get(block_id)
        if location is None:
            raise KeyError("Prefix cache block id is not present in radix tree.")
        node, index = location
        segments: list[tuple[bytes, ...]] = [node.segment[: index + 1]]
        while node.parent is not None:
            node = node.parent
            if node.segment:
                segments.append(node.segment)
        return tuple(block_id for segment in reversed(segments) for block_id in segment)

    def subtree_block_ids(self, root_block_id: bytes) -> tuple[bytes, ...]:
        location = self._locations.get(root_block_id)
        if location is None:
            raise KeyError("Prefix cache subtree root is not present.")
        root, index = location
        result: list[bytes] = []
        result.extend(root.segment[index:])
        stack = list(root.children.values())
        while stack:
            node = stack.pop()
            result.extend(node.segment)
            stack.extend(node.children.values())
        return tuple(result)

    def child_count(self, block_id: bytes) -> int:
        location = self._locations.get(block_id)
        if location is None:
            raise KeyError("Prefix cache block id is not present in radix tree.")
        node, index = location
        if index < len(node.segment) - 1:
            return 1
        return len(node.children)

    def child_block_ids(self, block_id: bytes) -> tuple[bytes, ...]:
        location = self._locations.get(block_id)
        if location is None:
            raise KeyError("Prefix cache block id is not present in radix tree.")
        node, index = location
        if index < len(node.segment) - 1:
            return (node.segment[index + 1],)
        return tuple(child.segment[0] for child in node.children.values())

    def leaf_block_ids(self) -> tuple[bytes, ...]:
        return tuple(self._leaf_block_ids)

    def stats(self) -> dict[str, int]:
        node_count = 0
        edge_count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            node_count += 1
            edge_count += len(node.children)
            stack.extend(node.children.values())
        return {
            "prefix_cache_tree_nodes": int(node_count),
            "prefix_cache_tree_edges": int(edge_count),
        }

    def _rebuild_locations(self) -> None:
        locations: dict[bytes, tuple[RadixTreeNode, int]] = {}
        leaf_block_ids: set[bytes] = set()
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            for index, block_id in enumerate(node.segment):
                locations[block_id] = (node, index)
            if node.segment and not node.children:
                leaf_block_ids.add(node.segment[-1])
            stack.extend(node.children.values())
        self._locations = locations
        self._leaf_block_ids = leaf_block_ids


@dataclass(frozen=True)
class PrefixCacheBlockedBlock:
    block_id: bytes | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": None if self.block_id is None else self.block_id.hex(),
            "reason": self.reason,
        }


@dataclass
class PrefixCacheDeleteResult:
    deleted_blocks: list[PrefixCacheBlock]
    blocked_blocks: list[PrefixCacheBlockedBlock]

    def to_dict(self) -> dict[str, Any]:
        return {
            "deleted_block_ids": [block.stable_block_id.hex() for block in self.deleted_blocks],
            "deleted_block_count": len(self.deleted_blocks),
            "blocked_blocks": [block.to_dict() for block in self.blocked_blocks],
        }


class RadixPrefixIndex:
    def __init__(
        self,
        *,
        block_size: int,
        fingerprint: bytes,
        max_blocks: int | None = None,
        backend: RadixTreeBackend | None = None,
    ):
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}.")
        if max_blocks is not None and int(max_blocks) <= 0:
            raise ValueError(f"max_blocks must be > 0 when set, got {max_blocks}.")
        self.fingerprint = bytes(fingerprint)
        self.max_blocks = None if max_blocks is None else int(max_blocks)
        self.backend = backend or RadixTreeBackend()
        self.blocks: dict[bytes, PrefixCacheBlock] = {}
        self._clock = 0
        self._mutation_epoch = 0
        self._capacity_epoch = 0
        self._resident_payload_epoch = 0
        self._lookup_epoch = 0
        self._insert_epoch = 0
        self._remove_epoch = 0
        self._removed_block_ids_for_capacity: set[bytes] | None = set()
        self._freeable_cache_epoch = -1
        self._freeable_block_ids_cache: frozenset[bytes] = frozenset()
        self._freeable_ids: set[bytes] | None = None
        self._blocked_freeable_children: dict[bytes, int] = {}
        self._evictable_cache_epoch = -1
        self._evictable_block_ids_cache: frozenset[bytes] = frozenset()
        self._device_freeable_cache_epoch = -1
        self._device_freeable_block_ids_cache: frozenset[bytes] = frozenset()
        self._device_reclaimable_cache_epoch = -1
        self._device_reclaimable_block_ids_cache: frozenset[bytes] = frozenset()

        self.lookup_requests = 0
        self.block_id_generation_requests = 0
        self.hit_requests = 0
        self.hit_tokens = 0
        self.hit_blocks = 0
        self.committed_blocks = 0
        self.evicted_blocks = 0
        self.device_demoted_blocks = 0
        self.host_evicted_blocks = 0
        self.deleted_blocks = 0
        self.duplicate_commits = 0
        self.control_inspect_requests = 0
        self.control_delete_requests = 0
        self.control_priority_updates = 0
        self.freeable_scans = 0
        self.freeable_snapshot_builds = 0
        self.freeable_cache_hits = 0
        self.evictable_scans = 0
        self.evictable_cache_hits = 0
        self.device_reclaimable_scans = 0
        self.device_reclaimable_cache_hits = 0
        self._routing_snapshot: PrefixCacheRoutingSnapshot | None = None
        self._routing_membership = _PrefixCacheRoutingMembership()

    def __len__(self) -> int:
        return len(self.blocks)

    def has_block(self, stable_block_id: bytes) -> bool:
        return stable_block_id in self.blocks

    def get_block(self, stable_block_id: bytes) -> PrefixCacheBlock | None:
        return self.blocks.get(stable_block_id)

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @property
    def mutation_epoch(self) -> int:
        return self._mutation_epoch

    @property
    def lookup_epoch(self) -> int:
        return self._lookup_epoch

    @property
    def capacity_epoch(self) -> int:
        return self._capacity_epoch

    @property
    def resident_payload_epoch(self) -> int:
        """Changes to existing blocks' device residency or physical payloads."""
        return self._resident_payload_epoch

    @property
    def insert_epoch(self) -> int:
        return self._insert_epoch

    @property
    def remove_epoch(self) -> int:
        return self._remove_epoch

    def take_removed_block_ids_for_capacity(self) -> set[bytes] | None:
        """Return removals since the last weighted capacity query.

        None means the bounded journal overflowed and callers must rebuild.
        """
        removed = self._removed_block_ids_for_capacity
        if removed is None or removed:
            self._removed_block_ids_for_capacity = set()
        return removed

    def _mark_mutated(self) -> None:
        self._mutation_epoch += 1

    def _mark_capacity_mutated(self) -> None:
        self._capacity_epoch += 1
        self._mark_mutated()

    def _refresh_freeable_path(self, block_id: bytes | None) -> None:
        """Update only ancestors whose subtree eviction eligibility changes."""
        freeable = self._freeable_ids
        if freeable is None:
            return  # Build lazily on the first capacity query.
        while block_id is not None:
            block = self.blocks[block_id]
            was_freeable = block_id in freeable
            is_freeable = (
                self._blocked_freeable_children[block_id] == 0
                and int(block.ref_count) == 0
                and int(block.eviction_priority) >= 0
                and block.residency.transfer is None
            )
            if is_freeable == was_freeable:
                break
            if is_freeable:
                freeable.add(block_id)
            else:
                freeable.remove(block_id)
            block_id = block.parent_block_id
            if block_id is not None:
                self._blocked_freeable_children[block_id] += -1 if is_freeable else 1

    def mark_payload_compacted(self, blocks: list[PrefixCacheBlock]) -> None:
        """Invalidate capacity views after an in-place physical payload change."""
        if not blocks:
            return
        for block in blocks:
            self._validate_indexed_block(block)
        self._resident_payload_epoch += 1
        self._mark_capacity_mutated()

    def _mark_inserted(self, *, capacity_changed: bool) -> None:
        self._insert_epoch += 1
        self._lookup_epoch += 1
        if capacity_changed:
            self._mark_capacity_mutated()
        else:
            self._mark_mutated()

    def _mark_removed(self, stable_block_id: bytes) -> None:
        self._remove_epoch += 1
        if self._removed_block_ids_for_capacity is not None:
            self._removed_block_ids_for_capacity.add(stable_block_id)
            if len(self._removed_block_ids_for_capacity) > 4096:
                self._removed_block_ids_for_capacity = None
        self._lookup_epoch += 1
        self._mark_capacity_mutated()

    def _validate_indexed_block(self, block: PrefixCacheBlock) -> None:
        if self.blocks.get(block.stable_block_id) is not block:
            raise RuntimeError("Cannot update a prefix block that is not indexed.")

    def set_block_ref_count(
        self,
        block: PrefixCacheBlock,
        ref_count: int,
        *,
        negative_error: str = "Prefix cache block ref_count became negative.",
    ) -> None:
        self._validate_indexed_block(block)
        ref_count = int(ref_count)
        if ref_count < 0:
            raise RuntimeError(negative_error)
        old_ref_count = int(block.ref_count)
        if old_ref_count == ref_count:
            return
        block.ref_count = ref_count
        if (old_ref_count == 0) != (ref_count == 0):
            self._refresh_freeable_path(block.stable_block_id)
            self._mark_capacity_mutated()
        else:
            self._mark_mutated()

    def acquire_block_ref(self, block: PrefixCacheBlock) -> None:
        self.set_block_ref_count(block, int(block.ref_count) + 1)

    def release_block_ref(
        self,
        block: PrefixCacheBlock,
        *,
        negative_error: str = "Prefix cache block ref_count became negative.",
    ) -> None:
        self.set_block_ref_count(
            block,
            int(block.ref_count) - 1,
            negative_error=negative_error,
        )

    def stable_block_id(
        self,
        token_ids: list[int] | tuple[int, ...],
        parent_block_id: bytes | None,
    ) -> bytes:
        if len(token_ids) != self.block_size:
            raise ValueError(
                f"prefix cache blocks must be full: got {len(token_ids)} tokens, block_size={self.block_size}."
            )
        return _stable_prefix_block_id(
            self.fingerprint,
            token_ids,
            parent_block_id,
        )

    def routing_snapshot(self, method: str) -> PrefixCacheRoutingSnapshot:
        method = str(method or "")
        snapshot = self._routing_snapshot
        if snapshot is None or snapshot.method != method:
            if self._routing_membership.live_blocks != len(self.blocks):
                raise RuntimeError(
                    "Prefix-cache routing membership is inconsistent with the live index: "
                    f"routing_blocks={self._routing_membership.live_blocks} "
                    f"live_blocks={len(self.blocks)}."
                )
            snapshot = PrefixCacheRoutingSnapshot(
                supported=True,
                enabled=True,
                method=method,
                block_size=self.block_size,
                fingerprint=self.fingerprint,
                routing_membership=self._routing_membership,
            )
            self._routing_snapshot = snapshot
        return snapshot

    def _record_routing_insert(self, stable_block_id: bytes) -> None:
        self._routing_membership = self._routing_membership.insert(
            stable_block_id
        )
        self._routing_snapshot = None

    def _record_routing_remove(self, stable_block_id: bytes) -> None:
        self._routing_membership = self._routing_membership.remove(
            stable_block_id
        )
        self._routing_snapshot = None

    def block_ids_for_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_tokens: int | None = None,
    ) -> list[bytes]:
        self.block_id_generation_requests += 1
        token_limit = len(token_ids) if max_tokens is None else min(int(max_tokens), len(token_ids))
        token_limit = (token_limit // self.block_size) * self.block_size
        if token_limit <= 0:
            return []
        # Preserve the stable ID wire format while doing token conversion and
        # packing once, outside the per-block hash loop.
        packed_tokens = _pack_token_ids(token_ids[:token_limit])
        block_bytes = self.block_size * 8
        parent_block_id: bytes | None = None
        block_ids: list[bytes] = []
        # Every non-root block starts with the same fingerprint and marker.
        # Copy that SHA state instead of initializing and updating it for
        # every block; the hashed byte sequence remains unchanged.
        child_hasher = hashlib.sha256(self.fingerprint + b"\x01")
        for start in range(0, len(packed_tokens), block_bytes):
            if parent_block_id is None:
                hasher = hashlib.sha256(self.fingerprint + b"\x00")
            else:
                hasher = child_hasher.copy()
                hasher.update(parent_block_id)
            hasher.update(packed_tokens[start: start + block_bytes])
            block_id = hasher.digest()
            block_ids.append(block_id)
            parent_block_id = block_id
        return block_ids

    def match_longest_prefix(
        self,
        token_ids: list[int],
        *,
        max_usable_tokens: int,
    ) -> tuple[int, bytes | None, int]:
        block_ids = self.block_ids_for_tokens(token_ids, max_tokens=max_usable_tokens)
        return self.match_longest_block_ids(block_ids)

    def match_longest_block_ids(
        self,
        block_ids: list[bytes] | tuple[bytes, ...],
    ) -> tuple[int, bytes | None, int]:
        result = self.backend.lookup(block_ids, len(block_ids))
        if result.hit_block_count <= 0:
            return 0, None, 0
        hit_len = result.hit_block_count * self.block_size
        return hit_len, result.last_block_id, result.hit_block_count

    def lookup_longest_prefix(
        self,
        token_ids: list[int],
        *,
        max_usable_tokens: int,
    ) -> tuple[int, bytes | None, int]:
        block_ids = self.block_ids_for_tokens(token_ids, max_tokens=max_usable_tokens)
        return self.lookup_longest_block_ids(block_ids)

    @cpu_timing.timed
    def lookup_longest_block_ids(
        self,
        block_ids: list[bytes] | tuple[bytes, ...],
    ) -> tuple[int, bytes | None, int]:
        self.lookup_requests += 1
        hit_len, last_block_id, hit_blocks = self.match_longest_block_ids(block_ids)
        if hit_blocks <= 0:
            return 0, None, 0
        self.hit_requests += 1
        self.hit_tokens += hit_len
        self.hit_blocks += hit_blocks
        return hit_len, last_block_id, hit_blocks

    def get_chain(self, last_block_id: bytes, block_count: int) -> list[PrefixCacheBlock]:
        block_count = int(block_count)
        if block_count <= 0:
            raise ValueError(f"block_count must be > 0, got {block_count}.")
        path = self.backend.path_to_block(last_block_id)
        if len(path) < block_count:
            raise RuntimeError(
                "Prefix cache chain is shorter than expected: "
                f"recovered_blocks={len(path)} expected_blocks={block_count}."
            )
        chain_ids = path[-block_count:]
        chain: list[PrefixCacheBlock] = []
        for block_id in chain_ids:
            block = self.blocks.get(block_id)
            if block is None:
                short_key = block_id.hex()[:16]
                raise RuntimeError(
                    "Prefix cache chain is incomplete: "
                    f"missing_block_id={short_key} recovered_blocks={len(chain)} expected_blocks={block_count}."
                )
            chain.append(block)
        return chain

    def ensure_insert_capacity(self, needed_blocks: int = 1) -> list[PrefixCacheBlock]:
        if self.max_blocks is None:
            return []
        needed_blocks = int(needed_blocks)
        if needed_blocks <= 0:
            raise ValueError(f"needed_blocks must be > 0, got {needed_blocks}.")
        over_capacity = len(self.blocks) + needed_blocks - self.max_blocks
        if over_capacity <= 0:
            return []
        evicted = self.evict_until_freeable(over_capacity)
        if len(evicted) != over_capacity:
            raise RuntimeError(
                "Prefix cache capacity exceeded and not enough blocks are evictable: "
                f"live_blocks={len(self.blocks)} max_blocks={self.max_blocks} "
                f"needed_blocks={needed_blocks} evicted_blocks={len(evicted)} "
                f"evictable_blocks={self.evictable_blocks()}."
            )
        return evicted

    def insert_block(self, block: PrefixCacheBlock) -> PrefixCacheBlock:
        block.residency.validate()
        existing = self.blocks.get(block.stable_block_id)
        if existing is not None:
            self.duplicate_commits += 1
            existing.last_access = self._tick()
            return existing
        if self.max_blocks is not None and len(self.blocks) >= self.max_blocks:
            raise RuntimeError(
                "Prefix cache capacity exceeded before insert: "
                f"live_blocks={len(self.blocks)} max_blocks={self.max_blocks} "
                f"evictable_blocks={self.evictable_blocks()}."
            )
        if block.parent_block_id is not None:
            parent = self.blocks.get(block.parent_block_id)
            if parent is None:
                raise KeyError("Cannot insert prefix cache block because parent is missing.")
            if block.residency.device_present and not parent.residency.device_present:
                raise RuntimeError(
                    "Cannot insert a device-resident prefix block below a host-only parent."
                )
            if block.residency.host_present and not parent.residency.host_present:
                raise RuntimeError(
                    "Cannot insert a host-resident prefix block below a device-only parent."
                )
        block.last_access = self._tick()
        self.blocks[block.stable_block_id] = block
        self.backend.insert_child(block.parent_block_id, block.stable_block_id)
        if self._freeable_ids is not None:
            self._blocked_freeable_children[block.stable_block_id] = 0
            # A new child is initially absent from the freeable set.
            if block.parent_block_id is not None:
                self._blocked_freeable_children[block.parent_block_id] += 1
            self._refresh_freeable_path(block.stable_block_id)
            self._refresh_freeable_path(block.parent_block_id)
        self.committed_blocks += 1
        self._record_routing_insert(block.stable_block_id)
        parent = (
            None
            if block.parent_block_id is None
            else self.blocks.get(block.parent_block_id)
        )
        block_is_locally_blocked = (
            int(block.ref_count) != 0
            or int(block.eviction_priority) < 0
            or block.residency.transfer is not None
        )
        parent_is_locally_blocked = (
            parent is not None
            and (
                int(parent.ref_count) != 0
                or int(parent.eviction_priority) < 0
                or parent.residency.transfer is not None
            )
        )
        capacity_changed = not (
            block_is_locally_blocked
            and (parent is None or parent_is_locally_blocked)
        )
        self._mark_inserted(capacity_changed=capacity_changed)
        return block

    def touch_chain(self, blocks: list[PrefixCacheBlock]) -> None:
        access = self._tick()
        for block in blocks:
            block.last_access = access

    def child_count(self, stable_block_id: bytes) -> int:
        return self.backend.child_count(stable_block_id)

    def device_child_count(self, stable_block_id: bytes) -> int:
        return sum(
            1
            for child_id in self.backend.child_block_ids(stable_block_id)
            if (child := self.blocks.get(child_id)) is not None
            and child.residency.device_present
        )

    def begin_d2h(self, block: PrefixCacheBlock) -> None:
        if self.blocks.get(block.stable_block_id) is not block:
            raise RuntimeError("Cannot back up a prefix block that is not indexed.")
        block.residency.validate()
        if int(block.ref_count) != 0:
            raise RuntimeError("Cannot back up a referenced prefix cache block.")
        if block.residency.host_present:
            raise RuntimeError("Cannot back up a prefix cache block that already has a host payload.")
        if block.residency.transfer is not None:
            raise RuntimeError("Cannot start D2H while another prefix transfer is active.")
        if block.parent_block_id is not None:
            parent = self.blocks.get(block.parent_block_id)
            if parent is None or not (
                parent.residency.host_present
                or parent.residency.transfer == PrefixTransferKind.D2H
            ):
                raise RuntimeError(
                    "Prefix D2H backup must preserve a host-resident path from the radix root."
                )
        block.residency.transfer = PrefixTransferKind.D2H
        block.residency.validate()
        self._refresh_freeable_path(block.stable_block_id)
        self._mark_capacity_mutated()

    def finish_d2h(self, block: PrefixCacheBlock) -> None:
        if block.residency.transfer != PrefixTransferKind.D2H:
            raise RuntimeError("Cannot finish D2H without an active D2H prefix transfer.")
        if block.parent_block_id is not None:
            parent = self.blocks.get(block.parent_block_id)
            if parent is None or not parent.residency.host_present:
                raise RuntimeError(
                    "Cannot publish a host prefix payload before its parent is host-resident."
                )
        block.residency.host_present = True
        block.residency.transfer = None
        block.residency.validate()
        self._refresh_freeable_path(block.stable_block_id)
        self._mark_capacity_mutated()

    def abort_d2h(self, block: PrefixCacheBlock) -> None:
        if block.residency.transfer != PrefixTransferKind.D2H:
            raise RuntimeError("Cannot abort D2H without an active D2H prefix transfer.")
        block.residency.transfer = None
        block.residency.validate()
        self._refresh_freeable_path(block.stable_block_id)
        self._mark_capacity_mutated()

    def begin_h2d(self, block: PrefixCacheBlock) -> None:
        if self.blocks.get(block.stable_block_id) is not block:
            raise RuntimeError("Cannot promote a prefix block that is not indexed.")
        block.residency.validate()
        if block.residency.device_present:
            raise RuntimeError("Cannot promote a prefix cache block that already has a device payload.")
        if not block.residency.host_present:
            raise RuntimeError("Cannot promote a prefix cache block without a host payload.")
        if block.residency.transfer is not None:
            raise RuntimeError("Cannot start H2D while another prefix transfer is active.")
        if block.parent_block_id is not None:
            parent = self.blocks.get(block.parent_block_id)
            if parent is None or not parent.residency.device_present:
                raise RuntimeError(
                    "Prefix H2D promotion must preserve a device-resident path from the radix root."
                )
        block.residency.device_present = True
        self._resident_payload_epoch += 1
        block.residency.transfer = PrefixTransferKind.H2D
        block.residency.validate()
        self._refresh_freeable_path(block.stable_block_id)
        self._mark_capacity_mutated()

    def finish_h2d(self, block: PrefixCacheBlock) -> None:
        if block.residency.transfer != PrefixTransferKind.H2D:
            raise RuntimeError("Cannot finish H2D without an active H2D prefix transfer.")
        block.residency.transfer = None
        block.residency.validate()
        self._refresh_freeable_path(block.stable_block_id)
        self._mark_capacity_mutated()

    def abort_h2d(self, block: PrefixCacheBlock) -> None:
        if block.residency.transfer != PrefixTransferKind.H2D:
            raise RuntimeError("Cannot abort H2D without an active H2D prefix transfer.")
        if self.device_child_count(block.stable_block_id) != 0:
            raise RuntimeError("Cannot abort H2D while a device-resident child depends on the block.")
        block.residency.device_present = False
        self._resident_payload_epoch += 1
        block.residency.transfer = None
        block.residency.validate()
        self._refresh_freeable_path(block.stable_block_id)
        self._mark_capacity_mutated()

    def can_evict(self, block: PrefixCacheBlock) -> bool:
        if (
            int(block.ref_count) != 0
            or int(block.eviction_priority) < 0
            or block.residency.transfer is not None
        ):
            return False
        return self.child_count(block.stable_block_id) == 0

    def can_demote_device(self, block: PrefixCacheBlock) -> bool:
        residency = block.residency
        if (
            int(block.ref_count) != 0
            or int(block.eviction_priority) < 0
            or not residency.device_present
            or not residency.host_present
            or residency.transfer is not None
        ):
            return False
        return self.device_child_count(block.stable_block_id) == 0

    def device_evictable_blocks(self) -> int:
        return sum(1 for block in self.blocks.values() if self.can_demote_device(block))

    def device_freeable_blocks(self) -> int:
        """Count device blocks reclaimable by repeated residency-leaf demotion."""
        return len(self.device_freeable_block_ids())

    def device_freeable_block_ids(self) -> frozenset[bytes]:
        """Return blocks reclaimable by repeated device-residency demotion."""
        return self._device_reclaimable_block_ids(include_inflight_d2h=False)

    def device_reclaimable_blocks(self) -> int:
        """Count blocks admission may reclaim after pending D2H completes."""
        return len(self.device_reclaimable_block_ids())

    def device_reclaimable_block_ids(self) -> frozenset[bytes]:
        """Return device blocks backed by host now or by an in-flight D2H.

        This is scheduler-facing capacity. Actual demotion must continue to use
        ``device_freeable_block_ids`` and wait for D2H completion first.
        """
        return self._device_reclaimable_block_ids(include_inflight_d2h=True)

    def _device_reclaimable_block_ids(
        self,
        *,
        include_inflight_d2h: bool,
    ) -> frozenset[bytes]:
        if include_inflight_d2h:
            if self._device_reclaimable_cache_epoch == self._capacity_epoch:
                self.device_reclaimable_cache_hits += 1
                return self._device_reclaimable_block_ids_cache
        elif self._device_freeable_cache_epoch == self._capacity_epoch:
            self.device_reclaimable_cache_hits += 1
            return self._device_freeable_block_ids_cache

        self.device_reclaimable_scans += 1
        freeable: set[bytes] = set()
        device_children = {
            block_id: self.device_child_count(block_id)
            for block_id, block in self.blocks.items()
            if block.residency.device_present
        }
        stack = [
            block_id
            for block_id, child_count in device_children.items()
            if child_count == 0
        ]
        while stack:
            block_id = stack.pop()
            block = self.blocks.get(block_id)
            if block is None or block_id in freeable:
                continue
            residency = block.residency
            host_ready = residency.host_present and residency.transfer is None
            host_pending = (
                include_inflight_d2h
                and residency.transfer == PrefixTransferKind.D2H
            )
            locally_freeable = (
                int(block.ref_count) == 0
                and int(block.eviction_priority) >= 0
                and residency.device_present
                and (host_ready or host_pending)
            )
            if not locally_freeable:
                continue
            freeable.add(block_id)
            parent_id = block.parent_block_id
            if parent_id is not None and parent_id in device_children:
                device_children[parent_id] -= 1
                if device_children[parent_id] == 0:
                    stack.append(parent_id)
        result = frozenset(freeable)
        if include_inflight_d2h:
            self._device_reclaimable_cache_epoch = self._capacity_epoch
            self._device_reclaimable_block_ids_cache = result
        else:
            self._device_freeable_cache_epoch = self._capacity_epoch
            self._device_freeable_block_ids_cache = result
        return result

    def demote_device_until_freeable(self, needed_blocks: int) -> list[PrefixCacheBlock]:
        return self.demote_device_until_weight(
            needed_blocks,
            lambda _block: 1,
        )

    def demote_device_until_weight(
        self,
        needed_weight: int,
        block_weight: Callable[[PrefixCacheBlock], int],
    ) -> list[PrefixCacheBlock]:
        """Demote device leaves in priority order with one candidate-tree scan."""
        demoted: list[PrefixCacheBlock] = []
        needed_weight = int(needed_weight)
        if needed_weight <= 0:
            return demoted
        demoted_weight = 0
        candidate_heap: list[tuple[int, int, bytes]] = []
        queued: set[bytes] = set()
        # Each edge is counted once. Recounting a parent's resident children
        # after every demotion makes a wide branch quadratic in its fanout.
        device_children = {
            block_id: 0 for block_id, block in self.blocks.items()
            if block.residency.device_present
        }
        for block_id in device_children:
            parent_id = self.blocks[block_id].parent_block_id
            if parent_id is not None:
                device_children[parent_id] += 1

        def demotable(block: PrefixCacheBlock) -> bool:
            residency = block.residency
            return (
                int(block.ref_count) == 0
                and int(block.eviction_priority) >= 0
                and residency.device_present
                and residency.host_present
                and residency.transfer is None
                and device_children[block.stable_block_id] == 0
            )

        def queue_if_demotable(
            block_id: bytes | None,
            *,
            heap_ready: bool = True,
        ) -> None:
            if block_id is None or block_id in queued:
                return
            block = self.blocks.get(block_id)
            if block is None or not demotable(block):
                return
            candidate = (
                -int(block.eviction_priority),
                int(block.last_access),
                block_id,
            )
            if heap_ready:
                heapq.heappush(candidate_heap, candidate)
            else:
                candidate_heap.append(candidate)
            queued.add(block_id)

        for block_id, child_count in device_children.items():
            if child_count == 0:
                queue_if_demotable(block_id, heap_ready=False)
        heapq.heapify(candidate_heap)

        while demoted_weight < needed_weight:
            while candidate_heap:
                _, _, block_id = heapq.heappop(candidate_heap)
                queued.discard(block_id)
                block = self.blocks.get(block_id)
                if block is not None and demotable(block):
                    break
            else:
                break
            weight = int(block_weight(block))
            if weight < 0:
                raise ValueError(
                    "Prefix cache demotion weight must be non-negative: "
                    f"block={block.stable_block_id.hex()[:16]} weight={weight}."
                )
            block.residency.device_present = False
            self._resident_payload_epoch += 1
            block.residency.validate()
            demoted.append(block)
            demoted_weight += weight
            self.device_demoted_blocks += 1
            self._mark_capacity_mutated()
            parent_id = block.parent_block_id
            if parent_id is not None:
                device_children[parent_id] -= 1
                if device_children[parent_id] == 0:
                    queue_if_demotable(parent_id)
        return demoted

    def can_evict_host(self, block: PrefixCacheBlock) -> bool:
        residency = block.residency
        if (
            int(block.ref_count) != 0
            or int(block.eviction_priority) < 0
            or residency.device_present
            or not residency.host_present
            or residency.transfer is not None
        ):
            return False
        return self.child_count(block.stable_block_id) == 0

    def evict_host_until_freeable(self, needed_blocks: int) -> list[PrefixCacheBlock]:
        return self.evict_host_until_weight(needed_blocks, lambda _block: 1)

    def evict_host_until_weight(
        self,
        needed_weight: int,
        block_weight: Callable[[PrefixCacheBlock], int],
    ) -> list[PrefixCacheBlock]:
        """Evict host leaves once per batch, including variable-size payloads."""
        evicted: list[PrefixCacheBlock] = []
        needed_weight = int(needed_weight)
        if needed_weight <= 0:
            return evicted
        evicted_weight = 0
        candidate_heap: list[tuple[int, int, bytes]] = []
        queued: set[bytes] = set()

        def queue_if_evictable(block_id: bytes | None, *, heap_ready: bool = True) -> None:
            if block_id is None or block_id in queued:
                return
            block = self.blocks.get(block_id)
            if block is None or not self.can_evict_host(block):
                return
            candidate = (-int(block.eviction_priority), int(block.last_access), block_id)
            if heap_ready:
                heapq.heappush(candidate_heap, candidate)
            else:
                candidate_heap.append(candidate)
            queued.add(block_id)

        for block_id in self.backend.leaf_block_ids():
            queue_if_evictable(block_id, heap_ready=False)
        heapq.heapify(candidate_heap)

        while evicted_weight < needed_weight:
            while candidate_heap:
                _, _, block_id = heapq.heappop(candidate_heap)
                queued.discard(block_id)
                block = self.blocks.get(block_id)
                if block is not None and self.can_evict_host(block):
                    break
            else:
                break
            weight = int(block_weight(block))
            if weight < 0:
                raise ValueError(
                    "Prefix cache host eviction weight must be non-negative: "
                    f"block={block.stable_block_id.hex()[:16]} weight={weight}."
                )
            parent_id = block.parent_block_id
            evicted.append(self._remove_block_from_index(block.stable_block_id))
            evicted_weight += weight
            self.host_evicted_blocks += 1
            queue_if_evictable(parent_id)
        return evicted

    def evictable_blocks(self) -> int:
        return len(self.evictable_block_ids())

    @cpu_timing.timed
    def evictable_block_ids(self) -> frozenset[bytes]:
        if self._evictable_cache_epoch == self._capacity_epoch:
            self.evictable_cache_hits += 1
            return self._evictable_block_ids_cache
        self.evictable_scans += 1
        self._evictable_block_ids_cache = frozenset(
            block_id
            for block_id in self.backend.leaf_block_ids()
            if (block := self.blocks.get(block_id)) is not None
            and int(block.ref_count) == 0
            and int(block.eviction_priority) >= 0
            and block.residency.transfer is None
        )
        self._evictable_cache_epoch = self._capacity_epoch
        return self._evictable_block_ids_cache

    @cpu_timing.timed
    def freeable_block_ids(self) -> frozenset[bytes]:
        """Return blocks removable by repeated leaf eviction without mutating the tree."""
        if self._freeable_cache_epoch == self._capacity_epoch:
            self.freeable_cache_hits += 1
            return self._freeable_block_ids_cache

        self.freeable_snapshot_builds += 1
        if self._freeable_ids is None:
            self.freeable_scans += 1
            # Initialize by virtual leaf removal once. Thereafter mutation
            # paths maintain membership and immediate non-freeable child counts.
            children = dict.fromkeys(self.blocks, 0)
            for block in self.blocks.values():
                if block.parent_block_id is not None:
                    children[block.parent_block_id] += 1
            pending = [block_id for block_id, count in children.items() if count == 0]
            freeable: set[bytes] = set()
            while pending:
                block_id = pending.pop()
                block = self.blocks[block_id]
                if (
                    int(block.ref_count) != 0
                    or int(block.eviction_priority) < 0
                    or block.residency.transfer is not None
                ):
                    continue
                freeable.add(block_id)
                parent = block.parent_block_id
                if parent is not None:
                    children[parent] -= 1
                    if children[parent] == 0:
                        pending.append(parent)
            self._freeable_ids = freeable
            self._blocked_freeable_children = children
        # Preserve immutable snapshots used by scheduler admission accounting.
        self._freeable_block_ids_cache = frozenset(self._freeable_ids)
        self._freeable_cache_epoch = self._capacity_epoch
        return self._freeable_block_ids_cache

    def freeable_blocks(self) -> int:
        return len(self.freeable_block_ids())

    def _remove_block_from_index(self, stable_block_id: bytes) -> PrefixCacheBlock:
        block = self.blocks.get(stable_block_id)
        if block is None:
            raise KeyError("Prefix cache block is not present.")
        if int(block.ref_count) != 0:
            raise RuntimeError("Cannot remove a referenced prefix cache block.")
        if int(block.eviction_priority) < 0:
            raise RuntimeError("Cannot remove a negative-priority prefix cache block.")
        if block.residency.transfer is not None:
            raise RuntimeError("Cannot remove a prefix cache block with an in-flight transfer.")
        if self.child_count(stable_block_id) != 0:
            raise RuntimeError("Cannot remove a prefix cache block with live children.")
        self.backend.remove_block(stable_block_id)
        del self.blocks[stable_block_id]
        if self._freeable_ids is not None:
            was_freeable = stable_block_id in self._freeable_ids
            self._freeable_ids.discard(stable_block_id)
            del self._blocked_freeable_children[stable_block_id]
            if not was_freeable and block.parent_block_id is not None:
                self._blocked_freeable_children[block.parent_block_id] -= 1
                self._refresh_freeable_path(block.parent_block_id)
        self._record_routing_remove(stable_block_id)
        self._mark_removed(stable_block_id)
        return block

    def rollback_inserted_leaf(self, block: PrefixCacheBlock) -> None:
        current = self.blocks.get(block.stable_block_id)
        if current is not block:
            raise RuntimeError(
                "Cannot roll back a prefix block that is not the current indexed block."
            )
        if int(block.ref_count) != 0:
            raise RuntimeError("Cannot roll back a referenced prefix block.")
        if self.child_count(block.stable_block_id) != 0:
            raise RuntimeError("Cannot roll back a prefix block with live children.")
        if self.committed_blocks <= 0:
            raise RuntimeError("Cannot roll back prefix commit statistics below zero.")
        self._remove_block_from_index(block.stable_block_id)
        self.committed_blocks -= 1

    def evict_until_freeable(self, needed_blocks: int) -> list[PrefixCacheBlock]:
        return self.evict_until_weight(
            needed_blocks,
            lambda _block: 1,
        )

    @cpu_timing.timed
    def evict_until_weight(
        self,
        needed_weight: int,
        block_weight: Callable[[PrefixCacheBlock], int],
    ) -> list[PrefixCacheBlock]:
        """Evict leaves in priority order with one candidate-tree scan."""
        evicted: list[PrefixCacheBlock] = []
        needed_weight = int(needed_weight)
        if needed_weight <= 0:
            return evicted
        evicted_weight = 0
        candidate_heap: list[tuple[int, int, bytes]] = []
        queued: set[bytes] = set()

        def queue_if_evictable(
            block_id: bytes | None,
            *,
            heap_ready: bool = True,
        ) -> None:
            if block_id is None or block_id in queued:
                return
            block = self.blocks.get(block_id)
            if block is None or not self.can_evict(block):
                return
            candidate = (
                -int(block.eviction_priority),
                int(block.last_access),
                block_id,
            )
            if heap_ready:
                heapq.heappush(candidate_heap, candidate)
            else:
                candidate_heap.append(candidate)
            queued.add(block_id)

        for block_id in self.backend.leaf_block_ids():
            queue_if_evictable(block_id, heap_ready=False)
        heapq.heapify(candidate_heap)

        while evicted_weight < needed_weight:
            while candidate_heap:
                _, _, block_id = heapq.heappop(candidate_heap)
                queued.discard(block_id)
                block = self.blocks.get(block_id)
                if block is not None and self.can_evict(block):
                    break
            else:
                break
            weight = int(block_weight(block))
            if weight < 0:
                raise ValueError(
                    "Prefix cache eviction weight must be non-negative: "
                    f"block={block.stable_block_id.hex()[:16]} weight={weight}."
                )
            parent_block_id = block.parent_block_id
            evicted.append(self._remove_block_from_index(block.stable_block_id))
            evicted_weight += weight
            self.evicted_blocks += 1
            queue_if_evictable(parent_block_id)
        return evicted

    def inspect_prefix(
        self,
        token_ids: list[int],
        *,
        include_subtree: bool = False,
    ) -> dict[str, Any]:
        self.control_inspect_requests += 1
        block_ids = self.block_ids_for_tokens(token_ids)
        if not block_ids:
            return {
                "matched": False,
                "reason": "prefix_shorter_than_block_size",
                "selector_block_count": 0,
                "hit_block_count": 0,
                "hit_len": 0,
                "last_block_id": None,
                "path_blocks": [],
            }
        result = self.backend.lookup(block_ids, len(block_ids))
        path_ids = block_ids[: result.hit_block_count]
        path_blocks = [self._block_status_dict(block_id) for block_id in path_ids if block_id in self.blocks]
        response: dict[str, Any] = {
            "matched": result.hit_block_count > 0,
            "selector_block_count": len(block_ids),
            "hit_block_count": int(result.hit_block_count),
            "hit_len": int(result.hit_block_count * self.block_size),
            "last_block_id": None if result.last_block_id is None else result.last_block_id.hex(),
            "path_blocks": path_blocks,
        }
        if include_subtree and result.last_block_id is not None:
            subtree_ids = self.backend.subtree_block_ids(result.last_block_id)
            response["subtree"] = self._subtree_summary(subtree_ids)
        return response

    def _block_status_dict(self, block_id: bytes) -> dict[str, Any]:
        block = self.blocks[block_id]
        payload = getattr(block.payload, "kv_payload", block.payload)
        retained_offsets = getattr(payload, "retained_offsets", None)
        effective_prune = None
        for path_block_id in self.backend.path_to_block(block_id):
            path_block = self.blocks.get(path_block_id)
            record = None if path_block is None else path_block.prune_record
            if record is not None:
                effective_prune = record
        status = {
            "block_id": block_id.hex(),
            "logical_block_idx": int(block.logical_block_idx),
            "logical_tokens": int(block.block_size),
            "resident_kv_tokens": int(
                block.block_size
                if retained_offsets is None
                else len(retained_offsets)
            ),
            "ref_count": int(block.ref_count),
            "eviction_priority": int(block.eviction_priority),
            "child_count": int(self.child_count(block_id)),
            "device_child_count": int(self.device_child_count(block_id)),
            "device_present": bool(block.residency.device_present),
            "host_present": bool(block.residency.host_present),
            "transfer": (
                None
                if block.residency.transfer is None
                else block.residency.transfer.value
            ),
            "last_access": int(block.last_access),
        }
        if effective_prune is not None:
            to_dict = getattr(effective_prune, "to_dict", None)
            status["prune"] = to_dict() if callable(to_dict) else effective_prune
        return status

    def _subtree_summary(self, block_ids: tuple[bytes, ...]) -> dict[str, int]:
        existing = [self.blocks[block_id] for block_id in block_ids if block_id in self.blocks]
        return {
            "block_count": int(len(existing)),
            "referenced_block_count": int(sum(1 for block in existing if int(block.ref_count) > 0)),
            "negative_priority_block_count": int(sum(1 for block in existing if int(block.eviction_priority) < 0)),
            "leaf_block_count": int(sum(1 for block in existing if self.child_count(block.stable_block_id) == 0)),
        }

    def _exact_subtree_ids_for_tokens(self, token_ids: list[int]) -> tuple[bytes, ...] | None:
        block_ids = self.block_ids_for_tokens(token_ids)
        if not block_ids:
            return None
        result = self.backend.lookup(block_ids, len(block_ids))
        if result.hit_block_count != len(block_ids) or result.last_block_id is None:
            return None
        return self.backend.subtree_block_ids(result.last_block_id)

    def preview_delete_subtree(self, token_ids: list[int]) -> PrefixCacheDeleteResult:
        subtree_ids = self._exact_subtree_ids_for_tokens(token_ids)
        if subtree_ids is None:
            return PrefixCacheDeleteResult(
                deleted_blocks=[],
                blocked_blocks=[PrefixCacheBlockedBlock(block_id=None, reason="not_found")],
            )
        sorted_ids = sorted(
            subtree_ids,
            key=lambda block_id: len(self.backend.path_to_block(block_id)),
            reverse=True,
        )
        deleted: list[PrefixCacheBlock] = []
        blocked: list[PrefixCacheBlockedBlock] = []
        planned_deleted_ids: set[bytes] = set()
        for block_id in sorted_ids:
            block = self.blocks.get(block_id)
            if block is None:
                continue
            reason = None
            if int(block.ref_count) > 0:
                reason = "referenced"
            elif int(block.eviction_priority) < 0:
                reason = "negative_priority"
            elif block.residency.transfer is not None:
                reason = "transfer_inflight"
            elif any(
                child_id not in planned_deleted_ids
                for child_id in self.backend.child_block_ids(block_id)
            ):
                reason = "has_children"
            if reason is not None:
                blocked.append(PrefixCacheBlockedBlock(block_id=block_id, reason=reason))
                continue
            deleted.append(block)
            planned_deleted_ids.add(block_id)
        return PrefixCacheDeleteResult(deleted_blocks=deleted, blocked_blocks=blocked)

    def safe_delete_subtree(self, token_ids: list[int]) -> PrefixCacheDeleteResult:
        self.control_delete_requests += 1
        plan = self.preview_delete_subtree(token_ids)
        deleted: list[PrefixCacheBlock] = []
        for planned in plan.deleted_blocks:
            current = self.blocks.get(planned.stable_block_id)
            if current is not planned:
                raise RuntimeError(
                    "Prefix subtree deletion plan became stale before it was applied."
                )
            deleted.append(self._remove_block_from_index(planned.stable_block_id))
            self.deleted_blocks += 1
        return PrefixCacheDeleteResult(
            deleted_blocks=deleted,
            blocked_blocks=plan.blocked_blocks,
        )

    def set_subtree_eviction_priority(self, token_ids: list[int], priority: int) -> dict[str, Any]:
        self.control_priority_updates += 1
        block_ids = self.block_ids_for_tokens(token_ids)
        if not block_ids:
            return {
                "matched": False,
                "reason": "prefix_shorter_than_block_size",
                "root_block_id": None,
                "updated_block_count": 0,
                "eviction_priority": int(priority),
            }
        result = self.backend.lookup(block_ids, len(block_ids))
        if result.hit_block_count != len(block_ids) or result.last_block_id is None:
            return {
                "matched": False,
                "reason": "not_found",
                "root_block_id": None,
                "updated_block_count": 0,
                "eviction_priority": int(priority),
            }
        subtree_ids = self.backend.subtree_block_ids(result.last_block_id)
        updated_block_count = 0
        changed = False
        capacity_changed = False
        for block_id in subtree_ids:
            block = self.blocks.get(block_id)
            if block is not None:
                updated_block_count += 1
                if int(block.eviction_priority) != int(priority):
                    old_priority = int(block.eviction_priority)
                    block.eviction_priority = int(priority)
                    if (old_priority < 0) != (int(priority) < 0):
                        self._refresh_freeable_path(block_id)
                    changed = True
                    capacity_changed = capacity_changed or (
                        (old_priority < 0) != (int(priority) < 0)
                    )
        if capacity_changed:
            self._mark_capacity_mutated()
        elif changed:
            self._mark_mutated()
        return {
            "matched": True,
            "root_block_id": result.last_block_id.hex(),
            "updated_block_count": int(updated_block_count),
            "eviction_priority": int(priority),
        }

    def stats(self) -> dict[str, int]:
        tree_stats = self.backend.stats()
        stats = {
            "prefix_cache_lookup_requests": int(self.lookup_requests),
            "prefix_cache_block_id_generation_requests": int(
                self.block_id_generation_requests
            ),
            "prefix_cache_hit_requests": int(self.hit_requests),
            "prefix_cache_hit_tokens": int(self.hit_tokens),
            "prefix_cache_hit_blocks": int(self.hit_blocks),
            "prefix_cache_committed_blocks": int(self.committed_blocks),
            "prefix_cache_duplicate_commits": int(self.duplicate_commits),
            "prefix_cache_deleted_blocks": int(self.deleted_blocks),
            "prefix_cache_evicted_blocks": int(self.evicted_blocks),
            "prefix_cache_device_demoted_blocks": int(self.device_demoted_blocks),
            "prefix_cache_host_evicted_blocks": int(self.host_evicted_blocks),
            "prefix_cache_live_blocks": int(len(self.blocks)),
            "prefix_cache_device_blocks": int(
                sum(1 for block in self.blocks.values() if block.residency.device_present)
            ),
            "prefix_cache_host_blocks": int(
                sum(1 for block in self.blocks.values() if block.residency.host_present)
            ),
            "prefix_cache_inflight_transfers": int(
                sum(1 for block in self.blocks.values() if block.residency.transfer is not None)
            ),
            "prefix_cache_device_freeable_blocks": int(self.device_freeable_blocks()),
            "prefix_cache_referenced_blocks": int(sum(1 for block in self.blocks.values() if int(block.ref_count) > 0)),
            "prefix_cache_negative_priority_blocks": int(
                sum(1 for block in self.blocks.values() if int(block.eviction_priority) < 0)
            ),
            "prefix_cache_leaf_blocks": int(len(self.backend.leaf_block_ids())),
            "prefix_cache_freeable_blocks": int(self.freeable_blocks()),
            "prefix_cache_control_inspect_requests": int(self.control_inspect_requests),
            "prefix_cache_control_delete_requests": int(self.control_delete_requests),
            "prefix_cache_control_priority_updates": int(self.control_priority_updates),
            "prefix_cache_mutation_epoch": int(self.mutation_epoch),
            "prefix_cache_capacity_epoch": int(self.capacity_epoch),
            "prefix_cache_lookup_epoch": int(self.lookup_epoch),
            "prefix_cache_insert_epoch": int(self.insert_epoch),
            "prefix_cache_remove_epoch": int(self.remove_epoch),
            "prefix_cache_freeable_scans": int(self.freeable_scans),
            "prefix_cache_freeable_snapshot_builds": int(self.freeable_snapshot_builds),
            "prefix_cache_freeable_cache_hits": int(self.freeable_cache_hits),
            "prefix_cache_evictable_scans": int(self.evictable_scans),
            "prefix_cache_evictable_cache_hits": int(self.evictable_cache_hits),
            "prefix_cache_device_reclaimable_scans": int(
                self.device_reclaimable_scans
            ),
            "prefix_cache_device_reclaimable_cache_hits": int(
                self.device_reclaimable_cache_hits
            ),
        }
        stats.update(tree_stats)
        return stats
