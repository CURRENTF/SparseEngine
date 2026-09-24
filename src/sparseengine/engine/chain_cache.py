from __future__ import annotations

import hashlib
import heapq
import json
import secrets
import sys
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from enum import Enum
from itertools import islice
from typing import Any, Iterable

from sparseengine.method_registry import prefill_sparse_method_fingerprint
from sparseengine.utils.log import logger



CHAIN_PREFIX_METHODS = frozenset(
    {"streamingllm", "snapkv", "h2o", "pyramidkv", "rkv", "skipkv"}
)
RADIX_PREFIX_METHODS = frozenset({"", "omnikv", "quest"})
PREFIX_CACHE_MODES = frozenset({"auto", "radix", "chain"})


class ChainState(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"


class ChainCacheError(RuntimeError):
    status_code = 500
    error_code = "chain_cache_error"

    def __init__(self, message: str, *, chain_id: str | None = None):
        super().__init__(message)
        self.chain_id = chain_id


class ChainNotFoundError(ChainCacheError):
    status_code = 404
    error_code = "chain_not_found"


class ChainBusyError(ChainCacheError):
    status_code = 409
    error_code = "chain_busy"


class ChainPrefixMismatchError(ChainCacheError):
    status_code = 409
    error_code = "chain_prefix_mismatch"


class ChainFingerprintMismatchError(ChainCacheError):
    status_code = 409
    error_code = "chain_fingerprint_mismatch"


class ChainGoneError(ChainCacheError):
    status_code = 410
    error_code = "chain_gone"


class ChainCapacityError(ChainCacheError):
    status_code = 503
    error_code = "chain_capacity_unavailable"


class ChainOwnerMismatchError(ChainCacheError):
    status_code = 500
    error_code = "chain_owner_mismatch"


class ChainModeError(ChainCacheError):
    status_code = 400
    error_code = "chain_mode_disabled"


@dataclass(frozen=True, slots=True)
class RequestAdmission:
    seq_id: int
    chain_id: str | None
    chain_status: str
    reused_tokens: int
    prefilled_tokens: int = 0
    prompt_token_ids: list[int] | None = None


@dataclass(slots=True)
class ChainRecord:
    chain_id: str
    seq_id: int
    fingerprint: bytes
    state: ChainState
    processed_token_count: int = 0
    processed_token_digest: bytes = b""
    token_ids: array = field(
        default_factory=lambda: array("I")
    )
    last_input_token_count: int = 0
    last_access: int = 0
    physical_slots_by_layer: tuple[int, ...] = ()
    resident_rows: int = 1
    reserved_slots_by_layer: tuple[int, ...] = ()
    reserved_rows: int = 0


@dataclass(frozen=True, slots=True)
class PreparedChainTokens:
    """Immutable rank-0 history prepared before the collective finish RPC."""

    chain_id: str
    seq_id: int
    processed_token_count: int
    processed_token_digest: bytes
    token_bytes: bytes


@dataclass(frozen=True, slots=True)
class ChainAdmissionPlan:
    chain_id: str
    seq_id: int
    status: str
    reused_tokens: int
    input_token_count: int = 0
    victim_chain_ids: tuple[str, ...] = ()
    reserved_slots_by_layer: tuple[int, ...] = ()
    reserved_rows: int = 0
    demote_chain_ids: tuple[str, ...] = ()
    decode_reserved_slots_by_layer: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ChainRoutingSnapshot:
    enabled: bool
    active_chain_ids: frozenset[str] = frozenset()
    idle_chain_ids: frozenset[str] = frozenset()
    tombstone_chain_ids: frozenset[str] = frozenset()

    def match(self, chain_id: str) -> dict[str, object]:
        chain_id = str(chain_id)
        if chain_id in self.active_chain_ids:
            return {
                "enabled": self.enabled,
                "present": True,
                "state": ChainState.ACTIVE.value,
                "tombstone": False,
            }
        if chain_id in self.idle_chain_ids:
            return {
                "enabled": self.enabled,
                "present": True,
                "state": ChainState.IDLE.value,
                "tombstone": False,
            }
        return {
            "enabled": self.enabled,
            "present": False,
            "state": None,
            "tombstone": chain_id in self.tombstone_chain_ids,
        }


def normalize_prefix_cache_mode(
    requested: str | None,
    *,
    enabled: bool,
    method: str,
    prefill_method: str = "",
) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode not in PREFIX_CACHE_MODES:
        supported = ", ".join(sorted(PREFIX_CACHE_MODES))
        raise ValueError(
            f"prefix_cache_mode must be one of {supported}; got {requested!r}."
        )
    if not bool(enabled):
        return "disabled"
    method = str(method or "")
    expected = (
        "chain"
        if prefill_method == "omnikv_prefill" and method in ("", "omnikv")
        else "radix"
        if method in RADIX_PREFIX_METHODS
        else "chain"
        if method in CHAIN_PREFIX_METHODS
        else None
    )
    if expected is None:
        raise ValueError(
            "prefix caching is not supported for "
            f"sparse_method={method!r}."
        )
    if mode == "auto":
        return expected
    if mode != expected:
        raise ValueError(
            f"prefix_cache_mode={mode!r} is incompatible with "
            f"sparse_method={method!r}; use {expected!r}."
        )
    return mode


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return str(value)


def build_chain_cache_fingerprint(config: Any) -> bytes:
    hf_config = getattr(config, "hf_config", None)
    method = str(
        getattr(
            config,
            "resolved_cache_sparse_method",
            getattr(config, "sparse_method", ""),
        )
        or ""
    )
    method_fields = {
        "omnikv": (
            "sink_keep_tokens",
            "recent_keep_tokens",
            "decode_keep_tokens",
            "sparse_attn_score_dtype",
        ),
        "streamingllm": (
            "sink_keep_tokens",
            "recent_keep_tokens",
        ),
        "snapkv": (
            "sink_keep_tokens",
            "recent_keep_tokens",
            "decode_keep_tokens",
            "observation_window_size",
            "snapkv_decode_eviction",
            "decode_eviction_interval",
            "snapkv_num_full_layers",
            "sparse_prefill_score_mode",
            "sparse_attn_score_dtype",
            "pool_kernel_size",
        ),
        "h2o": (
            "h2o_decode_budget",
            "h2o_decode_eviction",
            "h2o_decode_eviction_interval",
            "h2o_prefill_budget",
            "h2o_recent_ratio",
            "h2o_prefill_score_window",
            "sparse_prefill_score_mode",
            "sparse_attn_score_dtype",
        ),
        "pyramidkv": (
            "sink_keep_tokens",
            "recent_keep_tokens",
            "decode_keep_tokens",
            "observation_window_size",
            "decode_eviction_interval",
            "sparse_prefill_score_mode",
            "pyramid_layer_ratios",
            "pyramidkv_start_layer",
            "pyramidkv_start_ratio",
            "pyramidkv_least_layer",
            "pyramidkv_least_ratio",
            "sparse_attn_score_dtype",
            "pool_kernel_size",
        ),
        "rkv": (
            "sink_keep_tokens",
            "recent_keep_tokens",
            "decode_keep_tokens",
            "rkv_compression_interval",
            "rkv_observation_tokens",
            "rkv_alpha",
            "rkv_similarity_threshold",
            "rkv_recent_similar_keep",
            "rkv_kernel_size",
            "rkv_score_chunk_mb",
        ),
        "skipkv": (
            "sink_keep_tokens",
            "recent_keep_tokens",
            "decode_keep_tokens",
            "rkv_observation_tokens",
            "skipkv_compression_interval",
            "skipkv_alpha",
            "skipkv_similarity_threshold",
            "skipkv_segment_size",
            "skipkv_max_redundancy_tokens",
            "skipkv_redundancy_window",
            "skipkv_enable_sentence_scoring",
            "skipkv_sentence_score_weight",
            "skipkv_sentence_min_tokens",
            "skipkv_sentence_max_tokens",
            "skipkv_sentence_embedding_layer",
            "skipkv_max_tracked_sentences",
            "skipkv_enable_activation_steering",
            "skipkv_steering_vector_path",
            "skipkv_steering_layer",
            "skipkv_steering_alpha",
            "skipkv_steering_alpha_increment",
            "skipkv_steering_alpha_max",
        ),
    }
    payload: dict[str, Any] = {
        "schema": 1,
        "model": getattr(config, "model", None),
        "model_type": getattr(hf_config, "model_type", None),
        "dtype": str(hf_config.dtype),
        "tp_size": int(getattr(config, "tensor_parallel_size", 1)),
        "max_model_len": int(getattr(config, "max_model_len", 0)),
        "full_attention_layers": _jsonable(
            getattr(config, "full_attention_layers", ())
        ),
        "method": method,
        "salt": str(getattr(config, "prefix_cache_salt", "") or ""),
    }
    for field_name in method_fields.get(method, ()):
        payload[field_name] = _jsonable(getattr(config, field_name, None))
    payload.update(
        {
            key: _jsonable(value)
            for key, value in prefill_sparse_method_fingerprint(config).items()
        }
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


def stable_token_digest(
    token_ids: Iterable[int],
    *,
    count: int | None = None,
) -> bytes:
    # Preserve the wire format (signed int64, little endian), but pack/hash in
    # bulk instead of issuing a Python struct.pack + SHA update per token.
    limit = None if count is None else int(count)
    values = array("q", map(int, token_ids if limit is None else islice(token_ids, max(0, limit))))
    if limit is not None and len(values) != limit:
        raise ChainPrefixMismatchError(
            "Input is shorter than the chain's processed token boundary: "
            f"input_tokens={len(values)}, processed_tokens={limit}."
        )
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values).digest()


class ChainCacheIndex:
    """Logical lifecycle for non-branching, resident sparse-KV chains."""

    def __init__(
        self,
        *,
        max_tombstones: int = 1024,
        max_token_history_tokens: int | None = None,
    ):
        max_tombstones = int(max_tombstones)
        if max_tombstones <= 0:
            raise ValueError(
                f"chain_cache_max_tombstones must be > 0, got {max_tombstones}."
            )
        if max_token_history_tokens is not None:
            max_token_history_tokens = int(max_token_history_tokens)
            if max_token_history_tokens <= 0:
                raise ValueError(
                    "max_token_history_tokens must be > 0 when set, got "
                    f"{max_token_history_tokens}."
                )
        self.max_tombstones = max_tombstones
        self.max_token_history_tokens = max_token_history_tokens
        self._token_history_tokens = 0
        self.records: dict[str, ChainRecord] = {}
        self.active_records: dict[str, ChainRecord] = {}
        self._routing_snapshot_cache: ChainRoutingSnapshot | None = None
        self.seq_id_to_chain_id: dict[int, str] = {}
        self.tombstones: OrderedDict[str, int] = OrderedDict()
        self._clock = 0
        self._stats = {
            "chain_cache_created": 0,
            "chain_cache_resumed": 0,
            "chain_cache_finished": 0,
            "chain_cache_evicted": 0,
            "chain_cache_invalidated": 0,
            "chain_cache_prefix_mismatch": 0,
            "chain_cache_busy": 0,
        }

    @staticmethod
    def new_chain_id() -> str:
        return f"chain_{secrets.token_urlsafe(24)}"

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _set_record_state(self, record: ChainRecord, state: ChainState) -> None:
        self._routing_snapshot_cache = None
        record.state = state
        if state is ChainState.ACTIVE:
            self.active_records[record.chain_id] = record
        else:
            self.active_records.pop(record.chain_id, None)

    def _add_tombstone(self, chain_id: str) -> None:
        self._routing_snapshot_cache = None
        self.tombstones.pop(chain_id, None)
        self.tombstones[chain_id] = self._tick()
        while len(self.tombstones) > self.max_tombstones:
            self.tombstones.popitem(last=False)

    def _prepare_token_history(
        self,
        record: ChainRecord,
        token_ids: Iterable[int],
        *,
        processed_token_count: int,
    ) -> array:
        processed_token_count = int(processed_token_count)
        if processed_token_count < 0:
            raise ValueError(
                "processed_token_count must be >= 0, got "
                f"{processed_token_count}."
            )
        try:
            logical_tokens = array("I", map(int, token_ids))
        except OverflowError as exc:
            raise ChainOwnerMismatchError(
                "Chain token IDs must fit unsigned 32-bit storage.",
                chain_id=record.chain_id,
            ) from exc
        if len(logical_tokens) < processed_token_count:
            raise ChainOwnerMismatchError(
                "Finished token sequence is shorter than the processed boundary: "
                f"tokens={len(logical_tokens)}, "
                f"processed={processed_token_count}.",
                chain_id=record.chain_id,
            )
        self._check_token_history_capacity(record, len(logical_tokens))
        return logical_tokens

    def _check_token_history_capacity(self, record: ChainRecord, token_count: int) -> None:
        next_total = (
            self._token_history_tokens
            - len(record.token_ids)
            + token_count
        )
        if (
            self.max_token_history_tokens is not None
            and next_total > self.max_token_history_tokens
        ):
            raise ChainCapacityError(
                "Chain logical token history exceeds its configured resident "
                "capacity: "
                f"needed_tokens={next_total}, "
                f"capacity_tokens={self.max_token_history_tokens}.",
                chain_id=record.chain_id,
            )

    def _store_token_history(
        self,
        record: ChainRecord,
        logical_tokens: array,
    ) -> None:
        self._check_token_history_capacity(record, len(logical_tokens))
        self._token_history_tokens -= len(record.token_ids)
        record.token_ids = logical_tokens
        self._token_history_tokens += len(logical_tokens)

    def _drop_token_history(self, record: ChainRecord) -> None:
        self._token_history_tokens -= len(record.token_ids)
        record.token_ids = array("I")

    def lookup(self, chain_id: str) -> ChainRecord:
        chain_id = str(chain_id)
        record = self.records.get(chain_id)
        if record is not None:
            return record
        if chain_id in self.tombstones:
            raise ChainGoneError(
                f"Chain {chain_id!r} was evicted or invalidated.",
                chain_id=chain_id,
            )
        raise ChainNotFoundError(
            f"Unknown chain_id {chain_id!r}.",
            chain_id=chain_id,
        )

    def plan_admission(
        self,
        *,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
        fingerprint: bytes,
        required_slots_by_layer: tuple[int, ...] = (),
        row_deficit: int = 0,
        reserved_slots_by_layer: tuple[int, ...] = (),
        reserved_rows: int = 0,
    ) -> ChainAdmissionPlan:
        return self._plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            input_token_count=len(token_ids),
            input_prefix_digest=None,
            token_ids=token_ids,
            fingerprint=fingerprint,
            required_slots_by_layer=required_slots_by_layer,
            row_deficit=row_deficit,
            reserved_slots_by_layer=reserved_slots_by_layer,
            reserved_rows=reserved_rows,
        )

    def plan_admission_digest(
        self,
        *,
        chain_id: str,
        seq_id: int,
        input_token_count: int,
        input_prefix_digest: bytes,
        fingerprint: bytes,
        required_slots_by_layer: tuple[int, ...] = (),
        row_deficit: int = 0,
        reserved_slots_by_layer: tuple[int, ...] = (),
        reserved_rows: int = 0,
    ) -> ChainAdmissionPlan:
        """Validate a rank-local plan without broadcasting full token IDs."""
        return self._plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            input_token_count=int(input_token_count),
            input_prefix_digest=bytes(input_prefix_digest),
            token_ids=None,
            fingerprint=fingerprint,
            required_slots_by_layer=required_slots_by_layer,
            row_deficit=row_deficit,
            reserved_slots_by_layer=reserved_slots_by_layer,
            reserved_rows=reserved_rows,
        )

    def _plan_admission(
        self,
        *,
        chain_id: str,
        seq_id: int,
        input_token_count: int,
        input_prefix_digest: bytes | None,
        token_ids: list[int] | None,
        fingerprint: bytes,
        required_slots_by_layer: tuple[int, ...],
        row_deficit: int,
        reserved_slots_by_layer: tuple[int, ...],
        reserved_rows: int,
    ) -> ChainAdmissionPlan:
        record = self.records.get(chain_id)
        if record is None:
            if chain_id in self.tombstones:
                self.lookup(chain_id)
            status = "created"
            reused_tokens = 0
        else:
            if record.state is ChainState.ACTIVE:
                self._stats["chain_cache_busy"] += 1
                raise ChainBusyError(
                    f"Chain {chain_id!r} already has an active writer.",
                    chain_id=chain_id,
                )
            if record.fingerprint != fingerprint:
                raise ChainFingerprintMismatchError(
                    f"Chain {chain_id!r} was created with a different method/config fingerprint.",
                    chain_id=chain_id,
                )
            if record.seq_id != int(seq_id):
                raise ChainOwnerMismatchError(
                    f"Chain owner mismatch for {chain_id!r}: "
                    f"resident_seq_id={record.seq_id}, requested_seq_id={int(seq_id)}.",
                    chain_id=chain_id,
                )
            try:
                if token_ids is not None:
                    digest = stable_token_digest(
                        token_ids,
                        count=record.processed_token_count,
                    )
                else:
                    if int(input_token_count) < record.processed_token_count:
                        raise ChainPrefixMismatchError(
                            "Input is shorter than the chain's processed token "
                            "boundary: "
                            f"input_tokens={int(input_token_count)}, "
                            f"processed_tokens={record.processed_token_count}.",
                            chain_id=chain_id,
                        )
                    digest = bytes(input_prefix_digest or b"")
            except ChainPrefixMismatchError as exc:
                self._stats["chain_cache_prefix_mismatch"] += 1
                if exc.chain_id is None:
                    exc.chain_id = chain_id
                raise
            if digest != record.processed_token_digest:
                self._stats["chain_cache_prefix_mismatch"] += 1
                mismatch_details = ""
                if token_ids is not None:
                    expected = list(
                        record.token_ids[
                            : record.processed_token_count
                        ]
                    )
                    actual = [
                        int(token_id)
                        for token_id in token_ids[
                            : record.processed_token_count
                        ]
                    ]
                    first_mismatch = next(
                        (
                            index
                            for index, (expected_id, actual_id) in enumerate(
                                zip(expected, actual)
                            )
                            if expected_id != actual_id
                        ),
                        min(len(expected), len(actual)),
                    )
                    stable_boundary = min(
                        int(record.last_input_token_count),
                        len(expected),
                        len(actual),
                    )
                    stable_prefix_matches = (
                        expected[:stable_boundary]
                        == actual[:stable_boundary]
                    )
                    expected_id = (
                        expected[first_mismatch]
                        if first_mismatch < len(expected)
                        else None
                    )
                    actual_id = (
                        actual[first_mismatch]
                        if first_mismatch < len(actual)
                        else None
                    )
                    mismatch_details = (
                        f" input_tokens={len(token_ids)},"
                        f" last_input_token_count={record.last_input_token_count},"
                        f" stable_prefix_matches={stable_prefix_matches},"
                        f" first_mismatch={first_mismatch},"
                        f" expected_token_id={expected_id},"
                        f" actual_token_id={actual_id}."
                    )
                raise ChainPrefixMismatchError(
                    f"Input prefix does not match chain {chain_id!r} at "
                    f"processed_token_count={record.processed_token_count}."
                    f"{mismatch_details}",
                    chain_id=chain_id,
                )
            status = "resumed"
            reused_tokens = int(record.processed_token_count)

        slot_deficits = [max(0, int(value)) for value in required_slots_by_layer]
        remaining_row_deficit = max(0, int(row_deficit))
        victims: list[str] = []
        candidates = (
            candidate for candidate in (
                self.idle_resident_lru()
                if remaining_row_deficit > 0 or any(slot_deficits) else ()
            )
            if candidate.chain_id != chain_id
        )
        for victim in candidates:
            if remaining_row_deficit <= 0 and not any(slot_deficits):
                break
            victims.append(victim.chain_id)
            remaining_row_deficit = max(
                0, remaining_row_deficit - int(victim.resident_rows)
            )
            for layer_idx in range(len(slot_deficits)):
                resident = (
                    int(victim.physical_slots_by_layer[layer_idx])
                    if layer_idx < len(victim.physical_slots_by_layer)
                    else 0
                )
                slot_deficits[layer_idx] = max(
                    0, slot_deficits[layer_idx] - resident
                )
        if remaining_row_deficit > 0 or any(slot_deficits):
            raise ChainCapacityError(
                "No capacity is available for the chain request after evicting "
                "all IDLE chains; ACTIVE chains remain pinned.",
                chain_id=chain_id,
            )
        return ChainAdmissionPlan(
            chain_id=chain_id,
            seq_id=int(seq_id),
            status=status,
            reused_tokens=reused_tokens,
            input_token_count=int(input_token_count),
            victim_chain_ids=tuple(victims),
            reserved_slots_by_layer=tuple(
                max(0, int(value)) for value in reserved_slots_by_layer
            ),
            reserved_rows=max(0, int(reserved_rows)),
        )

    def apply_admission(
        self,
        plan: ChainAdmissionPlan,
        *,
        fingerprint: bytes,
    ) -> ChainRecord:
        for victim_chain_id in plan.victim_chain_ids:
            self.evict(victim_chain_id)
        if plan.status == "created":
            if plan.chain_id in self.records:
                raise ChainOwnerMismatchError(
                    f"Chain {plan.chain_id!r} appeared between plan and admission.",
                    chain_id=plan.chain_id,
                )
            if plan.seq_id in self.seq_id_to_chain_id:
                raise ChainOwnerMismatchError(
                    f"resident_seq_id={plan.seq_id} already owns chain "
                    f"{self.seq_id_to_chain_id[plan.seq_id]!r}.",
                    chain_id=plan.chain_id,
                )
            record = ChainRecord(
                chain_id=plan.chain_id,
                seq_id=int(plan.seq_id),
                fingerprint=bytes(fingerprint),
                state=ChainState.ACTIVE,
                last_input_token_count=int(plan.input_token_count),
                last_access=self._tick(),
            )
            self.records[plan.chain_id] = record
            self._set_record_state(record, ChainState.ACTIVE)
            self.seq_id_to_chain_id[int(plan.seq_id)] = plan.chain_id
            record.reserved_slots_by_layer = tuple(
                int(value) for value in plan.reserved_slots_by_layer
            )
            record.reserved_rows = int(plan.reserved_rows)
            self._stats["chain_cache_created"] += 1
            return record
        record = self.lookup(plan.chain_id)
        if record.state is not ChainState.IDLE:
            raise ChainBusyError(
                f"Chain {plan.chain_id!r} is not IDLE during admission.",
                chain_id=plan.chain_id,
            )
        self._set_record_state(record, ChainState.ACTIVE)
        record.last_input_token_count = int(plan.input_token_count)
        record.last_access = self._tick()
        record.reserved_slots_by_layer = tuple(
            int(value) for value in plan.reserved_slots_by_layer
        )
        record.reserved_rows = int(plan.reserved_rows)
        self._stats["chain_cache_resumed"] += 1
        return record

    def finish(
        self,
        chain_id: str,
        *,
        token_ids: list[int],
        processed_token_count: int,
        physical_slots_by_layer: tuple[int, ...],
        resident_rows: int = 1,
    ) -> ChainRecord:
        processed_token_count = int(processed_token_count)
        existing = self.lookup(chain_id)
        logical_tokens = self._prepare_token_history(
            existing,
            token_ids,
            processed_token_count=processed_token_count,
        )
        record = self.finish_digest(
            chain_id,
            processed_token_digest=stable_token_digest(
                token_ids,
                count=processed_token_count,
            ),
            processed_token_count=processed_token_count,
            physical_slots_by_layer=physical_slots_by_layer,
            resident_rows=resident_rows,
        )
        self._store_token_history(record, logical_tokens)
        return record

    def finish_digest(
        self,
        chain_id: str,
        *,
        processed_token_digest: bytes,
        processed_token_count: int,
        physical_slots_by_layer: tuple[int, ...],
        resident_rows: int = 1,
    ) -> ChainRecord:
        record = self.lookup(chain_id)
        if record.state is not ChainState.ACTIVE:
            raise ChainOwnerMismatchError(
                f"Cannot finish non-ACTIVE chain {chain_id!r}.",
                chain_id=chain_id,
            )
        processed_token_count = int(processed_token_count)
        if processed_token_count < record.processed_token_count:
            raise ChainOwnerMismatchError(
                "Chain processed boundary moved backwards: "
                f"previous={record.processed_token_count}, new={processed_token_count}.",
                chain_id=chain_id,
            )
        processed_token_digest = bytes(processed_token_digest)
        if len(processed_token_digest) != hashlib.sha256().digest_size:
            raise ValueError(
                "processed_token_digest must be a SHA-256 digest, got "
                f"{len(processed_token_digest)} bytes."
            )
        record.processed_token_digest = processed_token_digest
        record.processed_token_count = processed_token_count
        record.physical_slots_by_layer = tuple(
            int(value) for value in physical_slots_by_layer
        )
        record.resident_rows = int(resident_rows)
        record.reserved_slots_by_layer = ()
        record.reserved_rows = 0
        self._set_record_state(record, ChainState.IDLE)
        record.last_access = self._tick()
        self._stats["chain_cache_finished"] += 1
        return record

    def invalidate(self, chain_id: str) -> ChainRecord:
        record = self.lookup(chain_id)
        self._drop_token_history(record)
        self.records.pop(chain_id, None)
        self.active_records.pop(chain_id, None)
        self.seq_id_to_chain_id.pop(int(record.seq_id), None)
        self._add_tombstone(chain_id)
        self._stats["chain_cache_invalidated"] += 1
        return record

    def idle_resident_lru(self) -> Iterable[ChainRecord]:
        # Most pressure events need only a few victims. Heapify once and order
        # only the consumed prefix, preserving the same LRU and chain-ID ties.
        candidates = [
            (record.last_access, record.chain_id, record)
            for record in self.records.values()
            if record.state is ChainState.IDLE and record.resident_rows > 0
        ]
        heapq.heapify(candidates)
        while candidates:
            yield heapq.heappop(candidates)[2]

    def evict(self, chain_id: str) -> ChainRecord:
        record = self.lookup(chain_id)
        if record.state is ChainState.ACTIVE:
            raise ChainBusyError(
                f"Cannot evict ACTIVE chain {chain_id!r}.",
                chain_id=chain_id,
            )
        self._drop_token_history(record)
        self.records.pop(chain_id, None)
        self.active_records.pop(chain_id, None)
        self.seq_id_to_chain_id.pop(int(record.seq_id), None)
        self._add_tombstone(chain_id)
        self._stats["chain_cache_evicted"] += 1
        logger.warning("chain_evicted {}", json.dumps({
            "chain_id": chain_id,
            "seq_id": int(record.seq_id),
            "physical_slots_by_layer": record.physical_slots_by_layer,
            "resident_rows": int(record.resident_rows),
        }))
        return record

    def routing_match(self, chain_id: str) -> dict[str, object]:
        record = self.records.get(str(chain_id))
        if record is not None:
            return {
                "present": True,
                "state": record.state.value,
                "tombstone": False,
            }
        return {
            "present": False,
            "state": None,
            "tombstone": str(chain_id) in self.tombstones,
        }

    def routing_snapshot(self) -> ChainRoutingSnapshot:
        # Dispatcher refreshes before and after every step. Decode, offload
        # completion and residency changes do not change logical chain routing.
        if self._routing_snapshot_cache is not None:
            return self._routing_snapshot_cache
        snapshot = ChainRoutingSnapshot(
            enabled=True,
            active_chain_ids=frozenset(
                record.chain_id
                for record in self.records.values()
                if record.state is ChainState.ACTIVE
            ),
            idle_chain_ids=frozenset(
                record.chain_id
                for record in self.records.values()
                if record.state is ChainState.IDLE
            ),
            tombstone_chain_ids=frozenset(self.tombstones),
        )
        self._routing_snapshot_cache = snapshot
        return snapshot

    def stats(self) -> dict[str, int]:
        return {
            **self._stats,
            "chain_cache_entries": len(self.records),
            "chain_cache_active": len(self.active_records),
            "chain_cache_idle": len(self.records) - len(self.active_records),
            "chain_cache_tombstones": len(self.tombstones),
            "chain_cache_tombstone_capacity": self.max_tombstones,
            "chain_cache_token_history_tokens": self._token_history_tokens,
            "chain_cache_token_history_capacity": (
                -1
                if self.max_token_history_tokens is None
                else self.max_token_history_tokens
            ),
            "chain_cache_token_history_bytes": (
                self._token_history_tokens * array("I").itemsize
            ),
            "chain_cache_token_history_byte_capacity": (
                -1
                if self.max_token_history_tokens is None
                else self.max_token_history_tokens * array("I").itemsize
            ),
        }

    def reset(self) -> None:
        self._routing_snapshot_cache = None
        self.records.clear()
        self.active_records.clear()
        self.seq_id_to_chain_id.clear()
        self.tombstones.clear()
        self._clock = 0
        self._token_history_tokens = 0
        for name in self._stats:
            self._stats[name] = 0


class ChainCacheCoordinator:
    """Coordinates chain metadata; cache managers continue to own all payload."""

    def __init__(self, config: Any, cache_manager: Any):
        self.config = config
        self.cache_manager = cache_manager
        self.fingerprint = build_chain_cache_fingerprint(config)
        self.offload = None
        if bool(getattr(config, "enable_prefix_cache_offload", False)):
            self.offload = cache_manager.create_chain_offload(
                int(float(config.prefix_cache_host_size_gb) * 1024**3)
            )
        self.index = ChainCacheIndex(
            max_tombstones=int(
                getattr(config, "chain_cache_max_tombstones", 1024)
            ),
            max_token_history_tokens=(
                int(getattr(config, "max_model_len"))
                * int(
                    getattr(
                        config,
                        "max_num_seqs_in_gpu",
                        getattr(config, "max_num_seqs_in_batch", 1),
                    )
                    or 1
                )
            ),
        )
        if self.offload is not None:
            # CPU-only chains may outnumber GPU rows. Keep logical history
            # bounded separately, including the full (uncompressed) token list.
            self.index.max_token_history_tokens += self.offload.capacity_bytes // 4

    def owner_seq_id(self, chain_id: str) -> int:
        return int(self.index.lookup(chain_id).seq_id)

    def admission_requirements(
        self,
        *,
        chain_id: str,
        token_count: int,
        decode_reserved_slots_by_layer: tuple[int, ...] = (),
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
        if self.offload is not None:
            self.offload.poll()
        record = self.index.records.get(chain_id)
        reused = 0 if record is None else int(record.processed_token_count)
        existing_slots = (
            ()
            if record is None
            else tuple(int(value) for value in record.physical_slots_by_layer)
        )
        suffix_tokens = max(0, int(token_count) - reused)
        hook = getattr(self.cache_manager, "chain_capacity_deficits", None)
        if not callable(hook):
            return (), 0, (), 0
        outstanding_slots, outstanding_rows = (
            self._outstanding_active_reservations()
        )
        chain_reserved_slots = outstanding_slots
        outstanding_slots = tuple(
            (outstanding_slots[i] if i < len(outstanding_slots) else 0)
            + (decode_reserved_slots_by_layer[i] if i < len(decode_reserved_slots_by_layer) else 0)
            for i in range(max(len(outstanding_slots), len(decode_reserved_slots_by_layer)))
        )
        required_slots, required_rows, slot_deficits, row_deficit = hook(
            suffix_tokens=suffix_tokens,
            generation_tokens=0,
            existing_slots_by_layer=existing_slots,
            outstanding_reserved_slots_by_layer=outstanding_slots,
            outstanding_reserved_rows=outstanding_rows,
            needs_resident_row=record is None or record.resident_rows == 0,
        )
        if diagnostics is not None:
            budget_hook = getattr(self.cache_manager, "decode_window_budgets", None)
            diagnostics.update(
                kv_layer_indices=tuple(self.cache_manager.kv_transformer_layer_indices()),
                resident_sequence_capacity=getattr(self.config, "max_num_seqs_in_gpu", None),
                active_chains=len(self.index.active_records),
                idle_chains=len(self.index.records) - len(self.index.active_records),
                input_tokens=int(token_count),
                reused_tokens=reused,
                suffix_tokens=suffix_tokens,
                existing_slots_by_layer=existing_slots,
                cache_free_slot_budgets=budget_hook() if callable(budget_hook) else None,
                chain_reserved_slots_by_layer=chain_reserved_slots,
                decode_reserved_slots_by_layer=decode_reserved_slots_by_layer,
                outstanding_reserved_slots_by_layer=outstanding_slots,
                outstanding_reserved_rows=int(outstanding_rows),
                required_slots_by_layer=tuple(int(v) for v in required_slots),
                required_rows=int(required_rows),
                slot_deficits_by_layer=tuple(int(v) for v in slot_deficits),
                row_deficit=int(row_deficit),
            )
        return (
            tuple(int(value) for value in required_slots),
            int(required_rows),
            tuple(int(value) for value in slot_deficits),
            int(row_deficit),
        )

    def _outstanding_active_reservations(
        self,
        *,
        exclude_seq_ids: set[int] | None = None,
    ) -> tuple[tuple[int, ...], int]:
        has_residency = getattr(
            self.cache_manager, "chain_has_residency", None
        )
        physical_residency = getattr(
            self.cache_manager, "chain_physical_residency", None
        )
        outstanding: list[int] = []
        outstanding_rows = 0
        for record in self.index.active_records.values():
            if exclude_seq_ids is not None and int(record.seq_id) in exclude_seq_ids:
                continue
            # Final prefill clears these promises. Decoding chains then owe
            # nothing here, so their per-layer physical rows need no query.
            if not record.reserved_slots_by_layer and int(record.reserved_rows) <= 0:
                continue
            reserved = tuple(
                int(value) for value in record.reserved_slots_by_layer
            )
            if len(outstanding) < len(reserved):
                outstanding.extend([0] * (len(reserved) - len(outstanding)))
            resident = (
                bool(has_residency(int(record.seq_id)))
                if callable(has_residency)
                else False
            )
            current = (
                tuple(
                    int(value)
                    for value in physical_residency(int(record.seq_id))
                )
                if resident and callable(physical_residency)
                else tuple(
                    int(value)
                    for value in record.physical_slots_by_layer
                )
            )
            baseline = tuple(
                int(value) for value in record.physical_slots_by_layer
            )
            for layer_idx, reserved_slots in enumerate(reserved):
                current_slots = (
                    current[layer_idx] if layer_idx < len(current) else 0
                )
                baseline_slots = (
                    baseline[layer_idx] if layer_idx < len(baseline) else 0
                )
                realized = max(0, current_slots - baseline_slots)
                outstanding[layer_idx] += max(
                    0, reserved_slots - realized
                )
            outstanding_rows += max(
                0,
                int(record.reserved_rows) - (1 if resident else 0),
            )
        return tuple(outstanding), outstanding_rows

    def plan_admission(
        self,
        *,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
    ) -> ChainAdmissionPlan:
        ledger = getattr(self, "decode_reservations", None)
        diagnostics: dict[str, Any] = {
            "chain_id": chain_id, "seq_id": int(seq_id),
            "sparse_method": str(getattr(self.config, "sparse_method", "")),
        }
        reserved = {} if ledger is None else ledger.outstanding()
        decode_reserved = tuple(reserved.get(f"layer_{layer}", reserved.get("slots", 0))
                                for layer in self.cache_manager.kv_transformer_layer_indices()) if reserved else ()
        required_slots, required_rows, slots, rows = (
            self.admission_requirements(
                chain_id=chain_id,
                token_count=len(token_ids),
                decode_reserved_slots_by_layer=decode_reserved,
                diagnostics=diagnostics,
            )
        )
        diagnostics["pressure"] = [
            name for name, present in (("kv_slots", any(slots)), ("resident_rows", rows > 0))
            if present
        ]
        try:
            plan = self._offload_plan(self.index.plan_admission(
                chain_id=chain_id,
                seq_id=seq_id,
                token_ids=token_ids,
                fingerprint=self.fingerprint,
                required_slots_by_layer=slots,
                row_deficit=rows,
                reserved_slots_by_layer=required_slots,
                reserved_rows=required_rows,
            ))
        except ChainCapacityError as exc:
            diagnostics.update(outcome="rejected", error=str(exc))
            logger.warning("chain_admission {}", json.dumps(diagnostics))
            raise
        diagnostics.update(
            outcome="planned", chain_status=plan.status,
            victim_chain_ids=plan.victim_chain_ids,
            demote_chain_ids=plan.demote_chain_ids,
        )
        logger.log(
            "WARNING" if diagnostics["pressure"] else "DEBUG",
            "chain_admission {}", json.dumps(diagnostics),
        )
        return replace(plan, decode_reserved_slots_by_layer=decode_reserved)

    def _offload_plan(self, plan: ChainAdmissionPlan) -> ChainAdmissionPlan:
        if self.offload is None:
            return plan
        demote: list[str] = []
        evict: list[str] = []
        for chain_id in plan.victim_chain_ids:
            target = (demote if self.index.records[chain_id].seq_id in self.offload.snapshots
                      else evict)
            target.append(chain_id)
        return replace(plan, demote_chain_ids=tuple(demote), victim_chain_ids=tuple(evict))

    def validate_admission_plan(
        self,
        expected: ChainAdmissionPlan,
        *,
        input_token_count: int,
        input_prefix_digest: bytes,
    ) -> ChainAdmissionPlan:
        required_slots, required_rows, slots, rows = (
            self.admission_requirements(
                chain_id=expected.chain_id,
                token_count=int(input_token_count),
                decode_reserved_slots_by_layer=expected.decode_reserved_slots_by_layer,
            )
        )
        local = self._offload_plan(self.index.plan_admission_digest(
            chain_id=expected.chain_id,
            seq_id=expected.seq_id,
            input_token_count=int(input_token_count),
            input_prefix_digest=bytes(input_prefix_digest),
            fingerprint=self.fingerprint,
            required_slots_by_layer=slots,
            row_deficit=rows,
            reserved_slots_by_layer=required_slots,
            reserved_rows=required_rows,
        ))
        local = replace(local, decode_reserved_slots_by_layer=expected.decode_reserved_slots_by_layer)
        if local != expected:
            raise RuntimeError(
                "Chain-cache admission plan diverged from the rank-0 plan: "
                f"rank0={expected!r}, local={local!r}."
            )
        return local

    def apply_admission(self, plan: ChainAdmissionPlan) -> ChainRecord:
        if self.offload is not None:
            for chain_id in plan.demote_chain_ids:
                victim = self.index.lookup(chain_id)
                if victim.state is not ChainState.IDLE:
                    raise ChainBusyError("Cannot demote an ACTIVE chain.", chain_id=chain_id)
                if not self.offload.wait(victim.seq_id).valid:
                    raise RuntimeError("Cannot demote a chain without a valid CPU snapshot.")
                self.cache_manager.free_seq(victim.seq_id)
                victim.resident_rows = 0
                logger.info("chain_demoted {}", json.dumps({
                    "chain_id": chain_id, "seq_id": int(victim.seq_id),
                    "cause": "chain_admission", "admitting_chain_id": plan.chain_id,
                }))
            for chain_id in plan.victim_chain_ids:
                self.offload.drop(self.index.lookup(chain_id).seq_id)
        record = self.index.apply_admission(plan, fingerprint=self.fingerprint)
        return record

    def prepare_resumed_chain(self, record: ChainRecord) -> None:
        if self.offload is None:
            return
        if record.resident_rows == 0:
            self.offload.restore(record.seq_id)
            record.resident_rows = 1
            record.reserved_slots_by_layer = tuple(
                reserved - existing for reserved, existing in
                zip(record.reserved_slots_by_layer, record.physical_slots_by_layer)
            )
            record.reserved_rows = 0
        # Wait even when the GPU copy survived: the next writer must not race
        # the previous turn's D2H reader.
        self.offload.invalidate(record.seq_id)

    def save_finished_chain(self, record: ChainRecord) -> None:
        if self.offload is None:
            return
        state = self.cache_manager.snapshot_chain_method_state(record.seq_id)
        required = self.offload.required_bytes(record.seq_id, state)
        if required > self.offload.capacity_bytes:
            raise ChainCapacityError(
                f"Whole-chain CPU snapshot needs {required} bytes; host budget is {self.offload.capacity_bytes}.",
                chain_id=record.chain_id,
            )
        old_bytes = getattr(self.offload.snapshots.get(record.seq_id), "nbytes", 0)
        candidates = [
            # ACTIVE snapshots are obsolete allocations retained only for reuse.
            # Their validity does not depend on rank-local event completion.
            (r.state is not ChainState.ACTIVE, r.last_access, r.chain_id, r)
            for r in self.index.records.values() if r.chain_id != record.chain_id
            and r.seq_id in self.offload.snapshots
        ] if self.offload.used_bytes - old_bytes + required > self.offload.capacity_bytes else []
        heapq.heapify(candidates)
        while candidates and self.offload.used_bytes - old_bytes + required > self.offload.capacity_bytes:
            victim = heapq.heappop(candidates)[3]
            self.offload.drop(victim.seq_id)
            if victim.resident_rows == 0:
                self.index.evict(victim.chain_id)
        self.offload.save(record.seq_id, state)

    def finish(self, seq: Any) -> ChainRecord:
        chain_id = str(getattr(seq, "chain_id", "") or "")
        if not chain_id:
            raise ChainOwnerMismatchError(
                f"Sequence {getattr(seq, 'seq_id', None)} has no chain_id."
            )
        processed_token_count = max(0, int(seq.num_tokens) - 1)
        residency_hook = getattr(
            self.cache_manager, "chain_physical_residency", None
        )
        physical_slots = (
            tuple(int(value) for value in residency_hook(int(seq.seq_id)))
            if callable(residency_hook)
            else ()
        )
        return self.index.finish(
            chain_id,
            token_ids=list(seq.token_ids),
            processed_token_count=processed_token_count,
            physical_slots_by_layer=physical_slots,
        )

    def finish_values(
        self,
        *,
        chain_id: str,
        seq_id: int,
        processed_token_digest: bytes,
        processed_token_count: int,
    ) -> ChainRecord:
        record = self.index.lookup(chain_id)
        if int(record.seq_id) != int(seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, finished_seq_id={int(seq_id)}.",
                chain_id=chain_id,
            )
        residency_hook = getattr(
            self.cache_manager, "chain_physical_residency", None
        )
        physical_slots = (
            tuple(int(value) for value in residency_hook(int(seq_id)))
            if callable(residency_hook)
            else ()
        )
        return self.index.finish_digest(
            chain_id,
            processed_token_digest=bytes(processed_token_digest),
            processed_token_count=int(processed_token_count),
            physical_slots_by_layer=physical_slots,
        )

    def remember_processed_tokens(
        self,
        *,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
        processed_token_count: int,
    ) -> ChainRecord:
        prepared = self.prepare_processed_tokens(
            chain_id=chain_id, seq_id=seq_id, token_ids=token_ids,
            processed_token_count=processed_token_count,
        )
        return self.remember_prepared_tokens(prepared)

    def prepare_processed_tokens(
        self,
        *,
        chain_id: str,
        seq_id: int,
        token_ids: Iterable[int],
        processed_token_count: int,
    ) -> PreparedChainTokens:
        record = self.index.lookup(str(chain_id))
        processed_token_count = int(processed_token_count)
        if int(record.seq_id) != int(seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, finished_seq_id={int(seq_id)}.",
                chain_id=str(chain_id),
            )
        logical_tokens = self.index._prepare_token_history(
            record,
            token_ids,
            processed_token_count=processed_token_count,
        )
        return PreparedChainTokens(
            chain_id=str(chain_id), seq_id=int(seq_id),
            processed_token_count=processed_token_count,
            processed_token_digest=stable_token_digest(logical_tokens, count=processed_token_count),
            token_bytes=logical_tokens.tobytes(),
        )

    def remember_prepared_tokens(self, prepared: PreparedChainTokens) -> ChainRecord:
        record = self.index.lookup(prepared.chain_id)
        if record.seq_id != prepared.seq_id or record.state is not ChainState.IDLE:
            raise ChainOwnerMismatchError(
                "Prepared history requires its original owner in IDLE state.",
                chain_id=prepared.chain_id,
            )
        if (
            prepared.processed_token_count != int(record.processed_token_count)
            or prepared.processed_token_digest != record.processed_token_digest
        ):
            raise ChainOwnerMismatchError(
                "Finished logical tokens do not match the recorded chain boundary.",
                chain_id=prepared.chain_id,
            )
        logical_tokens = array("I")
        logical_tokens.frombytes(prepared.token_bytes)
        self.index._store_token_history(record, logical_tokens)
        return record

    def invalidate(self, chain_id: str) -> ChainRecord:
        if self.offload is not None:
            self.offload.drop(self.index.lookup(chain_id).seq_id)
        return self.index.invalidate(chain_id)

    def routing_match(self, chain_id: str) -> dict[str, object]:
        return self.index.routing_match(chain_id)

    def stats(self) -> dict[str, int]:
        stats = self.index.stats()
        if self.offload is not None:
            self.offload.poll()
            stats.update(chain_cache_host_bytes=self.offload.used_bytes,
                         chain_cache_d2h_bytes=self.offload.d2h_bytes,
                         chain_cache_h2d_bytes=self.offload.h2d_bytes,
                         chain_cache_d2h_inflight=sum(s.completion is not None for s in self.offload.snapshots.values()),
                         chain_cache_cpu_only=sum(r.resident_rows == 0 for r in self.index.records.values()))
        return stats

    def reset(self) -> None:
        if self.offload is not None:
            self.offload.reset()
        self.index.reset()
