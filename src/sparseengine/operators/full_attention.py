from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sparseengine.operators.decode_attention import (
    DecodeAttentionOpSpec,
    PreparedDecodeAttentionOp,
    prepare_decode_attention_op,
)
from sparseengine.operators.prefill_attention import (
    PrefillAttentionOpSpec,
    PreparedPrefillAttentionOp,
    prepare_prefill_attention_op,
)

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class FullAttentionOpSpec:
    """Semantic full-attention contract with phase-specific execution needs."""

    prefill: PrefillAttentionOpSpec
    decode: DecodeAttentionOpSpec

    def __post_init__(self) -> None:
        shared_fields = (
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "activation_dtype",
            "softmax_scale",
            "causal",
        )
        mismatches = [
            field
            for field in shared_fields
            if getattr(self.prefill, field) != getattr(self.decode, field)
        ]
        if mismatches:
            details = ", ".join(
                f"{field}={getattr(self.prefill, field)!r}/"
                f"{getattr(self.decode, field)!r}"
                for field in mismatches
            )
            raise ValueError(
                "Full attention prefill/decode contracts are incompatible: "
                f"{details}."
            )


class FullAttentionProvider:
    """Prepared full-attention owner composed from independent phase providers."""

    def __init__(
        self,
        spec: FullAttentionOpSpec,
        *,
        prefill_op: PreparedPrefillAttentionOp,
        decode_op: PreparedDecodeAttentionOp,
    ) -> None:
        if prefill_op.spec != spec.prefill:
            raise ValueError("Prepared prefill operator does not match FullAttentionOpSpec.")
        if decode_op.spec != spec.decode:
            raise ValueError("Prepared decode operator does not match FullAttentionOpSpec.")
        self.spec = spec
        self.prefill_op = prefill_op
        self.decode_op = decode_op
        self._closed = False

    @property
    def name(self) -> str:
        return f"prefill={self.prefill_op.name},decode={self.decode_op.name}"

    @property
    def prefill_name(self) -> str:
        return self.prefill_op.name

    @property
    def decode_name(self) -> str:
        return self.decode_op.name

    def bind(self, model: nn.Module) -> int:
        if self._closed:
            raise RuntimeError("Cannot bind a closed full-attention provider.")

        from sparseengine.layers.attention import Attention

        attention_layers = [
            module for module in model.modules() if isinstance(module, Attention)
        ]
        if not attention_layers:
            raise ValueError(
                "Cannot bind full-attention provider: model has no Attention layers."
            )

        expected = self.spec.prefill
        for attention in attention_layers:
            if attention.full_attention_provider is not None:
                raise RuntimeError(
                    "Attention layer already has a full-attention provider; "
                    "runtime provider rebinding is forbidden."
                )
            if attention.prefill_op is not None or attention.decode_op is not None:
                raise RuntimeError(
                    "Attention layer has phase operators without a full-attention "
                    "owner; refusing to overwrite a partial binding."
                )
            actual = (
                int(attention.num_heads),
                int(attention.num_kv_heads),
                int(attention.head_dim),
                float(attention.scale),
            )
            required = (
                int(expected.num_query_heads),
                int(expected.num_kv_heads),
                int(expected.head_dim),
                float(expected.softmax_scale),
            )
            if actual != required:
                raise ValueError(
                    "Full-attention provider does not match model layer contract: "
                    f"actual={actual} required={required}."
                )
        for attention in attention_layers:
            attention.full_attention_provider = self
            attention.prefill_op = self.prefill_op
            attention.decode_op = self.decode_op
        return len(attention_layers)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "composite_provider",
            "semantic_operator": "full_attention",
            "prefill_provider": self.prefill_name,
            "decode_provider": self.decode_name,
        }

    def prepare_weights(self, model: nn.Module) -> None:
        """Finalize optional semantic projection transformations after loading."""

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.prefill_op.close()
        finally:
            self.decode_op.close()


def prepare_full_attention_provider(
    spec: FullAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> FullAttentionProvider:
    prefill_op = prepare_prefill_attention_op(
        spec.prefill,
        device_index=device_index,
    )
    try:
        decode_op = prepare_decode_attention_op(
            spec.decode,
            device_index=device_index,
        )
    except Exception:
        prefill_op.close()
        raise
    try:
        return FullAttentionProvider(
            spec,
            prefill_op=prefill_op,
            decode_op=decode_op,
        )
    except Exception:
        try:
            prefill_op.close()
        finally:
            decode_op.close()
        raise


__all__ = [
    "FullAttentionOpSpec",
    "FullAttentionProvider",
    "prepare_full_attention_provider",
]
