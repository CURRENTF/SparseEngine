from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from sparseengine.utils.config import config_get, config_layer_get


def resolve_attention_qk_head_dim(hf_config: Any) -> int:
    dimensions = (
        config_get(hf_config, "qk_nope_head_dim", None),
        config_get(hf_config, "qk_rope_head_dim", None),
    )
    if all(value is not None for value in dimensions):
        head_dim = sum(map(int, dimensions))
    elif (value := config_get(hf_config, "head_dim", None)) is not None:
        head_dim = int(value)
    else:
        hidden_size = int(config_get(hf_config, "hidden_size", 0) or 0)
        num_heads = int(config_get(hf_config, "num_attention_heads", 0) or 0)
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError(
                "Attention QK head dimension requires valid head_dim or divisible "
                "hidden_size/num_attention_heads."
            )
        head_dim = hidden_size // num_heads
    if head_dim <= 0:
        raise ValueError(
            f"Attention QK head dimension must be positive, got {head_dim}."
        )
    return head_dim


def _coerce_int_list(
    name: str,
    value: Any,
    *,
    allow_none: bool = False,
) -> list[int] | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required.")
    if isinstance(value, str):
        return [int(part) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(
        f"{name} must be a list/tuple of ints or a comma-separated string, "
        f"got {value!r}."
    )


def _attention_type(value: Any) -> str:
    value = str(value).strip().lower()
    if value in {
        "full",
        "full_attention",
        "attention",
        "self_attention",
        "sliding_attention",
    }:
        return "full"
    if value in {
        "linear",
        "linear_attention",
        "recurrent",
        "recurrent_attention",
        "gated_delta",
        "gated_delta_net",
    }:
        return "linear"
    raise ValueError(f"Unsupported attention layer type {value!r}.")


@dataclass(frozen=True)
class RuntimeLayout:
    num_layers: int
    num_kv_layers: int
    full_attention_layer_indices: tuple[int, ...]
    linear_attention_layer_indices: tuple[int, ...]
    layer_idx_to_kv_idx: tuple[int | None, ...]
    kv_idx_to_layer_idx: tuple[int, ...]
    kv_num_heads: tuple[int, ...] = ()
    kv_head_dims: tuple[int, ...] = ()

    @property
    def heterogeneous_kv(self) -> bool:
        return len(set(zip(self.kv_num_heads, self.kv_head_dims))) > 1

    def local_kv_shapes(self, tp_size: int) -> tuple[tuple[int, int], ...]:
        tp_size = int(tp_size)
        if not self.kv_num_heads:
            return ()
        shapes = []
        for num_heads, head_dim in zip(self.kv_num_heads, self.kv_head_dims):
            if num_heads >= tp_size:
                if num_heads % tp_size:
                    raise ValueError(
                        f"KV heads must be divisible by TP: heads={num_heads}, TP={tp_size}."
                    )
                local_heads = num_heads // tp_size
            else:
                if tp_size % num_heads:
                    raise ValueError(
                        f"TP must be divisible by replicated KV heads: heads={num_heads}, TP={tp_size}."
                    )
                local_heads = 1
            shapes.append((local_heads, head_dim))
        return tuple(shapes)

    def local_kv_shape(self, layer_idx: int, tp_size: int) -> tuple[int, int] | None:
        if not self.kv_num_heads:
            return None
        return self.local_kv_shapes(tp_size)[self.kv_layer_index(layer_idx)]

    @classmethod
    def dense(cls, num_layers: int) -> RuntimeLayout:
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_layers}.")
        layers = tuple(range(num_layers))
        return cls(
            num_layers=num_layers,
            num_kv_layers=num_layers,
            full_attention_layer_indices=layers,
            linear_attention_layer_indices=(),
            layer_idx_to_kv_idx=tuple(range(num_layers)),
            kv_idx_to_layer_idx=layers,
        )

    @classmethod
    def from_config(
        cls,
        hf_config: Any,
        *,
        require_mixed: bool = False,
    ) -> RuntimeLayout:
        num_layers = int(config_get(hf_config, "num_hidden_layers"))
        if num_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_layers}.")
        layer_types = config_get(hf_config, "layer_types", None)
        full_layers = _coerce_int_list(
            "full_attention_layer_indices",
            config_get(
                hf_config,
                "full_attention_layer_indices",
                config_get(hf_config, "attention_layer_indices", None),
            ),
            allow_none=True,
        )
        linear_layers = _coerce_int_list(
            "linear_attention_layer_indices",
            config_get(hf_config, "linear_attention_layer_indices", None),
            allow_none=True,
        )

        if layer_types is not None:
            if len(layer_types) != num_layers:
                raise ValueError(
                    "layer_types length must equal num_hidden_layers: "
                    f"{len(layer_types)} != {num_layers}."
                )
            inferred_full, inferred_linear = [], []
            for layer_idx, layer_type in enumerate(layer_types):
                target = (
                    inferred_full
                    if _attention_type(layer_type) == "full"
                    else inferred_linear
                )
                target.append(layer_idx)
            full_layers = inferred_full if full_layers is None else full_layers
            linear_layers = inferred_linear if linear_layers is None else linear_layers

        if full_layers is None and linear_layers is None:
            if require_mixed:
                raise ValueError(
                    "Mixed-attention models require layer_types or explicit "
                    "full/linear attention layer indices."
                )
            return cls._with_attention_shapes(cls.dense(num_layers), hf_config)
        if full_layers is None:
            linear_set = set(linear_layers or ())
            full_layers = [idx for idx in range(num_layers) if idx not in linear_set]
        if linear_layers is None:
            full_set = set(full_layers or ())
            linear_layers = [idx for idx in range(num_layers) if idx not in full_set]

        full_tuple = tuple(sorted(int(idx) for idx in full_layers))
        linear_tuple = tuple(sorted(int(idx) for idx in linear_layers))
        full_set, linear_set = set(full_tuple), set(linear_tuple)
        expected = set(range(num_layers))
        if full_set & linear_set:
            raise ValueError(
                "RuntimeLayout full and linear layer sets overlap: "
                f"{sorted(full_set & linear_set)}."
            )
        if full_set | linear_set != expected:
            raise ValueError(
                "RuntimeLayout layer map is incomplete: "
                f"missing={sorted(expected - (full_set | linear_set))}, "
                f"extra={sorted((full_set | linear_set) - expected)}."
            )

        raw_layer_to_kv = config_get(hf_config, "layer_idx_to_kv_idx", None)
        if raw_layer_to_kv is None:
            layer_to_kv: list[int | None] = [None] * num_layers
            for kv_idx, layer_idx in enumerate(full_tuple):
                layer_to_kv[layer_idx] = kv_idx
            num_shared = int(config_get(hf_config, "num_kv_shared_layers", 0) or 0)
            if num_shared:
                if str(config_get(hf_config, "model_type", "")) != "gemma4_text":
                    raise NotImplementedError(
                        "Automatic KV sharing is only defined for Gemma 4."
                    )
                shared_start = num_layers - num_shared
                if shared_start <= 0:
                    raise ValueError(
                        "Gemma 4 num_kv_shared_layers must leave at least one "
                        f"physical KV layer, got {num_shared}/{num_layers}."
                    )
                layer_types = tuple(config_get(hf_config, "layer_types"))
                sources = {
                    layer_type: max(
                        idx
                        for idx in range(shared_start)
                        if layer_types[idx] == layer_type
                    )
                    for layer_type in set(layer_types[shared_start:])
                }
                physical_to_kv = {
                    layer_idx: kv_idx
                    for kv_idx, layer_idx in enumerate(full_tuple[:shared_start])
                }
                for layer_idx in range(shared_start, num_layers):
                    layer_to_kv[layer_idx] = physical_to_kv[
                        sources[layer_types[layer_idx]]
                    ]
        else:
            if len(raw_layer_to_kv) != num_layers:
                raise ValueError(
                    "layer_idx_to_kv_idx length must equal num_hidden_layers: "
                    f"{len(raw_layer_to_kv)} != {num_layers}."
                )
            layer_to_kv = [
                None if value is None or int(value) < 0 else int(value)
                for value in raw_layer_to_kv
            ]
            invalid_linear = [
                layer_idx
                for layer_idx in linear_tuple
                if layer_to_kv[layer_idx] is not None
            ]
            if invalid_linear:
                raise ValueError(
                    "Linear-attention layers must not have KV indices: "
                    f"{invalid_linear}."
                )

        assigned_layers = [
            layer_idx for layer_idx in full_tuple if layer_to_kv[layer_idx] is not None
        ]
        if len(assigned_layers) != len(full_tuple):
            raise ValueError(
                "RuntimeLayout must assign one KV index to each full-attention "
                f"layer: full={len(full_tuple)}, assigned={len(assigned_layers)}."
            )
        kv_indices = sorted({int(layer_to_kv[idx]) for idx in assigned_layers})
        if kv_indices != list(range(len(kv_indices))):
            raise ValueError(f"KV layer indices must be contiguous, got {kv_indices}.")
        kv_tuple = tuple(
            next(
                layer_idx
                for layer_idx in assigned_layers
                if layer_to_kv[layer_idx] == kv_idx
            )
            for kv_idx in kv_indices
        )
        configured_num_kv_layers = config_get(hf_config, "num_kv_layers", None)
        if configured_num_kv_layers is not None and int(
            configured_num_kv_layers
        ) != len(kv_tuple):
            raise ValueError(
                f"num_kv_layers={configured_num_kv_layers} does not match "
                f"physical KV layers={len(kv_tuple)}."
            )
        return cls._with_attention_shapes(
            cls(
                num_layers=num_layers,
                num_kv_layers=len(kv_tuple),
                full_attention_layer_indices=full_tuple,
                linear_attention_layer_indices=linear_tuple,
                layer_idx_to_kv_idx=tuple(layer_to_kv),
                kv_idx_to_layer_idx=kv_tuple,
            ),
            hf_config,
        )

    @classmethod
    def _with_attention_shapes(
        cls, layout: RuntimeLayout, hf_config: Any
    ) -> RuntimeLayout:
        if str(config_get(hf_config, "model_type", "")) != "gemma4_text":
            return layout
        layer_types = tuple(config_get(hf_config, "layer_types"))
        if len(layer_types) != layout.num_layers:
            raise ValueError(
                "Gemma 4 layer_types must match num_hidden_layers, "
                f"got {len(layer_types)} and {layout.num_layers}."
            )
        invalid_types = sorted(
            set(layer_types) - {"sliding_attention", "full_attention"}
        )
        if invalid_types:
            raise ValueError(f"Unsupported Gemma 4 layer types: {invalid_types}.")
        heads, dims = [], []
        for layer_idx in layout.kv_idx_to_layer_idx:
            is_full = str(layer_types[layer_idx]) == "full_attention"
            dims.append(
                int(
                    config_layer_get(
                        hf_config,
                        layer_idx,
                        "head_dim",
                        "global_head_dim" if is_full else "head_dim",
                    )
                )
            )
            heads.append(
                int(
                    config_layer_get(
                        hf_config,
                        layer_idx,
                        "num_key_value_heads",
                        "num_global_key_value_heads"
                        if is_full
                        and config_get(hf_config, "attention_k_eq_v", False)
                        else "num_key_value_heads",
                    )
                )
            )
        return replace(layout, kv_num_heads=tuple(heads), kv_head_dims=tuple(dims))

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_idx_to_kv_idx[int(layer_idx)] is not None

    def is_linear_attention(self, layer_idx: int) -> bool:
        return self.layer_idx_to_kv_idx[int(layer_idx)] is None

    def kv_layer_index(self, layer_idx: int) -> int:
        layer_idx = int(layer_idx)
        kv_idx = self.layer_idx_to_kv_idx[layer_idx]
        if kv_idx is None:
            raise RuntimeError(
                f"layer_idx={layer_idx} is linear_attention and has no KV cache."
            )
        return int(kv_idx)
