import re

import torch

from sparseengine.utils.weight_target import WeightTarget


_MLP_EXPERT_PREFIX = r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\."
MLP_EXPERT_SOURCE_RE = re.compile(_MLP_EXPERT_PREFIX + r"weight$")
MLP_EXPERT_TARGET_RE = re.compile(_MLP_EXPERT_PREFIX + r"expert_weight$")


def expert_weight_target(experts_at, layer_idx, expert_id, projection) -> WeightTarget:
    return WeightTarget(experts_at(int(layer_idx)), (int(expert_id), projection))


def load_expert_weight(
    target: WeightTarget | None,
    loaded_weight: torch.Tensor,
    loaded_scale: torch.Tensor | None,
) -> int:
    if target is None:
        return 0
    expert_id, projection = target.shard_id
    target.module.load_expert_weight(expert_id, projection, loaded_weight, loaded_scale)
    return 1


class ExpertWeightLoader:
    expert_target_pattern = MLP_EXPERT_TARGET_RE
    expert_block_name = "mlp"

    def _experts_at(self, layer_idx):
        return getattr(self.model.layers[layer_idx], self.expert_block_name).experts

    def resolve_special_weight(self, target_weight_name: str) -> WeightTarget | None:
        match = self.expert_target_pattern.match(target_weight_name)
        if match is None:
            return None
        return expert_weight_target(self._experts_at, *match.groups())

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        return load_expert_weight(
            self.resolve_special_weight(target_weight_name), loaded_weight, loaded_scale
        )


def nonexpert_parameter_names(model, expert_prefix: str = ".experts.") -> set[str]:
    packed_suffixes = (expert_prefix + "w13_weight", expert_prefix + "w2_weight")
    return {name for name, _ in model.named_parameters() if not name.endswith(packed_suffixes)}


def expert_matrix_shape(experts, *, down: bool) -> tuple[int, int]:
    hidden, intermediate = experts.hidden_size, experts.global_intermediate_size
    return (hidden, intermediate) if down else (intermediate, hidden)


def check_remote_shape(label, source_name, actual_shape, expected_shape) -> None:
    if actual_shape != expected_shape:
        raise ValueError(
            f"Remote {label} shape mismatch for {source_name!r}: "
            f"expected={expected_shape}, got={actual_shape}."
        )


def check_remote_fp8_dtype(model_name, dtype) -> None:
    if dtype != "F8_E4M3":
        raise TypeError(
            f"Remote {model_name} expert weight must be FP8 E4M3, got safetensors dtype {dtype}."
        )


def check_remote_bf16_scale(
    model_name, scale_label, source_name, scale_shape, scale_dtype, weight_shape
) -> None:
    expected_scale_shape = tuple(size // 128 for size in weight_shape)
    check_remote_shape(scale_label, source_name, scale_shape, expected_scale_shape)
    if scale_dtype != "BF16":
        raise TypeError(
            f"Remote {model_name} expert scale must be BF16, got safetensors dtype {scale_dtype}."
        )


def expert_scale_name(weight_name: str) -> str:
    return weight_name[:-len(".weight")] + ".weight_scale_inv"


def expected_remote_expert_weights(
    experts_at, num_layers, num_experts, projections, *, layer_prefix, block_name="mlp"
) -> set[str]:
    return {
        f"{layer_prefix}.{layer_idx}.{block_name}.experts.{expert_id}.{projection}.weight"
        for layer_idx in range(int(num_layers))
        for expert_id in range(int(num_experts))
        if not experts_at(layer_idx).is_local_expert(expert_id)
        for projection in projections
    }


def remote_expert_skip_differences(
    expected_weights, skipped_weights, skipped_scales, *, fp8
) -> tuple[list[str], list[str], list[str], list[str]]:
    expected_scales = {expert_scale_name(name) for name in expected_weights} if fp8 else set()
    return (
        sorted(expected_weights - skipped_weights),
        sorted(skipped_weights - expected_weights),
        sorted(expected_scales - skipped_scales),
        sorted(skipped_scales - expected_scales),
    )


def warmup_mha_projections(attn, hidden_states) -> None:
    attn.qkv_proj(hidden_states)
    attn.o_proj(hidden_states.new_zeros((hidden_states.shape[0], attn.q_size)))


def warmup_local_experts(
    experts, hidden_states, top_k, *, num_experts, weights_dtype, ids_dtype=torch.int64
) -> None:
    num_tokens = hidden_states.shape[0]
    device = hidden_states.device
    ids = (
        torch.arange(num_tokens * top_k, dtype=ids_dtype, device=device)
        .remainder(num_experts)
        .add(experts.local_expert_start)
        .view(num_tokens, top_k)
    )
    weights = torch.full((num_tokens, top_k), 1.0 / top_k, dtype=weights_dtype, device=device)
    experts(hidden_states, ids, weights)
