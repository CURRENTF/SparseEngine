"""Latent MLA Triton kernels with explicit Sparse-Engine contracts."""

from .copy_latent import (
    copy_latent_to_cache,
    validate_copy_slot_mapping,
    validate_copy_slot_mappings,
)
from .decode_schedule import (
    DEFAULT_GLM_MLA_DECODE_CONFIG,
    GLM_MLA_SOFTMAX_SCALE,
    MlaDecodeLaunchConfig,
    MlaDecodeWorkspace,
    allocate_mla_decode_workspace,
    prepare_mla_decode_schedule,
    required_workspace_blocks,
    run_mla_decode,
    select_glm_mla_decode_config,
    validate_mla_decode_metadata,
)
from .decode_stage1 import MLA_LATENT_DIM, MLA_ROPE_DIM, decode_stage1
from .decode_stage2 import decode_stage2
from .gather_latent import (
    gather_latent_history,
    validate_gather_metadata,
)

__all__ = [
    "DEFAULT_GLM_MLA_DECODE_CONFIG",
    "GLM_MLA_SOFTMAX_SCALE",
    "MLA_LATENT_DIM",
    "MLA_ROPE_DIM",
    "MlaDecodeLaunchConfig",
    "MlaDecodeWorkspace",
    "allocate_mla_decode_workspace",
    "copy_latent_to_cache",
    "decode_stage1",
    "decode_stage2",
    "gather_latent_history",
    "prepare_mla_decode_schedule",
    "required_workspace_blocks",
    "run_mla_decode",
    "select_glm_mla_decode_config",
    "validate_copy_slot_mapping",
    "validate_copy_slot_mappings",
    "validate_gather_metadata",
    "validate_mla_decode_metadata",
]
