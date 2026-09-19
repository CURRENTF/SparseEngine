import json
from types import SimpleNamespace

import torch

from sparseengine.utils.loader import sync_deltakv_config_from_checkpoint


def _base_config(tmp_path):
    return SimpleNamespace(
        deltakv_checkpoint_path=str(tmp_path),
        sparse_method="deltakv",
        deltakv_latent_dim=128,
        use_nonlinear_compressor=True,
        compressor_intermediate_size=2048,
        compressor_linear_bias=True,
        compressor_down_type="auto",
        compressor_up_type="auto",
        compressor_down_intermediate_size=-1,
        compressor_up_intermediate_size=-1,
    )


def test_syncs_from_checkpoint_config_json(tmp_path):
    cfg = _base_config(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "deltakv_latent_dim": 256,
                "compressor_down_type": "mlp_swiglu",
                "compressor_up_type": "linear",
                "compressor_down_intermediate_size": 3072,
                "compressor_up_intermediate_size": -1,
                "compressor_linear_bias": False,
                "use_nonlinear_compressor": True,
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {"model.layers.0.self_attn.compress_down.w3.weight": torch.randn(256, 3072)},
        tmp_path / "model.pt",
    )

    changed = sync_deltakv_config_from_checkpoint(cfg)

    assert changed is True
    assert cfg.deltakv_latent_dim == 256
    assert cfg.compressor_down_type == "mlp_swiglu"
    assert cfg.compressor_up_type == "linear"
    assert cfg.compressor_down_intermediate_size == 3072
    assert cfg.compressor_up_intermediate_size == -1
    assert cfg.compressor_linear_bias is False


def test_falls_back_to_weight_shape_when_config_missing(tmp_path):
    cfg = _base_config(tmp_path)
    torch.save(
        {
            "model.layers.0.self_attn.compress_down.w12.weight": torch.randn(6144, 256),
            "model.layers.0.self_attn.compress_down.w3.weight": torch.randn(256, 3072),
            "model.layers.0.self_attn.compress_up.weight": torch.randn(1024, 256),
        },
        tmp_path / "model.pt",
    )

    changed = sync_deltakv_config_from_checkpoint(cfg)

    assert changed is True
    assert cfg.deltakv_latent_dim == 256
    assert cfg.compressor_down_type == "mlp_swiglu"
    assert cfg.compressor_up_type == "linear"
    assert cfg.compressor_down_intermediate_size == 3072
    assert cfg.compressor_up_intermediate_size == -1
