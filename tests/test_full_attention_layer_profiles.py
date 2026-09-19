from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sparseengine.configs.full_attention_profiles import (
    _parse_profile_catalog,
    load_full_attention_layer_profiles,
    resolve_auto_full_attention_layers,
    resolve_full_attention_layer_profile,
)


def _profiles(*entries):
    return _parse_profile_catalog({"schema_version": 1, "profiles": list(entries)})


def _entry(profile_id="test", model_names=None, layers=None):
    return {
        "id": profile_id,
        "model_names": model_names or ["Model-X"],
        "sparse_methods": ["omnikv"],
        "full_attention_layers": [0, 3] if layers is None else layers,
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "Model-X",
        "org/Model-X",
        "/models/Model-X/",
        "/cache/models--org--Model-X/snapshots/revision",
        "model-x",
    ],
)
def test_profile_resolution_uses_exact_model_suffixes(model_name):
    profile = resolve_full_attention_layer_profile(
        model_name,
        "omnikv",
        profiles=_profiles(_entry()),
    )

    assert profile.profile_id == "test"


def test_profile_resolution_does_not_fuzzy_match_related_models():
    with pytest.raises(ValueError, match="No automatic full_attention_layers profile"):
        resolve_full_attention_layer_profile(
            "org/Model-X-Chat",
            "omnikv",
            profiles=_profiles(_entry()),
        )


def test_profile_catalog_rejects_ambiguous_aliases():
    with pytest.raises(ValueError, match="is ambiguous"):
        _profiles(
            _entry("first"),
            _entry("second", model_names=["model-x"]),
        )


@pytest.mark.parametrize("layers", [[], [True], [-1], [3, 0], [0, 0]])
def test_profile_catalog_rejects_invalid_layer_contracts(layers):
    entry = _entry()
    entry["full_attention_layers"] = layers
    with pytest.raises(ValueError, match="sorted list of unique non-negative integers"):
        _profiles(entry)


def test_non_profile_methods_resolve_auto_to_no_full_layers():
    config = SimpleNamespace(
        model="unregistered-model",
        sparse_method="quest",
        full_attention_layers="auto",
    )

    resolve_auto_full_attention_layers(config)

    assert config.full_attention_layers == []


def test_prefill_auto_uses_catalog_independently_and_excludes_sliding_layers():
    # Dense decode and an explicit decode override must both leave prefill's
    # catalog resolution intact; sliding entries cannot become score observers.
    profile = _profiles(_entry(layers=[0, 1, 3]))[0]
    for decode_layers in ("auto", [2]):
        config = SimpleNamespace(
            model="Model-X", sparse_method="", full_attention_layers=decode_layers,
            prefill_sparse_method="omnikv_prefill", omnikv_prefill_full_attention_layers="auto",
            outer_hf_config=SimpleNamespace(), runtime_layout=None,
            hf_config=SimpleNamespace(num_hidden_layers=4,
                layer_types=["sliding_attention", "full_attention"] * 2),
        )
        with patch("sparseengine.configs.full_attention_profiles.load_full_attention_layer_profiles", return_value=(profile,)):
            resolve_auto_full_attention_layers(config)
        assert config.omnikv_prefill_full_attention_layers == [1, 3]
        assert config.full_attention_layers == ([] if decode_layers == "auto" else [2])


def test_prefill_auto_rejects_missing_profile_instead_of_using_decode_layers():
    config = SimpleNamespace(
        model="Unknown", sparse_method="omnikv", full_attention_layers=[0, 2],
        prefill_sparse_method="omnikv_prefill", omnikv_prefill_full_attention_layers="auto",
        outer_hf_config=SimpleNamespace(), hf_config=SimpleNamespace(),
    )
    with patch("sparseengine.configs.full_attention_profiles.load_full_attention_layer_profiles", return_value=()):
        with pytest.raises(ValueError, match="No automatic"):
            resolve_auto_full_attention_layers(config)


def test_prefill_specific_calibration_does_not_replace_decode_profile():
    decode = _entry("decode", layers=[0, 2])
    prefill = _entry("prefill", layers=[1, 3])
    prefill["sparse_methods"] = ["omnikv_prefill"]
    profiles = _profiles(decode, prefill)
    assert resolve_full_attention_layer_profile("Model-X", "omnikv", profiles=profiles).profile_id == "decode"
    assert resolve_full_attention_layer_profile("Model-X", "omnikv_prefill", profiles=profiles).profile_id == "prefill"


def test_packaged_profile_catalog_satisfies_schema_contract():
    profiles = load_full_attention_layer_profiles()

    assert profiles
    assert all(profile.full_attention_layers for profile in profiles)


@pytest.mark.parametrize("sparse_method", ["omnikv", "deltakv"])
def test_config_auto_resolution_consumes_packaged_profile(tmp_path, sparse_method):
    from sparseengine.config import Config

    profile = load_full_attention_layer_profiles()[0]
    model_dir = tmp_path / profile.model_names[0]
    model_dir.mkdir()
    hf_config = SimpleNamespace(
        model_type="qwen2",
        dtype="float16",
        max_position_embeddings=32768,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=max(profile.full_attention_layers) + 1,
    )
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        config = Config(
            model=str(model_dir),
            sparse_method=sparse_method,
            allow_missing_deltakv_path=sparse_method == "deltakv",
        )

    assert config.full_attention_layers == list(profile.full_attention_layers)
    assert config.resolved_full_attention_profile == profile.profile_id
