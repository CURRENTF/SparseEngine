import importlib
from types import SimpleNamespace

import pytest
import torch

def test_explicit_cpu_platform_is_lazy_and_available(monkeypatch):
    monkeypatch.setenv("SPARSEENGINE_PLATFORM", "cpu")
    platforms = importlib.import_module("sparseengine.platforms")
    platforms._set_current_platform_for_tests(None)

    platform = platforms.get_current_platform()

    assert platform.name == "cpu"
    assert platform.get_device(3) == torch.device("cpu")
    assert platform.get_distributed_backend() == "gloo"
    assert not platform.supports_graph_capture()
    caps = platform.get_device_caps()
    assert caps.platform.name == "CPU"
    assert caps.device_type == "cpu"
    assert caps.supports_bfloat16
    assert not caps.supports_native_fp8
    assert not platform.supports_inference()
    with pytest.raises(RuntimeError, match="inference is not supported"):
        platform.validate_inference()

    platforms._set_current_platform_for_tests(None)


def test_cuda_device_caps_are_the_capability_source(monkeypatch):
    from sparseengine.platforms.cuda import CudaPlatform

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test H100")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=120),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    platform = CudaPlatform()

    caps = platform.get_device_caps(7)

    assert caps.device_index == 7
    assert caps.device_name == "Test H100"
    assert caps.compute_capability == (9, 0)
    assert caps.multi_processor_count == 120
    assert caps.supports_native_fp8
    assert platform.supports_fp8()


def test_unknown_explicit_platform_fails_fast(monkeypatch):
    monkeypatch.setenv("SPARSEENGINE_PLATFORM", "missing_test_platform")
    platforms = importlib.import_module("sparseengine.platforms")
    platforms._set_current_platform_for_tests(None)

    with pytest.raises(RuntimeError, match="SPARSEENGINE_PLATFORM"):
        platforms.get_current_platform()

    platforms._set_current_platform_for_tests(None)


def test_rocm_detection_fails_fast_until_backend_exists(monkeypatch):
    monkeypatch.setenv("SPARSEENGINE_PLATFORM", "rocm")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.0.0", raising=False)
    platforms = importlib.import_module("sparseengine.platforms")
    platforms._set_current_platform_for_tests(None)

    with pytest.raises(RuntimeError, match="ROCm.*not supported"):
        platforms.get_current_platform()

    platforms._set_current_platform_for_tests(None)
