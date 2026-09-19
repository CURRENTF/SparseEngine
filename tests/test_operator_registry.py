import json
import weakref
from dataclasses import dataclass
from unittest.mock import patch

import pytest

import sparseengine.operators.registry as operator_registry
from sparseengine.kernels.external.support import (
    ExternalKernelFamilyError,
    KernelFamilyHealth,
    KernelFamilyState,
    RequiredExternalKernelFamilyError,
)
from sparseengine.operators.registry import (
    NoProviderError,
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
    SupportStatus,
    operator_binding_report,
    operator_binding_reports,
    operator_runtime_stats,
    record_operator_binding,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class _Spec:
    enabled: bool = True


def _caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="test",
        compute_capability=(9, 0),
    )


def test_support_result_has_typed_atomic_eligibility() -> None:
    assert SupportResult.yes().status is SupportStatus.SUPPORTED
    assert (
        SupportResult.unsupported("wrong dtype").status
        is SupportStatus.UNSUPPORTED_CONTRACT
    )
    assert (
        SupportResult.dependency_absent("not installed").status
        is SupportStatus.DEPENDENCY_ABSENT
    )
    assert (
        SupportResult.dependency_broken("ABI mismatch").status
        is SupportStatus.DEPENDENCY_BROKEN
    )


def test_profile_overlay_is_separate_from_atomic_portfolio() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("portable",)),
        profile_order=("exact_profile",),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_atomic(ProviderRole.REPO_PORTABLE, profile_only=True)
    class ProfiledAtomic:
        name = "profiled_atomic"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

        def binding_metadata(self):
            return {"profile_id": "profile-v1"}

    @registry.register_profile
    class ExactProfile:
        name = "exact_profile"

        @classmethod
        def atomic_provider_names(cls, spec):
            del spec
            return ("profiled_atomic",)

        @classmethod
        def matches(cls, spec, caps):
            del caps
            return (
                ProfileMatch.yes("exact device/shape/toolchain match")
                if spec.enabled
                else ProfileMatch.no("shape is not profiled")
            )

        @classmethod
        def bind(cls, spec, caps):
            del spec, caps
            return ProfiledAtomic()

    profiled = OpResolver(registry).resolve(_Spec(enabled=True), _caps())
    unprofiled = OpResolver(registry).resolve(_Spec(enabled=False), _caps())
    forced = OpResolver(registry).resolve(
        _Spec(enabled=True),
        _caps(),
        force_atomic_provider="portable",
    )

    assert profiled.provider.name == "profiled_atomic"
    assert profiled.report.selected_provider == "profiled_atomic"
    assert profiled.report.selected_profile == "exact_profile"
    assert profiled.report.selection_basis == "profile_override"
    assert unprofiled.provider.name == "portable"
    assert unprofiled.report.profiles[0].matched is False
    assert unprofiled.report.candidates[1].supported is True
    assert forced.provider.name == "portable"
    assert forced.report.selected_profile is None
    assert forced.report.selection_basis == "benchmark_override"
    assert forced.report.profiles[0].matched is True


def test_forced_atomic_provider_fails_instead_of_falling_back() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("portable",)),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_atomic(ProviderRole.UPSTREAM_STANDARD, profile_only=True)
    class Unsupported:
        name = "unsupported"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.unsupported("wrong dtype")

    resolver = OpResolver(registry)
    with pytest.raises(NoProviderError, match="wrong dtype"):
        resolver.resolve(
            _Spec(),
            _caps(),
            force_atomic_provider="unsupported",
        )
    with pytest.raises(ValueError, match="Unknown forced"):
        resolver.resolve(
            _Spec(),
            _caps(),
            force_atomic_provider="missing",
        )


def test_profile_cannot_reference_ineligible_atomic_provider() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("portable",)),
        profile_order=("exact_profile",),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_atomic(ProviderRole.UPSTREAM_STANDARD, profile_only=True)
    class Ineligible:
        name = "ineligible"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.unsupported("wrong layout")

    @registry.register_profile
    class ExactProfile:
        name = "exact_profile"

        @classmethod
        def atomic_provider_names(cls, spec):
            del spec
            return ("ineligible",)

        @classmethod
        def matches(cls, spec, caps):
            raise AssertionError("ineligible profile must not be matched")

    resolved = OpResolver(registry).resolve(_Spec(), _caps())

    assert resolved.provider.name == "portable"
    assert "wrong layout" in resolved.report.profiles[0].reason


def test_profile_cannot_bind_an_undeclared_atomic_provider() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("portable",)),
        profile_order=("exact_profile",),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_atomic(ProviderRole.REPO_PORTABLE, profile_only=True)
    class Declared:
        name = "declared"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_profile
    class ExactProfile:
        name = "exact_profile"

        @classmethod
        def atomic_provider_names(cls, spec):
            del spec
            return ("declared",)

        @classmethod
        def matches(cls, spec, caps):
            return ProfileMatch.yes("matched")

        @classmethod
        def bind(cls, spec, caps):
            return Portable()

    with pytest.raises(RuntimeError, match="bound undeclared provider"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_dependency_absent_binds_degraded_portable_baseline() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(
            upstream_standard=("upstream",),
            repo_portable=("portable",),
        ),
    )

    @registry.register_atomic(ProviderRole.UPSTREAM_STANDARD)
    class Upstream:
        name = "upstream"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.dependency_absent("upstream package is absent")

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    resolved = OpResolver(registry).resolve(_Spec(), _caps())

    assert resolved.provider.name == "portable"
    assert resolved.report.selection_basis == "dependency_degraded"
    assert {
        candidate.provider: candidate.status
        for candidate in resolved.report.candidates
    }["upstream"] == "dependency_absent"


def test_absent_external_family_is_typed_and_does_not_abort_iteration() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(
            upstream_standard=("upstream",),
            repo_portable=("portable",),
        ),
    )
    health = KernelFamilyHealth(
        family="upstream",
        state=KernelFamilyState.ABSENT,
        version=None,
        reason="not installed",
    )

    @registry.register_atomic(ProviderRole.UPSTREAM_STANDARD)
    class Upstream:
        name = "upstream"

        @classmethod
        def supports(cls, spec, caps):
            raise ExternalKernelFamilyError(health, feature="test op")

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    resolved = OpResolver(registry).resolve(_Spec(), _caps())

    assert resolved.provider.name == "portable"
    assert {
        candidate.provider: candidate.status
        for candidate in resolved.report.candidates
    }["upstream"] == "dependency_absent"


def test_absent_required_external_family_fails_with_install_hint() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(
            upstream_standard=("upstream",),
            repo_portable=("portable",),
        ),
    )
    health = KernelFamilyHealth(
        family="upstream",
        state=KernelFamilyState.ABSENT,
        version=None,
        reason="not installed",
    )

    @registry.register_atomic(ProviderRole.UPSTREAM_STANDARD)
    class Upstream:
        name = "upstream"

        @classmethod
        def supports(cls, spec, caps):
            raise RequiredExternalKernelFamilyError(health, feature="test op")

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    with pytest.raises(
        RequiredExternalKernelFamilyError,
        match=r'pip install -e "\.\[cu129\]"',
    ):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_dependency_broken_fails_binding() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(
            upstream_standard=("upstream",),
            repo_portable=("portable",),
        ),
    )

    @registry.register_atomic(ProviderRole.UPSTREAM_STANDARD)
    class Upstream:
        name = "upstream"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.dependency_broken("undefined symbol")

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    with pytest.raises(RuntimeError, match="broken dependency.*undefined symbol"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_registry_rejects_role_policy_mismatch() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(upstream_standard=("same",)),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Provider:
        name = "same"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    with pytest.raises(ValueError, match="declared as repo_portable"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_registry_rejects_implicit_hidden_atomic_provider() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("portable",)),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Hidden:
        name = "hidden"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    with pytest.raises(ValueError, match="must appear in the default portfolio"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_registry_rejects_duplicate_atomic_provider_names() -> None:
    registry = OpRegistry("_test")

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class First:
        name = "same"

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_atomic(ProviderRole.REPO_PORTABLE)
        class Second:
            name = "same"


@pytest.mark.parametrize(
    ("atomic_names", "error"),
    [
        ((), "at least one atomic provider"),
        (("portable", "portable"), "duplicate atomic providers"),
        (("missing",), "unregistered atomic providers"),
    ],
)
def test_registry_rejects_invalid_profile_atomic_references(
    atomic_names,
    error,
) -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("portable",)),
        profile_order=("invalid",),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_profile
    class InvalidProfile:
        name = "invalid"

        @classmethod
        def atomic_provider_names(cls, spec):
            del spec
            return atomic_names

        @classmethod
        def matches(cls, spec, caps):
            raise AssertionError("invalid profile must not be matched")

    with pytest.raises(ValueError, match=error):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_profile_must_bind_a_plan_explicitly() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("portable",)),
        profile_order=("invalid",),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Portable:
        name = "portable"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register_profile
    class InvalidProfile:
        name = "invalid"

        @classmethod
        def atomic_provider_names(cls, spec):
            del spec
            return ("portable",)

        @classmethod
        def matches(cls, spec, caps):
            return ProfileMatch.yes("matched")

    with pytest.raises(TypeError, match="must define bind"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_binding_report_is_immutable_and_json_ready() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("configured",)),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Configured:
        name = "configured"

        def __init__(self, marker):
            self.marker = marker

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

        def binding_metadata(self):
            return {"layout_id": "test_layout", "marker": self.marker}

    with (
        patch.dict(operator_registry._OPERATOR_BINDINGS, {}, clear=True),
        patch.object(
            operator_registry,
            "_BINDING_REPORTS",
            weakref.WeakKeyDictionary(),
        ),
    ):
        resolved = OpResolver(registry).resolve(_Spec(), _caps(), marker="ready")
        assert operator_binding_report(resolved.provider) is resolved.report
        reports = operator_binding_reports()

    assert resolved.provider.marker == "ready"
    assert resolved.report.as_dict()["provider_metadata"] == {
        "layout_id": "test_layout",
        "marker": "ready",
    }
    assert reports == [{**resolved.report.as_dict(), "bound_provider_count": 1}]
    json.dumps(reports)


def test_resolver_rejects_invalid_binding_metadata() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("invalid",)),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Invalid:
        name = "invalid"

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

        def binding_metadata(self):
            return "not structured"

    with pytest.raises(TypeError, match="binding_metadata.*dict"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_resolver_uses_atomic_bind_hook() -> None:
    registry = OpRegistry(
        "_test",
        portfolio=PortfolioPolicy(repo_portable=("prepared",)),
    )

    @registry.register_atomic(ProviderRole.REPO_PORTABLE)
    class Prepared:
        name = "prepared"

        def __init__(self, spec, marker):
            self.spec = spec
            self.marker = marker

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

        @classmethod
        def bind(cls, spec, caps, *, marker):
            assert caps.device_name == "test"
            return cls(spec, marker)

    resolved = OpResolver(registry).resolve(_Spec(), _caps(), marker="ready")

    assert resolved.provider.spec == _Spec()
    assert resolved.provider.marker == "ready"


def test_operator_runtime_stats_aggregate_live_provider_kernel_paths() -> None:
    class Provider:
        name = "composite"

        def __init__(self, eager: int, captured: int, fallback: int):
            self.eager = eager
            self.captured = captured
            self.fallback = fallback

        def runtime_kernel_stats(self):
            return {
                "kernel_paths": {
                    "tilelang_score": {
                        "eager_dispatches": self.eager,
                        "cuda_graph_capture_dispatches": self.captured,
                    }
                },
                "fallback_reasons": {"noncontiguous:output": self.fallback},
            }

    first = Provider(2, 1, 0)
    second = Provider(3, 4, 1)
    with patch.dict(operator_registry._OPERATOR_BINDINGS, {}, clear=True):
        record_operator_binding("MLA attention", first)
        record_operator_binding("MLA attention", second)
        stats = operator_runtime_stats()

    assert stats == {
        "MLA attention": [
            {
                "implementation": "composite",
                "bound_provider_count": 2,
                "instrumented_provider_count": 2,
                "kernel_paths": {
                    "tilelang_score": {
                        "cuda_graph_capture_dispatches": 5,
                        "eager_dispatches": 5,
                    }
                },
                "fallback_reasons": {"noncontiguous:output": 1},
            }
        ]
    }


def test_empty_registry_fails_with_explicit_reason() -> None:
    with pytest.raises(RuntimeError, match="no default portfolio providers"):
        OpResolver(OpRegistry("_empty")).resolve(_Spec(), _caps())
