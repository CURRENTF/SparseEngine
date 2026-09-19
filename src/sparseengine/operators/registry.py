from __future__ import annotations

import re
import weakref
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

from sparseengine.platforms.interface import DeviceCaps
from sparseengine.utils.log import logger


SpecT = TypeVar("SpecT")
ProviderT = TypeVar("ProviderT", bound="OperatorProvider")


_OPERATOR_BINDINGS: dict[str, weakref.WeakSet[object]] = {}
_BINDING_REPORTS: weakref.WeakKeyDictionary[object, BindingReport] = (
    weakref.WeakKeyDictionary()
)


def record_operator_binding(
    operator_type: str,
    provider: object,
    *,
    report: BindingReport | None = None,
) -> None:
    _OPERATOR_BINDINGS.setdefault(operator_type, weakref.WeakSet()).add(provider)
    if report is not None:
        _BINDING_REPORTS[provider] = report


def _implementation_name(provider: object) -> str:
    return (
        getattr(provider, "implementation_name", None)
        or getattr(provider, "name", None)
        or provider.provider_name
    )


@dataclass(frozen=True, slots=True)
class _FrozenMapping:
    items: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _FrozenSequence:
    items: tuple[object, ...]


def _freeze_snapshot(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _FrozenMapping(
            tuple(
                (field.name, _freeze_snapshot(getattr(value, field.name)))
                for field in fields(value)
            )
        )
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return _FrozenMapping(
            tuple(
                (str(key), _freeze_snapshot(item))
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            )
        )
    if isinstance(value, (list, tuple)):
        return _FrozenSequence(tuple(_freeze_snapshot(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _thaw_snapshot(value: object) -> object:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_snapshot(item) for key, item in value.items}
    if isinstance(value, _FrozenSequence):
        return [_thaw_snapshot(item) for item in value.items]
    return value


@dataclass(frozen=True, slots=True)
class ProviderDecision:
    provider: str
    role: str
    portfolio_member: bool
    profile_only: bool
    status: str
    reason: str

    @property
    def supported(self) -> bool:
        return self.status == SupportStatus.SUPPORTED.value

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "role": self.role,
            "portfolio_member": self.portfolio_member,
            "profile_only": self.profile_only,
            "status": self.status,
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProfileDecision:
    profile: str
    matched: bool
    reason: str
    atomic_providers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "matched": self.matched,
            "reason": self.reason,
            "atomic_providers": list(self.atomic_providers),
        }


@dataclass(frozen=True, slots=True)
class BindingReport:
    operator_type: str
    spec_type: str
    spec: object
    device_caps: object
    selected_provider: str
    selected_profile: str | None
    selected_role: str | None
    selection_basis: str
    selected_reason: str
    candidates: tuple[ProviderDecision, ...]
    profiles: tuple[ProfileDecision, ...] = ()
    provider_metadata: _FrozenMapping = _FrozenMapping(())

    @property
    def rejected(self) -> tuple[ProviderDecision, ...]:
        return tuple(
            candidate for candidate in self.candidates if not candidate.supported
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "operator_type": self.operator_type,
            "spec_type": self.spec_type,
            "spec": _thaw_snapshot(self.spec),
            "device_caps": _thaw_snapshot(self.device_caps),
            "selected_provider": self.selected_provider,
            "selected_profile": self.selected_profile,
            "selected_role": self.selected_role,
            "selection_basis": self.selection_basis,
            "selected_reason": self.selected_reason,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "profiles": [profile.as_dict() for profile in self.profiles],
            "provider_metadata": _thaw_snapshot(self.provider_metadata),
        }


def operator_binding_reports() -> list[dict[str, object]]:
    """Return stable, JSON-ready decisions for every live resolver binding."""

    counts: dict[BindingReport, int] = {}
    for report in _BINDING_REPORTS.values():
        counts[report] = counts.get(report, 0) + 1
    rows = []
    for report, count in sorted(
        counts.items(),
        key=lambda item: (
            item[0].operator_type,
            item[0].selected_provider,
            repr(item[0].spec),
        ),
    ):
        row = report.as_dict()
        row["bound_provider_count"] = count
        rows.append(row)
    return rows


def operator_binding_report(provider: object) -> BindingReport | None:
    """Return the immutable resolver decision associated with one provider."""

    return _BINDING_REPORTS.get(provider)


def operator_runtime_stats() -> dict[str, list[dict[str, object]]]:
    """Return aggregate runtime kernel paths for every live bound provider."""

    result: dict[str, list[dict[str, object]]] = {}
    for operator_type, providers in sorted(_OPERATOR_BINDINGS.items()):
        grouped: dict[str, list[object]] = {}
        for provider in providers:
            grouped.setdefault(_implementation_name(provider), []).append(provider)
        entries: list[dict[str, object]] = []
        for implementation, bound_providers in sorted(grouped.items()):
            kernel_paths: dict[str, dict[str, int]] = {}
            fallback_reasons: dict[str, int] = {}
            instrumented = 0
            for provider in bound_providers:
                stats_fn = getattr(provider, "runtime_kernel_stats", None)
                if not callable(stats_fn):
                    continue
                instrumented += 1
                stats = stats_fn()
                for path, counts in stats.get("kernel_paths", {}).items():
                    aggregate = kernel_paths.setdefault(str(path), {})
                    for key, count in counts.items():
                        aggregate[str(key)] = int(aggregate.get(str(key), 0)) + int(count)
                for reason, count in stats.get("fallback_reasons", {}).items():
                    fallback_reasons[str(reason)] = (
                        int(fallback_reasons.get(str(reason), 0)) + int(count)
                    )
            entries.append(
                {
                    "implementation": implementation,
                    "bound_provider_count": len(bound_providers),
                    "instrumented_provider_count": instrumented,
                    "kernel_paths": {
                        path: dict(sorted(counts.items()))
                        for path, counts in sorted(kernel_paths.items())
                    },
                    "fallback_reasons": dict(sorted(fallback_reasons.items())),
                }
            )
        if entries:
            result[operator_type] = entries
    return result


def log_operator_implementations() -> None:
    entries = sorted(
        (
            operator_type,
            ", ".join(
                sorted({_implementation_name(provider) for provider in providers})
            ),
        )
        for operator_type, providers in _OPERATOR_BINDINGS.items()
        if providers
    )
    if not entries:
        return
    rows = "\n".join(
        f"  {operator_type}: {implementation}"
        for operator_type, implementation in entries
    )
    logger.info("Operator implementations:\n{}", rows)
    runtime_rows = []
    for operator_type, implementations in operator_runtime_stats().items():
        for implementation in implementations:
            for path, counts in implementation["kernel_paths"].items():
                runtime_rows.append(
                    f"  {operator_type}/{implementation['implementation']}/{path}: "
                    + ", ".join(
                        f"{key}={value}" for key, value in counts.items()
                    )
                )
    if runtime_rows:
        logger.info("Operator runtime kernels:\n{}", "\n".join(runtime_rows))
    binding_rows = []
    for report in operator_binding_reports():
        rejected = [
            f"{candidate['provider']}: {candidate['reason']}"
            for candidate in report["candidates"]
            if not candidate["supported"]
        ]
        binding_rows.append(
            f"  {report['operator_type']}: selected={report['selected_provider']} "
            f"basis={report['selection_basis']} "
            f"count={report['bound_provider_count']} "
            f"rejected={rejected or 'none'}"
        )
    if binding_rows:
        logger.info("Operator binding decisions:\n{}", "\n".join(binding_rows))


def runtime_version_at_least(
    version: str | None,
    minimum: tuple[int, int],
) -> bool:
    if version is None:
        return False
    match = re.match(r"^\s*(\d+)\.(\d+)", str(version))
    if match is None:
        return False
    current = tuple(map(int, match.groups()))
    return current >= minimum


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED_CONTRACT = "unsupported_contract"
    DEPENDENCY_ABSENT = "dependency_absent"
    DEPENDENCY_BROKEN = "dependency_broken"


class ProviderRole(str, Enum):
    UPSTREAM_STANDARD = "upstream_standard"
    REPO_PORTABLE = "repo_portable"
    REPO_NONSTANDARD = "repo_nonstandard"


class SelectionBasis(str, Enum):
    BENCHMARK_OVERRIDE = "benchmark_override"
    PROFILE_OVERRIDE = "profile_override"
    UPSTREAM_DEFAULT = "upstream_default"
    SEMANTIC_FALLBACK = "semantic_fallback"
    DEPENDENCY_DEGRADED = "dependency_degraded"


@dataclass(frozen=True, slots=True)
class SupportResult:
    status: SupportStatus
    reason: str

    @property
    def supported(self) -> bool:
        return self.status is SupportStatus.SUPPORTED

    @classmethod
    def yes(cls, reason: str = "supported") -> "SupportResult":
        return cls(SupportStatus.SUPPORTED, reason)

    @classmethod
    def unsupported(cls, reason: str) -> "SupportResult":
        return cls(SupportStatus.UNSUPPORTED_CONTRACT, reason)

    @classmethod
    def dependency_absent(cls, reason: str) -> "SupportResult":
        return cls(SupportStatus.DEPENDENCY_ABSENT, reason)

    @classmethod
    def dependency_broken(cls, reason: str) -> "SupportResult":
        return cls(SupportStatus.DEPENDENCY_BROKEN, reason)


@dataclass(frozen=True, slots=True)
class ProfileMatch:
    matched: bool
    reason: str

    @classmethod
    def yes(cls, reason: str) -> "ProfileMatch":
        return cls(True, reason)

    @classmethod
    def no(cls, reason: str) -> "ProfileMatch":
        return cls(False, reason)


@dataclass(frozen=True, slots=True)
class PortfolioPolicy:
    upstream_standard: tuple[str, ...] = ()
    repo_portable: tuple[str, ...] = ()
    repo_nonstandard: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = self.ordered_provider_names
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Portfolio policy contains duplicate providers: {duplicates}."
            )

    @property
    def ordered_provider_names(self) -> tuple[str, ...]:
        return (
            *self.upstream_standard,
            *self.repo_portable,
            *self.repo_nonstandard,
        )

    def role_for(self, provider_name: str) -> ProviderRole | None:
        if provider_name in self.upstream_standard:
            return ProviderRole.UPSTREAM_STANDARD
        if provider_name in self.repo_portable:
            return ProviderRole.REPO_PORTABLE
        if provider_name in self.repo_nonstandard:
            return ProviderRole.REPO_NONSTANDARD
        return None


class OperatorProvider(Protocol[SpecT]):
    name: str

    @classmethod
    def supports(cls, spec: SpecT, caps: DeviceCaps) -> SupportResult: ...


class ProfileOverlay(Protocol[SpecT, ProviderT]):
    name: str

    @classmethod
    def atomic_provider_names(cls, spec: SpecT) -> tuple[str, ...]: ...

    @classmethod
    def matches(cls, spec: SpecT, caps: DeviceCaps) -> ProfileMatch: ...

    @classmethod
    def bind(cls, spec: SpecT, caps: DeviceCaps, **provider_kwargs) -> ProviderT: ...


@dataclass(frozen=True, slots=True)
class AtomicProviderRegistration(Generic[ProviderT]):
    provider: type[ProviderT]
    role: ProviderRole
    profile_only: bool


class AtomicRegistry(Generic[SpecT, ProviderT]):
    def __init__(self, family: str) -> None:
        self.family = str(family)
        self._providers: dict[str, AtomicProviderRegistration[ProviderT]] = {}

    def register(self, role: ProviderRole, *, profile_only: bool = False):
        def decorator(provider: type[ProviderT]) -> type[ProviderT]:
            self._register(provider, role, profile_only=profile_only)
            return provider

        return decorator

    def _register(
        self,
        provider: type[ProviderT],
        role: ProviderRole,
        *,
        profile_only: bool,
    ) -> None:
        if not isinstance(role, ProviderRole):
            raise TypeError(
                f"Atomic provider role must be ProviderRole, got "
                f"{type(role).__name__}."
            )
        name = str(provider.name)
        if not name:
            raise ValueError("Atomic provider name must not be empty.")
        if name in self._providers:
            raise ValueError(
                f"Provider {name!r} is already registered for {self.family!r}."
            )
        self._providers[name] = AtomicProviderRegistration(
            provider,
            role,
            bool(profile_only),
        )

    @property
    def registrations(self) -> tuple[AtomicProviderRegistration[ProviderT], ...]:
        return tuple(self._providers.values())

    def registration(self, name: str) -> AtomicProviderRegistration[ProviderT]:
        try:
            return self._providers[str(name)]
        except KeyError as error:
            raise ValueError(
                f"Unknown atomic provider {name!r} for {self.family!r}."
            ) from error

    @property
    def providers(self) -> tuple[type[ProviderT], ...]:
        return tuple(item.provider for item in self._providers.values())


class ProfileRegistry(Generic[SpecT, ProviderT]):
    def __init__(self, family: str) -> None:
        self.family = str(family)
        self._profiles: dict[str, type[ProfileOverlay[SpecT, ProviderT]]] = {}

    def register(
        self,
        profile: type[ProfileOverlay[SpecT, ProviderT]],
    ) -> type[ProfileOverlay[SpecT, ProviderT]]:
        name = str(profile.name)
        if not name:
            raise ValueError("Profile name must not be empty.")
        if name in self._profiles:
            raise ValueError(
                f"Profile {name!r} is already registered for {self.family!r}."
            )
        self._profiles[name] = profile
        return profile

    def profile(self, name: str) -> type[ProfileOverlay[SpecT, ProviderT]]:
        try:
            return self._profiles[str(name)]
        except KeyError as error:
            raise ValueError(
                f"Unknown profile {name!r} for {self.family!r}."
            ) from error

    @property
    def profiles(self) -> tuple[type[ProfileOverlay[SpecT, ProviderT]], ...]:
        return tuple(self._profiles.values())


class OpRegistry(Generic[SpecT, ProviderT]):
    def __init__(
        self,
        family: str,
        *,
        portfolio: PortfolioPolicy | None = None,
        profile_order: tuple[str, ...] = (),
    ) -> None:
        self.family = str(family)
        self.portfolio = portfolio or PortfolioPolicy()
        self.profile_order = tuple(profile_order)
        if len(set(self.profile_order)) != len(self.profile_order):
            raise ValueError("Profile order must not contain duplicates.")
        self.atomic_registry: AtomicRegistry[SpecT, ProviderT] = AtomicRegistry(
            self.family
        )
        self.profile_registry: ProfileRegistry[SpecT, ProviderT] = ProfileRegistry(
            self.family
        )

    def register_atomic(
        self,
        role: ProviderRole,
        *,
        profile_only: bool = False,
    ):
        return self.atomic_registry.register(role, profile_only=profile_only)

    def register_profile(
        self,
        profile: type[ProfileOverlay[SpecT, ProviderT]],
    ) -> type[ProfileOverlay[SpecT, ProviderT]]:
        return self.profile_registry.register(profile)

    @property
    def providers(self) -> tuple[type[ProviderT], ...]:
        return self.atomic_registry.providers

    def validate(self) -> None:
        registered = {
            item.provider.name: item
            for item in self.atomic_registry.registrations
        }
        portfolio_names = set(self.portfolio.ordered_provider_names)
        for name, registration in registered.items():
            if registration.profile_only == (name in portfolio_names):
                expected = (
                    "be omitted from the default portfolio"
                    if registration.profile_only
                    else "appear in the default portfolio"
                )
                raise ValueError(
                    f"Atomic provider {name!r} must {expected} for "
                    f"{self.family!r}."
                )
        for name in self.portfolio.ordered_provider_names:
            if name not in registered:
                raise ValueError(
                    f"Portfolio provider {name!r} is not registered for "
                    f"{self.family!r}."
                )
            expected_role = self.portfolio.role_for(name)
            actual_role = registered[name].role
            if actual_role is not expected_role:
                raise ValueError(
                    f"Portfolio provider {name!r} is declared as {actual_role.value}, "
                    f"but policy places it under {expected_role.value}."
                )
        registered_profiles = {
            profile.name for profile in self.profile_registry.profiles
        }
        if registered_profiles != set(self.profile_order):
            missing = sorted(registered_profiles - set(self.profile_order))
            unknown = sorted(set(self.profile_order) - registered_profiles)
            raise ValueError(
                f"Profile order mismatch for {self.family!r}: "
                f"missing={missing}, unknown={unknown}."
            )


@dataclass(frozen=True)
class ResolvedProvider(Generic[ProviderT]):
    provider: ProviderT
    rejected: tuple[tuple[str, str], ...]
    report: BindingReport


class NoProviderError(RuntimeError):
    """No registered provider satisfies the requested semantic contract."""


class OpResolver(Generic[SpecT, ProviderT]):
    def __init__(self, registry: OpRegistry[SpecT, ProviderT]) -> None:
        self.registry = registry

    def resolve(
        self,
        spec: SpecT,
        caps: DeviceCaps,
        *,
        force_atomic_provider: str | None = None,
        **provider_kwargs,
    ) -> ResolvedProvider[ProviderT]:
        self.registry.validate()
        results: dict[str, SupportResult] = {}
        rejected: list[tuple[str, str]] = []
        decisions: list[ProviderDecision] = []
        portfolio_names = set(self.registry.portfolio.ordered_provider_names)
        for registration in self.registry.atomic_registry.registrations:
            provider = registration.provider
            try:
                result = provider.supports(spec, caps)
            except Exception as error:
                result = self._external_dependency_result(error)
                if result is None:
                    raise
            if not isinstance(result, SupportResult):
                raise TypeError(
                    f"Atomic provider {provider.name!r} supports() must return "
                    f"SupportResult, got {type(result).__name__}."
                )
            if result.status is SupportStatus.DEPENDENCY_BROKEN:
                raise RuntimeError(
                    f"Atomic provider {provider.name!r} has a broken dependency: "
                    f"{result.reason}"
                )
            results[str(provider.name)] = result
            decisions.append(
                ProviderDecision(
                    provider=str(provider.name),
                    role=registration.role.value,
                    portfolio_member=provider.name in portfolio_names,
                    profile_only=registration.profile_only,
                    status=result.status.value,
                    reason=result.reason,
                )
            )
            if not result.supported:
                rejected.append((provider.name, result.reason))

        profile_decisions: list[ProfileDecision] = []
        selected_profile: type[ProfileOverlay[SpecT, ProviderT]] | None = None
        selected_profile_atomic_names: tuple[str, ...] = ()
        selected_reason = ""
        for profile_name in self.registry.profile_order:
            profile = self.registry.profile_registry.profile(profile_name)
            atomic_names = tuple(profile.atomic_provider_names(spec))
            if not atomic_names:
                raise ValueError(
                    f"Profile {profile.name!r} must declare at least one atomic "
                    "provider."
                )
            if len(set(atomic_names)) != len(atomic_names):
                raise ValueError(
                    f"Profile {profile.name!r} declares duplicate atomic "
                    f"providers: {atomic_names}."
                )
            unknown = sorted(set(atomic_names) - set(results))
            if unknown:
                raise ValueError(
                    f"Profile {profile.name!r} references unregistered atomic "
                    f"providers: {unknown}."
                )
            unavailable = [
                name
                for name in atomic_names
                if not results[name].supported
            ]
            if unavailable:
                reason = "; ".join(
                    f"{name}: "
                    + results[name].reason
                    for name in unavailable
                )
                match = ProfileMatch.no(
                    f"required atomic providers are not eligible: {reason}"
                )
            else:
                match = profile.matches(spec, caps)
                if not isinstance(match, ProfileMatch):
                    raise TypeError(
                        f"Profile {profile.name!r} matches() must return "
                        f"ProfileMatch, got {type(match).__name__}."
                    )
            profile_decisions.append(
                ProfileDecision(
                    profile=str(profile.name),
                    matched=match.matched,
                    reason=match.reason,
                    atomic_providers=atomic_names,
                )
            )
            if match.matched and selected_profile is None:
                selected_profile = profile
                selected_profile_atomic_names = atomic_names
                selected_reason = match.reason

        selected_type: type[ProviderT] | type[ProfileOverlay[SpecT, ProviderT]]
        selected_role: ProviderRole | None
        selection_basis: SelectionBasis
        if force_atomic_provider is not None:
            forced_name = str(force_atomic_provider).strip()
            if not forced_name:
                raise ValueError("force_atomic_provider must not be empty.")
            if forced_name not in results:
                available = sorted(results)
                raise ValueError(
                    f"Unknown forced {self.registry.family} atomic provider "
                    f"{forced_name!r}; available={available}."
                )
            selected_registration = self.registry.atomic_registry.registration(
                forced_name
            )
            result = results[forced_name]
            if not result.supported:
                raise NoProviderError(
                    f"Forced {self.registry.family} provider {forced_name!r} "
                    f"does not support spec={spec!r} on "
                    f"device={caps.device_name!r}: {result.reason}."
                )
            selected_profile = None
            selected_profile_atomic_names = ()
            selected_type = selected_registration.provider
            selected_role = selected_registration.role
            selected_reason = f"benchmark override: {result.reason}"
            selection_basis = SelectionBasis.BENCHMARK_OVERRIDE
        elif selected_profile is not None:
            selected_type = selected_profile
            selected_role = None
            selection_basis = SelectionBasis.PROFILE_OVERRIDE
        else:
            selected_registration = None
            for provider_name in self.registry.portfolio.ordered_provider_names:
                result = results[provider_name]
                if result.supported:
                    selected_registration = self.registry.atomic_registry.registration(
                        provider_name
                    )
                    selected_reason = result.reason
                    break
            if selected_registration is None:
                details = "; ".join(
                    f"{name}: {reason}" for name, reason in rejected
                )
                raise NoProviderError(
                    f"No {self.registry.family} provider supports spec={spec!r} on "
                    f"device={caps.device_name!r}: "
                    f"{details or 'no default portfolio providers registered'}."
                )
            selected_type = selected_registration.provider
            selected_role = selected_registration.role
            selected_index = self.registry.portfolio.ordered_provider_names.index(
                selected_registration.provider.name
            )
            dependency_degraded = any(
                results[name].status is SupportStatus.DEPENDENCY_ABSENT
                for name in self.registry.portfolio.ordered_provider_names[
                    :selected_index
                ]
            ) or any(
                decision.status == SupportStatus.DEPENDENCY_ABSENT.value
                and decision.role == ProviderRole.UPSTREAM_STANDARD.value
                and decision.profile_only
                for decision in decisions
            )
            if dependency_degraded:
                selection_basis = SelectionBasis.DEPENDENCY_DEGRADED
            elif selected_role is ProviderRole.UPSTREAM_STANDARD:
                selection_basis = SelectionBasis.UPSTREAM_DEFAULT
            else:
                selection_basis = SelectionBasis.SEMANTIC_FALLBACK

        bind_fn = getattr(selected_type, "bind", None)
        if selected_profile is not None and not callable(bind_fn):
            raise TypeError(
                f"Profile {selected_profile.name!r} must define bind(); profiles "
                "cannot be instantiated as atomic providers."
            )
        if callable(bind_fn):
            selected = bind_fn(spec, caps, **provider_kwargs)
        else:
            selected = selected_type(**provider_kwargs)
        metadata_fn = getattr(selected, "binding_metadata", None)
        metadata = metadata_fn() if callable(metadata_fn) else {}
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Provider {selected_type.name!r} binding_metadata() must return "
                f"a dict, got {type(metadata).__name__}."
            )
        if selected_profile is not None:
            selected_name = str(getattr(selected, "name", ""))
            allowed_names = {
                selected_profile.name,
                *selected_profile_atomic_names,
            }
            if selected_name not in allowed_names:
                raise RuntimeError(
                    f"Profile {selected_profile.name!r} bound undeclared provider "
                    f"{selected_name!r}; allowed={sorted(allowed_names)}."
                )
            routes = metadata.get("routes")
            if isinstance(routes, list):
                route_names = {
                    str(route.get("provider"))
                    for route in routes
                    if isinstance(route, dict) and "provider" in route
                }
                undeclared_routes = sorted(
                    route_names - set(selected_profile_atomic_names)
                )
                if undeclared_routes:
                    raise RuntimeError(
                        f"Profile {selected_profile.name!r} contains undeclared "
                        f"atomic routes {undeclared_routes}."
                    )
        report = BindingReport(
            operator_type=self.registry.family,
            spec_type=type(spec).__name__,
            spec=_freeze_snapshot(spec),
            device_caps=_freeze_snapshot(caps),
            selected_provider=str(getattr(selected, "name", selected_type.name)),
            selected_profile=(
                str(selected_profile.name)
                if selected_profile is not None
                else None
            ),
            selected_role=(selected_role.value if selected_role is not None else None),
            selection_basis=selection_basis.value,
            selected_reason=selected_reason,
            candidates=tuple(
                sorted(decisions, key=lambda decision: decision.provider)
            ),
            profiles=tuple(profile_decisions),
            provider_metadata=_freeze_snapshot(metadata),
        )
        record_operator_binding(self.registry.family, selected, report=report)
        return ResolvedProvider(selected, tuple(rejected), report)

    @staticmethod
    def _external_dependency_result(error: Exception) -> SupportResult | None:
        from sparseengine.kernels.external.support import (
            ExternalKernelContractError,
            ExternalKernelFamilyError,
            KernelFamilyState,
            RequiredExternalKernelFamilyError,
        )

        if isinstance(error, RequiredExternalKernelFamilyError):
            raise error
        if isinstance(error, ExternalKernelFamilyError):
            if error.health.state is KernelFamilyState.ABSENT:
                return SupportResult.dependency_absent(str(error))
            raise error
        if isinstance(error, ExternalKernelContractError):
            raise error
        return None
