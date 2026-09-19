"""Regressions for compiler admission, cache hits, and process hook ownership."""

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from sparseengine.utils.compilation_guard import (
    RuntimeCompilationError,
    RuntimeCompilationGuard,
    validate_compilation_limit,
)


@pytest.mark.parametrize("limit", [-1, True, 1.5, "5"])
def test_invalid_budget_fails_before_startup(limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_compilation_limit(limit)


def test_budget_blocks_before_compiler_and_remains_exhausted():
    calls = []

    class Compiler:
        def build(self):
            calls.append("compiled")

    guard = RuntimeCompilationGuard(2, rank=3)
    guard._wrap(Compiler, "build", "test", lambda instance, args: "dynamic_width")
    try:
        Compiler().build()
        Compiler().build()
        for _ in range(2):
            with pytest.raises(RuntimeCompilationError, match="rank=3.*dynamic_width"):
                Compiler().build()
        assert len(calls) == 2
    finally:
        guard.close()
    Compiler().build()
    assert len(calls) == 3


def test_concurrent_compilers_share_one_budget():
    guard = RuntimeCompilationGuard(3, rank=0)

    def attempt(_):
        try:
            guard.before_compile("test", "kernel")
            return True
        except RuntimeCompilationError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(12))) == 3


def test_upstream_compilation_failure_is_not_hidden():
    class Compiler:
        def build(self):
            raise ValueError("invalid CUDA source")

    guard = RuntimeCompilationGuard(1, rank=0)
    guard._wrap(Compiler, "build", "test", lambda instance, args: "bad_source")
    try:
        with pytest.raises(ValueError, match="invalid CUDA source"):
            Compiler().build()
        with pytest.raises(RuntimeCompilationError):
            Compiler().build()
    finally:
        guard.close()


def test_arm_is_idempotent_and_close_preserves_foreign_hooks(monkeypatch):
    module = importlib.import_module("triton.compiler.compiler")
    compiler = next(iter(module.backends.values())).compiler

    previous_calls = []
    previous = lambda *args, **kwargs: previous_calls.append(True)
    monkeypatch.setattr(compiler, "add_stages", previous)
    guard = RuntimeCompilationGuard(1, rank=0)
    guard.arm()
    try:
        compiler.add_stages(None, {}, None, None)
        guard.arm()
        with pytest.raises(RuntimeCompilationError):
            compiler.add_stages(None, {}, None, None)
        assert previous_calls == [True]
        with pytest.raises(RuntimeError, match="already active"):
            RuntimeCompilationGuard(1, rank=0).arm()
        foreign = lambda *args: None
        compiler.add_stages = foreign
    finally:
        guard.close()
    assert compiler.add_stages is foreign


@pytest.mark.parametrize("class_name", ["JitSpec", "JitSpecNvcc"])
def test_flashinfer_legacy_and_split_specs_enforce_budget_and_restore(class_name):
    # 0.6.15 exports only JitSpec; the newer installation alone cannot catch
    # startup failure on that supported API. Exercise both with the old call
    # signature, including verbose/need_lock and the original error propagation.
    calls = []

    class Spec:
        name = "legacy_module"

        def build(self, verbose, need_lock=True):
            calls.append((verbose, need_lock))
            raise ValueError("compiler failed")

    module = SimpleNamespace(__name__="flashinfer.jit.core", **{class_name: Spec})
    original = Spec.build
    guard = RuntimeCompilationGuard(1, rank=0)
    guard._instrument(module)
    try:
        with pytest.raises(ValueError, match="compiler failed"):
            Spec().build(True, need_lock=False)
        with pytest.raises(RuntimeCompilationError, match="backend=flashinfer kernel=legacy_module"):
            Spec().build(verbose=False)
        assert calls == [(True, False)]
    finally:
        guard.close()
    assert Spec.build is original


def test_backend_imported_after_arm_is_instrumented(tmp_path, monkeypatch):
    import sparseengine.utils.compilation_guard as module

    name = "guard_test_backend"
    (tmp_path / f"{name}.py").write_text("loaded = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(module, "_TARGETS", {name})
    guard = RuntimeCompilationGuard(0, rank=0)
    seen = []
    monkeypatch.setattr(guard, "_instrument", lambda backend: seen.append(backend.loaded))
    guard.arm()
    try:
        importlib.import_module(name)
        assert seen == [True]
    finally:
        guard.close()
        sys.modules.pop(name, None)


def test_runner_arms_each_rank_without_resetting_budget():
    from sparseengine.engine.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(runtime_compilation_limit=1)
    runner.rank = 2
    runner.arm_runtime_compilation_guard()
    try:
        runner._compilation_guard.before_compile("test", "kernel")
        runner.arm_runtime_compilation_guard()
        with pytest.raises(RuntimeCompilationError, match="rank=2"):
            runner._compilation_guard.before_compile("test", "kernel")
    finally:
        runner._compilation_guard.close()


@pytest.mark.parametrize("module_name,class_name,method,args", [
    ("tilelang.jit.kernel", "JITKernel", "_compile_and_create_adapter",
     (SimpleNamespace(attrs={"global_symbol": "tile_kernel"}), [])),
    ("flashinfer.jit.core", "JitSpecNvcc", "build", ()),
    ("flashinfer.jit.cute_dsl_core", "JitSpecCuteDsl", "build", ()),
])
def test_backend_build_is_blocked_before_side_effects(
    monkeypatch, module_name, class_name, method, args,
):
    module = pytest.importorskip(module_name)
    if class_name == "JitSpecNvcc" and not hasattr(module, class_name):
        class_name = "JitSpec"
    cls = getattr(module, class_name)
    calls = []
    monkeypatch.setattr(cls, method, lambda *args, **kwargs: calls.append(True))
    guard = RuntimeCompilationGuard(0, rank=0)
    guard.arm()
    try:
        with pytest.raises(RuntimeCompilationError):
            getattr(cls, method)(SimpleNamespace(name="module_kernel"), *args)
        assert calls == []
    finally:
        guard.close()
    getattr(cls, method)(SimpleNamespace(name="module_kernel"), *args)
    assert calls == [True]


def test_triton_persistent_cache_hit_is_allowed_but_new_ir_is_blocked(tmp_path, monkeypatch):
    # Compile offline with an explicit target: this tests the actual compiler
    # admission boundary, without launching a CUDA kernel or claiming numerics.
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    @triton.jit
    def kernel(output, VALUE: tl.constexpr):
        tl.store(output, VALUE)

    monkeypatch.setattr(triton.knobs.cache, "dir", str(tmp_path))
    target = GPUTarget("cuda", 80, 32)
    cached = ASTSource(kernel, {"output": "*fp32", "VALUE": "constexpr"}, {"VALUE": 1})
    unseen = ASTSource(kernel, {"output": "*fp32", "VALUE": "constexpr"}, {"VALUE": 2})
    triton.compile(cached, target=target)
    guard = RuntimeCompilationGuard(0, rank=0)
    guard.arm()
    try:
        triton.compile(cached, target=target)
        assert guard.count == 0
        with pytest.raises(RuntimeCompilationError, match="backend=triton kernel=kernel"):
            triton.compile(unseen, target=target)
    finally:
        guard.close()
