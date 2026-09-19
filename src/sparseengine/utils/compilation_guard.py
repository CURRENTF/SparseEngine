"""Per-process compilation budget, armed only after engine startup finishes."""

import functools
import importlib.abc
import sys
import threading
import traceback

from sparseengine.utils.log import logger


class RuntimeCompilationError(RuntimeError):
    """The post-startup compilation budget was exhausted."""


def validate_compilation_limit(limit: int) -> None:
    if type(limit) is not int or limit < 0:
        raise ValueError("runtime_compilation_limit must be a nonnegative integer")


_TARGETS = {
    "triton.compiler.compiler",
    "tilelang.jit.kernel",
    "flashinfer.jit.core",
    "flashinfer.jit.cute_dsl_core",
}


class _GuardLoader(importlib.abc.Loader):
    def __init__(self, loader, guard):
        self.loader = loader
        self.guard = guard

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.guard._instrument(module)


class RuntimeCompilationGuard(importlib.abc.MetaPathFinder):
    """Count compiler entries after cache lookup, independently on each rank.

    Hooks touch compilation paths only. The import finder instruments optional
    backends loaded by the first real request without importing them at startup.
    """

    def __init__(self, limit: int, rank: int):
        validate_compilation_limit(limit)
        self.limit = limit
        self.rank = rank
        self.count = 0
        self._lock = threading.Lock()
        self._patches = []
        self._armed = False

    def before_compile(self, backend: str, kernel: str) -> None:
        with self._lock:
            self.count += 1
            count = self.count
        detail = (
            f"Runtime kernel compilation rank={self.rank} count={count} "
            f"limit={self.limit} backend={backend} kernel={kernel}\n"
            + "".join(traceback.format_stack(limit=16)[:-1])
        )
        if count > self.limit:
            raise RuntimeCompilationError(
                detail + "Compilation blocked before code generation. "
                "Warm up this specialization or remove unintended specialization."
            )
        logger.warning("{}", detail)

    def _patch(self, owner, name, replacement):
        previous = getattr(owner, name)  # Unsupported upstream APIs fail explicitly.
        self._patches.append((owner, name, previous, replacement))
        setattr(owner, name, replacement)

    def _wrap(self, owner, name, backend, describe):
        original = getattr(owner, name)

        @functools.wraps(original)
        def guarded(instance, *args, **kwargs):
            self.before_compile(backend, describe(instance, args))
            return original(instance, *args, **kwargs)

        self._patch(owner, name, guarded)

    def _instrument(self, module):
        if module.__name__ == "triton.compiler.compiler":
            def triton_kernel(instance, args):
                # Both Triton 3.5 and 3.6 call backend.add_stages only after a
                # persistent-cache miss. The source lives in the compiler frame.
                frame = sys._getframe(1)
                kernel = "unknown"
                try:
                    while frame is not None:
                        if frame.f_globals.get("__name__") == "triton.compiler.compiler":
                            source = frame.f_locals.get("src")
                            kernel = str(getattr(source, "name", "unknown"))
                            break
                        frame = frame.f_back
                finally:
                    del frame
                return kernel

            for compiler in dict.fromkeys(backend.compiler for backend in module.backends.values()):
                self._wrap(compiler, "add_stages", "triton", triton_kernel)
        elif module.__name__ == "tilelang.jit.kernel":
            self._wrap(
                module.JITKernel, "_compile_and_create_adapter", "tilelang",
                lambda instance, args: str(args[0].attrs["global_symbol"]),
            )
        elif module.__name__ in {"flashinfer.jit.core", "flashinfer.jit.cute_dsl_core"}:
            if module.__name__.endswith(".core"):
                # FlashInfer 0.6.15 predates the split into concrete JIT specs.
                cls = module.JitSpecNvcc if hasattr(module, "JitSpecNvcc") else module.JitSpec
            else:
                cls = module.JitSpecCuteDsl
            self._wrap(cls, "build", "flashinfer", lambda instance, args: instance.name)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        # Preserve the existing finder order, including editable-install finders.
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = _GuardLoader(spec.loader, self)
                return spec
        return None

    def arm(self):
        if self._armed:
            return  # Re-arming must never reset a consumed budget.
        if any(isinstance(finder, RuntimeCompilationGuard) for finder in sys.meta_path):
            raise RuntimeError("A runtime compilation guard is already active in this process")
        try:
            for name in sorted(_TARGETS):
                if name in sys.modules:
                    self._instrument(sys.modules[name])
            sys.meta_path.insert(0, self)
            self._armed = True
        except BaseException:
            self.close()
            raise
        logger.info("Runtime compilation guard armed: rank={} limit={}", self.rank, self.limit)

    def close(self):
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for owner, name, previous, replacement in reversed(self._patches):
            if getattr(owner, name) is replacement:
                setattr(owner, name, previous)
        self._patches.clear()
        self._armed = False
