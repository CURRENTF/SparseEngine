"""Small autotune shim for vendored LightLLM kernels.

The upstream kernels accept an optional ``run_config`` argument and decorate
several launch helpers with LightLLM's autotuner. SparseEngine keeps the kernel
sources vendored for local customization, but does not vendor LightLLM's global
autotune/cache stack. This shim preserves the call signature and lets each
kernel use its upstream default launch config.
"""

from collections.abc import Callable


def autotune(
    kernel_name: str,
    configs_gen_func: Callable,
    static_key_func: Callable,
    run_key_func: Callable,
    run_key_distance_func: Callable | None = None,
    mutates_args: list[str] | None = None,
):
    del kernel_name, configs_gen_func, static_key_func, run_key_func, run_key_distance_func, mutates_args

    def decorator(fn: Callable) -> Callable:
        return fn

    return decorator
