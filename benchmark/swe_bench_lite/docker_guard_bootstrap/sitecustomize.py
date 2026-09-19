"""Install the guarded Docker environment into mini-SWE-agent's docker alias."""

from __future__ import annotations

import os
import sys


if int(os.environ.get("SPARSEENGINE_DOCKER_WRITABLE_LAYER_LIMIT_BYTES", "0")) > 0:
    try:
        import minisweagent.environments

        minisweagent.environments._ENVIRONMENT_MAPPING[
            "docker"
        ] = "benchmark.swe_bench_lite.guarded_docker_environment.GuardedDockerEnvironment"
        if int(os.environ.get("SPARSEENGINE_DOCKER_MEMORY_LIMIT_BYTES", "0")) > 0:
            from benchmark.swe_bench_lite.docker_memory_guard import install_sdk_limits
            install_sdk_limits()
    except Exception as exc:
        sys.stderr.write(f"Failed to install Docker writable-layer guard: {exc}\n")
        raise SystemExit(70) from exc
