"""Install the guarded Docker environment into mini-SWE-agent's docker alias."""

from __future__ import annotations

import os
import sys


if os.environ.get("SPARSEENGINE_SWE_SCORER_IPV4") == "1":
    from benchmark.swe_bench_lite.scorer_network import install_ipv4_metadata_transport

    install_ipv4_metadata_transport()


if os.environ.get("SPARSEENGINE_SWE_PODMAN_ROOTLESS") == "1":
    try:
        from benchmark.swe_bench_lite.podman_compat import (
            install_root_owned_uploads,
            install_rootless_container_creation,
        )

        install_root_owned_uploads()
        install_rootless_container_creation()
    except Exception as exc:
        sys.stderr.write(f"Failed to install Podman compatibility adapters: {exc}\n")
        raise SystemExit(70) from exc


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
