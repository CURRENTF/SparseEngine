"""SWE-bench container compatibility for a single-UID rootless Podman service."""
from io import BytesIO
import shlex
import subprocess
import tarfile


def install_rootless_container_creation() -> None:
    """Preserve the scorer's create contract while disabling unavailable cgroups."""
    import docker
    from docker.models.containers import ContainerCollection

    original_from_env = docker.from_env

    def from_env(**kwargs):
        # Shared Podman storage can take minutes to create/inspect containers.
        # This bounds infrastructure waits; official test execution is unchanged.
        kwargs.setdefault("timeout", 600)
        return original_from_env(**kwargs)

    docker.from_env = from_env

    def create(self, image, command=None, **kwargs):
        if kwargs.pop("detach", False) is not True:
            raise ValueError("Rootless scorer containers must be detached")
        name = kwargs.pop("name")
        user = kwargs.pop("user", None)
        platform = kwargs.pop("platform", None)
        capabilities = kwargs.pop("cap_add", [])
        if kwargs:
            raise ValueError(f"Unsupported rootless scorer create options: {sorted(kwargs)}")
        argv = ["docker", "create", "--cgroups=disabled", "--name", name]
        if user is not None:
            argv.extend(["--user", str(user)])
        if platform is not None:
            argv.extend(["--platform", platform])
        for capability in capabilities:
            argv.extend(["--cap-add", capability])
        argv.append(image)
        if isinstance(command, str):
            argv.extend(shlex.split(command))
        elif command is not None:
            argv.extend(command)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Rootless scorer container creation failed: {exc.stderr}") from exc
        return self.get(result.stdout.strip())

    ContainerCollection.create = create


def install_root_owned_uploads() -> None:
    from docker.models.images import ImageCollection
    import swebench.harness.docker_utils as docker_utils

    def list_images(self, name=None, all=False, filters=None):
        # Both scorer inventory paths only consume IDs/tags. Construct models
        # from the inventory response instead of inspecting every image again:
        # foreign images may be deleted concurrently from the shared store.
        return [
            self.prepare_model(image)
            for image in self.client.api.images(name=name, all=all, filters=filters)
        ]

    ImageCollection.list = list_images

    def root_owner(member: tarfile.TarInfo) -> tarfile.TarInfo:
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        return member

    def copy_to_container(container, src, dst):
        # Host UID/GID need not exist in the service's user namespace. Only
        # ownership changes; file contents, paths and modes stay unchanged.
        with BytesIO() as payload:
            with tarfile.open(fileobj=payload, mode="w") as archive:
                archive.add(src, arcname=dst.name, filter=root_owner)
            code, output = container.exec_run(f"mkdir -p {shlex.quote(str(dst.parent))}")
            if code:
                raise RuntimeError(f"Cannot create container upload directory: {output!r}")
            if not container.put_archive(str(dst.parent), payload.getvalue()):
                raise RuntimeError(f"Cannot upload {src} to {dst}")

    docker_utils.copy_to_container = copy_to_container
