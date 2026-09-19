import socket

import pytest

from sparseengine.engine import model_runner
from sparseengine.engine.model_runner import DEFAULT_MASTER_PORT, select_master_port


def test_explicit_master_port_is_preferred(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    monkeypatch.setenv("SPARSEENGINE_MASTER_PORT", str(port))
    assert select_master_port() == port


def test_default_master_port_is_preferred(monkeypatch):
    monkeypatch.delenv("SPARSEENGINE_MASTER_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    monkeypatch.setattr(model_runner, "DEFAULT_MASTER_PORT", port)
    assert select_master_port() == port


def test_explicit_occupied_master_port_fails(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        monkeypatch.setenv("SPARSEENGINE_MASTER_PORT", str(port))

        with pytest.raises(RuntimeError, match=f"SPARSEENGINE_MASTER_PORT={port} is already in use"):
            select_master_port()


def test_explicit_master_port_is_reusable_after_completed_connection(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            connection.close()
            client.recv(1)

    monkeypatch.setenv("SPARSEENGINE_MASTER_PORT", str(port))
    assert select_master_port() == port


def test_occupied_default_master_port_uses_available_port(monkeypatch):
    monkeypatch.delenv("SPARSEENGINE_MASTER_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", DEFAULT_MASTER_PORT))
        except OSError:
            pytest.skip(f"default port {DEFAULT_MASTER_PORT} is already occupied")

        selected = select_master_port()

    assert selected != DEFAULT_MASTER_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", selected))


@pytest.mark.parametrize("value", ["", "abc", "0", "65536"])
def test_invalid_explicit_master_port_fails(monkeypatch, value):
    monkeypatch.setenv("SPARSEENGINE_MASTER_PORT", value)
    with pytest.raises(ValueError, match="SPARSEENGINE_MASTER_PORT"):
        select_master_port()
