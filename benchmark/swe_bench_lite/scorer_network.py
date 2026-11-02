"""Opt-in IPv4 transport for official scorer metadata on IPv6-blocked hosts."""
import socket
from urllib.parse import urlsplit


def install_ipv4_metadata_transport() -> None:
    import requests
    from urllib3.util import connection

    connection.allowed_gai_family = lambda: socket.AF_INET
    original = requests.sessions.Session.request

    def request(self, method, url, **kwargs):
        # SWE-bench fetches revision-pinned environment files without a timeout.
        if urlsplit(url).hostname == "raw.githubusercontent.com" and kwargs.get("timeout") is None:
            kwargs["timeout"] = (15, 120)
        return original(self, method, url, **kwargs)

    requests.sessions.Session.request = request
