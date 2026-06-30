import socket

import pytest


@pytest.fixture
def unused_tcp_port():
    """Return an OS-assigned free TCP port.

    Replaces the fixture pytest-aiohttp would provide (that plugin is not
    installed); used by tests that spin up a local aiohttp server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
