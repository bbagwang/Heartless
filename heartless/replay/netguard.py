"""Network guard for replays: any real outbound connection attempt raises and is recorded.

A replay must be a closed world (stored data only); a code path that silently reached Binance, Telegram or the public
archive would make results depend on the network and leak a test run to the outside. The guard patches the socket
layer (TCP connect, DNS), asyncio's connection factory and the httpx network transports for the duration of a
`with block_network(...)` block. Local IPC (AF_UNIX sockets, pipes used by process pools) stays allowed, and httpx
clients built on a MockTransport (the Telegram capture) are unaffected.
"""
from __future__ import annotations

import asyncio
import socket
import traceback
from contextlib import contextmanager
from typing import Callable, Iterator

import httpx


class NetworkBlocked(RuntimeError):
    """A replay tried to open a real network connection."""


@contextmanager
def block_network(record: Callable[[str], None] | None = None) -> Iterator[list[str]]:
    attempts: list[str] = []

    def deny(what: str) -> NetworkBlocked:
        where = "".join(traceback.format_stack(limit=8)[:-2]).strip().splitlines()
        msg = f"network access blocked during replay: {what} (from {where[-2].strip() if len(where) >= 2 else '?'})"
        attempts.append(msg)
        if record is not None:
            record(msg)
        return NetworkBlocked(msg)

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_create_connection = socket.create_connection
    orig_getaddrinfo = socket.getaddrinfo
    orig_loop_create_connection = asyncio.base_events.BaseEventLoop.create_connection
    orig_async_transport = httpx.AsyncHTTPTransport.handle_async_request
    orig_sync_transport = httpx.HTTPTransport.handle_request

    def connect(self, address):
        if self.family == socket.AF_UNIX:
            return orig_connect(self, address)
        raise deny(f"socket.connect{address!r}")

    def connect_ex(self, address):
        if self.family == socket.AF_UNIX:
            return orig_connect_ex(self, address)
        raise deny(f"socket.connect_ex{address!r}")

    def create_connection(address, *a, **k):
        raise deny(f"socket.create_connection{address!r}")

    def getaddrinfo(host, port, *a, **k):
        raise deny(f"getaddrinfo({host!r}, {port!r})")

    async def loop_create_connection(self, protocol_factory, host=None, port=None, *a, **k):
        raise deny(f"loop.create_connection({host!r}, {port!r})")

    async def async_transport(self, request):
        raise deny(f"httpx {request.method} {request.url}")

    def sync_transport(self, request):
        raise deny(f"httpx {request.method} {request.url}")

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo
    asyncio.base_events.BaseEventLoop.create_connection = loop_create_connection
    httpx.AsyncHTTPTransport.handle_async_request = async_transport
    httpx.HTTPTransport.handle_request = sync_transport
    try:
        yield attempts
    finally:
        socket.socket.connect = orig_connect
        socket.socket.connect_ex = orig_connect_ex
        socket.create_connection = orig_create_connection
        socket.getaddrinfo = orig_getaddrinfo
        asyncio.base_events.BaseEventLoop.create_connection = orig_loop_create_connection
        httpx.AsyncHTTPTransport.handle_async_request = orig_async_transport
        httpx.HTTPTransport.handle_request = orig_sync_transport
