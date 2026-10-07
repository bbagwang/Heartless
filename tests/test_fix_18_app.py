"""Regression tests for the web auth throttle in heartless/web/app.py (round 18).

Offline, against a minimal fake orchestrator. Covered:
  * token-less requests (login page, healthcheck probes, scanners) never count as attempts and never lock the owner out
  * the valid token is accepted even while the client IP is locked; only presented-but-wrong tokens are throttled
  * a non-ASCII token yields 401 and a counted attempt instead of a TypeError / HTTP 500
  * the lock expires after the window and the per-IP table is bounded (rotating source addresses)
"""
from __future__ import annotations

import secrets

import httpx
import pytest
from starlette.requests import Request

import heartless.web.app as webmod
from heartless.config import Settings
from heartless.web.app import FAIL_MAX_ATTEMPTS, FAIL_MAX_IPS, FAIL_WINDOW_S, WebServer


class _App:
    def __init__(self):
        self.s = Settings(_env_file=None, TELEGRAM_BOT_TOKEN="", WEB_ENABLED=False)
        self.web_token = secrets.token_urlsafe(16)
        self.health = {"ok": True}
        self.killed = 0

    def status(self):
        return {"mode": "paper"}

    async def kill(self):
        self.killed += 1
        return 0


class _Clock:
    """Stand-in for the `time` module inside heartless.web.app so the lock window can be advanced deterministically."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def time(self):
        return self.t


def _client(web: WebServer, ip: str = "127.0.0.1") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=web.api, client=(ip, 1234)), base_url="http://test")


def _request(ip: str, query: str = "") -> Request:
    scope = {"type": "http", "http_version": "1.1", "method": "GET", "scheme": "http", "path": "/", "root_path": "",
             "query_string": query.encode(), "headers": [], "client": (ip, 1), "server": ("test", 80)}
    return Request(scope)


async def test_anonymous_requests_never_lock_out_the_owner():
    app = _App()
    web = WebServer(app)
    async with _client(web) as c:
        # a scanner / Docker HEALTHCHECK / stale dashboard tab: far more than ten token-less hits from one IP
        for _ in range(FAIL_MAX_ATTEMPTS + 15):
            r = await c.get("/")
            assert r.status_code == 401 and "token" in r.text
            r = await c.get("/api/status")
            assert r.status_code == 401
        assert web._fail == {}  # anonymous visits are not attempts
        # the owner, from the same (proxy) IP, keeps every control
        r = await c.get("/api/health", headers={"x-token": app.web_token})
        assert r.status_code == 200 and r.json() == {"ok": True}
        r = await c.post("/api/kill", headers={"x-token": app.web_token})
        assert r.status_code == 200 and r.json()["ok"] and app.killed == 1
        r = await c.get("/", params={"token": app.web_token})
        assert r.status_code == 303 and "hl_token" in r.headers.get("set-cookie", "")  # token leaves the URL
        r = await c.get("/api/status", cookies={"hl_token": app.web_token})
        assert r.status_code == 200 and r.json()["mode"] == "paper"


async def test_wrong_tokens_are_throttled_but_the_valid_token_still_passes():
    app = _App()
    web = WebServer(app)
    async with _client(web) as c:
        for i in range(FAIL_MAX_ATTEMPTS):
            r = await c.get("/api/status", headers={"x-token": f"wrong-{i}"})
            assert r.status_code == 401
        assert len(web._fail["127.0.0.1"]) == FAIL_MAX_ATTEMPTS
        # the budget is spent: the next wrong token is refused with 429 (API and login page alike) and not recorded
        r = await c.get("/api/status", headers={"x-token": "wrong-again"})
        assert r.status_code == 429
        r = await c.get("/", params={"token": "wrong-again"})
        assert r.status_code == 429
        assert len(web._fail["127.0.0.1"]) == FAIL_MAX_ATTEMPTS
        # ... but the owner's valid token is never refused, from the very same IP, by header, query or cookie
        r = await c.get("/api/health", headers={"x-token": app.web_token})
        assert r.status_code == 200
        r = await c.post("/api/kill", headers={"x-token": app.web_token})
        assert r.status_code == 200 and app.killed == 1
        r = await c.get("/", params={"token": app.web_token})
        assert r.status_code == 303  # valid token accepted: cookie set, token moved out of the URL
        r = await c.get("/api/status", cookies={"hl_token": app.web_token})
        assert r.status_code == 200
        # a token-less request from the locked IP still just sees the login page (401), not 429
        c.cookies.clear()  # drop the hl_token cookie the login above set in the client jar
        r = await c.get("/")
        assert r.status_code == 401
        # the lock is still in place for wrong tokens afterwards
        r = await c.get("/api/status", headers={"x-token": "wrong-again"})
        assert r.status_code == 429
    # the throttle is per source address: another client is unaffected
    async with _client(web, ip="10.0.0.2") as other:
        r = await other.get("/api/status", headers={"x-token": "wrong"})
        assert r.status_code == 401
        r = await other.get("/api/status", headers={"x-token": app.web_token})
        assert r.status_code == 200


async def test_non_ascii_token_is_unauthorized_not_a_server_error():
    app = _App()
    web = WebServer(app)
    async with _client(web) as c:
        r = await c.get("/?token=%C3%A9")  # 'é' -> compare_digest(str) used to raise TypeError -> HTTP 500
        assert r.status_code == 401 and "token" in r.text
        assert len(web._fail["127.0.0.1"]) == 1  # ... and the attempt is counted
        r = await c.get("/api/status", params={"token": "페어링"})
        assert r.status_code == 401
        r = await c.get("/api/status", headers={b"cookie": "hl_token=ñ".encode("utf-8")})  # httpx refuses non-ASCII jars
        assert r.status_code == 401
        r = await c.get("/api/status", headers={b"x-token": "é".encode("utf-8")})  # latin-1 decoded by starlette
        assert r.status_code == 401
        assert len(web._fail["127.0.0.1"]) == 4
        # a non-ASCII web_token on the server side is compared without error too
        app.web_token = "토큰-é"
        r = await c.get("/api/status", params={"token": "토큰-é"})
        assert r.status_code == 200
        r = await c.get("/api/status", params={"token": "토큰-e"})
        assert r.status_code == 401


async def test_lock_expires_after_the_window(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(webmod, "time", clock)
    app = _App()
    web = WebServer(app)
    async with _client(web) as c:
        for _ in range(FAIL_MAX_ATTEMPTS):
            assert (await c.get("/api/status", headers={"x-token": "wrong"})).status_code == 401
        assert (await c.get("/api/status", headers={"x-token": "wrong"})).status_code == 429
        clock.t += FAIL_WINDOW_S - 1
        assert (await c.get("/api/status", headers={"x-token": "wrong"})).status_code == 429
        clock.t += 2  # the oldest strikes are now outside the window
        r = await c.get("/api/status", headers={"x-token": "wrong"})
        assert r.status_code == 401
        assert len(web._fail["127.0.0.1"]) == 1  # expired strikes were dropped, only the fresh one remains


def test_fail_table_is_bounded_and_evicts_expired_buckets_first(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(webmod, "time", clock)
    app = _App()
    web = WebServer(app)
    # a few early strikes that will expire
    for i in range(5):
        assert web._check_token(_request(f"10.9.9.{i}", "token=wrong")) is False
    clock.t += FAIL_WINDOW_S + 1
    # an attacker rotating source addresses: the table never grows past the cap
    for i in range(FAIL_MAX_IPS + 500):
        assert web._check_token(_request(f"2001:db8::{i:x}", "token=wrong")) is False
        assert len(web._fail) <= FAIL_MAX_IPS
    assert len(web._fail) == FAIL_MAX_IPS
    assert not any(k.startswith("10.9.9.") for k in web._fail)  # expired buckets went first
    # the stalest live buckets were dropped, the newest kept
    assert "2001:db8::0" not in web._fail and f"2001:db8::{FAIL_MAX_IPS + 499:x}" in web._fail
    # eviction never turns a lock into a bypass for an IP that is still present
    hot = f"2001:db8::{FAIL_MAX_IPS + 499:x}"
    for _ in range(FAIL_MAX_ATTEMPTS - 1):
        assert web._check_token(_request(hot, "token=wrong")) is False
    with pytest.raises(webmod.HTTPException) as ei:
        web._check_token(_request(hot, "token=wrong"))
    assert ei.value.status_code == 429
    # the valid token passes regardless of the table state
    assert web._check_token(_request(hot, f"token={app.web_token}")) is True
