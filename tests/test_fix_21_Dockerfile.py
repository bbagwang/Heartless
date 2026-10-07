"""Regression tests for the Dockerfile HEALTHCHECK (fix round 21).

The probe used to be ``urllib.request.urlopen('http://127.0.0.1:8080/').status in (200, 401)``. An anonymous ``GET /`` is
answered ``401`` (login page) and ``urlopen`` raises ``HTTPError`` on every 4xx, so the membership test was never reached
and the command exited 1 on every run: three probes after start the container was ``unhealthy`` for good, and any autoheal
sidecar / ``depends_on: condition: service_healthy`` restarted the trading process in a loop. The probe also used to be
counted as a wrong-token login for 127.0.0.1, which flipped ``/`` to ``429`` after ten probes.

These tests run the *exact* shell-form command from the Dockerfile the way Docker does (``/bin/sh -c <cmd>``) against a real
uvicorn-served ``WebServer`` on an ephemeral port, and against stub responders / a closed port, so the Dockerfile is
exercised without a Docker daemon.
"""
from __future__ import annotations

import http.client
import http.server
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from heartless.config import Settings
from heartless.web.app import FAIL_MAX_ATTEMPTS, WebServer

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"


# --- Dockerfile parsing -------------------------------------------------------------------------------------------------

def _instructions() -> list[str]:
    """Dockerfile instructions with ``\\``-newline continuations joined, comment lines dropped (what the Docker parser does)."""
    lines = [ln for ln in DOCKERFILE.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    joined = re.sub(r"\\\r?\n", "", "\n".join(lines))
    return [ln.strip() for ln in joined.splitlines() if ln.strip()]


def _healthcheck() -> str:
    hc = [i for i in _instructions() if i.upper().startswith("HEALTHCHECK")]
    assert len(hc) == 1, "exactly one HEALTHCHECK instruction expected"
    return hc[0]


def _probe_cmd() -> str:
    """The shell-form command that Docker executes as ``/bin/sh -c <cmd>`` on every probe."""
    m = re.search(r"\bCMD\s+(.+)$", _healthcheck(), re.S)
    assert m, "HEALTHCHECK must carry a CMD"
    return m.group(1).strip()


def _run_probe(**env: str) -> subprocess.CompletedProcess:
    """Run the probe like Docker does; ``python`` resolves to the test interpreter and only the given WEB_* vars are set."""
    base = {k: v for k, v in os.environ.items() if not k.startswith("WEB_")}
    base["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{base.get('PATH', '')}"
    base.update(env)
    return subprocess.run(["/bin/sh", "-c", _probe_cmd()], env=base, capture_output=True, text=True, timeout=60)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- fixtures -----------------------------------------------------------------------------------------------------------

class _App:
    """The minimum the dashboard needs: settings and the owner token (the API routes touch the rest lazily)."""

    def __init__(self):
        self.s = Settings(_env_file=None, WEB_ENABLED=True, TELEGRAM_BOT_TOKEN="")
        self.web_token = secrets.token_urlsafe(16)


@pytest.fixture
def live_web():
    """A real ``WebServer`` served by uvicorn in a background thread on an ephemeral loopback port."""
    web = WebServer(_App())
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(web.api, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    th = threading.Thread(target=server.run, name="uvicorn-test", daemon=True)
    th.start()
    deadline = time.time() + 20
    while not server.started:
        assert th.is_alive(), "uvicorn thread died during startup"
        assert time.time() < deadline, "uvicorn did not start in time"
        time.sleep(0.02)
    try:
        yield web, port
    finally:
        server.should_exit = True
        th.join(timeout=10)


@pytest.fixture
def stub_server():
    """A loopback HTTP server answering every GET with a configurable status code."""
    class Handler(http.server.BaseHTTPRequestHandler):
        code = 500

        def do_GET(self):  # noqa: N802 - http.server API
            body = b"stub"
            self.send_response(type(self).code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence
            return None

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield Handler, srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


# --- the probe against the real dashboard -------------------------------------------------------------------------------

def test_anonymous_root_is_401_and_probe_passes(live_web):
    """The precondition that broke the old probe (anonymous / -> 401) still holds, and the probe now exits 0 on it."""
    web, port = live_web
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/")
    assert conn.getresponse().status == 401
    conn.close()

    r = _run_probe(WEB_PORT=str(port))
    assert r.returncode == 0, f"healthcheck failed against a healthy dashboard: {r.stderr.strip()}"


def test_probe_is_not_throttled_as_failed_login(live_web):
    """More probes than FAIL_MAX_ATTEMPTS keep passing and leave the wrong-token table empty (no 429 after ten minutes)."""
    web, port = live_web
    for i in range(FAIL_MAX_ATTEMPTS + 2):
        r = _run_probe(WEB_PORT=str(port))
        assert r.returncode == 0, f"probe #{i + 1} failed: {r.stderr.strip()}"
    assert web._fail == {}, "the anonymous healthcheck must not be booked as a failed login attempt"
    # and the lock-out that the old probe used to trigger is not in effect for the loopback client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/")
    assert conn.getresponse().status == 401
    conn.close()


def test_probe_honours_web_port_from_environment(live_web):
    """WEB_PORT is configurable in the app, so the probe must follow it rather than a hard-coded 8080."""
    _, port = live_web
    assert _run_probe(WEB_PORT=str(port)).returncode == 0
    assert _run_probe(WEB_PORT=str(_free_port())).returncode != 0  # same server, wrong port -> unhealthy


# --- the probe must still fail when the dashboard is actually unhealthy -------------------------------------------------

def test_probe_fails_when_nothing_listens():
    r = _run_probe(WEB_PORT=str(_free_port()))
    assert r.returncode != 0


@pytest.mark.parametrize("code", [404, 500, 502, 503])
def test_probe_fails_on_non_healthy_status(stub_server, code):
    handler, port = stub_server
    handler.code = code
    r = _run_probe(WEB_PORT=str(port))
    assert r.returncode != 0, f"status {code} must be reported unhealthy"


@pytest.mark.parametrize("code", [200, 401])
def test_probe_accepts_healthy_statuses_without_raising(stub_server, code):
    """Both statuses the Dockerfile lists as healthy really yield exit 0 (the 401 branch is the one urlopen could never reach)."""
    handler, port = stub_server
    handler.code = code
    r = _run_probe(WEB_PORT=str(port))
    assert r.returncode == 0, r.stderr.strip()


# --- WEB_ENABLED ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "off", " f "])
def test_probe_is_healthy_when_dashboard_disabled(value):
    """With WEB_ENABLED off there is no HTTP server at all; the container must not be flagged unhealthy for that."""
    r = _run_probe(WEB_ENABLED=value, WEB_PORT=str(_free_port()))
    assert r.returncode == 0, r.stderr.strip()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_truthy_or_blank_web_enabled_still_probes(value):
    """Only an explicit false disables the probe; truthy / blank values keep it live (and it fails on a closed port)."""
    r = _run_probe(WEB_ENABLED=value, WEB_PORT=str(_free_port()))
    assert r.returncode != 0


# --- static guards ------------------------------------------------------------------------------------------------------

def test_healthcheck_options():
    hc = _healthcheck()
    assert "--retries=" in hc and "--interval=" in hc and "--timeout=" in hc
    # the web server starts only after the candle backfill, which can take minutes: failures during the start period must
    # not count towards the unhealthy threshold
    m = re.search(r"--start-period=(\d+)s", hc)
    assert m and int(m.group(1)) >= 120, "HEALTHCHECK needs a start-period covering the backfill before the web server starts"


def test_probe_does_not_use_urlopen():
    """urlopen raises on every 4xx, so a bare ``urlopen(...).status`` test can never see the 401 the login page answers."""
    assert "urlopen" not in _probe_cmd()
