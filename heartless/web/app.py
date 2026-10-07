"""Owner-only web dashboard + JSON API (FastAPI, served inside the bot's event loop)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from heartless.util.timeutil import now_ms

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
# brute-force throttle for *presented* wrong tokens, keyed by client IP
FAIL_WINDOW_S = 900
FAIL_MAX_ATTEMPTS = 10
FAIL_MAX_IPS = 1024

LOGIN_HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Heartless</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#171a21;padding:32px;border-radius:12px;width:min(360px,90vw)}input{width:100%;padding:10px;margin:12px 0;border-radius:8px;border:1px solid #333;background:#0f1115;color:#fff}
button{width:100%;padding:10px;border-radius:8px;border:0;background:#4c8dff;color:#fff;font-weight:600}</style></head>
<body><form method="get" action="/"><h2>Heartless</h2><p>텔레그램 /web 명령으로 받은 토큰을 입력하세요.</p>
<input name="token" placeholder="access token" autofocus><button>열기</button></form></body></html>"""


class WebServer:
    def __init__(self, app):
        self.app = app
        self.s = app.s
        self.api = FastAPI(title="Heartless", docs_url=None, redoc_url=None, openapi_url=None)
        self._fail: dict[str, list[float]] = {}
        self._routes()
        self.server: uvicorn.Server | None = None

    # --- auth ----------------------------------------------------------------------------------
    def _check_token(self, request: Request) -> bool:
        tok = request.query_params.get("token") or request.headers.get("x-token") or request.cookies.get("hl_token") or ""
        # bytes comparison: compare_digest() rejects non-ASCII str with a TypeError (-> HTTP 500); as bytes any input
        # simply mismatches and is counted like every other wrong token
        if tok and secrets.compare_digest(tok.encode("utf-8"), self.app.web_token.encode("utf-8")):
            return True  # the valid token is never refused: the lock below only throttles presented-but-wrong tokens
        if not tok:
            return False  # anonymous visits (login page, Docker healthcheck, scanners) are not attempts
        ip = request.client.host if request.client else "?"
        now = time.time()
        fails = [t for t in self._fail.get(ip, []) if now - t < FAIL_WINDOW_S]
        if len(fails) >= FAIL_MAX_ATTEMPTS:
            raise HTTPException(429, "too many attempts")
        if ip not in self._fail and len(self._fail) >= FAIL_MAX_IPS:
            # bound the table: drop expired buckets first, then the stalest ones (rotating source addresses)
            self._fail = {k: v for k, v in self._fail.items() if v and now - v[-1] < FAIL_WINDOW_S}
            stale = sorted(self._fail, key=lambda k: self._fail[k][-1])
            for k in stale[: max(0, len(self._fail) - FAIL_MAX_IPS + 1)]:
                del self._fail[k]
        fails.append(now)
        self._fail[ip] = fails
        return False

    def auth(self):
        def dep(request: Request):
            if not self._check_token(request):
                raise HTTPException(401, "unauthorized")
            return True
        return dep

    # --- routes --------------------------------------------------------------------------------
    def _routes(self) -> None:
        api, app = self.api, self.app
        auth = Depends(self.auth())

        @api.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            if not self._check_token(request):
                return HTMLResponse(LOGIN_HTML, status_code=401)
            if request.query_params.get("token"):
                # move the secret from the URL (browser history, referrers, proxy logs) into an HttpOnly cookie
                secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
                resp = RedirectResponse("/", status_code=303)
                resp.set_cookie("hl_token", request.query_params["token"], httponly=True, samesite="lax",
                                secure=secure, max_age=30 * 86400)
                return resp
            return FileResponse(STATIC / "index.html")

        @api.get("/api/status")
        async def status(_: bool = auth):
            return JSONResponse(_clean(app.status()))

        @api.get("/api/pnl")
        async def pnl(period: str = "today", _: bool = auth):
            return JSONResponse(_clean(app.pnl_report(period if period in ("today", "week", "month", "all") else "today")))

        @api.get("/api/trades")
        async def trades(limit: int = 100, engine: str | None = None, _: bool = auth):
            eng = engine or (app.primary_engine.name if app.primary_engine else "paper")
            return JSONResponse(_clean(app.store.load_trades(engine=eng, limit=min(limit, 1000))))

        @api.get("/api/equity")
        async def equity(engine: str | None = None, days: int = 7, _: bool = auth):
            eng = engine or (app.primary_engine.name if app.primary_engine else "paper")
            rows = app.store.load_equity(eng, since=now_ms() - days * 86_400_000, limit=5000)
            return JSONResponse([[r[0], r[1], r[2]] for r in rows])

        @api.get("/api/engines")
        async def engines(_: bool = auth):
            return JSONResponse(_clean({n: e.snapshot(app.marks()) for n, e in app.engines.items()}))

        @api.get("/api/alphas")
        async def alphas(_: bool = auth):
            return JSONResponse(_clean({"trust": app.bandit.snapshot(), "params": app.params.to_dict()}))

        @api.get("/api/research")
        async def research(_: bool = auth):
            return JSONResponse(_clean({"last": app.research.last_summary, "runs": app.store.load_research_runs(10),
                                        "versions": app.store.load_params_versions(limit=20),
                                        "challengers": [{"name": c.name, "version": c.params.version, "source": c.source,
                                                         "started": c.started} for c in app.challengers]}))

        @api.get("/api/events")
        async def events(limit: int = 100, _: bool = auth):
            return JSONResponse(_clean(app.store.load_events(min(limit, 500))))

        @api.get("/api/health")
        async def health(_: bool = auth):
            return JSONResponse(_clean(app.health))

        @api.post("/api/pause")
        async def pause(_: bool = auth):
            await app.pause("web")
            return {"ok": True}

        @api.post("/api/resume")
        async def resume(_: bool = auth):
            await app.resume()
            return {"ok": True}

        @api.post("/api/close")
        async def close(payload: dict, _: bool = auth):
            n = await app.close_symbol(str(payload.get("symbol", "")).upper(), payload.get("engine"))
            return {"ok": n > 0, "closed": n}

        @api.post("/api/close_all")
        async def close_all(payload: dict | None = None, _: bool = auth):
            n = await app.close_all((payload or {}).get("engine"))
            return {"ok": True, "closed": n}

        @api.post("/api/kill")
        async def kill(_: bool = auth):
            n = await app.kill()
            return {"ok": True, "closed": n}

        @api.post("/api/mode")
        async def mode(payload: dict, _: bool = auth):
            m = payload.get("mode")
            if m not in ("paper", "live"):
                raise HTTPException(400, "mode must be paper|live")
            return {"ok": True, "message": await app.set_mode(m)}

        @api.post("/api/research/run")
        async def run_research(_: bool = auth):
            if app.research.running:
                return {"ok": False, "message": "already running"}
            asyncio.create_task(app.research.cycle(force=True))
            return {"ok": True}

        @api.post("/api/report")
        async def report(_: bool = auth):
            asyncio.create_task(app.daily_report())
            return {"ok": True}

        @api.get("/favicon.ico")
        async def favicon():
            return RedirectResponse("data:,")

    async def run(self) -> None:
        # behind a reverse proxy (nginx / Docker bridge) every client shares one request.client.host; set the env var
        # FORWARDED_ALLOW_IPS to the proxy's address so uvicorn applies X-Forwarded-For and the throttle sees real clients
        config = uvicorn.Config(self.api, host=self.s.web_host, port=self.s.web_port, log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        # the bot owns signal handling
        self.server.capture_signals = lambda: contextlib.nullcontext()  # type: ignore[method-assign]
        self.server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        if self.s.telegram_bot_token:
            log.info("web dashboard on http://%s:%s/ (token via Telegram /web)", self.s.web_host, self.s.web_port)
        else:
            log.info("web dashboard on http://%s:%s/?token=%s", self.s.web_host, self.s.web_port, self.app.web_token)
        await self.server.serve()


def _clean(obj):
    """Make arbitrary objects JSON-serialisable."""
    import dataclasses
    import math
    from enum import Enum

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _clean(dataclasses.asdict(obj))
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "__fspath__"):
        return str(obj)
    return obj
