"""Owner-only web dashboard + JSON API (FastAPI, served inside the bot's event loop)."""
from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import secrets
import time
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from heartless.util.timeutil import MS_DAY, fmt_ts, now_ms

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
# brute-force throttle for *presented* wrong tokens, keyed by client IP
FAIL_WINDOW_S = 900
FAIL_MAX_ATTEMPTS = 10
FAIL_MAX_IPS = 1024
# the research page renders on the server and runs no script; this policy also keeps the browser from loading anything
# from another origin (no CDN, fonts or images from outside)
RESEARCH_CSP = ("default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; base-uri 'none'; "
                "form-action 'self'; frame-ancestors 'none'")
EQUITY_DAYS = 30          # equity curve span on the research page
EQUITY_MAX_POINTS = 600   # points drawn (the stored curve has one row a minute)

LOGIN_HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Heartless</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#171a21;padding:32px;border-radius:12px;width:min(360px,90vw)}input{width:100%;padding:10px;margin:12px 0;border-radius:8px;border:1px solid #333;background:#0f1115;color:#fff}
button{width:100%;padding:10px;border-radius:8px;border:0;background:#4c8dff;color:#fff;font-weight:600}</style></head>
<body><form method="get" action="__ACTION__"><h2>Heartless</h2><p>텔레그램 /web 명령으로 받은 토큰을 입력하세요.</p>
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

    def _page(self, request: Request, render):
        """An owner-only HTML page: the login form without a valid token; a token in the URL moves into an HttpOnly cookie
        (303 back to the same path, so the secret leaves the address bar, history and referrers); else `render()`."""
        if not self._check_token(request):
            return HTMLResponse(LOGIN_HTML.replace("__ACTION__", html.escape(request.url.path)), status_code=401)
        if request.query_params.get("token"):
            secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
            resp = RedirectResponse(request.url.path, status_code=303)
            resp.set_cookie("hl_token", request.query_params["token"], httponly=True, samesite="lax",
                            secure=secure, max_age=30 * 86400)
            return resp
        return render()

    # --- routes --------------------------------------------------------------------------------
    def _routes(self) -> None:
        api, app = self.api, self.app
        auth = Depends(self.auth())

        @api.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self._page(request, lambda: FileResponse(STATIC / "index.html"))

        @api.get("/research", response_class=HTMLResponse)
        async def research_page(request: Request):
            def render():
                resp = HTMLResponse(self.research_html())
                resp.headers["Content-Security-Policy"] = RESEARCH_CSP
                resp.headers["Cache-Control"] = "no-store"
                return resp
            return self._page(request, render)

        @api.get("/api/learning")
        async def learning(_: bool = auth):
            return JSONResponse(_clean(self.learning_data()))

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

        @api.get("/api/settings")
        async def get_settings(_: bool = auth):
            from heartless.core.secrets import mask

            return {"binance_api_key": mask(app.s.binance_api_key), "has_secret": bool(app.s.binance_api_secret),
                    "testnet": app.s.binance_testnet, "telegram": bool(app.s.telegram_bot_token), "mode": app.mode,
                    "live_capable": app.s.live_capable}

        @api.post("/api/settings/keys")
        async def set_keys(payload: dict, request: Request, _: bool = auth):
            if request.headers.get("x-forwarded-proto", request.url.scheme) != "https" and request.client and \
                    request.client.host not in ("127.0.0.1", "::1", "localhost"):
                raise HTTPException(400, "API 키는 HTTPS 또는 로컬 접속에서만 입력할 수 있습니다")
            msg = await app.set_credentials(str(payload.get("api_key", "")), str(payload.get("api_secret", "")),
                                            payload.get("testnet"))
            return {"ok": msg.startswith("키 등록 완료"), "message": msg}

        @api.post("/api/settings/telegram")
        async def set_telegram(payload: dict, request: Request, _: bool = auth):
            from heartless.core.secrets import save_secrets

            token = str(payload.get("token", "")).strip()
            if ":" not in token:
                raise HTTPException(400, "올바른 텔레그램 봇 토큰이 아닙니다")
            save_secrets(app.s.data_dir, {"TELEGRAM_BOT_TOKEN": token})
            return {"ok": True, "message": "저장했습니다. 봇을 재시작하면 텔레그램이 연결되고, 로그의 페어링 코드로 /start 하세요"}

        @api.post("/api/report")
        async def report(_: bool = auth):
            asyncio.create_task(app.daily_report())
            return {"ok": True}

        @api.get("/favicon.ico")
        async def favicon():
            return RedirectResponse("data:,")

    # --- research / learning page ------------------------------------------------------------------
    def learning_data(self) -> dict:
        """Everything the research page shows, as plain data (also served as /api/learning): the alpha overview and the
        evidence registry, champion version history, challenger checkpoints, the kill switch, recent research and
        discovery cycles, the seal ledger and the primary engine's equity curve. Read-only."""
        from heartless.learning import research as R
        from heartless.learning import seal
        from heartless.strategy import evidence

        app = self.app
        now = now_ms()
        prim = app.primary_engine.name if app.primary_engine else "paper"
        overview = app.alpha_overview()
        versions = []
        for v in app.store.load_params_versions(limit=30):
            enabled = (v.get("params") or {}).get("enabled") or {}
            decisions = (v.get("metrics") or {}).get("decisions") or []
            versions.append({"id": v.get("id"), "created": v.get("created"), "role": v.get("role"),
                             "source": v.get("source"), "note": v.get("note"),
                             "enabled": sorted(a for a, on in enabled.items() if on),
                             "decisions": [{"kind": d.get("kind") or d.get("path"), "verdict": d.get("verdict"),
                                            "rule": d.get("rule"), "alpha": d.get("alpha")} for d in decisions[:3]]})
        runs = []
        for r in app.store.load_research_runs(10):
            sm = r.get("summary") or {}
            alphas = {}
            for a, x in (sm.get("alphas") or {}).items():
                base, best = x.get("base") or {}, x.get("best") or {}
                alphas[a] = {"enabled": x.get("enabled", True), "base_n": (base.get("test") or {}).get("n", 0),
                             "base_avg_r": (base.get("test") or {}).get("avg_r", 0.0),
                             "best_n": (best.get("test") or {}).get("n", 0),
                             "best_avg_r": (best.get("test") or {}).get("avg_r", 0.0),
                             "improved": not best.get("is_base") and best.get("score", -1e9) > base.get("score", -1e9)}
            sel = sm.get("selection_detail")
            runs.append({"ts": r.get("ts"), "seconds": sm.get("seconds"), "symbols": len(sm.get("symbols") or []),
                         "window": sm.get("window"), "alphas": alphas,
                         # E2: which alphas the cycle tuned (enabled, untested, the failed-alpha rotation)
                         "selection": R.describe_selection(sel) if sel else ""})
        cycles = []
        for ev in app.store.load_events(40, topic="research_done"):
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {"message": ev.get("message")}
            cycles.append({"ts": ev.get("ts"), "seconds": msg.get("seconds"), "message": str(msg.get("message") or "")[:600]})
        disc = app.store.get("discovery.last") or {}
        ledger, ledger_error = [], ""
        try:
            if seal.LEDGER.exists():
                for line in seal.LEDGER.read_text(encoding="utf-8").splitlines()[-50:]:
                    if line.strip():
                        ledger.append(json.loads(line))
        except Exception as e:  # noqa: BLE001 - a damaged ledger must not take the page down
            ledger_error = f"{type(e).__name__}: {e}"
        rows = app.store.load_equity(prim, since=now - EQUITY_DAYS * MS_DAY, limit=EQUITY_DAYS * 1440 + 10)
        step = max(1, len(rows) // EQUITY_MAX_POINTS)
        curve = [[r[0], r[2]] for r in rows[::step]]
        if rows and (not curve or curve[-1][0] != rows[-1][0]):
            curve.append([rows[-1][0], rows[-1][2]])
        return {"generated": now, "mode": app.mode, "champion": app.params.version, "engine": prim,
                "portfolio": app.portfolio_status() if hasattr(app, "portfolio_status") else None,
                "overview": overview, "judgements": evidence.registry().get("judgements", {}),
                "versions": versions, "challengers": app.challenger_outlook(),
                "kill_switch": {"settings": (overview.get("learning") or {}).get("kill_switch", ""),
                                "enabled": bool(app.s.kill_switch), "state": app.killswitch.champion_state(),
                                "trips": app.killed_alphas()},
                "research": {"runs": runs, "cycles": cycles, "running": app.research.running,
                             "last": app.research.last_cycle_ts or None, "failures": app.research.research_failures},
                "discovery": {"ts": disc.get("ts"), "tested": disc.get("tested", 0), "t_bar": disc.get("t_bar"),
                              "seconds": disc.get("seconds"),
                              "passed": [{"id": r.get("id"), "side": r.get("side"), "tf": r.get("tf"),
                                          "conds": r.get("conds"), "valid": (r.get("stats") or {}).get("valid")}
                                         for r in (disc.get("passed") or [])[:10]]},
                "seal": {"sealed": sorted(getattr(seal, "SEALED", {}) or {}), "spent": sorted(getattr(seal, "SPENT", {}) or {}),
                         "ledger": ledger, "ledger_file": f"docs/results/{seal.LEDGER.name}", "ledger_error": ledger_error},
                "equity": curve}

    def research_html(self) -> str:
        """The research / learning page, rendered on the server (every value escaped) into static/research.html."""
        d = self.learning_data()
        body = _research_body(d, self.s.timezone)
        return (STATIC / "research.html").read_text(encoding="utf-8").replace("<!--BODY-->", body)

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


# --- research page rendering (server side, no script) ------------------------------------------------------------------
def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def _f(v, fmt: str = "+.3f") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "–"


def _cls(v) -> str:
    try:
        return "pos" if float(v) > 0 else "neg" if float(v) < 0 else ""
    except (TypeError, ValueError):
        return ""


def _table(head: list[str], rows: list[list[str]], empty: str) -> str:
    """A table of pre-escaped cells (or the escaped `empty` note when there are no rows)."""
    if not rows:
        return f'<p class="muted small">{_e(empty)}</p>'
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    tb = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>'


def equity_svg(points: list, tz: str, width: int = 760, height: int = 220) -> str:
    """Inline SVG line chart of [(ts, equity)]: horizontal grid with value labels, first/middle/last dates, a dashed
    start line and the last value. Colours come from the page's CSS tokens (light and dark)."""
    if len(points) < 2:
        return '<p class="muted small">자산 기록이 아직 부족합니다 (엔진이 1분마다 저장)</p>'
    L, R, T, B = 64, 16, 12, 28
    xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
    x0, x1 = xs[0], xs[-1] if xs[-1] > xs[0] else xs[0] + 1
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    def X(v: float) -> float:
        return L + (v - x0) / (x1 - x0) * (width - L - R)

    def Y(v: float) -> float:
        return T + (1 - (v - lo) / (hi - lo)) * (height - T - B)

    g = []
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        g.append(f'<line class="g" x1="{L}" x2="{width - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for i, anchor in ((0, "start"), (2, "middle"), (4, "end")):
        v = x0 + (x1 - x0) * i / 4
        g.append(f'<text x="{X(v):.1f}" y="{height - 8}" text-anchor="{anchor}">{_e(fmt_ts(v, tz, "%m-%d %H:%M"))}</text>')
    g.append(f'<line class="ax" x1="{L}" x2="{width - R}" y1="{height - B}" y2="{height - B}"/>')
    g.append(f'<line class="base" x1="{L}" x2="{width - R}" y1="{Y(ys[0]):.1f}" y2="{Y(ys[0]):.1f}"/>')
    path = " ".join(f"{'L' if i else 'M'}{X(x):.1f} {Y(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    g.append(f'<path class="line" d="{path}"/>')
    g.append(f'<circle class="dot" cx="{X(xs[-1]):.1f}" cy="{Y(ys[-1]):.1f}" r="4"/>'
             f'<text x="{X(xs[-1]) - 6:.1f}" y="{Y(ys[-1]) - 8:.1f}" text-anchor="end">{ys[-1]:,.2f}</text>')
    chg = (ys[-1] / ys[0] - 1) * 100 if ys[0] else 0.0
    label = f"자산 곡선: {ys[0]:,.2f} → {ys[-1]:,.2f} USDT ({chg:+.2f}%)"
    return (f'<svg class="eq" viewBox="0 0 {width} {height}" role="img" aria-label="{_e(label)}">'
            f'<title>{_e(label)}</title>{"".join(g)}</svg>')


def _evidence_cells(e: dict | None) -> list[str]:
    """TRAIN / VALID (or P1 OOS) / 2025 cells of the evidence table."""
    if not e:
        return ["–", "–", "–"]
    tr, va, s25 = e.get("train") or {}, e.get("valid") or {}, e.get("sealed2025") or {}

    def cell(x: dict, bar=None) -> str:
        if not x:
            return "–"
        return (f'n={_e(x.get("n", 0))} <span class="{_cls(x.get("avg_r"))}">{_f(x.get("avg_r"))}R</span> '
                f't {_f(x.get("t_day"), "+.2f")}' + (f'<span class="muted">/{_f(bar, "g")}</span>' if bar is not None else ""))
    if not tr and not va:  # P1: one OOS window
        oos = {"n": e.get("n"), "avg_r": e.get("avg_r"), "t_day": e.get("t_day")}
        return [cell(oos) + f'<div class="muted small">{_e(e.get("oos_window"))}</div>', "–", "–"]
    from heartless.notify.formatter import P2_VALID_T_BAR, SEALED2025_POOLED_T_BAR

    c25 = "–"
    if s25:
        pooled = s25.get("pooled_valid_2025") or {}
        c25 = cell(s25) + (f'<div class="muted small">VALID+2025 t {_f(pooled.get("t_day"), "+.2f")}'
                           f'/{SEALED2025_POOLED_T_BAR:g}</div>')
    return [cell(tr, tr.get("t_day_bar")), cell(va, P2_VALID_T_BAR), c25]


def _research_body(d: dict, tz: str) -> str:
    from heartless.notify.formatter import ROLE_KO, VERDICT_KO, evidence_stage, fmt_portfolio, outlook_stat

    ov = d.get("overview") or {}
    ls = ov.get("learning") or {}
    out = [f'<header><div><h1>Heartless 연구·학습</h1><div class="muted small">{_e(fmt_ts(d.get("generated"), tz, "%Y-%m-%d %H:%M"))} '
           f'({_e(tz)}) 기준 · 모드 {_e(str(d.get("mode", "")).upper())} · 챔피언 <code>{_e(d.get("champion"))}</code></div></div>'
           f'<div><a href="/">← 대시보드</a> · <a href="/api/learning">JSON</a></div></header>']
    # summary
    summary = [f'<p class="headline">{_e(ls.get("headline"))}</p>']
    if ls.get("no_alpha") and ls.get("criteria"):
        summary.append(f'<p class="small">거래 시작 조건: {_e(ls["criteria"])}</p>')
    summary.append(f'<p class="small muted">{_e(ls.get("evidence"))} · {_e(ls.get("kill_switch"))}</p>')
    pf = fmt_portfolio({"portfolio": d.get("portfolio")})
    if pf:  # Telegram HTML: only <code> tags, plain text once they are dropped
        summary.append(f'<p class="small">{_e(pf.replace("<code>", "").replace("</code>", ""))}</p>')
    out.append('<div class="card"><h2>지금 상태</h2>' + "".join(summary) + "</div>")
    # equity
    eq = d.get("equity") or []
    kpis = ""
    if len(eq) >= 2:
        a, b = float(eq[0][1]), float(eq[-1][1])
        peak, mdd = a, 0.0
        for _, v in eq:
            peak = max(peak, float(v))
            mdd = min(mdd, (float(v) / peak - 1) * 100 if peak else 0.0)
        kpis = (f'<div class="kpis"><div><b>{b:,.2f}</b><span>현재 자산</span></div>'
                f'<div><b class="{_cls(b - a)}">{(b / a - 1) * 100 if a else 0:+.2f}%</b><span>{EQUITY_DAYS}일 변화</span></div>'
                f'<div><b>{mdd:.2f}%</b><span>최대 낙폭(표시 구간)</span></div></div>')
    out.append(f'<div class="card"><h2>자산 곡선 ({_e(d.get("engine"))}, 최근 {EQUITY_DAYS}일)</h2>'
               f'{equity_svg(eq, tz)}{kpis}</div>')
    # alphas
    rows = []
    for r in sorted(ov.get("rows") or [], key=lambda x: (not x.get("enabled"), not x.get("challengers"))):
        e, live, k = r.get("evidence"), r.get("live") or {}, r.get("kill_switch")
        verdict = r.get("verdict")
        vcls = {"passed": "on", "failed": "bad"}.get(verdict, "off")
        if r.get("evidence_stale"):
            stage = "재설계로 무효"
        else:
            stage = evidence_stage(e)
        ks = "–"
        if k:
            ks = (f'CUSUM {_f(k.get("cusum"), "+.2f")}/-{_f(k.get("h"), "g")}σ · 최근 {_e(k.get("window_n", 0))}건 '
                  f'{_f(k.get("window_sum"), "+.1f")}/{_f(k.get("window_r"), "g")}R' + (' <span class="badge bad">작동</span>' if k.get("tripped") else ""))
        elif r.get("killed"):
            ks = f'<span class="badge bad">꺼짐</span> <code>{_e(r["killed"].get("version"))}</code>'
        chs = " ".join(f'<span class="chip">{_e(c["name"])} {_e(ROLE_KO.get(c["role"], c["role"]))}</span>'
                       for c in r.get("challengers") or []) or "–"
        rows.append([f'<code>{_e(r["alpha"])}</code>',
                     '<span class="badge on">켜짐</span>' if r.get("enabled") else '<span class="badge off">꺼짐</span>',
                     chs, _e(stage), f'<span class="badge {vcls}">{_e(VERDICT_KO.get(verdict, verdict))}</span>',
                     *_evidence_cells(e),
                     f'{_e(live.get("n", 0))}건 <span class="{_cls(live.get("avg_r"))}">{_f(live.get("avg_r"), "+.2f")}R</span> '
                     f't {_f(live.get("t_day"), "+.2f")}' if live.get("n") else '<span class="muted">0건</span>', ks])
    out.append('<div class="card"><h2>알파 · 증거 기록 · 운영 성적</h2>'
               + _table(["알파", "챔피언", "챌린저", "단계", "판정", "TRAIN / P1 OOS", "VALID", "2025", f"운영({_e(d.get('engine'))})", "킬 스위치"],
                        rows, "등록된 알파가 없습니다")
               + '<p class="small muted">t는 일 단위 군집 t, "/값"은 사전 등록 기준. 판정과 수치는 heartless/strategy/evidence.json(출하 데이터)에서 옵니다.</p></div>')
    # judgements
    jrows = [[f'<code>{_e(k)}</code>', _e(j.get("what")), _e(j.get("window") or f'{j.get("train_window", "")} / {j.get("valid_window", "")}'),
              _e(j.get("universe")), _e(j.get("decided")), f'<code>{_e(j.get("source"))}</code>']
             for k, j in (d.get("judgements") or {}).items()]
    out.append('<div class="card"><h2>사전 등록 판정</h2>' + _table(["판정", "내용", "기간", "유니버스", "결정일", "문서"], jrows, "판정 기록 없음") + "</div>")
    # challengers
    crow = []
    for c in d.get("challengers") or []:
        cur = c.get("current") or {}
        chips = []
        for x in c.get("checkpoints") or []:
            css = "done" if x in (c.get("passed") or []) else "next" if x == c.get("next_checkpoint") else ""
            chips.append(f'<span class="chip {css}">{_e(x)}일</span>')
        nxt = (f'{_e(c["next_checkpoint"])}일 ({_e(fmt_ts(c.get("next_at"), tz, "%m-%d"))})' if c.get("next_checkpoint") is not None
               else "판정점 모두 지남")
        stat = outlook_stat(c)
        crow.append([f'<code>{_e(c.get("name"))}</code><div class="muted small">{_e(c.get("source"))} · <code>{_e(c.get("version"))}</code></div>',
                     _e(fmt_ts(c.get("started"), tz, "%m-%d %H:%M")), "".join(chips), nxt,
                     "상대" if c.get("path") == "relative" else "절대",
                     f'{_e(cur.get("n", 0))}/{_e(c.get("min_trades"))}건 · {_e(cur.get("days", 0))}/{_e(c.get("min_days"))}일 · '
                     f'<span class="{_cls(cur.get("avg_r"))}">{_f(cur.get("avg_r"), "+.2f")}R</span>',
                     f'{stat} {_f(c.get("stat"), "+.2f")} / {_f(c.get("need"), "g")}',
                     _e(c.get("verdict") or "대기") + (f'<div class="muted small">{_e(c.get("rule"))}</div>' if c.get("rule") else "")])
    out.append('<div class="card"><h2>챌린저 판정점</h2>'
               + _table(["챌린저", "시작", "판정점", "다음", "경로", "현재 기록", "통계 / 다음 기준", "최근 판정"], crow,
                        "활성 챌린저 없음 (리서치·발굴이 후보를 찾으면 생성)") + "</div>")
    # kill switch + versions
    ks = d.get("kill_switch") or {}
    krows = [[f'<code>{_e(k.get("alpha"))}</code>', _e(k.get("engine")), _e(k.get("n", 0)),
              f'<span class="{_cls(k.get("avg_r"))}">{_f(k.get("avg_r"), "+.3f")}R</span> (기대 {_f(k.get("expected"), "+.3f")})',
              f'{_f(k.get("cusum"), "+.2f")} / -{_f(k.get("h"), "g")}σ', f'{_f(k.get("window_sum"), "+.1f")} / {_f(k.get("window_r"), "g")}R',
              '<span class="badge bad">작동</span>' if k.get("tripped") else '<span class="badge on">정상</span>']
             for k in ks.get("state") or []]
    trows = [[_e(fmt_ts(t.get("at"), tz, "%m-%d %H:%M")), f'<code>{_e(t.get("alpha"))}</code>', _e(t.get("engine")),
              f'<code>{_e(t.get("version"))}</code>', _e(t.get("rule")), "꺼짐 유지" if t.get("still_off") else "다시 켜짐(승격)"]
             for t in ks.get("trips") or []]
    out.append('<div class="grid"><div class="card"><h2>킬 스위치</h2>'
               + f'<p class="small muted">{_e(ks.get("settings"))}</p>'
               + _table(["알파", "엔진", "거래", "평균R", "CUSUM", "최근 합", "상태"], krows, "감시 중인 알파 없음 (챔피언에 켜진 알파 0개)")
               + "<h3>작동 이력</h3>" + _table(["시각", "알파", "엔진", "버전", "규칙", "현재"], trows, "작동 이력 없음") + "</div>")
    vrows = [[_e(fmt_ts(v.get("created"), tz, "%m-%d %H:%M")), f'<code>{_e(v.get("id"))}</code>',
              f'<span class="badge {"on" if v.get("role") == "champion" else "off"}">{_e(v.get("role"))}</span>', _e(v.get("source")),
              _e(", ".join(v.get("enabled") or []) or "없음"), f'<span class="note">{_e(str(v.get("note") or "")[:300])}</span>']
             for v in d.get("versions") or []]
    out.append('<div class="card"><h2>챔피언·챌린저 버전 이력</h2>'
               + _table(["생성", "버전", "역할", "출처", "켜진 알파", "메모"], vrows, "저장된 버전 없음 (출하 기본값 사용 중)")
               + "</div></div>")
    # research & discovery
    rs = d.get("research") or {}
    rrows = []
    for r in rs.get("runs") or []:
        al = r.get("alphas") or {}
        imp = [a for a, x in al.items() if x.get("improved")]
        sel = f'<div class="muted small">{_e(r["selection"])}</div>' if r.get("selection") else ""
        rrows.append([_e(fmt_ts(r.get("ts"), tz, "%m-%d %H:%M")), _e(f'{(r.get("seconds") or 0) / 60:.0f}분'), _e(r.get("symbols", 0)),
                      _e(len(al)) + sel, _e(", ".join(imp) or "없음")])
    crows = [[_e(fmt_ts(c.get("ts"), tz, "%m-%d %H:%M")), f'<span class="note">{_e(c.get("message"))}</span>'] for c in rs.get("cycles") or []]
    disc = d.get("discovery") or {}
    tested = int(disc.get("tested") or 0)
    dline = (f'<p class="small">마지막 발굴 {_e(fmt_ts(disc.get("ts"), tz, "%m-%d %H:%M"))} · 규칙 {tested:,}개 검사 · '
             f'다중검정 기준 t≥{_f(disc.get("t_bar"), ".2f")} · 통과 {_e(len(disc.get("passed") or []))}개</p>') if disc.get("ts") else \
        '<p class="small muted">알파 발굴이 아직 실행되지 않았습니다</p>'
    prow = [[f'<code>{_e(x.get("id"))}</code>', _e(x.get("side")), _e(x.get("tf")),
             _e(", ".join(f"{c[0]} {c[1]} {c[2]}" for c in x.get("conds") or [] if isinstance(c, (list, tuple)) and len(c) >= 3)),
             _e(f'n={(x.get("valid") or {}).get("n", 0)} {_f((x.get("valid") or {}).get("avg_r"), "+.2f")}R')]
            for x in disc.get("passed") or []]
    out.append('<div class="grid"><div class="card"><h2>리서치 사이클</h2>'
               + ('<p class="small">진행 중</p>' if rs.get("running") else "")
               + _table(["시각", "소요", "종목", "평가 알파", "개선 후보"], rrows, "리서치 사이클 기록 없음")
               + "<h3>최근 결과 메시지</h3>" + _table(["시각", "내용"], crows[:15], "기록 없음") + "</div>"
               + '<div class="card"><h2>알파 발굴</h2>' + dline
               + _table(["규칙", "방향", "격자", "조건", "검증 구간"], prow, "검증을 통과한 규칙 없음") + "</div></div>")
    # seal ledger
    sl = d.get("seal") or {}
    lrows = []
    for x in reversed(sl.get("ledger") or []):
        argv = x.get("argv") or []
        cmd = " ".join(str(a) for a in argv[1:]) if argv else (x.get("note") or "")
        lrows.append([_e(x.get("utc")), _e(x.get("period")), _e(x.get("event") or "unseal"),
                      f'<span class="note">{_e(str(cmd)[:400])}</span>', f'<code>{_e(str(x.get("commit") or "")[:10])}</code>'])
    state = (f'봉인 중: {_e(", ".join(sl.get("sealed") or []) or "없음")} · 소진: {_e(", ".join(sl.get("spent") or []) or "없음")}')
    out.append('<div class="card"><h2>봉인 원장</h2>' + f'<p class="small">{state} · <code>{_e(sl.get("ledger_file"))}</code></p>'
               + (f'<p class="small neg">원장을 읽지 못했습니다: {_e(sl.get("ledger_error"))}</p>' if sl.get("ledger_error") else "")
               + _table(["UTC", "구간", "기록", "명령 / 메모", "커밋"], lrows,
                        "원장 기록 없음 (개봉한 적이 없거나, 배포 이미지에 docs/가 없음)") + "</div>")
    return "\n".join(out)
