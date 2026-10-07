"""Offline integration test: the orchestrator wired to a fake Binance REST, fed synthetic bars and ticks,
exercising engines, status/pnl reporting, Telegram formatting and the web API."""
import asyncio
import tempfile
from pathlib import Path

import httpx
import pytest

from heartless.app import Heartless
from heartless.config import Settings
from heartless.core.models import Candle
from heartless.notify import formatter as F
from heartless.util.timeutil import now_ms
from synth import synth_candles

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _exchange_info():
    return {"symbols": [{"symbol": s, "baseAsset": s[:-4], "quoteAsset": "USDT", "pricePrecision": 2, "quantityPrecision": 3,
                         "status": "TRADING", "contractType": "PERPETUAL", "onboardDate": 1,
                         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                     {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                     {"filterType": "MIN_NOTIONAL", "notional": "5"}]} for s in SYMS]}


class FakeRest:
    def __init__(self, candles):
        self.candles = candles
        self.used_weight = 0
        self.ws_base = "wss://fake"
        self.time_offset = 0

    async def sync_time(self): return None
    async def exchange_info(self): return _exchange_info()
    async def ticker_24h(self): return [{"symbol": s, "quoteVolume": "500000000"} for s in SYMS]
    async def klines_range(self, symbol, start, end, interval="1m"):
        ca = self.candles[symbol]
        return [ca.candle_at(i) for i in range(ca.n) if start <= ca.open_time[i] <= end]
    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        return [{"fundingTime": t, "fundingRate": "0.0001", "markPrice": "100"} for t in range(start or 0, end or 0, 8 * 3600_000)]
    async def open_interest_hist(self, symbol, period="15m", limit=20): return []
    async def close(self): return None


@pytest.fixture
def app_env(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    now = now_ms()
    n_hist = 6000
    start = (now - (n_hist + 400) * 60_000) // 60_000 * 60_000
    candles = {s: synth_candles(n_hist + 300, seed=i, start_ms=start, price=100 * (i + 1)) for i, s in enumerate(SYMS)}
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), HISTORY_DAYS=5, UNIVERSE_SIZE=3,
                        ALWAYS_INCLUDE="BTCUSDT,ETHUSDT,SOLUSDT", WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="")
    settings.data_dir.mkdir(exist_ok=True)
    app = Heartless(settings)
    app.rest = FakeRest(dict(candles))
    return app, candles, n_hist


async def _bootstrap(app, candles, n_hist):
    # feed only the first n_hist bars through "backfill"; the rest arrive as live bars
    for s in SYMS:
        full = candles[s]
        app.rest.candles[s] = full.slice(0, n_hist)
    await app._load_symbols()
    await app.refresh_universe(initial=True)
    await app._backfill(app.universe)
    app._build_views(app.universe)
    await app._create_engines()
    assert set(app.engines) >= {"paper"}
    return {s: candles[s] for s in SYMS}


async def _feed_live(app, candles, n_hist, n_live):
    from heartless.core.models import Ticker
    for i in range(n_hist, n_hist + n_live):
        for s in SYMS:
            c = candles[s].candle_at(i)
            t = app.tickers.setdefault(s, Ticker(s))
            t.bid, t.ask, t.mark, t.last, t.ts = c.close * 0.9999, c.close * 1.0001, c.close, c.close, c.close_time
            await app._tick(s, t)
            await app._process_bar(s, c)
        if i % 30 == 0:
            for eng in app.engines.values():
                await eng.on_equity_tick()


async def test_orchestrator_end_to_end(app_env):
    app, candles, n_hist = app_env
    await _bootstrap(app, candles, n_hist)
    await _feed_live(app, candles, n_hist, 150)
    st = app.status()
    assert st["mode"] == "paper" and st["primary"] == "paper" and len(st["universe"]) == 3
    assert "paper" in st["engines"]
    rep = app.pnl_report("all")
    assert "stats" in rep and rep["engine"] == "paper"
    # formatting must not crash with real objects
    assert "Heartless" in F.fmt_status(st, "Asia/Seoul")
    F.fmt_positions(st, {s: 0.01 for s in SYMS})
    F.fmt_pnl(rep, "Asia/Seoul")
    F.fmt_alphas(app.bandit.snapshot(), app.params)
    F.fmt_daily(app.pnl_report("today"), app.pnl_report("week"), st, "Asia/Seoul")
    # a challenger can be installed and retired
    slot = await app.install_challenger(app.params.clone(note="test"), source="trend_pullback")
    assert slot and slot.name in app.engines
    await app.retire_challenger(slot, reason="test")
    assert slot.name not in app.engines
    # control API
    await app.pause()
    assert app.engines["paper"].risk.state.paused
    await app.resume()
    assert not app.engines["paper"].risk.state.paused
    assert "키" in await app.set_mode("live")  # no keys -> refused
    await app.close_all()


async def test_web_api_with_token(app_env):
    app, candles, n_hist = app_env
    await _bootstrap(app, candles, n_hist)
    await _feed_live(app, candles, n_hist, 30)
    from heartless.web.app import WebServer

    web = WebServer(app)
    transport = httpx.ASGITransport(app=web.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/status")
        assert r.status_code == 401
        r = await client.get("/", params={"token": "wrong"})
        assert r.status_code == 401
        r = await client.get("/api/status", headers={"x-token": app.web_token})
        assert r.status_code == 200 and r.json()["mode"] == "paper"
        r = await client.get("/", params={"token": app.web_token})
        assert r.status_code == 200 and "Heartless" in r.text and "hl_token" in r.headers.get("set-cookie", "")
        for path in ("/api/pnl?period=week", "/api/trades", "/api/equity", "/api/engines", "/api/alphas", "/api/research",
                     "/api/events", "/api/health"):
            r = await client.get(path, headers={"x-token": app.web_token})
            assert r.status_code == 200, path
        r = await client.post("/api/pause", headers={"x-token": app.web_token})
        assert r.json()["ok"] and app.engines["paper"].risk.state.paused
        r = await client.post("/api/mode", json={"mode": "bogus"}, headers={"x-token": app.web_token})
        assert r.status_code == 400
