"""F7 runtime: concurrent start-up checks, prefetched reads, connection reuse, clock drift, the latency budget, the
optional speed-ups (orjson, uvloop) and the replay's performance report (hot-path profile, order-book proxy)."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

from heartless.config import Settings
from heartless.exchange import binance_rest as BR
from heartless.exchange.binance_rest import BinanceError, BinanceRest
from heartless.exchange.binance_ws import JSON_ENGINE, MarketStream, StreamLag, loads
from heartless.util import perf

LAT = 0.4  # seconds per mocked request: eleven of them one after another would take 4.4 s


class SlowRest(BinanceRest):
    """BinanceRest with the network replaced: every request takes LAT seconds and returns canned Binance JSON."""

    def __init__(self, server_skew_ms: int = 0):
        super().__init__("k", "s", False)
        self.sent: list[str] = []
        self.sent_at: list[tuple[str, float]] = []  # (path, loop time the request started)
        self.skew = server_skew_ms

    async def _send(self, method, path, params=None, signed=False, retries=3):
        self.sent.append(path)
        self.sent_at.append((path, asyncio.get_running_loop().time()))
        await asyncio.sleep(LAT)
        if path == "/fapi/v1/time":
            return {"serverTime": int(time.time() * 1000) + self.skew}
        if path == "/fapi/v1/exchangeInfo":
            return {"symbols": [{"symbol": s, "pair": s, "contractType": "PERPETUAL", "status": "TRADING",
                                 "baseAsset": s[:-4], "quoteAsset": "USDT", "marginAsset": "USDT",
                                 "pricePrecision": 2, "quantityPrecision": 3, "onboardDate": 1_600_000_000_000,
                                 "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                                             {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                             {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                             {"filterType": "MIN_NOTIONAL", "notional": "5"}]}
                                for s in ("BTCUSDT", "ETHUSDT")]}
        if path == "/fapi/v1/premiumIndex":
            return [{"symbol": "BTCUSDT", "nextFundingTime": 1_800_000_000_000}]
        if path == "/fapi/v1/fundingInfo":
            return [{"symbol": "ETHUSDT", "fundingIntervalHours": 4}]
        if path == "/fapi/v1/ticker/24hr":
            return [{"symbol": s, "quoteVolume": "1e10", "lastPrice": p, "priceChangePercent": "1"}
                    for s, p in (("BTCUSDT", "60000"), ("ETHUSDT", "3000"))]
        if path == "/fapi/v1/positionSide/dual":
            return {"dualSidePosition": False}
        if path in ("/fapi/v3/account",):
            return {"totalWalletBalance": "1000", "totalUnrealizedProfit": "0", "availableBalance": "1000"}
        if path in ("/fapi/v3/positionRisk", "/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders"):
            return []
        raise BinanceError(-1121, f"not mocked: {path}", 404)


def _app(**kw):
    from heartless.app import Heartless

    tmp = Path(tempfile.mkdtemp())
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 ALWAYS_INCLUDE="BTCUSDT", ARCHIVE_BACKFILL=False, **kw)
    s.data_dir.mkdir(exist_ok=True)
    return Heartless(s)


def test_startup_checks_run_concurrently_under_three_seconds():
    async def main():
        app = _app()
        app.rest = SlowRest()
        t0 = time.perf_counter()
        timings = await app._f7_startup_checks()
        await app.refresh_universe(initial=True)  # reads the prefetched 24h tickers
        live = await app._f7_prefetch_account()
        # what LiveAccount.start and the engine's reconcile read next: all prefetched, no round trip
        sent = len(app.rest.sent)
        await app.rest.get_position_mode()
        await app.rest.account()
        await app.rest.position_risk()
        await app.rest.open_orders()
        await app.rest.open_algo_orders()
        elapsed = time.perf_counter() - t0
        assert len(app.rest.sent) == sent  # nothing was sent twice
        await app.rest.close()
        app.store.close()
        return app, timings, live, elapsed

    app, timings, live, elapsed = asyncio.run(main())
    # 11 requests (time, exchangeInfo, premiumIndex, fundingInfo, 24h tickers, 5 account reads) in two bursts
    assert elapsed < 3.0 and elapsed < 11 * LAT
    assert set(app.symbols) == {"BTCUSDT", "ETHUSDT"} and app.universe
    assert app.funding_next["BTCUSDT"] == 1_800_000_000_000 and app.funding_interval["ETHUSDT"] == 4 * 3_600_000
    assert app.rest.sent.count("/fapi/v1/ticker/24hr") == 1 and app.rest.sent.count("/fapi/v1/premiumIndex") == 1
    assert {"time", "exchangeInfo", "prefetch", "funding_schedule", "total"} <= set(timings)
    assert timings["total"] < 2 * LAT * 1e3 + 300 and timings["funding_schedule"] < 100  # served from the prefetch
    # the five market checks start together (red-team F-L9: the total bound above also passed with the clock sync
    # awaited before the others, 0.8 s instead of 0.4 s)
    market = ("/fapi/v1/time", "/fapi/v1/exchangeInfo", "/fapi/v1/premiumIndex", "/fapi/v1/fundingInfo",
              "/fapi/v1/ticker/24hr")
    starts = [t for p, t in app.rest.sent_at if p in market]
    assert len(starts) == 5 and max(starts) - min(starts) < 0.05
    assert timings["total"] < LAT * 1e3 + 300
    assert live["live total"] < 2 * LAT * 1e3 and len([k for k in live if k != "live total"]) == 5
    assert app.startup_timings["total"] == timings["total"]


def test_startup_failures_surface_as_before():
    class NoInfo(SlowRest):
        async def _send(self, method, path, params=None, signed=False, retries=3):
            if path == "/fapi/v1/exchangeInfo":
                raise BinanceError(-1000, "down", 503)
            return await super()._send(method, path, params, signed, retries)

    async def main():
        app = _app()
        app.rest = NoInfo()
        try:
            with pytest.raises(BinanceError):
                await app._f7_startup_checks()
        finally:
            await app.rest.close()
            app.store.close()

    asyncio.run(main())


def test_prefetched_answers_are_used_once_and_expire():
    async def main():
        r = SlowRest()
        await r.prefetch([("GET", "/fapi/v1/fundingInfo", None, False), ("GET", "/nope", None, False)])
        assert r.sent == ["/fapi/v1/fundingInfo", "/nope"]
        assert (await r.request("GET", "/fapi/v1/fundingInfo"))[0]["symbol"] == "ETHUSDT"
        assert r.sent == ["/fapi/v1/fundingInfo", "/nope"]  # served from the prefetch
        await r.request("GET", "/fapi/v1/fundingInfo")
        assert r.sent.count("/fapi/v1/fundingInfo") == 2  # used once
        with pytest.raises(BinanceError):  # a failed prefetch is not kept: the consumer sees the error itself
            await r.request("GET", "/nope")
        await r.prefetch([("GET", "/fapi/v1/premiumIndex", {"symbol": None}, False)], ttl=0.0)
        await asyncio.sleep(0.01)
        await r.request("GET", "/fapi/v1/premiumIndex")
        assert r.sent.count("/fapi/v1/premiumIndex") == 2  # expired
        await r.prefetch([("GET", "/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"}, False)])
        await r.request("GET", "/fapi/v1/premiumIndex")  # other parameters: not the same request
        assert r.sent.count("/fapi/v1/premiumIndex") == 4
        await r.close()

    asyncio.run(main())

    class Local(BinanceRest):  # answers locally (like the replay): prefetch leaves it alone
        async def request(self, method, path, params=None, signed=False, retries=3):
            return {}

    async def local():
        r = Local()
        assert await r.prefetch([("GET", "/x", None, False)]) == {}
        await r.close()

    asyncio.run(local())


def test_connection_pool_keeps_idle_connections_and_keep_warm_syncs_time(caplog):
    r = BinanceRest()
    pool = getattr(getattr(r._client, "_transport", None), "_pool", None)
    if pool is not None:
        assert getattr(pool, "_keepalive_expiry", BR.KEEPALIVE_S) == BR.KEEPALIVE_S
    assert BR.POOL_LIMITS.keepalive_expiry == BR.KEEPALIVE_S > BR.WARM_INTERVAL_S
    asyncio.run(r.close())

    perf.reset()

    async def main():
        rr = SlowRest(server_skew_ms=-2500)  # the exchange clock 2.5 s behind ours
        task = asyncio.create_task(rr.keep_warm(0.01))
        await asyncio.sleep(1.0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await rr.close()
        return rr

    with caplog.at_level(logging.WARNING):
        rr = asyncio.run(main())
    assert rr.sent.count("/fapi/v1/time") >= 1 and rr.time_rtt_ms >= LAT * 1e3 * 0.9
    snap = perf.DRIFT.snapshot()
    assert 2000 < snap["offset_ms"]["rest"] < 3500 and snap["warnings"] == 1
    assert any("clock drift" in m for m in caplog.messages)
    perf.reset()


def test_stream_lag_reports_the_window_minimum_to_the_drift_monitor():
    perf.reset()
    lag = StreamLag("stream")
    lag.observe(1_000_000, recv_ms=1_000_050)
    lag.observe(1_000_000, recv_ms=1_000_020)  # minimum of the first window
    assert "stream" not in perf.DRIFT.last
    lag.observe(1_060_000, recv_ms=1_061_000 + 30)  # a new window: the previous minimum is reported
    assert perf.DRIFT.last["stream"] == 20 and lag.min_last_window == 20
    lag.observe(None)
    perf.reset()


def test_the_market_stream_parses_bytes_and_text_frames():
    got = []
    ms = MarketStream("wss://x", got.append, lambda d: None, lambda d: None)
    k = {"e": "kline", "E": 5, "s": "BTCUSDT", "k": {"x": True}}
    asyncio.run(ms._dispatch(json.dumps({"stream": "btcusdt@kline_1m", "data": k}).encode()))
    asyncio.run(ms._dispatch(json.dumps({"stream": "btcusdt@kline_1m", "data": k})))
    asyncio.run(ms._dispatch(b"{broken"))
    assert len(got) == 2 and JSON_ENGINE in ("orjson", "json") and loads(b"[1, 2.5]") == [1, 2.5]


def test_uvloop_is_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "uvloop", None)  # import fails
    monkeypatch.setattr(perf, "_UVLOOP", False)  # not looked up yet
    assert perf.loop_factory() is None and perf._UVLOOP is None
    fake = types.ModuleType("uvloop")
    fake.new_event_loop = asyncio.new_event_loop
    monkeypatch.setitem(sys.modules, "uvloop", fake)
    monkeypatch.setattr(perf, "_UVLOOP", False)
    assert perf.loop_factory() is asyncio.new_event_loop and perf.loop_factory(False) is None
    assert asyncio.run(asyncio.sleep(0, "ok"), loop_factory=perf.loop_factory()) == "ok"


def test_latency_budget_percentiles_and_lines():
    b = perf.LatencyBudget(size=100)
    for v in range(1, 201):  # only the last 100 samples count
        b.record("paper", "sent_to_ack", float(v))
    b.record("paper", "close_to_decision", 5.0)
    b.record("paper", "close_to_decision", float("nan"))  # ignored
    snap = b.snapshot()["paper"]
    assert list(snap) == ["close_to_decision", "sent_to_ack"]  # stage order
    assert snap["sent_to_ack"]["n"] == 200 and snap["sent_to_ack"]["p50"] == pytest.approx(150.5)
    assert snap["sent_to_ack"]["p99"] == pytest.approx(199.01) and snap["sent_to_ack"]["max"] == 200
    assert snap["close_to_decision"]["n"] == 1
    assert b.lines()[0].startswith("paper: 봉 마감→결정 p50 5.00ms")
    assert perf.percentile([], 50) != perf.percentile([], 50)  # NaN


def test_order_calls_are_timed_against_the_bar_close():
    perf.reset()

    class Acc:
        async def market_order(self, symbol, side, qty, reduce_only=False, client_id=""):
            await asyncio.sleep(0.02)
            return "ok"

        async def place_stop(self, symbol, side, trigger_price, qty=None, **k):
            return "stop"

    async def main():
        app = _app()
        acc = Acc()
        app._f7_time_orders(acc, "paper")
        app._f7_bar["BTCUSDT"] = 1_000
        assert await acc.market_order("BTCUSDT", "BUY", 1.0) == "ok"
        app._f7_bar.pop("BTCUSDT")
        assert await acc.market_order(symbol="ETHUSDT", side="SELL", qty=1.0) == "ok"  # not decided on a bar
        assert await acc.place_stop("BTCUSDT", "SELL", 90.0) == "stop"
        app.store.close()
        await app.rest.close()

    asyncio.run(main())
    snap = perf.LATENCY.snapshot()["paper"]
    assert snap["close_to_sent"]["n"] == 1 and snap["sent_to_ack"]["n"] == 3
    assert snap["sent_to_ack"]["max"] >= 15
    perf.reset()


def test_context_carries_the_book_features():
    from heartless.core.models import SymbolInfo
    from heartless.exchange.orderbook import EMPTY_BOOK, OrderBooks

    off = _app()
    off.symbols = {"BTCUSDT": SymbolInfo("BTCUSDT", "BTC", "USDT", 0.1, 0.001, 0.001, 5.0, 1, 3)}
    assert off._context("BTCUSDT", 1_000).book is EMPTY_BOOK and off.books is None and off.book_stream is None
    on = _app(ORDERBOOK_STREAM=True, ORDERBOOK_DEPTH_BPS=25)
    assert isinstance(on.books, OrderBooks) and on.book_stream is not None and on.book_stream.speed_ms == 100
    on.symbols = dict(off.symbols)
    on.books.set_symbols(["BTCUSDT"])
    asyncio.run(on.books.on_message({"e": "depthUpdate", "s": "BTCUSDT", "u": 3, "U": 1, "pu": 0,
                                     "b": [["100.0", "2"], ["99.0", "1"]], "a": [["100.1", "1"], ["101.0", "1"]]}))
    b = on._context("BTCUSDT", 1_000).book  # levels reach past 25 bp on both sides: the depth within k is known
    assert b.source == "stream" and b.k_bps == 25 and b.depth_bid_k == pytest.approx(200.0) and b.covered_bps > 25
    assert "orderbook" in on._f7_health() and "latency" in on._f7_health() and "features" in off._f7_health()
    assert off._f7_health() is off._f7_health()  # refreshed at most once a minute
    for a in (off, on):
        a.store.close()
        asyncio.run(a.rest.close())


# --- the replay's performance report ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def perf_replay(tmp_path_factory):
    from tests.test_p5_replay import SEED_DAYS, SMALL_FRAMES, START, build_source, busy_params

    from heartless.data import features
    from heartless.replay.runner import ReplayConfig, run_replay
    from heartless.util.timeutil import MS_DAY, MS_HOUR

    d = tmp_path_factory.mktemp("f7replay")
    seen = {}

    def peek(app):
        seen["book"] = app._context("BTCUSDT", app.views["BTCUSDT"].cursor_time).book
        seen["live"] = {s: v.live is not None for s, v in app.views.items()}

    with pytest.MonkeyPatch.context() as mp:
        for tf, n in SMALL_FRAMES.items():
            mp.setitem(features.LIVE_WINDOW, tf, n)
        cfg = ReplayConfig(source_db=build_source(d / "src.db", 1), start_ms=START, end_ms=START + MS_DAY - 1,
                           work_dir=str(d / "bot"), seed_days=SEED_DAYS, params=busy_params(),
                           settings={"RESEARCH_INTERVAL_MINUTES": 10**9, "DISCOVERY_INTERVAL_HOURS": 10**8,
                                     "HISTORY_DAYS": 12, "ORDERBOOK_STREAM": True},
                           commands=[], profile=True, profile_top=15, out_json=str(d / "summary.json"),
                           inject=[(START + 12 * MS_HOUR + 30_000, peek)])
        return run_replay(cfg), seen


def test_replay_reports_latency_features_profile_and_the_book_proxy(perf_replay):
    s, seen = perf_replay
    assert s["passed"], s["invariants"]
    p = s["perf"]
    f = p["features"]
    assert f["incremental_views"] == f["views"] == 2 and f["batch_views"] == 0
    assert f["bars"] >= 2 * 1440 - 5 and f["builds"] == 2 and f["verified"] == f["reanchors"] > 0
    assert p["latency"]["app"]["feature_update"]["n"] >= 2 * 1440 - 5
    assert p["latency"]["paper"]["close_to_decision"]["n"] >= 2 * 1440 - 5
    assert p["orderbook"]["mode"] == "proxy" and p["orderbook"]["calls"] > 0
    assert p["profile"]["top"] and all({"function", "calls", "tottime_s", "cumtime_s"} <= set(r)
                                       for r in p["profile"]["top"])
    assert "performance:" in s["report"] and "hot path" in s["report"]
    assert seen["book"].source == "proxy" and seen["book"].spread_bps > 0 and seen["book"].imbalance == 0.0
    assert all(seen["live"].values())


# --- review fixes ----------------------------------------------------------------------------------------------------
def test_the_context_book_age_is_never_negative():
    """ctx.book is built on a bar close that is in the past by the processing delay; a book updated after that close
    is a few milliseconds old, not negative (the staleness used to be bar close - local receive time)."""
    from heartless.core.models import SymbolInfo
    from heartless.util.timeutil import now_ms

    app = _app(ORDERBOOK_STREAM=True)
    app.symbols = {"BTCUSDT": SymbolInfo("BTCUSDT", "BTC", "USDT", 0.1, 0.001, 0.001, 5.0, 1, 3)}
    app.books.set_symbols(["BTCUSDT"])
    asyncio.run(app.books.on_message({"e": "depthUpdate", "s": "BTCUSDT", "u": 3, "U": 1, "pu": 0,
                                      "b": [["100.0", "2"]], "a": [["100.1", "1"]]}))
    bar_close = now_ms() - 45_000
    age = app._context("BTCUSDT", bar_close).book.staleness_ms
    assert 0 <= age < 5_000
    app.store.close()
    asyncio.run(app.rest.close())


def test_new_credentials_drop_prefetched_answers():
    """Prefetched answers are keyed by request, not by account: after the owner swaps the keys, neither a stored answer
    nor one still in flight may reach the new account's start-up reads."""
    async def main():
        r = SlowRest()
        await r.prefetch([("GET", "/fapi/v3/account", None, True)])
        r.set_credentials("k2", "s2")
        await r.request("GET", "/fapi/v3/account", signed=True)
        assert r.sent.count("/fapi/v3/account") == 2  # asked again with the new keys
        task = asyncio.create_task(r.prefetch([("GET", "/fapi/v3/positionRisk", None, True)]))
        await asyncio.sleep(LAT / 4)
        r.set_credentials("k3", "s3")  # while the prefetch is on its way
        took = await task
        assert took["/fapi/v3/positionRisk"] < 0 and not r._prefetched
        await r.request("GET", "/fapi/v3/positionRisk", signed=True)
        assert r.sent.count("/fapi/v3/positionRisk") == 2
        await r.close()

    asyncio.run(main())


def test_health_reads_each_views_incremental_state_once():
    """A feature step in a worker thread can drop view.live (fallback to batch) at any moment; the health summary must
    not read it twice (a check, then .stats on what has become None)."""
    class Flaky:
        incremental = True

        def __init__(self):
            self.reads = 0

        @property
        def live(self):
            self.reads += 1
            return types.SimpleNamespace(stats={"reanchors": 3}) if self.reads == 1 else None

    app = _app()
    app.views = {"BTCUSDT": Flaky()}
    out = app._f7_health_now()
    assert "incremental 1/1 symbols, 3 re-anchors" in out["features"]
    app.store.close()
    asyncio.run(app.rest.close())
