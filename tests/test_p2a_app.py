"""P2a market context wired into the orchestrator and the replay harness.

The real app keeps the online MarketContextTracker for CONTEXT_UNIVERSE: the traded members come from the trading
path, the others ("context-only") are streamed and stored as 1m klines only, funding settlements reach it through the
funding refresh and open interest through the metrics poll. At every hour close the snapshot is published before any
engine decides, so ctx.market for every engine is exactly what the offline market_context build gives for the same
data -- the property the P2 research relies on (alphas researched offline see in live trading what they saw in the
lab)."""
from __future__ import annotations

import asyncio
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.app import CONTEXT_GRACE_MS, CONTEXT_HISTORY_DAYS, Heartless
from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import context as C
from heartless.data import features
from heartless.exchange.binance_ws import MarketStream
from heartless.execution.engine import TradingEngine
from heartless.replay.clock import run_virtual
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE, now_ms

START = 1_772_323_200_000  # 2026-03-01 00:00 UTC: first stored bar
HIST_DAYS = 35  # stored before the app starts: covers the tracker's whole warm-up window (C.WARMUP_HOURS)
T0 = START + HIST_DAYS * MS_DAY + 7 * MS_HOUR  # app start 07:00 UTC: the 08:00 funding settlement happens live
LIVE_HOURS = 3
T_END = T0 + LIVE_HOURS * MS_HOUR - 1  # close of the last streamed bar
TRADED = ["BTCUSDT", "ETHUSDT"]  # ETHUSDT is traded but outside the context universe: market-wide features only
CONTEXT_ONLY = ["SOLUSDT", "XRPUSDT"]  # in the context universe, listed, not traded
CTX_UNIVERSE = ["BTCUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]  # DOGEUSDT is not listed: never any data
SMALL_FRAMES = {"1m": 240, "5m": 240, "15m": 240, "1h": 260}
# a context-only bar that closes an hour arrives late: inside the grace (the hour close waits for it) and, at the last
# hour close, after it (the snapshot goes out without it)
LATE = {("SOLUSDT", T0 + MS_HOUR - 1): 3_000, ("XRPUSDT", T_END): CONTEXT_GRACE_MS + 5_000}


def _history() -> dict:
    rng = np.random.default_rng(7)
    n = (T_END + 1 - START) // MS_MINUTE
    common = rng.normal(0, 8e-4, n)
    out = {}
    for k, sym in enumerate(TRADED + CONTEXT_ONLY):
        ret = 0.6 * common + rng.normal(2e-5 * (k - 1.5), 9e-4, n)
        close = (100 + 50 * k) * np.exp(np.cumsum(ret))
        opn = np.concatenate(([close[0]], close[:-1]))
        hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 3e-4, n)))
        lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 3e-4, n)))
        vol = np.abs(rng.normal(50, 20, n))
        t = START + np.arange(n, dtype=np.int64) * MS_MINUTE
        bars = [Candle(int(t[i]), float(opn[i]), float(hi[i]), float(lo[i]), float(close[i]), float(vol[i]),
                       float(vol[i] * close[i]), 5, float(vol[i] * 0.5), int(t[i]) + 59_999) for i in range(n)]
        funding = [(START + j * 8 * MS_HOUR, float(rng.normal(1e-4, 2e-4))) for j in range((T_END - START) // (8 * MS_HOUR) + 1)]
        metrics = [{"ts": START + j * 300_000, "oi": float(1e6 * (1 + 0.1 * math.sin(j / 300 + k))),
                    "oi_value": float(5e7 * (1 + 0.1 * math.sin(j / 200 + k))), "top_ls_accounts": 1.1,
                    "top_ls_positions": 1.2, "ls_accounts": 1.3, "taker_ls_vol": 0.9}
                   for j in range((T_END + 1 - START) // 300_000)]
        out[sym] = {"bars": bars, "by_open": {b.open_time: b for b in bars}, "funding": funding, "metrics": metrics}
    return out


class _Rest:
    """Public Binance REST answered from the synthetic history: only what exists at the (virtual) current time."""

    used_weight = 0
    ws_base = "wss://fake"

    def __init__(self, hist: dict):
        self.h = hist
        self.calls: list[tuple[str, str]] = []

    async def sync_time(self): return None
    async def close(self): return None

    async def exchange_info(self):
        return {"symbols": [{"symbol": s, "baseAsset": s[:-4], "quoteAsset": "USDT", "pricePrecision": 2,
                             "quantityPrecision": 3, "status": "TRADING", "contractType": "PERPETUAL", "onboardDate": 1,
                             "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                         {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                         {"filterType": "MIN_NOTIONAL", "notional": "5"}]} for s in self.h]}

    async def ticker_24h(self):  # only the traded pair is liquid enough for the universe
        return [{"symbol": s, "quoteVolume": "900000000"} for s in TRADED]

    async def klines_range(self, symbol, start, end, interval="1m"):
        self.calls.append(("klines", symbol))
        now = now_ms()
        return [b for b in self.h[symbol]["bars"] if start <= b.open_time <= end and b.close_time <= now]

    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        self.calls.append(("funding", symbol))
        hi = min(end if end is not None else 2**62, now_ms())
        return [{"symbol": symbol, "fundingTime": t, "fundingRate": repr(r), "markPrice": "0"}
                for t, r in self.h[symbol]["funding"] if (start or 0) <= t <= hi]

    async def open_interest_hist(self, symbol, period="5m", limit=30):
        self.calls.append(("oi", symbol))
        rows = [m for m in self.h[symbol]["metrics"] if m["ts"] + 300_000 <= now_ms()][-limit:]
        return [{"symbol": symbol, "sumOpenInterest": repr(m["oi"]), "sumOpenInterestValue": repr(m["oi_value"]),
                 "timestamp": m["ts"] + 300_000} for m in rows]  # REST stamps the bucket end

    async def _empty(self, symbol, period="5m", limit=30):
        return []

    top_long_short_ratio = top_long_short_account_ratio = global_long_short_account_ratio = _empty
    taker_long_short_ratio = _empty


class _Stream(MarketStream):
    """Closed 1m klines of every subscribed symbol (traded and kline-only) at their close time, as websocket frames;
    the bars in LATE are held back by their delay."""

    def __init__(self, hist: dict, app):
        super().__init__("fake://", app._on_kline, app._on_book, app._on_mark)
        self.h = hist
        self.finished = asyncio.Event()
        self.sent: dict[str, int] = {}

    async def _send(self, sym: str, c: Candle, delay: int = 0) -> None:
        if delay:
            await asyncio.sleep(delay / 1000)
        k = {"t": c.open_time, "T": c.close_time, "s": sym, "i": "1m", "o": repr(c.open), "c": repr(c.close),
             "h": repr(c.high), "l": repr(c.low), "v": repr(c.volume), "n": c.trades, "x": True,
             "q": repr(c.quote_volume), "V": repr(c.taker_buy_volume)}
        await self._dispatch(json.dumps({"stream": f"{sym.lower()}@kline_1m", "data": {"e": "kline", "s": sym, "k": k}}))
        self.sent[sym] = self.sent.get(sym, 0) + 1

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        held = []
        m = T0
        while m + MS_MINUTE - 1 <= T_END:
            await loop.sleep_until_ms(m + MS_MINUTE - 1)
            for sym in [*self.symbols, *self.kline_only]:
                c = self.h[sym]["by_open"][m]
                delay = LATE.get((sym, c.close_time), 0)
                if delay:
                    held.append(asyncio.create_task(self._send(sym, c, delay)))
                else:
                    await self._send(sym, c)
            m += MS_MINUTE
        await asyncio.gather(*held)
        await asyncio.sleep(1)  # the worker finishes the last bars
        self.finished.set()


def _bits(a: Mapping, b: Mapping) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = float(a[k]), float(b[k])
        if math.isnan(x) != math.isnan(y) or (not math.isnan(x) and np.float64(x).view(np.int64) !=
                                              np.float64(y).view(np.int64)):
            return False
    return True


@pytest.fixture(scope="module")
def small_frames():
    with pytest.MonkeyPatch.context() as mp:
        for tf, n in SMALL_FRAMES.items():
            mp.setitem(features.LIVE_WINDOW, tf, n)
        yield


@pytest.fixture(scope="module")
def live_run(small_frames):
    """The whole orchestrator (app.run(): startup, warm-up, stream, kline worker, scheduler) on the virtual-time loop
    over LIVE_HOURS of streamed bars, with every engine's on_bar recorded; plus the offline build of the same data."""
    hist = _history()
    tmp = Path(tempfile.mkdtemp())
    # what a bot that has been running keeps besides candles and funding (which its own backfill fetches): the 5m
    # metrics it polled before this start
    st = Store(tmp / "heartless.db")
    for sym, d in hist.items():
        st.save_metrics(sym, [SimpleNamespace(**m) for m in d["metrics"] if m["ts"] <= T0 - 300_000])
    p = StrategyParams.default().with_alpha("trend_pullback", {}, note="challenger under test")
    st.save_params_version(p.version, p.created, "challenger", "test", p.note, p.to_dict(),
                           {"slot": "challenger-1", "started": T0})
    st.close()
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), HISTORY_DAYS=2, UNIVERSE_SIZE=2,
                        ALWAYS_INCLUDE=",".join(TRADED), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="", ARCHIVE_BACKFILL=False,
                        TIMEZONE="UTC", RESEARCH_INTERVAL_MINUTES=10**9, DISCOVERY_INTERVAL_HOURS=10**8,
                        CONTEXT_UNIVERSE=",".join(CTX_UNIVERSE))
    seen: list[tuple[str, str, int, Mapping]] = []
    orig = TradingEngine.on_bar

    async def on_bar(self, view, ctx):
        seen.append((self.name, view.symbol, ctx.now, ctx.market))
        return await orig(self, view, ctx)

    out: dict = {}

    async def main(loop):
        await loop.sleep_until_ms(T0 + 2_500)
        app = Heartless(settings)
        app.rest = _Rest(hist)
        app.market = stream = _Stream(hist, app)
        task = asyncio.create_task(app.run())
        await stream.finished.wait()
        out.update(app=app, stream=stream, last=app.context.snapshot(T_END), stats=dict(app.context_stats),
                   engines=sorted(app.engines), universe=list(app.universe), context_only=list(app.context_only),
                   calls=list(app.rest.calls), funding=app.store.load_funding("SOLUSDT", T0, T_END),
                   stored=app.store.candle_range("XRPUSDT"), health=app.health)
        await app.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TradingEngine, "on_bar", on_bar)
        run_virtual(main, T0)
    src = Store(tmp / "offline_src.db")
    for sym, d in hist.items():
        src.save_candles(sym, d["bars"])
        src.save_funding(sym, [(t, r, 0.0) for t, r in d["funding"]])
        src.save_metrics(sym, [SimpleNamespace(**m) for m in d["metrics"]])
    dst = Store(tmp / "offline_ctx.db")
    C.build(src, dst, universe=CTX_UNIVERSE)
    table = C.ContextTable.load(dst, CTX_UNIVERSE + ["ETHUSDT"], T0 - 2 * MS_HOUR, T_END)
    return out, seen, table


def test_every_engine_decides_on_the_offline_snapshot_of_the_hour(live_run):
    out, seen, table = live_run
    assert out["universe"] == TRADED and out["context_only"] == CONTEXT_ONLY
    assert out["engines"] == ["challenger-1", "paper"]
    bars = [x for x in seen if x[2] < T_END]  # the last hour close is checked below (a bar arrived after the grace)
    assert len(bars) == 2 * len(TRADED) * (LIVE_HOURS * 60 - 1)  # every engine, every traded symbol, every minute
    hours = set()
    for eng, sym, t, market in bars:
        expect = table.view(sym, t)
        assert market["ts"] == C.hour_close(C.hour_of(t)) == expect["ts"], (eng, sym, t)
        assert _bits(market, expect), (eng, sym, C._iso(t))
        hours.add(market["ts"])
    assert hours == {T0 - 1, T0 + MS_HOUR - 1, T0 + 2 * MS_HOUR - 1}
    btc = [m for _, s, t, m in bars if s == "BTCUSDT" and t == T0 + MS_HOUR - 1]
    eth = [m for _, s, t, m in bars if s == "ETHUSDT" and t == T0 + MS_HOUR - 1]
    assert len(btc) == len(eth) == 2 and btc[0] is btc[1]  # both engines read the one snapshot of the hour
    # defined values (the warm-up covered the long windows): own features for BTC, market-wide only for ETH
    assert all(np.isfinite(btc[0][k]) for k in ("ret_168h", "ret_24h_rank", "above_ema200", "funding_z", "oi_chg_24h"))
    assert np.isfinite(btc[0]["breadth_ema200"]) and btc[0]["n"] == 3
    assert "ret_1h" not in eth[0] and _bits({k: eth[0][k] for k in C.MARKET_FEATURES},
                                            {k: btc[0][k] for k in C.MARKET_FEATURES})


def test_the_hour_close_waits_for_a_late_context_bar_inside_the_grace(live_run):
    out, seen, table = live_run
    T = T0 + MS_HOUR - 1  # SOLUSDT's bar closing at T arrived 3 s late: the decisions at T waited for it
    decided = [m for _, _, t, m in seen if t == T]
    assert decided and all(_bits(m, table.view(s, T)) for (_, s, t, m) in seen if t == T)
    # the 08:00 settlement was fetched by the live funding refresh and is in the 08:59 snapshot, as offline
    assert (T0 + MS_HOUR, ) == tuple(t for t, _ in out["funding"] if t > T0 and t <= T0 + MS_HOUR)
    assert table.view("BTCUSDT", T0 + 2 * MS_HOUR - 1)["funding"] == out["app"].context.view(
        "BTCUSDT", T0 + 2 * MS_HOUR - 1)["funding"]


def test_a_bar_later_than_the_grace_is_left_out_and_counted(live_run):
    out, seen, table = live_run
    st = out["stats"]
    assert st["late"] == 1 and st["errors"] == 0 and st["last"] == T_END
    assert st["snapshots"] == LIVE_HOURS  # one per live hour close (the start hour came from the warm-up)
    last = out["last"]
    j = last.symbols.index("XRPUSDT")
    offline = table.symbols["XRPUSDT"][T_END]
    names = list(C.SYMBOL_FEATURES)
    assert last.sym[j][names.index("ret_1h")] != offline[names.index("ret_1h")]  # the :59 bar is missing
    k = last.symbols.index("BTCUSDT")
    assert last.sym[k][names.index("ret_1h")] == table.symbols["BTCUSDT"][T_END][names.index("ret_1h")]
    # every engine still decided at T_END, after the grace, on that one snapshot
    at_end = [m for _, _, t, m in seen if t == T_END]
    assert len(at_end) == 2 * len(TRADED) and all(m["ts"] == T_END for m in at_end)


def test_context_only_symbols_are_streamed_and_stored_without_trading_state(live_run):
    out, _, _ = live_run
    app, stream = out["app"], out["stream"]
    assert stream.kline_only == CONTEXT_ONLY and stream.symbols == TRADED
    assert all(stream.sent[s] == LIVE_HOURS * 60 for s in TRADED + CONTEXT_ONLY)
    for s in CONTEXT_ONLY:  # no views, frames, engines' symbols or metrics series for them
        assert s not in app.candles and s not in app.views and s not in app.metrics_series
    lo, hi, n = out["stored"]
    # the backfill reached back past HISTORY_DAYS (2) for the tracker's warm-up (all CONTEXT_HISTORY_DAYS that exist),
    # and every streamed bar (also the one after the grace) was stored
    assert lo == START < T0 - 2 * MS_DAY and T0 - START < CONTEXT_HISTORY_DAYS * MS_DAY and hi == T_END - 59_999
    kinds = {(k, s) for k, s in out["calls"]}
    assert ("oi", "XRPUSDT") in kinds and ("funding", "XRPUSDT") in kinds and ("klines", "XRPUSDT") in kinds
    assert not any(s == "DOGEUSDT" for _, s in out["calls"])  # not listed on the exchange: never requested
    assert "3/4 symbols" in out["health"]["market_context"]


def test_a_tracker_failure_never_reaches_the_engines():
    tmp = Path(tempfile.mkdtemp())
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="")
    s.data_dir.mkdir(exist_ok=True)
    app = Heartless(s)

    def boom(*a, **k):
        raise RuntimeError("tracker bug")

    app.context.view = boom
    app.context.on_bar = boom
    assert app._market_context("BTCUSDT", now_ms()) is C.EMPTY
    app._context_bar("BTCUSDT", Candle(0, 1, 1, 1, 1, 1, 1, 1, 1, 59_999))
    assert app.context_stats["errors"] == 2
    app.store.close()


# --- the replay harness serves the context universe ---------------------------------------------------------------
def test_a_replay_streams_the_context_symbols_and_publishes_the_offline_snapshot_every_hour(tmp_path, small_frames):
    from heartless.replay.runner import ReplayConfig, context_digest, run_replay

    seed_days, hours = 14, 6
    start = START + 30 * MS_DAY
    t0 = start - seed_days * MS_DAY
    n = (seed_days * 24 + hours) * 60
    rng = np.random.default_rng(5)
    src = Store(tmp_path / "src.db")
    for k, sym in enumerate(TRADED + CONTEXT_ONLY):  # all four are in the default CONTEXT_UNIVERSE
        close = (100 + 30 * k) * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
        opn = np.concatenate(([close[0]], close[:-1]))
        vol = np.abs(rng.normal(40, 10, n))
        src.save_candles(sym, [Candle(t0 + i * MS_MINUTE, float(opn[i]), float(max(opn[i], close[i]) * 1.0002),
                                      float(min(opn[i], close[i]) * 0.9998), float(close[i]), float(vol[i]),
                                      float(vol[i] * close[i]), 3, float(vol[i] / 2), t0 + i * MS_MINUTE + 59_999)
                                for i in range(n)])
        src.save_funding(sym, [(t, float(rng.normal(1e-4, 1e-4)), 0.0) for t in range(t0, t0 + n * MS_MINUTE, 8 * MS_HOUR)])
        src.save_metrics(sym, [SimpleNamespace(ts=t, oi=1e6 + 50 * j + 1e4 * k, oi_value=1e8 + j, top_ls_accounts=1.1,
                                               top_ls_positions=1.2, ls_accounts=1.3, taker_ls_vol=1.0)
                               for j, t in enumerate(range(t0, t0 + n * MS_MINUTE, 300_000))])
    cfg = ReplayConfig(source_db=str(tmp_path / "src.db"), start_ms=start, end_ms=start + hours * MS_HOUR - 1,
                       work_dir=str(tmp_path / "bot"), symbols=TRADED, seed_days=seed_days, seed_metrics=True,
                       settings={"RESEARCH_INTERVAL_MINUTES": 10**9, "DISCOVERY_INTERVAL_HOURS": 10**8,
                                 "HISTORY_DAYS": 12}, commands=[])
    s = run_replay(cfg)
    assert s["passed"], s["report"]
    cx = s["context"]
    assert cx["served_extra"] == CONTEXT_ONLY and cx["context_only"] == CONTEXT_ONLY
    assert sorted(s["universe_final"]) == TRADED and s["symbols"] == TRADED  # never traded, never a universe candidate
    assert cx["context_bars"] == hours * 60 * len(CONTEXT_ONLY) and cx["context_gaps"] == 0
    assert s["market"]["bars"] == hours * 60 * len(TRADED)  # the traded bar count is unchanged
    assert cx["snapshots"] == cx["hours_expected"] == hours + 1 and cx["missing"] == 0  # + the hour it started in
    assert cx["late"] == 0 and cx["errors"] == 0 and cx["symbols_present"] == {"min": 4, "max": 4, "mean": 4.0}
    assert "market context: 7 snapshots for 7 hour closes" in s["report"]
    # bit-identical to the offline build over the source store
    universe = Settings(_env_file=None).context_universe_list
    dst = Store(tmp_path / "ctx.db")
    C.build(src, dst, universe=universe)
    table = C.ContextTable.load(dst, universe, start - MS_HOUR, start + hours * MS_HOUR)
    for ts, digest in cx["digests"].items():
        T = int(ts)
        rows = [(C.MARKET_KEY, T, table.market[T].tobytes())]
        rows += [(sym, T, table.symbols[sym][T].tobytes()) for sym in universe if T in table.symbols.get(sym, {})]
        assert context_digest(rows) == digest, C._iso(T)


def test_the_market_stream_subscribes_klines_only_for_the_context_only_symbols():
    ms = MarketStream("wss://x", None, None, None)
    ms.set_symbols(["ethusdt", "BTCUSDT"], kline_only=["SOLUSDT", "BTCUSDT"])
    assert ms.symbols == ["BTCUSDT", "ETHUSDT"] and ms.kline_only == ["SOLUSDT"] and ms._restart.is_set()
    url = ms._url()
    assert "solusdt@kline_1m" in url and "solusdt@bookTicker" not in url and "btcusdt@bookTicker" in url
    ms._restart.clear()
    ms.set_symbols(["BTCUSDT", "ETHUSDT"])  # no kline_only given: kept, and nothing changed
    assert ms.kline_only == ["SOLUSDT"] and not ms._restart.is_set()
    ms.set_symbols(["BTCUSDT", "ETHUSDT", "SOLUSDT"])  # a context-only symbol that joins the universe is traded
    assert ms.kline_only == [] and ms._restart.is_set() and ms._url().count("solusdt@kline_1m") == 1
