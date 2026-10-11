"""C2-B: the running bot persists its market context. Every hourly snapshot it publishes is appended to the live context
store (CONTEXT_DB, else <data_dir>/context.db), the hours of the research window that store lacks are built from the
stored candles with the offline builder in the background at start-up, the store is pruned with the data retention,
and the research worker's walk-forward backtests (load_backtester(market="auto")) read it: a research backtest sees
exactly the ctx.market the live engines decided on."""
from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from heartless.app import Heartless, live_context_path
from heartless.config import Settings
from heartless.core.store import Store
from heartless.data import context as C
from heartless.data import features
from heartless.execution.engine import TradingEngine
from heartless.learning.backtester import load_backtester
from heartless.replay.clock import run_virtual
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR
from tests.test_p2a_app import (CTX_UNIVERSE, LIVE_HOURS, SMALL_FRAMES, START, T0, T_END, TRADED, _bits, _history, _Rest,
                                _Stream)

LOOKBACK_DAYS = 3  # the research window the start-up fill covers


@pytest.fixture(scope="module")
def run():
    """The whole orchestrator over LIVE_HOURS of streamed bars (as tests/test_p2a_app.py), every engine decision's
    ctx.market recorded, then the background context fill awaited; plus the offline build of the same data."""
    hist = _history()
    tmp = Path(tempfile.mkdtemp())
    st = Store(tmp / "heartless.db")
    for sym, d in hist.items():
        st.save_metrics(sym, [SimpleNamespace(**m) for m in d["metrics"] if m["ts"] <= T0 - 300_000])
    st.close()
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), HISTORY_DAYS=2, UNIVERSE_SIZE=2,
                        ALWAYS_INCLUDE=",".join(TRADED), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="", ARCHIVE_BACKFILL=False,
                        TIMEZONE="UTC", RESEARCH_INTERVAL_MINUTES=10**9, DISCOVERY_INTERVAL_HOURS=10**8,
                        CONTEXT_UNIVERSE=",".join(CTX_UNIVERSE), RESEARCH_LOOKBACK_DAYS=LOOKBACK_DAYS)
    seen: dict[tuple[str, int], Mapping] = {}
    orig = TradingEngine.on_bar

    async def on_bar(self, view, ctx):
        if self.name == "paper":
            seen[(view.symbol, ctx.now)] = ctx.market
        return await orig(self, view, ctx)

    out: dict = {}

    async def main(loop):
        await loop.sleep_until_ms(T0 + 2_500)
        app = Heartless(settings)
        app.rest = _Rest(hist)
        app.market = stream = _Stream(hist, app)
        task = asyncio.create_task(app.run())
        await stream.finished.wait()
        while app._ctx_fill_task is None or not app._ctx_fill_task.done():
            await asyncio.sleep(1)
        out.update(app=app, worker=app.research.worker_settings(), stats=dict(app.context_stats),
                   fill=app._ctx_fill_task.result(), symbols=dict(app.symbols), health=dict(app.health))
        await app.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    with pytest.MonkeyPatch.context() as mp:
        for tf, n in SMALL_FRAMES.items():
            mp.setitem(features.LIVE_WINDOW, tf, n)
        mp.setattr(TradingEngine, "on_bar", on_bar)
        run_virtual(main, T0)
    src = Store(tmp / "offline_src.db")
    for sym, d in hist.items():
        src.save_candles(sym, d["bars"])
        src.save_funding(sym, [(t, r, 0.0) for t, r in d["funding"]])
        src.save_metrics(sym, [SimpleNamespace(**m) for m in d["metrics"]])
    dst = Store(tmp / "offline_ctx.db")
    C.build(src, dst, universe=CTX_UNIVERSE)
    return out, seen, tmp, dst


def test_every_published_snapshot_is_appended_to_the_live_context_store(run):
    out, seen, tmp, _ = run
    path = tmp / "context.db"
    assert live_context_path(out["app"].s) == path and path.exists()
    st = Store(path)
    table = C.ContextTable.load(st, CTX_UNIVERSE + ["ETHUSDT"], T0 - MS_HOUR, T_END)
    assert table is not None
    # the warm-up hour (stored at start-up) and each live hour close
    assert sorted(table.market) == [T0 - 1 + k * MS_HOUR for k in range(LIVE_HOURS + 1)]
    for (sym, t), market in seen.items():
        assert market["ts"] == C.hour_close(C.hour_of(t))
        assert _bits(table.view(sym, t), market), (sym, C._iso(t))
    assert out["stats"]["stored"] == LIVE_HOURS + 1 and out["stats"]["errors"] == 0
    meta = st.get("context.meta")
    assert C.meta_ok(meta, CTX_UNIVERSE) and C.covered(meta, T0 - LOOKBACK_DAYS * MS_DAY, T_END)
    st.close()


def test_the_research_window_is_filled_from_stored_candles_at_start_up(run):
    out, _, tmp, offline = run
    st = Store(tmp / "context.db")
    lo = C.hour_close(C.hour_of(T0 - LOOKBACK_DAYS * MS_DAY))
    live = C.ContextTable.load(st, CTX_UNIVERSE, lo, T0 - 2)
    ref = C.ContextTable.load(offline, CTX_UNIVERSE, lo, T0 - 2)
    hours = sorted(live.market)
    assert hours[0] == lo and hours[-1] == T0 - MS_HOUR - 1 and len(hours) == (T0 - 1 - lo) // MS_HOUR
    assert out["fill"] == len(hours) and out["stats"]["filled"] == len(hours) and out["stats"]["fill_errors"] == 0
    for T in hours:  # bit-identical to the offline build of the same data
        assert live.market[T].tobytes() == ref.market[T].tobytes(), C._iso(T)
        for sym in CTX_UNIVERSE:
            a, b = live.symbols[sym].get(T), ref.symbols[sym].get(T)
            assert (a is None) == (b is None) and (a is None or a.tobytes() == b.tobytes()), (sym, C._iso(T))
    assert "built 72 hours in context.db" in out["health"]["market_context"]
    st.close()


def test_a_research_backtest_sees_the_live_snapshots(run):
    out, seen, tmp, _ = run
    worker = out["worker"]
    assert worker["CONTEXT_DB"] == str(tmp / "context.db") and "BINANCE_API_KEY" not in worker
    settings = Settings(_env_file=None, **worker)  # as optimizer.run_research_cycle builds it in the worker
    store = Store(tmp / "heartless.db")
    bt = load_backtester(settings, store, out["symbols"], T0 - 2 * MS_DAY, until=T_END, symbol_list=TRADED,
                         market="auto")
    assert bt.market is not None and len(bt.market) >= LIVE_HOURS + 1
    got: dict[tuple[str, int], Mapping] = {}
    orig = TradingEngine.on_bar

    async def on_bar(self, view, ctx):
        got[(view.symbol, ctx.now)] = ctx.market
        return await orig(self, view, ctx)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TradingEngine, "on_bar", on_bar)
        bt.run(StrategyParams.default(), T0, T_END)
    common = [k for k in seen if k in got]
    assert len(common) >= len(TRADED) * (LIVE_HOURS * 60 - 2)
    for k in common:
        assert got[k] and got[k]["ts"] == seen[k]["ts"] and "breadth_ema50" in got[k], k
        assert _bits(got[k], seen[k]), (k[0], C._iso(k[1]))
    store.close()


def test_the_live_context_store_is_pruned_with_the_retention(tmp_path):
    st = Store(tmp_path / "ctx.db")
    meta = C.meta_for(CTX_UNIVERSE)
    hours = [START + k * MS_HOUR - 1 for k in range(1, 49)]
    for T in hours:
        st.add_market_context([(C.MARKET_KEY, T, b"\0" * 8)], T, T, meta)
    assert st.get("context.meta")["built"] == [[hours[0], hours[-1]]]
    cut = START + 24 * MS_HOUR + 30 * 60_000  # mid-hour
    assert st.prune_market_context(cut) == 24
    assert st.market_context_stats()["first"] == START + 25 * MS_HOUR - 1
    assert st.get("context.meta")["built"] == [[START + 25 * MS_HOUR - 1, hours[-1]]]
    st.close()


def test_the_prune_reaches_the_live_context_store(tmp_path):
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 DATA_RETENTION_DAYS=10, HISTORY_DAYS=2)
    app = Heartless(s)
    now = START + 40 * MS_DAY
    st = app._ctx_db()
    meta = app._ctx_meta(st)
    for d in range(40):
        T = START + d * MS_DAY + MS_HOUR - 1
        st.add_market_context([(C.MARKET_KEY, T, b"\0" * 8)], T, T, meta)
    app._prune(now)
    assert st.market_context_stats()["hours"] == 15  # retention 10 days + 5 days of margin, as every other series
    asyncio.run(app.stop())
