"""Wave D integration of C2-A (research infrastructure) and C2-B (live context store, challenger slots).

* ONE context-store path resolution (Settings.context_db_path) serves the bot's live append, its research worker and
  the lab / backtester.
* Because one path can name the other side's file (CONTEXT_DB exported for lab commands, a lab run on the bot's data
  dir), the live and the offline stores refuse each other instead of deleting each other's rows: the bot never appends
  to or prunes a store built from another candle database, and a research build never rebuilds a live store.
* The start-up fill of the live store writes under the store's build lock (C2-A), in a worker thread.
* The live 1m history follows the longest lookback of the enabled alphas of the shipped champion and of a C2-B
  challenger slot, and a deeper need is handled at the next scheduler tick.
Synthetic data only."""
from __future__ import annotations

import asyncio
import math
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.app import Heartless, live_context_path
from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import context as C
from heartless.learning.backtester import stored_context
from heartless.learning.research import ChallengerSlot
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

T = 1_780_000_000_000 // MS_HOUR * MS_HOUR + MS_HOUR - 1  # an hour close (2026-05)
ROW = np.zeros(len(C.MARKET_FEATURES)).tobytes()


def _app(tmp_path, **kw) -> Heartless:
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path / "live"), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 ARCHIVE_BACKFILL=False, **kw)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return Heartless(s)


def _snap(ts: int):
    return SimpleNamespace(ts=ts, rows=lambda: [(C.MARKET_KEY, ts, ROW)])


# --- one path resolution ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ctx", ["", "elsewhere/ctx.db"])
def test_live_append_research_worker_and_backtester_resolve_one_context_store(tmp_path, ctx):
    app = _app(tmp_path, CONTEXT_DB=str(tmp_path / ctx) if ctx else "")
    s = app.s
    assert app.live_context_path == live_context_path(s) == s.context_db_path
    assert s.context_db_path == (tmp_path / ctx if ctx else s.data_dir / "context.db")
    worker = Settings(_env_file=None, **app.research.worker_settings())
    assert worker.context_db_path == s.context_db_path
    app._persist_context(_snap(T))
    assert app.context_stats["stored"] == 1 and not app._ctx_off
    # the research worker's backtester (load_backtester(market="auto") -> stored_context) reads that very store
    src = Store(s.db_path)
    table = stored_context(worker, src, [], T - MS_HOUR, T + MS_HOUR)
    src.close()
    assert table is not None and list(table.market) == [T]
    asyncio.run(app.stop())


# --- the bot never touches an offline store ---------------------------------------------------------------------------
@pytest.mark.parametrize("universe", ["same", "other"])
def test_the_bot_never_appends_to_or_prunes_an_offline_store(tmp_path, universe):
    offline = tmp_path / "longhist" / "context.db"
    offline.parent.mkdir(parents=True)
    probe = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path / "x"))
    uni = probe.context_universe_list if universe == "same" else ["BTCUSDT", "ETHUSDT"]
    source = {"path": os.path.realpath(offline.parent / "heartless.db"), "ranges": {}, "fingerprint": "f"}
    old = [T - 4 * 365 * MS_DAY + k * MS_HOUR for k in range(3)]  # years behind any retention cutoff
    st = Store(offline)
    st.save_market_context([(C.MARKET_KEY, t, ROW) for t in old])
    meta = {**C.meta_for(uni, source), "built": [[old[0], old[-1]]]}
    st.set("context.meta", meta)
    st.close()

    app = _app(tmp_path, CONTEXT_DB=str(offline))
    app._persist_context(_snap(T))
    app._prune(T)
    assert app._ctx_off and "not from this bot's candle database" in app._ctx_off
    assert asyncio.run(app._fill_context_gaps()) == 0
    with pytest.raises(C.ContextMismatch):
        app._ctx_db()
    # the research worker gets no context from it either (built from another candle database)
    worker = Settings(_env_file=None, **app.research.worker_settings())
    src = Store(app.s.db_path)
    assert stored_context(worker, src, [], old[0], T) is None
    src.close()
    asyncio.run(app.stop())
    st = Store(offline)
    assert st.market_context_stats()["hours"] == 3 and st.get("context.meta") == meta
    st.close()


def test_an_older_live_store_is_adopted_with_its_rows(tmp_path):
    app = _app(tmp_path)
    path = app.live_context_path
    st = Store(path)
    st.add_market_context([(C.MARKET_KEY, T, ROW)], T, T, C.meta_for(app.context.symbols))  # C2-B: no source
    st.close()
    st = app._ctx_db()
    assert st.get("context.meta")["source"] == C.live_source(app.s.db_path) == {
        "path": os.path.realpath(app.s.db_path), "live": True}
    assert st.market_context_stats()["hours"] == 1
    asyncio.run(app.stop())


# --- research builds never rebuild a live store -----------------------------------------------------------------------
def test_a_research_build_refuses_the_live_store(tmp_path):
    app = _app(tmp_path)
    s = app.s
    t0 = T - 3 * MS_DAY + 1
    app.store.save_candles("BTCUSDT", [Candle(t0 + i * MS_MINUTE, 100.0, 100.1, 99.9, 100.0, 1.0, 100.0, 1, 0.5,
                                              t0 + i * MS_MINUTE + 59_999) for i in range(3 * 1440)])
    app._persist_context(_snap(T))
    st = app._ctx_db()
    cfg = {"db": str(s.context_db_path), "universe": s.context_universe_list}
    with pytest.raises(C.ContextMismatch, match="live context store"):
        C.prepare(str(s.db_path), cfg, T - 2 * MS_DAY, T)
    src = Store(s.db_path)
    assert C.usable(src, st, s.context_universe_list)[:2] == (False, "mismatch")
    assert C.prepare(str(s.db_path), {**cfg, "build": False}, T - 2 * MS_DAY, T) is None
    # a live store of an older definition is refused as well, before anything is deleted
    meta = st.get("context.meta")
    st.set("context.meta", {**meta, "version": "outdated"})
    with pytest.raises(C.ContextMismatch):
        C.build(src, st, T - 2 * MS_DAY, T, s.context_universe_list)
    src.close()
    assert st.market_context_stats()["hours"] == 1 and st.get("context.meta")["source"]["live"] is True
    asyncio.run(app.stop())


# --- the start-up fill takes the build lock ---------------------------------------------------------------------------
def test_the_start_up_fill_writes_under_the_build_lock(tmp_path):
    app = _app(tmp_path)
    st = app._ctx_db()
    a, b = T - 3 * MS_HOUR, T
    done = threading.Event()
    th = threading.Thread(target=lambda: (app._build_context_gap(a, b), done.set()))
    with C.build_lock(st.path, shared=True):  # a reader (the research worker's ContextTable.load)
        th.start()
        assert not done.wait(0.4)  # the fill waits for it
    th.join(10)
    assert done.is_set() and [a, b] in st.get("context.meta")["built"]
    asyncio.run(app.stop())


# --- live 1m history depth --------------------------------------------------------------------------------------------
def test_live_history_follows_the_longest_lookback_of_champion_and_a_challenger_slot(tmp_path, monkeypatch):
    app = _app(tmp_path)
    shipped = [ALPHA_BY_NAME[a] for a, on in StrategyParams.default().enabled.items() if on and a in ALPHA_BY_NAME]
    longest = max([float(getattr(a, "lookback_days", 0) or 0) for a in shipped] or [0.0])
    assert app._alpha_history_days() == (math.ceil(longest + 2) if longest else 0)  # shipped defaults
    ch = StrategyParams.default()
    ch.enabled["vol_breakout_dbox"] = ch.enabled["ts_trend_tsmom"] = True
    now = T
    app.challengers.append(ChallengerSlot(name="challenger-1", params=ch, started=now, source="research"))
    assert app._alpha_history_days() == 92  # vol_breakout_dbox: 90 days + the lab's 2-day margin
    calls = []

    async def ensure():
        calls.append(app._alpha_history_days())
        app._depth_need = app._alpha_history_days()

    monkeypatch.setattr(app, "_ensure_history_depth", ensure)
    app._last_depth_check, app._depth_need = now, 0  # the hourly check just ran under the shipped champion

    async def ticks():
        app._history_depth_tick(now + MS_MINUTE)  # deeper need: at once, not an hour later
        await asyncio.sleep(0)
        app._history_depth_tick(now + 2 * MS_MINUTE)  # handled: back to hourly
        await asyncio.sleep(0)

    asyncio.run(ticks())
    assert calls == [92]
    asyncio.run(app.stop())
