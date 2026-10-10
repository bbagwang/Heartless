"""P4 housekeeping found by the P5 soak: slim EventBus history, research/discovery workers terminated on shutdown, every
time series pruned with the candle retention, MetricsSeries.merge keeping fields a newer partial row lacks, the seal
error wording, the metrics archive backfill, and the research-window defaults."""
import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.bus import HISTORY_TEXT_CHARS, EventBus
from heartless.core.store import Store
from heartless.data.extras import MetricsSeries
from heartless.learning import research as R
from heartless.learning import seal
from heartless.util.timeutil import MS_DAY, now_ms


# --- EventBus history -----------------------------------------------------------------------------------------------
def test_bus_history_keeps_slim_entries_and_handlers_get_the_full_payload():
    bus = EventBus()
    got = []
    bus.subscribe("research_done", got.append)
    big = {"alphas": {f"a{i}": list(range(1000)) for i in range(50)}}
    payload = {"message": "x" * 5000, "seconds": 12.5, "notify": True, "result": big, "position": object()}
    asyncio.run(bus.publish("research_done", payload))
    assert got[0] is payload and got[0]["result"] is big  # subscribers are unaffected
    ts, topic, entry = bus.history[-1]
    assert topic == "research_done" and entry["seconds"] == 12.5 and entry["notify"] is True
    assert len(entry["message"]) == HISTORY_TEXT_CHARS + 1 and sorted(entry["_dropped"]) == ["position", "result"]
    assert "result" not in entry
    for i in range(2500):
        asyncio.run(bus.publish("tick", {"i": i}))
    assert len(bus.history) <= 2000


# --- research / discovery worker shutdown -------------------------------------------------------------------------
class _App:
    def __init__(self, tmp):
        self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp))
        self.store = Store(Path(tmp) / "h.db")


def test_shutdown_terminates_a_busy_worker_process(tmp_path):
    rm = R.ResearchManager(_App(tmp_path))
    rm.pool = R._new_pool()
    fut = rm.pool.submit(time.sleep, 120)  # a research cycle that would run for minutes
    deadline = time.time() + 60
    while not rm.pool._processes and time.time() < deadline:
        time.sleep(0.05)
    procs = list(rm.pool._processes.values())
    while not all(p.is_alive() for p in procs) and time.time() < deadline:
        time.sleep(0.05)
    assert procs and all(p.is_alive() for p in procs)
    t0 = time.time()
    rm.shutdown()
    assert rm.pool is None and rm.stopped is True
    assert not any(p.is_alive() for p in procs), "worker left running after shutdown"
    assert time.time() - t0 < 10
    with pytest.raises(Exception):
        fut.result(timeout=10)  # the cycle's future fails (BrokenProcessPool / cancelled); the cycle ends quietly


def test_worker_exits_when_its_parent_disappears(monkeypatch):
    exits = []
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R.os, "getppid", lambda: 1)  # re-parented: the bot is gone
    monkeypatch.setattr(R.os, "_exit", lambda code: (exits.append(code), (_ for _ in ()).throw(SystemExit)))
    with pytest.raises(SystemExit):
        R._watch_parent(os.getpid())
    assert exits == [0]


def test_a_cycle_whose_worker_was_terminated_ends_quietly(tmp_path, monkeypatch):
    app = _App(tmp_path)
    app.events = []
    app.params = None
    app.symbols, app.universe = {}, []

    async def emit(kind, payload):
        app.events.append(kind)

    app.emit = emit
    app.params = SimpleNamespace(to_dict=lambda: {})
    rm = R.ResearchManager(app)

    async def run():
        loop = asyncio.get_running_loop()

        def broken(executor, fn, *a):
            rm.stopped = True  # shutdown() ran while the worker was busy
            fut = loop.create_future()
            fut.set_exception(RuntimeError("A process in the process pool was terminated abruptly"))
            return fut

        monkeypatch.setattr(loop, "run_in_executor", broken)
        rm.pool = SimpleNamespace(shutdown=lambda **k: None)
        assert await rm.cycle(force=True) is None
        assert rm.running is False and rm.research_failures == 0 and "error" not in app.events

    asyncio.run(run())


# --- retention -------------------------------------------------------------------------------------------------------
async def test_every_time_series_is_pruned_with_the_candle_retention(tmp_path):
    from heartless.app import Heartless
    from heartless.core.models import Candle

    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 DATA_RETENTION_DAYS=30, HISTORY_DAYS=5)
    app = Heartless(s)
    now = now_ms()
    old, new = now - 40 * MS_DAY, now - 10 * MS_DAY
    st = app.store
    for t in (old, new):
        st.save_candles("BTCUSDT", [Candle(t, 1, 1, 1, 1, 1, 1, 1, 1, t + 59_999)])
        st.save_funding("BTCUSDT", [(t, 0.0001, 1.0)])
        st.save_metrics("BTCUSDT", [SimpleNamespace(ts=t, oi=1.0, oi_value=1.0, top_ls_accounts=1.0, top_ls_positions=1.0,
                                                    ls_accounts=1.0, taker_ls_vol=1.0)])
        st.save_equity("paper", t, 1.0, 1.0)
        st.save_fill_quality({"engine": "paper", "ts": t, "symbol": "BTCUSDT", "position_id": "p", "kind": "ENTRY",
                              "side": "BUY", "maker": False, "qty": 1.0, "price": 1.0, "ref": 1.0, "notional": 1.0,
                              "slip_bps": 0.0, "fee": 0.0, "model_fee": 0.0})
    app._prune(now)  # cutoff: max(5, 30) + 5 = 35 days back
    for table, col in (("candles", "open_time"), ("funding", "funding_time"), ("metrics", "ts"), ("equity", "ts"),
                       ("fill_quality", "ts")):
        assert [r[0] for r in st.query(f"SELECT {col} FROM {table} ORDER BY {col}")] == [new], table
    await app.stop()


def test_research_window_and_retention_defaults():
    s = Settings(_env_file=None)
    assert s.research_lookback_days == 120 and s.data_retention_days == 180 and s.promotion_min_days == 14
    assert s.data_retention_days >= s.research_lookback_days + 30  # room for warm-up and the backfill margin
    assert (s.kill_switch, s.kill_switch_k, s.kill_switch_h, s.kill_switch_window, s.kill_switch_window_r) == \
        (True, 0.25, 5.0, 30, -8.0)


# --- MetricsSeries.merge --------------------------------------------------------------------------------------------
def _row(ts, **v):
    base = {"ts": ts, "oi": None, "oi_value": None, "top_ls_accounts": None, "top_ls_positions": None,
            "ls_accounts": None, "taker_ls_vol": None}
    base.update(v)
    return base


def test_metrics_merge_keeps_fields_a_newer_partial_row_does_not_provide():
    old = MetricsSeries.from_rows([_row(0, oi=1.0, oi_value=10.0, top_ls_positions=1.1, ls_accounts=1.2, taker_ls_vol=0.9),
                                   _row(300_000, oi=2.0, oi_value=20.0, top_ls_positions=1.3, ls_accounts=1.4)])
    # the next poll: one series lags a bucket behind (no ratios for the newest bucket), another answered empty
    new = MetricsSeries.from_rows([_row(300_000, oi=2.5, taker_ls_vol=1.05), _row(600_000, oi=3.0, oi_value=30.0)])
    m = old.merge(new)
    assert m.ts.tolist() == [0, 300_000, 600_000]
    assert m.oi.tolist() == [1.0, 2.5, 3.0]  # newer values win where they exist
    assert m.oi_value.tolist() == [10.0, 20.0, 30.0]  # ... and older values survive where the newer row is empty
    assert m.top_ls_positions[1] == pytest.approx(1.3) and m.ls_accounts[1] == pytest.approx(1.4)
    assert m.taker_ls_vol[1] == pytest.approx(1.05) and np.isnan(m.top_ls_positions[2])
    assert old.merge(MetricsSeries.empty()) is old and MetricsSeries.empty().merge(new) is new
    # a steady one-bucket lag no longer leaves every bucket with a single series
    s = MetricsSeries.empty()
    for b in range(1, 6):
        s = s.merge(MetricsSeries.from_rows([_row(b * 300_000, oi=float(b)), _row((b - 1) * 300_000, ls_accounts=1.5)]))
    assert np.isfinite(s.oi[1:5]).all() and np.isfinite(s.ls_accounts[1:5]).all()  # buckets 1-4 got both series


# --- seal wording --------------------------------------------------------------------------------------------------
def test_the_seal_error_only_points_to_the_commands_that_can_unseal():
    a = seal.SEALED["2025"][0]
    with pytest.raises(seal.SealedError) as e:
        seal.check_window(a, a + MS_DAY)
    msg = str(e.value)
    assert "Only `heartless lab` and `heartless discover` can open it" in msg and "--unseal 2025" in msg
    assert "other commands have no unseal option" in msg


# --- metrics archive backfill -----------------------------------------------------------------------------------------
async def test_metrics_are_backfilled_from_the_archive_in_the_background(tmp_path):
    from heartless.app import Heartless

    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 DATA_RETENTION_DAYS=10, HISTORY_DAYS=2)
    app = Heartless(s)
    calls = []
    now = now_ms()

    class FakeArchive:
        async def metrics(self, symbol, start, end, now_ms_):
            calls.append((symbol, start, end))
            if symbol == "BADUSDT":
                raise RuntimeError("archive down")
            step = 300_000
            return [SimpleNamespace(ts=t, oi=1.0, oi_value=1.0, top_ls_accounts=1.0, top_ls_positions=1.0,
                                    ls_accounts=1.0, taker_ls_vol=1.0)
                    for t in range(start - start % step + step, min(end, now - MS_DAY), step)]

    app.archive = FakeArchive()
    # ETH already holds the recent polled rows: only the history before them is fetched
    app.store.save_metrics("ETHUSDT", [SimpleNamespace(ts=now - MS_DAY, oi=2.0, oi_value=2.0, top_ls_accounts=1.0,
                                                       top_ls_positions=1.0, ls_accounts=1.0, taker_ls_vol=1.0)])
    await app._backfill_metrics(["BTCUSDT", "BADUSDT", "ETHUSDT"])  # a failing symbol never raises
    lo, hi = app.store.metrics_range("BTCUSDT")
    assert lo <= now - 9 * MS_DAY and hi >= now - MS_DAY - 300_000
    eth = [c for c in calls if c[0] == "ETHUSDT"]
    assert len(eth) == 1 and eth[0][2] == now - MS_DAY - 300_000  # up to the oldest stored row
    assert app.store.metrics_range("BADUSDT") == (None, None)
    await app.stop()
