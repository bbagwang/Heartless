"""E1 funding schedule: the backtester charges funding at each symbol's STORED settlement times (8 h for most symbols,
4 / 2 / 1 h while Binance settles one more often, e.g. SOLUSDT in November 2022) instead of on a fixed 8-hour grid, and
the engines' minutes-to-funding / settlement interval follow the same schedule; fetch / archive derive the interval per
symbol and funding_fade measures "since the last settlement" in the symbol's own interval. For 8-hourly settlements on
the UTC grid everything is exactly what it was."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest
from synth import synth_candles, synth_symbols

from heartless.config import Settings
from heartless.core.models import Regime, Ticker
from heartless.core.store import Store
from heartless.data.archive import MAX_FUNDING_INTERVAL_MS, settlement_interval
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.learning.backtester import FUNDING_INTERVAL, Backtester, funding_schedule
from heartless.strategy.alphas.funding_fade import FundingFade
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_HOUR, MS_MINUTE

START = 1_700_006_400_000  # synth_candles default start: 00:00 UTC, on the 8-hour grid
H = MS_HOUR


def _times(*hours: float) -> np.ndarray:
    return np.array([START + int(h * H) for h in hours], dtype=np.int64)


# --- the schedule ---------------------------------------------------------------------------------------------------

def test_schedule_on_the_8h_grid_is_the_grid():
    t = _times(0, 8, 16, 24)
    for ts in (START + 1, START + 5 * H, START + 8 * H - 1):
        assert funding_schedule(t, ts) == (START + 8 * H, 8 * H)
    assert funding_schedule(t, START + 24 * H + 1) == (START + 32 * H, 8 * H)  # past the stored span: the last interval
    assert funding_schedule(t[:0], START + 3 * H) == (START + 8 * H, FUNDING_INTERVAL)  # nothing stored: UTC grid


def test_schedule_follows_a_shorter_interval_and_crosses_holes_on_the_last_one():
    t = _times(0, 8, 10, 12, 16, 32)  # 8 h, then 2 h, 2 h, 4 h, then a hole (16 h)
    assert funding_schedule(t, START + 9 * H) == (START + 10 * H, 2 * H)
    assert funding_schedule(t, START + 13 * H) == (START + 16 * H, 4 * H)
    # across the hole the settlements continue on the last interval (4 h) from the last stored one
    assert funding_schedule(t, START + 17 * H) == (START + 20 * H, 4 * H)
    assert funding_schedule(t, START + 21 * H) == (START + 24 * H, 4 * H)


def test_settlement_interval_from_stored_times():
    assert settlement_interval([START, START + 4 * H]) == 4 * H
    assert settlement_interval([START, START + 8 * H, START + 9 * H]) == H
    assert settlement_interval([START]) == MAX_FUNDING_INTERVAL_MS
    assert settlement_interval([START, START + 16 * H]) == MAX_FUNDING_INTERVAL_MS  # a hole is not an interval


# --- the backtester -------------------------------------------------------------------------------------------------

def _record(monkeypatch, rows: list[tuple[int, float]], bars: int = 1500):
    s = Settings(_env_file=None)
    bt = Backtester(s, synth_symbols(["BTCUSDT"]), {"BTCUSDT": synth_candles(bars, seed=1)}, {"BTCUSDT": rows})
    seen: dict[int, tuple[float, float, float]] = {}
    charged: list[tuple[float, int]] = []
    orig_bar, orig_apply = TradingEngine.on_bar, PaperAccount.apply_funding

    async def rec_bar(self, view, ctx):
        seen[ctx.now] = (ctx.funding_rate, ctx.minutes_to_funding, ctx.funding_interval_min)
        return await orig_bar(self, view, ctx)

    async def rec_apply(self, symbol, rate, mark, ts):
        charged.append((rate, ts))
        return await orig_apply(self, symbol, rate, mark, ts)

    monkeypatch.setattr(TradingEngine, "on_bar", rec_bar)
    monkeypatch.setattr(PaperAccount, "apply_funding", rec_apply)
    bt.run(StrategyParams.default())
    return bt, seen, charged


def test_backtester_charges_every_stored_settlement_of_a_two_hourly_period(monkeypatch):
    # 8-hourly until 08:00, then every 2 hours (as SOLUSDT in November 2022), 1500 bars = 25 h
    times = [0, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    rows = [(START + h * H, 0.0001 * (i + 1)) for i, h in enumerate(times)]
    _, seen, charged = _record(monkeypatch, rows)
    # each settlement after the first bar is charged once, on the first bar closing after it, at its own rate
    assert charged == [(r, t + 59_999) for t, r in rows[1:]]
    # the engines see the 2-hour schedule: minutes to the next settlement and the interval itself
    fr, mtf, iv = seen[START + 11 * H - 1]  # 10:59:59.999, one hour before the 12:00 settlement
    assert mtf == pytest.approx(60.0, abs=1e-3) and iv == 120.0
    assert fr == pytest.approx(0.5 * 0.0003 + 0.5 * 0.0004)  # halfway from the 10:00 rate to the 12:00 one
    _, mtf, iv = seen[START + 5 * H - 1]  # inside the 8-hour period
    assert mtf == pytest.approx(180.0, abs=1e-3) and iv == 480.0


def test_backtester_on_the_8h_grid_matches_the_previous_bucket_rule(monkeypatch):
    rows = [(START + h * H, 0.0001 * (1 + h / 8)) for h in (0, 8, 16, 24)]
    _, seen, charged = _record(monkeypatch, rows)
    # previous rule: charge at the first bar of each new 8-hour bucket the last settled rate; minutes to the next grid
    # point; the blend weight (t % 8h) / 8h between the settled and the next charged rate
    assert charged == [(r, t + 59_999) for t, r in rows[1:]]
    assert [t for _, t in charged] == [START + 8 * H + 59_999, START + 16 * H + 59_999, START + 24 * H + 59_999]
    for t, (fr, mtf, iv) in seen.items():
        bucket = t // FUNDING_INTERVAL
        assert mtf == ((bucket + 1) * FUNDING_INTERVAL - t) / MS_MINUTE and iv == 480.0
        prev = [r for ts, r in rows if ts <= t][-1]
        nxt = [r for ts, r in rows if ts > t]
        if nxt:
            w = (t % FUNDING_INTERVAL) / FUNDING_INTERVAL
            assert fr == (1 - w) * prev + w * nxt[0]  # bit for bit


def test_a_missing_settlement_is_not_charged_at_the_previous_rate(monkeypatch):
    rows = [(START, 0.0001), (START + 8 * H, 0.0002), (START + 24 * H, 0.0009)]  # 16:00 is missing
    _, _, charged = _record(monkeypatch, rows)
    assert charged == [(0.0002, START + 8 * H + 59_999), (0.0009, START + 24 * H + 59_999)]


# --- funding_fade: minutes since the last settlement in the symbol's own interval ----------------------------------

def test_funding_fade_settlement_window_uses_the_symbol_interval(monkeypatch):
    seen: list[float] = []
    a = FundingFade()

    class H_:  # a ready history: the test only looks at the window gate
        def ready(self):
            return True

        def z_tlp(self):
            seen.append(1.0)
            return 0.0

        def z_fr(self):
            return 0.0

    monkeypatch.setattr(a, "_observe", lambda ctx: H_())

    class V:
        def closed(self, tf):
            return True

    p = dict(StrategyParams.default().alphas["funding_fade"], settle_win=120)

    def asks(mtf: float, interval: float) -> bool:
        seen.clear()
        ctx = Context(symbol="X", info=None, ticker=Ticker("X"), regime=Regime.RANGE, minutes_to_funding=mtf,
                      funding_interval_min=interval, extras={"oi_chg_24h": 1.0})
        a.evaluate(V(), ctx, p)
        return bool(seen)  # reached the crowding check, i.e. inside the settlement window

    assert not asks(200.0, 480.0)  # 8-hourly: settled 280 min ago, next in 200 -> outside +-120
    assert asks(200.0, 240.0)  # 4-hourly: settled 40 min ago -> inside
    assert asks(100.0, 480.0) and asks(400.0, 480.0) and not asks(300.0, 480.0)


# --- fetch / archive / app ------------------------------------------------------------------------------------------

class _Rest:
    def __init__(self):
        self.calls: list[tuple] = []

    async def klines_range(self, sym, start, end, tf):
        return []

    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        self.calls.append((symbol, start, end))
        return []


def test_fetch_asks_for_funding_one_symbol_interval_after_the_last_settlement(monkeypatch):
    from heartless import cli

    now = START + 100 * H
    monkeypatch.setattr(cli, "now_ms", lambda: now)
    st = Store(Path(tempfile.mkdtemp()) / "f.db")
    # 4-hourly symbol, last settlement 5 h ago (the next one was due an hour ago); 8-hourly symbol, also 5 h ago (not
    # due for 3 more hours). The former 8-hour rule asked for neither.
    st.save_funding("SOLUSDT", [(now - 9 * H, 0.0001, 0.0), (now - 5 * H, 0.0001, 0.0)])
    st.save_funding("BTCUSDT", [(now - 13 * H, 0.0001, 0.0), (now - 5 * H, 0.0001, 0.0)])
    rest = _Rest()
    asyncio.run(cli._ensure_candles(rest, st, ["SOLUSDT", "BTCUSDT"], days=1))
    assert [c[0] for c in rest.calls] == ["SOLUSDT"]
    st.close()


def test_archive_sync_refetches_funding_with_a_hole(tmp_path):
    from heartless.data.archive import sync_symbol
    from tests.test_p0_fetch import NOW, FakeArchive, _ms

    a, b = _ms(2024, 3, 1), _ms(2024, 4, 1) - 1
    st = Store(tmp_path / "a.db")
    r1 = asyncio.run(sync_symbol(st, FakeArchive(), "BTCUSDT", a, b, NOW, metrics=False))
    assert r1["funding"] == 93
    missing = _ms(2024, 3, 15, 8)
    st.execute("DELETE FROM funding WHERE symbol=? AND funding_time=?", ("BTCUSDT", missing))
    # 92 of 93 rows passes the count check (>= 95% of the 8-hour minimum), but the 16-hour spacing is a missing
    # settlement whatever the symbol's interval: refetched
    r2 = asyncio.run(sync_symbol(st, FakeArchive(), "BTCUSDT", a, b, NOW, metrics=False))
    assert r2["funding"] == 93 and [t for t, _ in st.load_funding("BTCUSDT", missing, missing)] == [missing]
    r3 = asyncio.run(sync_symbol(st, FakeArchive(), "BTCUSDT", a, b, NOW, metrics=False))
    assert r3["funding"] == 0  # complete: nothing asked
    st.close()


def test_app_context_carries_the_symbol_settlement_interval():
    from heartless.app import Heartless
    from heartless.core.models import SymbolInfo

    tmp = Path(tempfile.mkdtemp())
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 ARCHIVE_BACKFILL=False)
    app = Heartless(s)
    app.symbols = {"SOLUSDT": SymbolInfo("SOLUSDT", "SOL", "USDT", 0.01, 0.001, 0.001, 5.0, 2, 3)}
    app.tickers["SOLUSDT"] = Ticker("SOLUSDT", mark=100.0)
    assert app._context("SOLUSDT", START + H).funding_interval_min == 480.0
    app.funding_interval["SOLUSDT"] = 4 * H
    assert app._context("SOLUSDT", START + H).funding_interval_min == 240.0
    app.store.close()
