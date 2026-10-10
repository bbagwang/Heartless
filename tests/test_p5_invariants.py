"""P5 replay invariants: every check stays quiet on a consistent state and detects an injected violation.

The checker runs against a real TradingEngine + PaperAccount + Store (a position opened through the engine's own
entry path, so its stop and take-profit are real simulator algos); each test then corrupts one book."""
from __future__ import annotations

import asyncio
import logging
import socket
from types import SimpleNamespace

import httpx
import pytest
from synth import synth_symbols

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import (Decision, EntryStyle, PositionStatus, Regime, Side, Signal, Ticker, TradeRecord)
from heartless.core.store import Store
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.notify import formatter as F
from heartless.replay.invariants import InvariantChecker
from heartless.replay.netguard import NetworkBlocked, block_network
from heartless.replay.telegram import SentMessage
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util import timeutil
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

SYM = "BTCUSDT"
T0 = 1_772_323_200_000  # 2026-03-01 00:00 UTC


class Clock:
    def __init__(self, t: int = T0 + 10 * MS_HOUR):
        self.t = t

    def __call__(self) -> int:
        return self.t


class View:
    frames: dict = {}
    price = 100.0

    def ready(self) -> bool:
        return True

    def closed(self, tf: str) -> bool:
        return False


def book(price: float, ts: int) -> Ticker:
    return Ticker(SYM, bid=price * (1 - 1e-4), ask=price * (1 + 1e-4), mark=price, last=price, ts=ts)


def decision() -> Decision:
    sig = Signal(alpha="htf_trend", symbol=SYM, side=Side.LONG, confidence=0.7, reason="t", stop=98.0,
                 take_profit=104.0, tp1=102.0, entry_style=EntryStyle.MARKET, atr=2.0, timeframe="1h")
    return Decision(symbol=SYM, side=Side.LONG, score=0.7, confidence=0.7, alphas=["htf_trend"], primary=sig,
                    reason="t", regime=Regime.RANGE)


@pytest.fixture
def world(tmp_path):
    """(checker, app, engine, account, clock) with one OPEN long whose SL / TP1 / TP work on the account. The process
    clock follows the test clock (stored trades and the 'today' of reports use it)."""
    clock = Clock()
    prev = timeutil.set_clock(clock)
    store = Store(tmp_path / "bot.db")
    syms = synth_symbols([SYM])
    settings = Settings(_env_file=None, TIMEZONE="UTC")
    acc = PaperAccount("paper", 10_000.0, clock=clock)
    acc.set_symbols(syms)
    eng = TradingEngine("paper", acc, StrategyParams.default(), settings, syms, store=store, bus=EventBus(),
                        clock=clock, persist=True, notify=True)
    app = SimpleNamespace(engines={"paper": eng}, paper_accounts={"paper": acc}, symbols=syms, store=store, s=settings,
                          primary_engine=eng, candles={}, bus=eng.bus, metrics_series={}, telegram=None)
    asyncio.run(_open(eng, acc, clock))
    yield InvariantChecker(app), app, eng, acc, clock
    timeutil.set_clock(prev)
    store.close()


async def _open(eng, acc, clock) -> None:
    await eng.start()
    tk = book(100.0, clock())
    await acc.on_ticker(tk)
    await eng._open(decision(), View(), Context(symbol=SYM, info=eng.symbols[SYM], ticker=tk, regime=Regime.RANGE,
                                                now=clock()), await eng.equity())
    pos = eng.positions[SYM]
    assert pos.status is PositionStatus.OPEN and pos.sl_algo_id and pos.tp_algo_id


def checks(chk: InvariantChecker) -> dict[str, int]:
    asyncio.run(chk.check_minute())
    return {k: v for k, v in chk.counts.items() if v}


def test_a_consistent_book_raises_nothing(world):
    chk, app, eng, acc, clock = world
    assert checks(chk) == {}
    assert all(chk.runs[k] == 1 for k in ("positions", "stops", "orphans", "ledger"))
    # a partial take-profit, a stop move to breakeven and a full exit keep all books aligned
    clock.t += MS_MINUTE
    asyncio.run(acc.on_ticker(book(102.5, clock())))  # TP1 fills, stop to breakeven
    assert eng.positions[SYM].tp1_done and checks(chk) == {}
    clock.t += MS_MINUTE
    asyncio.run(acc.on_ticker(book(104.5, clock())))  # runner TP fills
    assert SYM not in eng.positions and checks(chk) == {}
    chk.check_trades(final=True)
    assert chk.total_violations == 0


def test_positions_detects_an_account_that_drifted_from_the_engine(world):
    chk, app, eng, acc, clock = world
    acc.positions[SYM].qty *= 2
    assert checks(chk).get("positions") == 1 and "qty: engine" in chk.samples[-1].detail
    acc.positions[SYM].qty = 0.0
    checks(chk)
    assert chk.counts["positions"] == 2 and "account is flat" in chk.samples[-1].detail


def test_stops_detects_a_position_whose_stop_is_gone_or_elsewhere(world):
    chk, app, eng, acc, clock = world
    pos = eng.positions[SYM]
    asyncio.run(acc.cancel_algo(SYM, pos.sl_algo_id))  # the exchange dropped the stop
    assert checks(chk).get("stops") == 1 and "no working stop" in chk.samples[-1].detail
    asyncio.run(eng._ensure_stop(_clear(pos)))  # re-armed: quiet again
    checks(chk)
    assert chk.counts["stops"] == 1
    pos.stop = 97.0  # the engine believes in a stop the account does not have
    checks(chk)
    assert chk.counts["stops"] == 2 and "engine stop 97.0" in chk.samples[-1].detail


def _clear(pos):
    pos.sl_algo_id = ""
    return pos


def test_orphans_detects_stray_algos_and_orders(world):
    chk, app, eng, acc, clock = world
    asyncio.run(acc.place_stop(SYM, "SELL", 90.0, close_position=True))  # a second, unowned stop
    found = checks(chk)
    assert found.get("orphans") == 2  # belongs to nothing + two stops on one symbol
    chk2 = InvariantChecker(app)
    asyncio.run(acc.cancel_all(SYM))
    asyncio.run(eng._place_brackets(eng.positions[SYM]))
    asyncio.run(acc.limit_order("ETHUSDT", "BUY", 0.01, 90.0, client_id="HLEPAPEstray"))
    assert checks(chk2).get("orphans") == 1 and "without a pending/open position" in chk2.samples[-1].detail


def test_ledger_detects_a_corrupted_balance_and_a_corrupted_engine_book(world):
    chk, app, eng, acc, clock = world
    assert checks(chk) == {}  # baseline taken
    acc.wallet += 5.0
    assert checks(chk).get("ledger") == 2  # own counters and engine books both disagree with the wallet
    acc.wallet -= 5.0
    eng.positions[SYM].fees += 1.0  # the engine booked a fee the account never charged
    checks(chk)
    assert chk.counts["ledger"] == 3 and "stored trades + open positions" in chk.samples[-1].detail
    eng.positions[SYM].fees -= 1.0
    assert checks(chk).get("ledger") == 3  # consistent again
    acc.wallet += 0.001  # e.g. a funding payment on a 10 USDT position that no book recorded
    checks(chk)
    assert chk.counts["ledger"] == 5


def test_exceptions_are_violations(world):
    chk, *_ = world
    chk.loop_exception({"message": "Task exception was never retrieved", "exception": ValueError("boom")})
    rec = logging.LogRecord("heartless.app", logging.ERROR, __file__, 1, "bar processing failed for %s", ("X",), None)
    chk.error_log(rec)
    assert chk.counts["exceptions"] == 2 and "ValueError" in chk.samples[0].detail
    assert "bar processing failed for X" in chk.samples[1].detail


def test_reported_pnl_is_checked_against_stored_trades(world):
    chk, app, eng, acc, clock = world
    clock.t += MS_MINUTE
    asyncio.run(acc.on_ticker(book(97.0, clock())))  # stopped out: one stored losing trade today
    assert SYM not in eng.positions
    trade = eng.closed[-1]
    pos = SimpleNamespace(side=Side.LONG, symbol=SYM, alpha="htf_trend")
    today = {"stats": {"n": 1, "net": trade.pnl, "win_rate": 0.0, "profit_factor": 0.0}, "by_alpha": {}}
    week = {"stats": {"n": 1, "net": trade.pnl, "win_rate": 0.0, "avg_r": trade.r_multiple, "max_dd": 0.0}}
    status = {"engines": {"paper": {"equity": acc.equity(), "positions": []}}, "primary": "paper", "mode": "paper",
              "params_version": "v", "challengers": []}
    good = [SentMessage(clock(), "daily_report", F.fmt_daily(today, week, status, "UTC")),
            SentMessage(clock(), "cmd:/pnl", F.fmt_pnl({"period": "today", "engine": "paper", "stats": today["stats"]},
                                                       "UTC")),
            SentMessage(clock(), "position_closed", F.fmt_position_closed(pos, trade, 0.01, "🧪 PAPER", trade.pnl,
                                                                          acc.equity()))]
    for m in good:
        chk.on_message(m)
    assert chk.runs["pnl_reports"] == 4 and chk.counts["pnl_reports"] == 0  # daily = today + 7d lines
    # a report that is off by one trade or by a cent is caught
    today["stats"]["net"] += 0.02
    chk.on_message(SentMessage(clock(), "daily_report", F.fmt_daily(today, week, status, "UTC")))
    today["stats"]["net"] -= 0.02  # the right amount, but one trade too many: the count alone must be caught
    today["stats"]["n"] = 2
    chk.on_message(SentMessage(clock(), "cmd:/pnl", F.fmt_pnl({"period": "today", "engine": "paper",
                                                               "stats": today["stats"]}, "UTC")))
    chk.on_message(SentMessage(clock(), "position_closed", "garbled"))
    assert chk.counts["pnl_reports"] == 3
    # yesterday's trades are not "today" any more
    clock.t += MS_DAY
    chk.on_message(SentMessage(clock(), "position_closed", F.fmt_position_closed(pos, trade, 0.01, "🧪 PAPER", 0.0,
                                                                                acc.equity())))
    assert chk.counts["pnl_reports"] == 3


def test_memory_growth_and_slope_are_bounded(world):
    chk, app, *_ = world
    for i in range(10 * 24):
        chk.memory.append({"ts": T0 + i * MS_HOUR, "rss_mb": 100.0 + (i % 5) * 0.1})
    flat = chk.check_memory(limit_mb=50, slope_mb_per_day=2, warmup_ms=MS_DAY)
    assert chk.counts["memory"] == 0 and abs(flat["slope_mb_per_day"]) < 0.5
    chk.memory = [{"ts": T0 + i * MS_HOUR, "rss_mb": 100.0 + i * 0.5} for i in range(10 * 24)]  # 12 MB/day leak
    leak = chk.check_memory(limit_mb=50, slope_mb_per_day=2, warmup_ms=MS_DAY)
    assert chk.counts["memory"] == 2 and leak["slope_mb_per_day"] == pytest.approx(12.0, rel=0.01)


def test_trades_must_be_complete_and_nothing_left_open_after_the_flatten(world):
    chk, app, eng, acc, clock = world
    rec = TradeRecord(position_id="Pbroken", engine="paper", symbol=SYM, side="LONG", alpha="x", alphas=["x"],
                      regime="RANGE", entry_time=clock(), exit_time=clock(), entry_price=100.0, exit_price=float("nan"),
                      qty=1.0, notional=100.0, pnl=0.0, gross=0.0, fees=0.0, funding=0.0, r_multiple=0.0,
                      risk_amount=1.0, exit_reason="x", confidence=0.0, params_version="v", bars_held=0, max_fav_r=0.0,
                      max_adv_r=0.0)
    app.store.save_trade(rec)
    chk.check_trades()
    assert chk.counts["trades"] == 1 and "exit_price" in chk.samples[-1].detail
    chk.check_trades(final=True)  # the OPEN position was never flattened: store, engine and account all say so
    details = " | ".join(v.detail for v in chk.samples)
    assert chk.counts["trades"] == 4 and "left OPEN" in details and "engine still tracks" in details
    assert "account not flat" in details


async def test_network_guard_blocks_and_records_real_connections():
    seen: list[str] = []
    with block_network(seen.append) as attempts:
        with pytest.raises(NetworkBlocked):
            async with httpx.AsyncClient() as c:
                await c.get("https://fapi.binance.com/fapi/v1/time")
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("api.telegram.org", 443))
        with pytest.raises(NetworkBlocked):
            await asyncio.get_running_loop().create_connection(asyncio.Protocol, "example.com", 80)
        # a capture transport is not network
        mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True})))
        assert (await mock.get("https://api.telegram.org/x")).json() == {"ok": True}
        await mock.aclose()
    assert len(seen) == len(attempts) == 3 and "fapi.binance.com" in seen[0]
    assert socket.create_connection.__module__ == "socket"  # restored
