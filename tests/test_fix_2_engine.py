"""Regression tests for the execution engine fixes (round 2): close-path safety, CLOSING recovery, partial-fill
remainders, PnL booking without an exit fill, R geometry, zombie stops, cancel/fill races and exit-reason labels.

Everything runs against an in-memory, live-shaped fake Account (``is_paper = False``) so the live-only code paths
(REST-synthesized fills, reconcile, exception recovery) are exercised without any network."""
from __future__ import annotations

import itertools

import httpx
import pytest

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import AccountState, Decision, EntryStyle, Fill, Position, PositionStatus, Regime, Side, Signal, Ticker
from heartless.exchange.base import Account, OrderResult, PositionSnapshot
from heartless.exchange.paper import PaperAccount
from heartless.execution import engine as engine_mod
from heartless.execution.engine import TradingEngine
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from synth import synth_symbols

SYM = "BTCUSDT"
T0 = 1_700_000_000_000


class FakeLive(Account):
    """Minimal exchange double: orders, algo orders and positions are plain dicts; behaviour is scripted per test."""
    is_paper = False

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []
        self.orders: dict[str, dict] = {}
        self.algos: dict[str, dict] = {}
        self.positions: dict[str, PositionSnapshot] = {}
        self.price = 100.0
        self._ids = itertools.count(1)
        self.hedge_mode = False
        self.market_script: list = []  # per call: Exception to raise, OrderResult to return, or None (fill normally)
        self.positions_raise: Exception | None = None
        self.limit_raise: Exception | None = None
        self.on_place_stop = None  # async hook run while a stop placement is "in flight"
        self.on_open_algo_orders = None  # async hook run while reconcile awaits the algo list

    # --- queries ---
    async def get_state(self):
        return AccountState(10_000, 10_000, 10_000, 0.0, T0)

    async def get_positions(self):
        self.calls.append(("get_positions",))
        if self.positions_raise:
            raise self.positions_raise
        return dict(self.positions)

    async def open_orders(self, symbol=None):
        return [dict(o) for o in self.orders.values() if o["status"] in ("NEW", "PARTIALLY_FILLED")
                and (symbol is None or o["symbol"] == symbol)]

    async def open_algo_orders(self, symbol=None):
        if self.on_open_algo_orders:
            await self.on_open_algo_orders()
        return [dict(a) for a in self.algos.values() if a["status"] == "NEW" and (symbol is None or a["symbol"] == symbol)]

    # --- trading ---
    def _new_order(self, symbol, side, qty, client_id, status, filled=0.0):
        oid = str(next(self._ids))
        self.orders[oid] = {"orderId": oid, "clientOrderId": client_id, "symbol": symbol, "side": side, "origQty": qty,
                            "status": status, "executedQty": filled, "avgPrice": self.price if filled else 0.0}
        return oid

    async def market_order(self, symbol, side, qty, reduce_only=False, client_id=""):
        self.calls.append(("market_order", symbol, side, qty, reduce_only, client_id))
        step = self.market_script.pop(0) if self.market_script else None
        if isinstance(step, Exception):
            raise step
        if isinstance(step, OrderResult):
            return step
        oid = self._new_order(symbol, side, qty, client_id, "FILLED", qty)
        return OrderResult(oid, client_id, "FILLED", qty, self.price, {})

    async def limit_order(self, symbol, side, qty, price, post_only=True, reduce_only=False, client_id=""):
        self.calls.append(("limit_order", symbol, side, qty, price, client_id))
        oid = self._new_order(symbol, side, qty, client_id, "NEW")
        if self.limit_raise:
            raise self.limit_raise  # the order was accepted by the exchange but the response never arrived
        return OrderResult(oid, client_id, "NEW", 0.0, 0.0, {})

    def _find(self, symbol, order_id, client_id):
        for o in self.orders.values():
            if o["symbol"] == symbol and ((order_id and o["orderId"] == order_id) or (client_id and o["clientOrderId"] == client_id)):
                return o
        return None

    async def cancel_order(self, symbol, order_id="", client_id=""):
        self.calls.append(("cancel_order", symbol, order_id, client_id))
        o = self._find(symbol, order_id, client_id)
        if o and o["status"] in ("NEW", "PARTIALLY_FILLED"):
            o["status"] = "CANCELED"
            return True
        return False

    async def query_order(self, symbol, order_id="", client_id=""):
        self.calls.append(("query_order", symbol, order_id, client_id))
        o = self._find(symbol, order_id, client_id)
        if o is None:
            return OrderResult(order_id, client_id, "UNKNOWN")
        return OrderResult(o["orderId"], o["clientOrderId"], o["status"], float(o["executedQty"]), float(o["avgPrice"]))

    async def place_stop(self, symbol, side, trigger_price, qty=None, close_position=False, client_id=""):
        self.calls.append(("place_stop", symbol, side, trigger_price))
        if self.on_place_stop:
            hook, self.on_place_stop = self.on_place_stop, None
            await hook()
        aid = "A" + str(next(self._ids))
        self.algos[aid] = {"algoId": aid, "symbol": symbol, "side": side, "orderType": "STOP_MARKET", "status": "NEW",
                           "triggerPrice": trigger_price, "closePosition": close_position}
        return aid

    async def place_take_profit(self, symbol, side, trigger_price, qty, client_id=""):
        self.calls.append(("place_take_profit", symbol, side, trigger_price, qty))
        aid = "A" + str(next(self._ids))
        self.algos[aid] = {"algoId": aid, "symbol": symbol, "side": side, "orderType": "TAKE_PROFIT_MARKET", "status": "NEW",
                           "triggerPrice": trigger_price, "quantity": qty}
        return aid

    async def cancel_algo(self, symbol, algo_id):
        self.calls.append(("cancel_algo", symbol, algo_id))
        a = self.algos.get(algo_id)
        if a and a["status"] == "NEW":
            a["status"] = "CANCELED"
            return True
        return False

    async def cancel_all(self, symbol):
        self.calls.append(("cancel_all", symbol))
        for a in self.algos.values():
            if a["symbol"] == symbol and a["status"] == "NEW":
                a["status"] = "CANCELED"

    # --- helpers for tests ---
    def open_stops(self, symbol=SYM):
        return [a for a in self.algos.values() if a["symbol"] == symbol and a["status"] == "NEW" and a["orderType"] == "STOP_MARKET"]

    def open_tps(self, symbol=SYM):
        return [a for a in self.algos.values() if a["symbol"] == symbol and a["status"] == "NEW" and a["orderType"] == "TAKE_PROFIT_MARKET"]

    def names(self):
        return [c[0] for c in self.calls]


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms


class FakeCursor:
    ok = False

    def v(self, name):
        return float("nan")


class FakeView:
    frames: dict = {}
    price = 100.0

    def tf(self, name):
        return FakeCursor()

    def ready(self):
        return True


def make_engine(acc=None, clock=None):
    acc = acc or FakeLive()
    clock = clock or Clock()
    s = Settings(_env_file=None)
    eng = TradingEngine("live", acc, StrategyParams.default(), s, synth_symbols([SYM]), bus=EventBus(), clock=clock,
                        persist=False)
    return eng, acc, clock


def pending(eng, acc, qty=1.0, style="LIMIT", entry=100.0, stop=99.0, tp1=None, tp=None, side=Side.LONG, tp1_frac=0.5):
    """A PENDING position whose entry order rests on the fake exchange (mirrors _open without the strategy)."""
    cid = "HLELIVE" + "%08x" % next(acc._ids)
    oid = acc._new_order(SYM, side.order_side, qty, cid, "NEW") if isinstance(acc, FakeLive) else ""
    r_unit = abs(entry - stop)
    pos = Position(id="P1", engine=eng.name, symbol=SYM, side=side, qty=0.0, entry_price=entry, entry_time=eng.clock(),
                   stop=stop, take_profit=tp, tp1=tp1, initial_stop=stop, alpha="trend_pullback", alphas=["trend_pullback"],
                   reason="t", confidence=0.7, regime="RANGE", risk_amount=qty * (r_unit + entry * (2 * eng.s.taker_fee + 0.0003)),
                   r_unit=r_unit, notional=qty * entry, leverage=5, params_version="v", atr=1.0, trail_atr_mult=0.0,
                   max_hold_bars=0, status=PositionStatus.PENDING, entry_client_id=cid, entry_order_id=oid, original_qty=qty,
                   entry_style=style, limit_price=entry if style == "LIMIT" else None, pending_since=eng.clock(),
                   extra={"tp1_frac": tp1_frac, "last_mark": entry})
    eng.positions[SYM] = pos
    return pos


def entry_fill(pos, qty, price=100.0, ts=T0, maker_fee=0.0002):
    return Fill(SYM, pos.side.order_side, qty, price, qty * price * maker_fee, ts, pos.entry_client_id, pos.entry_order_id, kind="ENTRY")


async def opened(eng, acc, qty=1.0, **kw):
    """PENDING -> OPEN through a full entry fill; the exchange shows the position and the brackets rest."""
    pos = pending(eng, acc, qty=qty, **kw)
    o = acc.orders[pos.entry_order_id]
    o.update(status="FILLED", executedQty=qty, avgPrice=kw.get("entry", 100.0))
    await eng.on_fill(entry_fill(pos, qty, kw.get("entry", 100.0)))
    assert pos.status is PositionStatus.OPEN
    acc.positions[SYM] = PositionSnapshot(SYM, qty * pos.side.sign, pos.entry_price, mark=pos.entry_price)
    acc.calls.clear()
    return pos


def events(eng, topic):
    return [p for _, t, p in eng.bus.history if t == topic]


def ctx_for(eng, price=100.0, now=T0):
    return Context(symbol=SYM, info=eng.symbols[SYM], ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=now),
                   regime=Regime.RANGE, now=now)


# --- 1. close_position must never leave a live position without a stop ------------------------------------------

async def test_close_timeout_keeps_stop_and_hands_position_back():
    eng, acc, clock = make_engine()
    pos = await opened(eng, acc)
    stop_id = pos.sl_algo_id
    acc.market_script = [httpx.ReadTimeout("timed out")]
    ok = await eng.close_position(SYM, "시간 초과 청산(time stop)")
    assert ok is False
    assert pos.status is PositionStatus.OPEN
    # the exchange stop was never cancelled: the position was protected throughout
    assert acc.algos[stop_id]["status"] == "NEW" and len(acc.open_stops()) == 1
    assert "cancel_algo" not in acc.names()
    assert events(eng, "error"), "the owner must be told"
    # the next backstop tick retries and succeeds
    acc.price = 90.0
    clock.advance(3_000)
    await eng.on_tick(SYM, 90.0, 89.9, 90.1)
    assert SYM not in eng.positions and eng.closed and eng.closed[-1].symbol == SYM
    assert not acc.open_stops() and not acc.open_tps()


async def test_close_sends_market_order_before_cancelling_brackets():
    eng, acc, _ = make_engine()
    await opened(eng, acc, tp=103.0)
    assert await eng.close_position(SYM, "수동 청산(owner)") is True
    names = acc.names()
    assert names.index("market_order") < names.index("cancel_algo")
    assert SYM not in eng.positions and not acc.open_stops() and not acc.open_tps()
    assert eng.closed[-1].exit_reason == "수동 청산(owner)"


async def test_close_unknown_outcome_is_not_treated_as_flat():
    """LiveAccount returns status UNKNOWN when it cannot tell whether the send reached Binance."""
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)
    acc.market_script = [OrderResult(client_id="x", status="UNKNOWN", raw={"uncertain": True})]
    assert await eng.close_position(SYM, "x") is False
    assert pos.status is PositionStatus.OPEN and len(acc.open_stops()) == 1


# --- 2. a CLOSING position is recoverable (on_tick / owner retry / reconcile) ------------------------------------

async def test_closing_position_recovers_when_exchange_is_unreachable_then_reconciles():
    eng, acc, clock = make_engine()
    pos = await opened(eng, acc)
    acc.market_script = [httpx.ReadTimeout("t")]
    acc.positions_raise = httpx.ConnectError("down")
    assert await eng.close_position(SYM, "펀딩 회피 청산") is False
    assert pos.status is PositionStatus.CLOSING and len(acc.open_stops()) == 1  # still protected
    # a second attempt within the retry window is throttled (no order burst from on_tick)
    assert await eng.close_position(SYM, "펀딩 회피 청산") is False
    assert acc.names().count("market_order") == 1
    # the exchange comes back: the exchange stop had fired meanwhile (flat) and the fill event was lost
    acc.positions_raise = None
    acc.positions.clear()
    pos.extra["last_mark"] = 99.0
    clock.advance(engine_mod.CLOSE_RETRY_MS + 1)
    await eng.reconcile()
    assert SYM not in eng.positions
    rec = eng.closed[-1]
    assert rec.exit_reason.startswith("펀딩 회피 청산") and "reconcile" in rec.exit_reason
    assert rec.gross == pytest.approx(-1.0) and rec.r_multiple < -0.9


async def test_reconcile_retries_a_stuck_close_when_position_still_exists():
    eng, acc, clock = make_engine()
    pos = await opened(eng, acc)
    acc.market_script = [httpx.ReadTimeout("t")]
    acc.positions_raise = httpx.ConnectError("down")
    await eng.close_position(SYM, "시간 초과 청산(time stop)")
    assert pos.status is PositionStatus.CLOSING
    acc.positions_raise = None
    for a in acc.algos.values():  # and the exchange stop got lost too
        a["status"] = "CANCELED"
    clock.advance(engine_mod.CLOSE_RETRY_MS + 1)
    await eng.reconcile()
    # stop was re-placed before the retry, the retry filled, the position is booked with the original reason
    assert "place_stop" in acc.names()
    assert SYM not in eng.positions and eng.closed[-1].exit_reason == "시간 초과 청산(time stop)"
    assert not acc.open_stops()


async def test_owner_can_retry_close_of_stuck_closing_row():
    eng, acc, clock = make_engine()
    pos = await opened(eng, acc)
    pos.status = PositionStatus.CLOSING  # e.g. restored from the DB after a crash mid-close
    pos.exit_reason = "x"
    assert await eng.close_position(SYM, "수동 청산(owner)") is True
    assert SYM not in eng.positions and eng.closed[-1].exit_reason == "수동 청산(owner)"


# --- 3/16. partially filled post-only entry: the remainder is owned and cancelled ------------------------------

async def test_partial_entry_remainder_cancelled_on_exit_and_late_fill_ignored():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0)
    acc.orders[pos.entry_order_id].update(status="PARTIALLY_FILLED", executedQty=0.4, avgPrice=100.0)
    await eng.on_fill(entry_fill(pos, 0.4))
    assert pos.status is PositionStatus.OPEN and pos.qty == pytest.approx(0.4)
    acc.positions[SYM] = PositionSnapshot(SYM, 0.4, 100.0)
    # the exchange stop fires on the 0.4
    await eng.on_fill(Fill(SYM, "SELL", 0.4, 99.0, 0.02, T0 + 1, "HLSLIVE1", pos.sl_algo_id, reduce_only=True, kind="SL"))
    assert SYM not in eng.positions
    assert any(c[0] == "cancel_order" and c[2] == pos.entry_order_id for c in acc.calls)
    assert acc.orders[pos.entry_order_id]["status"] == "CANCELED" and await acc.open_orders() == []
    assert eng.closed[-1].qty == pytest.approx(0.4)
    # a late fill of the dead remainder cannot create exposure the engine does not know about
    await eng.on_fill(entry_fill(pos, 0.6))
    assert SYM not in eng.positions


async def test_partial_entry_remainder_cancelled_before_market_close():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0)
    acc.orders[pos.entry_order_id].update(status="PARTIALLY_FILLED", executedQty=0.4, avgPrice=100.0)
    await eng.on_fill(entry_fill(pos, 0.4))
    acc.positions[SYM] = PositionSnapshot(SYM, 0.4, 100.0)
    acc.calls.clear()
    assert await eng.close_position(SYM, "수동 청산(owner)") is True
    names = acc.names()
    assert names.index("cancel_order") < names.index("market_order")
    mo = next(c for c in acc.calls if c[0] == "market_order")
    assert mo[3] == pytest.approx(0.4) and mo[4] is True
    assert await acc.open_orders() == [] and SYM not in eng.positions


async def test_partial_entry_remainder_times_out_in_manage():
    eng, acc, clock = make_engine()
    pos = pending(eng, acc, qty=1.0)
    acc.orders[pos.entry_order_id].update(status="PARTIALLY_FILLED", executedQty=0.4, avgPrice=100.0)
    await eng.on_fill(entry_fill(pos, 0.4))
    clock.advance(2 * 60_000)
    await eng.on_bar(FakeView(), ctx_for(eng, now=clock()))
    assert acc.orders[pos.entry_order_id]["status"] == "CANCELED"
    assert pos.status is PositionStatus.OPEN and pos.original_qty == pytest.approx(0.4) and pos.qty == pytest.approx(0.4)


async def test_reconcile_cancels_orphan_entry_orders():
    eng, acc, _ = make_engine()
    acc._new_order(SYM, "BUY", 0.6, "HLELIVEorphan", "PARTIALLY_FILLED", 0.4)
    await eng.reconcile()
    assert await acc.open_orders() == []


# --- 4/15. _finalize books quantity that left without a processed fill ------------------------------------------

async def test_rejected_close_on_flat_exchange_books_real_loss():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)
    acc.positions.clear()  # the stop already fired on the exchange, the fill event lagged
    pos.extra["last_mark"] = 98.9
    acc.market_script = [OrderResult(client_id="c", status="REJECTED", raw={"code": -2022})]
    assert await eng.close_position(SYM, "소프트웨어 백스톱 손절") is True
    rec = eng.closed[-1]
    assert rec.gross == pytest.approx(-1.1) and rec.pnl < rec.gross and rec.r_multiple < -0.9
    assert rec.exit_price == 98.9
    # the late stop fill for the same quantity is not double counted
    await eng.on_fill(Fill(SYM, "SELL", 1.0, 98.9, 0.05, T0 + 5, "HLSLIVE1", "A1", reduce_only=True, kind="SL"))
    assert len(eng.closed) == 1 and eng.stats.realized_today == pytest.approx(rec.pnl)


async def test_reconcile_flat_books_remaining_quantity():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc, side=Side.SHORT, entry=100.0, stop=101.0)
    acc.positions.clear()
    pos.extra["last_mark"] = 101.2
    await eng.reconcile()
    rec = eng.closed[-1]
    assert rec.gross == pytest.approx(-1.2) and rec.r_multiple < -0.9
    assert eng.risk.state.symbol_cooldown.get(SYM, 0) > 0  # the loss registers a cool-down


# --- 5. risk_amount / r_unit follow every entry fill ----------------------------------------------------------

async def test_multi_trade_entry_keeps_one_r_geometry():
    eng, acc, s = make_engine()
    pos = pending(eng, acc, qty=1.0, style="MARKET")
    acc.orders[pos.entry_order_id].update(status="FILLED", executedQty=1.0, avgPrice=100.0)
    await eng.on_fill(entry_fill(pos, 0.4, 100.0, maker_fee=0.0005))
    await eng.on_fill(entry_fill(pos, 0.6, 100.0, maker_fee=0.0005))
    assert pos.qty == pytest.approx(1.0)
    expected = 1.0 * (pos.r_unit + pos.entry_price * (2 * eng.s.taker_fee + 0.0003))
    assert pos.risk_amount == pytest.approx(expected)
    await eng.on_fill(Fill(SYM, "SELL", 1.0, 99.0, 1.0 * 99.0 * 0.0005, T0 + 9, "HLSLIVE1", pos.sl_algo_id, reduce_only=True, kind="SL"))
    assert eng.closed[-1].r_multiple == pytest.approx(-1.0, abs=0.05)


# --- 6. _replace_stop must not orphan a stop placed while the position exited ---------------------------------

async def test_replace_stop_cancels_new_stop_if_position_closed_meanwhile():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)
    old_id = pos.sl_algo_id

    async def stop_fires_during_placement():
        await eng.on_fill(Fill(SYM, "SELL", 1.0, 99.0, 0.05, T0 + 1, "HLSLIVE1", old_id, reduce_only=True, kind="SL"))

    acc.on_place_stop = stop_fires_during_placement
    await eng._replace_stop(pos, 99.5)
    assert pos.status is PositionStatus.CLOSED and SYM not in eng.positions
    assert not acc.open_stops(), "the freshly placed stop must not survive the position"


async def test_place_brackets_cancels_orders_if_position_exits_during_placement():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0, style="MARKET", tp=103.0)

    async def exits_immediately():
        await eng.on_fill(Fill(SYM, "SELL", 1.0, 98.0, 0.05, T0 + 1, "", "99", reduce_only=True, kind="UNKNOWN"))

    acc.on_place_stop = exits_immediately
    await eng.on_fill(entry_fill(pos, 1.0))
    assert SYM not in eng.positions and not acc.open_stops() and not acc.open_tps()


# --- 7. cancelling a pending entry that just filled -----------------------------------------------------------

async def test_kill_switch_flattens_entry_that_filled_under_the_cancel():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0)
    acc.orders[pos.entry_order_id].update(status="FILLED", executedQty=1.0, avgPrice=100.0)  # filled, event in flight
    acc.positions[SYM] = PositionSnapshot(SYM, 1.0, 100.0)
    assert await eng.close_position(SYM, "전체 수동 청산(owner)") is True
    assert pos.status is not PositionStatus.CANCELLED
    assert any(c[0] == "market_order" and c[4] is True for c in acc.calls), "the filled entry must be flattened"
    assert SYM not in eng.positions and eng.closed[-1].qty == pytest.approx(1.0)
    # the stream fill that arrives afterwards is a duplicate of what the query credited
    await eng.on_fill(entry_fill(pos, 1.0))
    assert SYM not in eng.positions


async def test_cancel_pending_keeps_filled_entry_as_open_position():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0)
    acc.orders[pos.entry_order_id].update(status="FILLED", executedQty=1.0, avgPrice=100.0)
    await eng._cancel_pending(pos, "진입가 미도달로 주문 취소")
    assert pos.status is PositionStatus.OPEN and len(acc.open_stops()) == 1 and SYM in eng.positions
    await eng.on_fill(entry_fill(pos, 1.0))  # late stream fill: de-duplicated
    assert pos.qty == pytest.approx(1.0) and pos.filled_qty == pytest.approx(1.0)


# --- 8. entry exception after the order may have been accepted ------------------------------------------------

def _decision(eng, side=Side.LONG):
    sig = Signal("trend_pullback", SYM, side, 0.8, "t", 99.0, 103.0, None, EntryStyle.MARKET, None, 0, 0.0, 1.0, "5m",
                 {"ref_price": 100.0})
    return Decision(SYM, side, 0.8, 0.8, ["trend_pullback"], sig, "t", Regime.TREND_UP, 1.0, 2.0)


async def test_entry_timeout_after_acceptance_is_verified_not_forgotten():
    eng, acc, _ = make_engine()
    orig = acc.market_order

    async def timeout_after_fill(symbol, side, qty, reduce_only=False, client_id=""):
        acc._new_order(symbol, side, qty, client_id, "FILLED", qty)  # Binance executed it ...
        raise httpx.ReadTimeout("t")  # ... but the response never came back

    acc.market_order = timeout_after_fill
    await eng._open(_decision(eng), FakeView(), ctx_for(eng), 10_000.0)
    pos = eng.positions.get(SYM)
    assert pos is not None and pos.status is PositionStatus.OPEN and pos.qty > 0
    assert len(acc.open_stops()) == 1
    qty = pos.qty
    await eng.on_fill(Fill(SYM, "BUY", qty, 100.0, 0.05, T0 + 1, pos.entry_client_id, pos.entry_order_id, kind="ENTRY"))
    assert pos.qty == pytest.approx(qty), "the late stream fill is a duplicate"
    acc.market_order = orig


async def test_entry_exception_before_any_order_is_a_clean_cancel():
    eng, acc, _ = make_engine()

    async def boom(symbol, leverage):
        raise RuntimeError("leverage")

    acc.prepare_symbol = boom
    await eng._open(_decision(eng), FakeView(), ctx_for(eng), 10_000.0)
    assert SYM not in eng.positions and "query_order" not in acc.names()


# --- 9. reconcile must not adopt a phantom from a stale snapshot ---------------------------------------------

async def test_reconcile_does_not_adopt_position_closed_during_its_awaits():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)

    async def stop_fires_now():
        acc.positions.clear()
        await eng.on_fill(Fill(SYM, "SELL", 1.0, 99.0, 0.05, T0 + 1, "HLSLIVE1", pos.sl_algo_id, reduce_only=True, kind="SL"))

    acc.on_open_algo_orders = stop_fires_now
    await eng.reconcile()
    assert SYM not in eng.positions
    assert len(eng.closed) == 1 and eng.closed[0].alpha != "adopted"
    assert not acc.open_stops()


async def test_reconcile_still_adopts_a_genuine_external_position():
    eng, acc, _ = make_engine()
    acc.positions["BTCUSDT"] = PositionSnapshot(SYM, 2.0, 100.0, leverage=5, mark=100.0)
    await eng.reconcile()
    pos = eng.positions[SYM]
    assert pos.alpha == "adopted" and pos.qty == 2.0 and len(acc.open_stops()) == 1


# --- 10. missing-stop detection is side-aware ---------------------------------------------------------------

async def test_reconcile_ignores_opposite_side_zombie_stop():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc, side=Side.SHORT, entry=100.0, stop=101.0)
    for a in acc.algos.values():
        a["status"] = "CANCELED"  # our BUY stop placement is gone
    pos.sl_algo_id = ""
    zombie = "AZ"
    acc.algos[zombie] = {"algoId": zombie, "symbol": SYM, "side": "SELL", "orderType": "STOP_MARKET", "status": "NEW", "triggerPrice": 95.0}
    await eng.reconcile()
    placed = [c for c in acc.calls if c[0] == "place_stop"]
    assert placed and placed[-1][2] == "BUY"
    assert acc.algos[zombie]["status"] == "CANCELED"
    assert pos.sl_algo_id and acc.algos[pos.sl_algo_id]["side"] == "BUY"


async def test_reconcile_syncs_sl_algo_id_to_the_real_stop():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)
    real = pos.sl_algo_id
    pos.sl_algo_id = ""
    await eng.reconcile()
    assert pos.sl_algo_id == real and acc.names().count("place_stop") == 0


# --- 11/17. engine-initiated closes keep their reason ---------------------------------------------------------

async def test_engine_close_reason_is_not_relabelled_in_paper():
    acc = PaperAccount(initial_balance=10_000)
    clock = Clock()
    eng = TradingEngine("paper", acc, StrategyParams.default(), Settings(_env_file=None), synth_symbols([SYM]), bus=EventBus(),
                        clock=clock, persist=False)
    await acc.on_ticker(Ticker(SYM, bid=100.0, ask=100.02, mark=100.01, ts=T0))
    pos = pending(eng, acc, qty=1.0, style="MARKET", tp1=101.0, tp=103.0)
    pos.entry_order_id = ""
    res = await acc.market_order(SYM, "BUY", 1.0, client_id=pos.entry_client_id)
    assert pos.status is PositionStatus.OPEN and res.status == "FILLED"
    # mark just below tp1: a time stop here used to be recorded as '익절(TP)'
    await acc.on_ticker(Ticker(SYM, bid=100.8, ask=100.82, mark=100.81, ts=T0 + 1))
    assert await eng.close_position(SYM, "시간 초과 청산(time stop)") is True
    assert eng.closed[-1].exit_reason == "시간 초과 청산(time stop)"
    assert not events(eng, "partial_tp")


async def test_split_close_fill_does_not_trigger_partial_tp_logic():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc, tp1=101.0, tp=103.0)
    pos.status = PositionStatus.CLOSING
    pos.exit_reason = "펀딩 회피 청산"
    acc.calls.clear()
    await eng.on_fill(Fill(SYM, "SELL", 0.5, 100.9, 0.02, T0 + 1, "HLCLIVE1", "77", reduce_only=True, kind="CLOSE"))
    assert pos.tp1_done is False and "place_stop" not in acc.names() and not events(eng, "partial_tp")
    await eng.on_fill(Fill(SYM, "SELL", 0.5, 100.9, 0.02, T0 + 1, "HLCLIVE1", "77", reduce_only=True, kind="CLOSE"))
    assert eng.closed[-1].exit_reason == "펀딩 회피 청산"


# --- 12. re-bracketing after TP1 keeps a runner take-profit ---------------------------------------------------

async def test_place_brackets_after_tp1_places_full_runner_tp():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc, tp1=101.0, tp=103.0)
    assert len(acc.open_tps()) == 2
    # TP1 fills half (the exchange finishes that algo order)
    tp1_id = pos.extra["tp1_algo_id"]
    acc.algos[tp1_id]["status"] = "FINISHED"
    await eng.on_fill(Fill(SYM, "SELL", 0.5, 101.0, 0.02, T0 + 1, "HLTLIVE1", tp1_id, reduce_only=True, kind="TP"))
    assert pos.tp1_done and pos.qty == pytest.approx(0.5)
    await eng._cancel_brackets(pos)
    acc.calls.clear()
    await eng._place_brackets(pos)
    tps = acc.open_tps()
    assert len(tps) == 1 and tps[0]["triggerPrice"] == 103.0 and tps[0]["quantity"] == pytest.approx(0.5)
    assert "tp1_algo_id" not in pos.extra


# --- 13. re-quote is exception safe ---------------------------------------------------------------------------

async def test_requote_transport_error_adopts_the_resting_order():
    eng, acc, clock = make_engine()
    pos = pending(eng, acc, qty=1.0)
    old_cid = pos.entry_client_id
    clock.advance(2 * 60_000 + 1)
    acc.limit_raise = httpx.ReadTimeout("t")
    await eng.on_bar(FakeView(), ctx_for(eng, price=99.9, now=clock()))
    assert pos.status is PositionStatus.PENDING
    resting = await acc.open_orders()
    assert len(resting) == 1, "exactly one entry order may rest"
    assert pos.entry_client_id == resting[0]["clientOrderId"] != old_cid
    assert pos.entry_order_id == resting[0]["orderId"]
    # a fill of that order is booked, not discarded as stale
    acc.orders[pos.entry_order_id].update(status="FILLED", executedQty=1.0, avgPrice=99.9)
    await eng.on_fill(Fill(SYM, "BUY", 1.0, 99.9, 0.02, clock(), pos.entry_client_id, pos.entry_order_id, kind="ENTRY"))
    assert pos.status is PositionStatus.OPEN and pos.qty == pytest.approx(1.0)
    # a second bar does not quote another order on top of it
    acc.limit_raise = None
    acc.calls.clear()
    clock.advance(2 * 60_000 + 1)
    await eng.on_bar(FakeView(), ctx_for(eng, price=99.9, now=clock()))
    assert "limit_order" not in acc.names() and len(await acc.open_orders()) == 0


# --- 14. de-duplication of a partial fill credited from the re-quote query -----------------------------------

async def test_requote_partial_fill_is_not_double_counted():
    eng, acc, clock = make_engine()
    pos = pending(eng, acc, qty=1.0)
    oid = pos.entry_order_id
    acc.orders[oid].update(status="PARTIALLY_FILLED", executedQty=0.4, avgPrice=100.0)  # filled 0.4 as we cancel
    clock.advance(2 * 60_000 + 1)
    await eng.on_bar(FakeView(), ctx_for(eng, price=99.9, now=clock()))
    assert pos.status is PositionStatus.OPEN and pos.qty == pytest.approx(0.4)
    # the stream TRADE event for the same 0.4 arrives afterwards
    await eng.on_fill(Fill(SYM, "BUY", 0.4, 100.0, 0.008, clock(), pos.entry_client_id, oid, kind="ENTRY"))
    assert pos.qty == pytest.approx(0.4) and pos.filled_qty == pytest.approx(0.4)
    tps = acc.open_tps()
    assert all(t["quantity"] <= 0.4 + 1e-9 for t in tps)
