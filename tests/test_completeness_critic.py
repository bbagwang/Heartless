"""Completeness-critic regression tests: gaps left after the nine-lens review on the live money path.

Covered (all offline, against the live-shaped fake exchange from test_fix_2_engine):
  * an entry whose stop is already at/through the trade reference is skipped (LONG and SHORT); a take-profit that
    sits behind the reference is dropped instead of being sent (Binance -2021 / instant self-trigger)
  * restart recovery: a PENDING live entry that executed while the bot was down is credited by reconcile (-> OPEN,
    exchange stop placed) instead of sitting unprotected until a later bar
  * a market entry whose result never came back is verified on the next bar, not three bars later
  * app._tick isolates engine failures so a raising paper engine cannot skip the live software backstop
  * a REST-synthesized close fill that is capped by the remaining quantity only books its share of the fee
  * the live fill parser treats closePosition (cp) fills as reduce-only
"""
from __future__ import annotations

import asyncio

import pytest

from heartless.app import Heartless
from heartless.core.models import Decision, EntryStyle, Fill, PositionStatus, Regime, Side, Signal, Ticker
from heartless.exchange.base import PositionSnapshot
from heartless.exchange.live import LiveAccount
from heartless.util.timeutil import MS_MINUTE
from tests.test_fix_2_engine import SYM, T0, FakeView, _decision, ctx_for, make_engine, opened, pending


def _short_decision(stop=101.0, tp=97.0):
    sig = Signal("trend_pullback", SYM, Side.SHORT, 0.8, "t", stop, tp, None, EntryStyle.MARKET, None, 0, 0.0, 1.0, "5m",
                 {"ref_price": 100.0})
    return Decision(SYM, Side.SHORT, 0.8, 0.8, ["trend_pullback"], sig, "t", Regime.TREND_DOWN, 1.0, 2.0)


# --- 1. bracket sanity against the actual trade reference ---------------------------------------------------

async def test_long_entry_is_skipped_when_the_stop_is_already_through():
    eng, acc, _ = make_engine()
    # signal: stop 99 computed from the bar close at 100; the touch has since fallen to 98 (ask 98.0098 < stop)
    await eng._open(_decision(eng), FakeView(), ctx_for(eng, price=98.0), 10_000.0)
    assert SYM not in eng.positions
    assert "market_order" not in acc.names() and "place_stop" not in acc.names()
    assert eng.stats.skipped.get("stop") == 1


async def test_short_entry_with_stop_below_reference_is_skipped_and_a_sane_short_opens():
    eng, acc, _ = make_engine()
    # _decision's stop (99) lies below the short's reference bid (99.99): already through -> skip
    await eng._open(_decision(eng, Side.SHORT), FakeView(), ctx_for(eng), 10_000.0)
    assert SYM not in eng.positions and "market_order" not in acc.names()
    # a short whose stop is above and TP below the reference trades normally: SELL entry, BUY stop above entry
    await eng._open(_short_decision(), FakeView(), ctx_for(eng), 10_000.0)
    pos = eng.positions[SYM]
    assert pos.status is PositionStatus.OPEN and pos.side is Side.SHORT
    assert ("market_order", SYM, "SELL", pytest.approx(pos.qty), False, pos.entry_client_id) in acc.calls
    stops = acc.open_stops()
    assert len(stops) == 1 and stops[0]["side"] == "BUY" and stops[0]["triggerPrice"] > pos.entry_price
    assert all(t["side"] == "BUY" and t["triggerPrice"] < pos.entry_price for t in acc.open_tps())


async def test_take_profit_behind_the_reference_is_dropped_but_the_entry_still_opens_with_a_stop():
    eng, acc, _ = make_engine()
    # price ran from 100 to 104 since the bar close: TP 103 is already passed, stop 99 is still protective
    await eng._open(_decision(eng), FakeView(), ctx_for(eng, price=104.0), 10_000.0)
    pos = eng.positions[SYM]
    assert pos.status is PositionStatus.OPEN and pos.take_profit is None and pos.tp1 is None
    assert len(acc.open_stops()) == 1 and not acc.open_tps()


# --- 2. restart: a PENDING entry that filled while the bot was down -------------------------------------------

async def test_reconcile_credits_pending_entry_that_filled_while_down():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0, style="MARKET")  # the row a restart restores: PENDING, order id known
    o = acc.orders[pos.entry_order_id]
    o.update(status="FILLED", executedQty=1.0, avgPrice=100.0)  # Binance executed it; the stream event was missed
    acc.positions[SYM] = PositionSnapshot(SYM, 1.0, 100.0, mark=100.0)
    assert not acc.open_stops()
    await eng.reconcile()
    assert pos.status is PositionStatus.OPEN and pos.qty == pytest.approx(1.0)
    assert len(acc.open_stops()) == 1, "the position must be protected right after reconcile, not bars later"
    assert pos.entry_order_id in pos.extra.get("settled_orders", [])
    # a late stream fill for the same order is a duplicate
    await eng.on_fill(Fill(SYM, "BUY", 1.0, 100.0, 0.05, T0 + 1, pos.entry_client_id, pos.entry_order_id, kind="ENTRY"))
    assert pos.qty == pytest.approx(1.0)
    assert eng.closed == [] and SYM in eng.positions


async def test_reconcile_leaves_a_pending_entry_alone_when_the_exchange_is_flat():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0, style="LIMIT")
    await eng.reconcile()
    assert pos.status is PositionStatus.PENDING and "query_order" not in acc.names()
    assert acc.orders[pos.entry_order_id]["status"] == "NEW", "the resting limit order is _manage_pending's business"


# --- 3. market entry verification happens on the next bar -----------------------------------------------------

async def test_unconfirmed_market_entry_is_verified_on_the_next_bar():
    eng, acc, clock = make_engine()
    pos = pending(eng, acc, qty=1.0, style="MARKET")
    acc.orders[pos.entry_order_id].update(status="FILLED", executedQty=1.0, avgPrice=100.0)
    clock.advance(MS_MINUTE)
    await eng.on_bar(FakeView(), ctx_for(eng, now=clock()))
    assert pos.status is PositionStatus.OPEN and len(acc.open_stops()) == 1


async def test_market_entry_that_never_reached_the_exchange_is_cancelled_on_the_next_bar():
    eng, acc, clock = make_engine()
    pos = pending(eng, acc, qty=1.0, style="MARKET")
    del acc.orders[pos.entry_order_id]  # the exchange has no such order (lookup -> UNKNOWN)
    clock.advance(MS_MINUTE)
    await eng.on_bar(FakeView(), ctx_for(eng, now=clock()))
    assert pos.status is PositionStatus.CANCELLED and SYM not in eng.positions


# --- 4. app._tick isolates engine failures --------------------------------------------------------------------

async def test_tick_runs_the_live_backstop_even_when_a_paper_engine_raises():
    live_eng, live_acc, clock = make_engine()
    pos = await opened(live_eng, live_acc)  # LONG 1.0 @ 100, stop 99
    clock.advance(5_000)  # past the 2.5 s grace the backstop grants a freshly placed exchange stop
    bad_eng, bad_acc, _ = make_engine()
    await opened(bad_eng, bad_acc)
    raised = []

    async def boom(symbol, mark, bid, ask):
        raised.append(symbol)
        raise RuntimeError("simulator broke")

    bad_eng.on_tick = boom
    app = Heartless.__new__(Heartless)
    app.paper_accounts = {}
    app.engines = {"paper": bad_eng, "live": live_eng}  # paper engines are created first
    live_acc.calls.clear()
    await app._tick(SYM, Ticker(SYM, bid=98.0, ask=98.01, mark=98.0, last=98.0, ts=T0 + 10_000))
    assert raised == [SYM]
    assert any(c[0] == "market_order" and c[2] == "SELL" and c[4] is True for c in live_acc.calls), \
        "the live engine's software backstop must still flatten a position through its stop"
    assert pos.status in (PositionStatus.CLOSING, PositionStatus.CLOSED)


# --- 5. synthesized close fee is not double counted -------------------------------------------------------------

async def test_synthesized_close_fill_books_only_the_uncredited_share_of_the_fee():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)
    entry_fees = pos.fees
    pos.status = PositionStatus.CLOSING
    pos.exit_reason = "x"
    # stream trade for 0.4 of the close arrives first ...
    await eng.on_fill(Fill(SYM, "SELL", 0.4, 101.0, 0.0202, T0 + 1, "HLCLIVE1", "77", reduce_only=True, kind="CLOSE"))
    assert pos.qty == pytest.approx(0.6)
    # ... then the REST result for the whole order (1.0 @ 101, fee for the full quantity) is synthesized
    await eng.on_fill(Fill(SYM, "SELL", 1.0, 101.0, 0.0505, T0 + 2, "HLCLIVE1", "77", reduce_only=True, kind="CLOSE"))
    assert pos.status is PositionStatus.CLOSED
    assert pos.fees == pytest.approx(entry_fees + 0.0202 + 0.0505 * 0.6)
    assert pos.realized == pytest.approx(1.0)  # 1.0 contract closed once at +1.0


# --- 6. live fill parser: closePosition fills are reduce-only ----------------------------------------------------

def test_close_position_fill_is_classified_reduce_only():
    acc = LiveAccount(rest=object())
    got: list[Fill] = []

    async def grab(f: Fill) -> None:
        got.append(f)

    acc.on_fill(grab)
    ev = {"e": "ORDER_TRADE_UPDATE", "o": {"s": SYM, "S": "SELL", "x": "TRADE", "l": "1", "L": "99", "n": "0.05", "N": "USDT",
                                           "o": "MARKET", "ot": "MARKET", "c": "autoid", "R": False, "cp": True, "i": 5,
                                           "T": T0, "m": False}}
    asyncio.run(acc._on_user_event(ev))
    assert len(got) == 1 and got[0].reduce_only is True and got[0].kind == "CLOSE"


# --- 7. reconcile keeps exactly one same-side stop ----------------------------------------------------------------

async def test_reconcile_cancels_a_stale_same_side_stop_and_keeps_the_engines_own():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)  # LONG @100, own stop A at 99
    own = pos.sl_algo_id
    # the previous trade's closePosition SELL stop whose cancel failed (transport error) and survived the re-entry
    acc.algos["Z1"] = {"algoId": "Z1", "symbol": SYM, "side": "SELL", "orderType": "STOP_MARKET", "status": "NEW",
                       "triggerPrice": 105.0, "closePosition": True}
    await eng.reconcile()
    assert pos.sl_algo_id == own and acc.algos[own]["status"] == "NEW"
    assert acc.algos["Z1"]["status"] == "CANCELED", "a second same-side stop would close the position at 105, a level nobody chose"
    assert pos.status is PositionStatus.OPEN and len(acc.open_stops()) == 1


async def test_reconcile_prefers_the_stop_nearest_the_intended_level_when_its_id_is_stale():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)
    for a in list(acc.algos.values()):  # the process restarted: the row's ids point at nothing
        if a["orderType"] == "STOP_MARKET":
            del acc.algos[a["algoId"]]
    pos.sl_algo_id = "gone"
    acc.algos["FAR"] = {"algoId": "FAR", "symbol": SYM, "side": "SELL", "orderType": "STOP_MARKET", "status": "NEW",
                        "triggerPrice": 90.0, "closePosition": True}
    acc.algos["NEAR"] = {"algoId": "NEAR", "symbol": SYM, "side": "SELL", "orderType": "STOP_MARKET", "status": "NEW",
                         "triggerPrice": 99.0, "closePosition": True}
    await eng.reconcile()
    assert pos.sl_algo_id == "NEAR" and acc.algos["NEAR"]["status"] == "NEW" and acc.algos["FAR"]["status"] == "CANCELED"


# --- 8. a reducing fill on a PENDING row is foreign ---------------------------------------------------------------

async def test_reducing_fill_on_pending_entry_is_ignored_not_booked():
    eng, acc, _ = make_engine()
    pos = pending(eng, acc, qty=1.0, style="LIMIT")
    await eng.on_fill(Fill(SYM, "SELL", 1.0, 97.0, 0.05, T0 + 1, "HLSLIVE9", "old-stop", reduce_only=True, kind="SL"))
    assert pos.status is PositionStatus.PENDING and pos.realized == 0.0 and pos.fees == 0.0
    assert eng.closed == [] and SYM in eng.positions
    assert acc.orders[pos.entry_order_id]["status"] == "NEW", "the working entry must not be cancelled by a foreign fill"


# --- 9. a late entry fill after close is reported -------------------------------------------------------------------

async def test_late_entry_fill_after_close_is_reported_for_reconcile():
    eng, acc, _ = make_engine()
    pos = await opened(eng, acc)
    pos.status = PositionStatus.CLOSED
    await eng.on_fill(Fill(SYM, "BUY", 0.3, 100.0, 0.01, T0 + 5, pos.entry_client_id, "new-order-id", kind="ENTRY"))
    assert pos.qty == pytest.approx(1.0), "a closed row must not absorb quantity it no longer manages"
    errs = [p for _, t, p in eng.bus.history if t == "error"]
    assert errs and SYM in errs[-1]["message"]
