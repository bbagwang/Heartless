"""P5 known issue: the paper account reports working orders in Binance's shape, so engine.reconcile() run against a
simulated account (e.g. a paper exchange standing in for Binance) recognises its stops and entry orders.

PaperAccount.open_algo_orders used to label the order type only as `kind`; reconcile reads `orderType`, found no stop
and placed another one on every pass (and could never cancel a leftover)."""
from __future__ import annotations

from synth import synth_symbols

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import Decision, EntryStyle, PositionStatus, Regime, Side, Signal, Ticker
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_MINUTE

SYM = "BTCUSDT"
T0 = 1_772_323_200_000


class Clock:
    t = T0

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


async def exchange_like_engine():
    """An engine on a PaperAccount that plays the exchange: is_paper=False makes the engine reconcile against it."""
    clock = Clock()
    acc = PaperAccount("sim", 10_000.0, clock=clock)
    acc.is_paper = False
    syms = synth_symbols([SYM])
    acc.set_symbols(syms)
    eng = TradingEngine("live", acc, StrategyParams.default(), Settings(_env_file=None), syms, bus=EventBus(),
                        clock=clock, persist=False)
    await eng.start()
    tk = book(100.0, clock())
    await acc.on_ticker(tk)
    sig = Signal(alpha="htf_trend", symbol=SYM, side=Side.LONG, confidence=0.7, reason="t", stop=98.0, take_profit=None,
                 tp1=None, entry_style=EntryStyle.MARKET, atr=2.0, timeframe="1h")
    d = Decision(symbol=SYM, side=Side.LONG, score=0.7, confidence=0.7, alphas=["htf_trend"], primary=sig, reason="t",
                 regime=Regime.RANGE)
    await eng._open(d, View(), Context(symbol=SYM, info=syms[SYM], ticker=tk, regime=Regime.RANGE, now=clock()),
                    await eng.equity())
    return eng, acc, clock


async def test_paper_orders_have_the_binance_shape():
    eng, acc, clock = await exchange_like_engine()
    pos = eng.positions[SYM]
    (stop,) = await acc.open_algo_orders(SYM)
    assert stop["algoId"] == pos.sl_algo_id and stop["orderType"] == "STOP_MARKET" and stop["side"] == "SELL"
    assert stop["triggerPrice"] == pos.stop and stop["closePosition"] is True and stop["algoStatus"] == "NEW"
    assert stop["clientAlgoId"].startswith("HLS") and stop["algoType"] == "CONDITIONAL"
    res = await acc.limit_order(SYM, "BUY", 0.5, 95.0, client_id="HLELIVEabc")
    (order,) = await acc.open_orders(SYM)
    assert order["orderId"] == res.order_id and order["clientOrderId"] == "HLELIVEabc" and order["origQty"] == 0.5
    assert order["executedQty"] == 0.0 and order["type"] == "LIMIT" and order["timeInForce"] == "GTX"
    assert order["price"] == 95.0 and order["status"] == "NEW" and order["reduceOnly"] is False


async def test_reconcile_against_a_paper_account_keeps_a_stopped_position_consistent():
    eng, acc, clock = await exchange_like_engine()
    pos = eng.positions[SYM]
    sl = pos.sl_algo_id
    stray = await acc.limit_order(SYM, "BUY", 0.5, 95.0, client_id=f"HLE{eng.tag}stale")  # an orphan entry order
    for _ in range(3):
        clock.t += MS_MINUTE
        await eng.reconcile()
    # read shape-independently so the pre-fix behaviour (a new stop on every pass) shows as such
    stops = [a for a in await acc.open_algo_orders(SYM) if a.get("orderType", a.get("kind")) == "STOP_MARKET"]
    assert [a.get("algoId", a.get("algo_id")) for a in stops] == [sl]  # recognised: no duplicate stops
    assert pos.sl_algo_id == sl and pos.status is PositionStatus.OPEN and pos.qty == acc.positions[SYM].qty
    assert await acc.open_orders() == [] and (await acc.query_order(SYM, stray.order_id)).status == "CANCELED"
    # the stop fires on the simulated exchange; reconcile then finds a flat account and nothing to adopt or cancel
    clock.t += MS_MINUTE
    await acc.on_ticker(book(97.5, clock()))
    assert SYM not in eng.positions and eng.closed[-1].exit_reason == "손절(SL)"
    await eng.reconcile()
    assert SYM not in eng.positions and not acc.positions[SYM].qty
    assert await acc.open_algo_orders() == [] and await acc.open_orders() == []
