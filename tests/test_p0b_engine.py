"""P0b engine correctness (docs/ROADMAP.md, docs/RESEARCH.md section 9), driven through the paper simulator:

* a re-quoted post-only entry is re-sized for its new price, so a chased fill never risks more than the trade's
  budget at its stop, never grows, and is cancelled when what fits is below the exchange minimum,
* partially filled entries: the filled part counts against the budget, the remainder is the re-sized order,
* Signal.entry_ttl_bars: a resting retest limit that is never moved and expires after N bars of its timeframe,
* the regime-flip exit is a per-signal opt-in (not an alpha name) and survives persistence and restart.
"""
from __future__ import annotations

import pytest
from synth import synth_symbols

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import (Decision, EntryStyle, Position, PositionStatus, Regime, Side, Signal, SymbolInfo,
                                   Ticker)
from heartless.core.store import Store
from heartless.exchange.base import PositionSnapshot
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_HOUR, MS_MINUTE
from tests.test_fix_2_engine import FakeLive

SYM = "BTCUSDT"
T0 = 1_700_000_000_000
NAN = float("nan")
BUDGET = 10_000 * 0.5 / 100  # default RISK_PER_TRADE_PCT on the default 10k paper account
COST = 2 * 0.0005 + 0.0003  # sizing cost per unit of price: round-trip taker fees + slippage allowance


class Clock:
    def __init__(self, t: int = T0):
        self.t = t

    def __call__(self) -> int:
        return self.t

    def advance(self, ms: int) -> None:
        self.t += ms


class Cursor:
    def __init__(self, ok: bool = False, **vals):
        self.ok = ok
        self.vals = vals

    def v(self, name: str, k: int = 0) -> float:
        return float(self.vals.get(name, NAN))


class View:
    """Just enough MarketView for position management; `m15` feeds the regime-flip check."""
    frames: dict = {}

    def __init__(self, price: float, m15: Cursor | None = None):
        self.price = price
        self._m15 = m15

    def tf(self, name: str) -> Cursor:
        return self._m15 if name == "15m" and self._m15 is not None else Cursor()

    def ready(self) -> bool:
        return True

    def closed(self, tf: str) -> bool:
        return False  # no alpha decides here: the tests drive entries explicitly


class PartialPaper(PaperAccount):
    """PaperAccount whose next resting-order execution fills only `partial` of the remaining quantity. Each order
    executes at most once per tick (the simulator re-checks orders after the engine placed brackets)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.partial: float | None = None
        self._executed: set[str] = set()

    async def on_ticker(self, t: Ticker) -> None:
        self._executed = set()
        await super().on_ticker(t)

    async def _fill_order(self, o, px, maker, ts=None):
        if o.order_id in self._executed:
            return
        self._executed.add(o.order_id)
        if self.partial and not o.reduce_only:
            qty = round((o.qty - o.filled_qty) * self.partial, 3)
            self.partial = None
            await self._execute(o.symbol, o.side, qty, px, maker=maker, client_id=o.client_id, order_id=o.order_id,
                                reduce_only=False, ts=ts)
            o.avg_price = (o.avg_price * o.filled_qty + px * qty) / (o.filled_qty + qty)
            o.filled_qty += qty
            o.status = "PARTIALLY_FILLED"
            return
        await super()._fill_order(o, px, maker, ts)


def make(balance: float = 10_000.0, info: SymbolInfo | None = None, store: Store | None = None, acc=None,
         clock: Clock | None = None):
    clock = clock or Clock()
    acc = acc if acc is not None else PaperAccount(name="p", initial_balance=balance, clock=clock)
    syms = {SYM: info or synth_symbols([SYM])[SYM]}
    acc.set_symbols(syms)
    eng = TradingEngine("paper", acc, StrategyParams.default(), Settings(_env_file=None), syms, store=store,
                        bus=EventBus(), clock=clock, persist=store is not None)
    return eng, acc, clock


def book(price: float, ts: int) -> Ticker:
    return Ticker(SYM, bid=price * (1 - 1e-4), ask=price * (1 + 1e-4), mark=price, last=price, ts=ts)


def ctx(eng: TradingEngine, tk: Ticker) -> Context:
    return Context(symbol=SYM, info=eng.symbols[SYM], ticker=tk, regime=Regime.RANGE, now=tk.ts)


def decision(limit: float = 100.0, stop: float = 98.0, side: Side = Side.LONG, ttl: int | None = None,
             regime_exit: bool = False, tf: str = "1h", alpha: str = "htf_trend",
             style: EntryStyle = EntryStyle.LIMIT) -> Decision:
    sig = Signal(alpha=alpha, symbol=SYM, side=side, confidence=0.7, reason="t", stop=stop, take_profit=None, tp1=None,
                 entry_style=style, limit_price=limit if style is EntryStyle.LIMIT else None, atr=2.0, timeframe=tf,
                 tags={"ref_price": limit}, exit_on_regime_change=regime_exit, entry_ttl_bars=ttl)
    return Decision(symbol=SYM, side=side, score=0.7, confidence=0.7, alphas=[alpha], primary=sig, reason="t",
                    regime=Regime.RANGE, exit_on_regime_change=regime_exit, entry_ttl_bars=ttl)


async def enter(eng: TradingEngine, acc, d: Decision, price: float = 100.0) -> Position | None:
    tk = book(price, eng.clock())
    await acc.on_ticker(tk)
    await eng._open(d, View(price), ctx(eng, tk), await eng.equity())
    return eng.positions.get(SYM)


async def move(eng: TradingEngine, acc, clock: Clock, price: float, minutes: int = 1,
               m15: Cursor | None = None) -> None:
    """`minutes` 1m bars at `price`: the simulator sees each book first (fills, triggers), then the engine's bar."""
    for _ in range(minutes):
        clock.advance(MS_MINUTE)
        tk = book(price, clock())
        await acc.on_ticker(tk)
        await eng.on_bar(View(price, m15), ctx(eng, tk))


def events(eng: TradingEngine, topic: str) -> list:
    return [p for _, t, p in eng.bus.history if t == topic]


def stop_risk(qty: float, entry: float, stop: float) -> float:
    """The sizing rule's stop-out loss (incl. the fee/slippage allowance) for qty bought at entry."""
    return qty * (abs(entry - stop) + entry * COST)


async def working(acc) -> list[dict]:
    return await acc.open_orders(SYM)


# --- post-only re-quote resizing --------------------------------------------------------------------------------

async def test_requote_shrinks_the_order_so_a_chased_fill_stays_within_the_risk_budget():
    eng, acc, clock = make()
    pos = await enter(eng, acc, decision(limit=100.0, stop=98.0))
    q0 = pos.original_qty
    assert pos.status is PositionStatus.PENDING and stop_risk(q0, 100.0, 98.0) <= BUDGET
    await move(eng, acc, clock, 101.0, minutes=2)  # never trades down to 100: re-quoted at the new bid
    price = pos.limit_price
    assert pos.requotes == 1 and price == pytest.approx(100.98)
    q1 = pos.original_qty
    assert q1 < q0
    assert stop_risk(q1, price, 98.0) <= BUDGET + 1e-9 < stop_risk(q1 + 0.001, price, 98.0)  # largest qty that fits
    assert pos.risk_amount == pytest.approx(stop_risk(q1, price, 98.0))
    assert pos.notional == pytest.approx(q1 * price)
    rest = await working(acc)
    assert len(rest) == 1 and rest[0]["qty"] == pytest.approx(q1) and rest[0]["price"] == pytest.approx(price)
    # the chased order fills; the position is the re-sized quantity and its stop-out stays inside the budget
    await move(eng, acc, clock, 100.9)
    assert pos.status is PositionStatus.OPEN and pos.qty == pytest.approx(q1)
    assert pos.entry_price == pytest.approx(price)
    assert pos.risk_amount <= BUDGET + 1e-9
    await move(eng, acc, clock, 98.0)  # stopped out at the structural stop
    trade = eng.closed[-1]
    assert SYM not in eng.positions and trade.exit_reason == "손절(SL)"
    loss = -trade.pnl
    assert 0.9 * BUDGET < loss <= BUDGET
    # without the re-sizing the original quantity would have lost ~1.45x the budget on the same path
    assert loss * q0 / q1 > 1.4 * BUDGET


async def test_requote_never_grows_the_order_when_the_new_price_is_closer_to_the_stop():
    eng, acc, clock = make()
    pos = await enter(eng, acc, decision(limit=100.0, stop=98.0))
    q0 = pos.original_qty
    await move(eng, acc, clock, 99.995, minutes=2)  # the ask stays above 100 (no fill), the bid dips below it
    assert pos.requotes == 1 and pos.limit_price == pytest.approx(99.98)
    assert pos.original_qty == q0  # the budget would allow more, but a re-quote only ever shrinks an entry
    rest = await working(acc)
    assert len(rest) == 1 and rest[0]["qty"] == q0


async def test_requote_cancels_the_entry_when_the_resized_quantity_is_below_the_exchange_minimum():
    info = SymbolInfo(SYM, "BTC", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=20.0,
                      price_precision=2, quantity_precision=3)
    eng, acc, clock = make(balance=100.0, info=info)
    pos = await enter(eng, acc, decision(limit=100.0, stop=98.0))
    budget = 100.0 * 0.5 / 100
    assert pos.original_qty == pytest.approx(0.234) and pos.original_qty * 100.0 >= 20.0
    await move(eng, acc, clock, 101.1, minutes=2)  # what fits the budget at 101.08 is ~0.155 (15.7 USDT < 20)
    assert SYM not in eng.positions and pos.status is PositionStatus.CANCELLED
    assert "최소" in pos.exit_reason and "최소" in events(eng, "entry_cancelled")[-1]["reason"]
    assert await working(acc) == []
    # the old behaviour (re-quote the full 0.234) would have risked 1.5x the budget
    assert stop_risk(0.234, 101.08, 98.0) > 1.4 * budget


async def test_resized_order_that_fills_partially_keeps_its_remainder_within_the_budget():
    eng, acc, clock = make(acc=PartialPaper(name="p", initial_balance=10_000.0))
    acc._clock = clock
    pos = await enter(eng, acc, decision(limit=100.0, stop=98.0))
    await move(eng, acc, clock, 101.0, minutes=2)
    q1, price = pos.original_qty, pos.limit_price
    acc.partial = 0.5
    await move(eng, acc, clock, 100.9)  # half of the re-sized order fills
    assert pos.status is PositionStatus.OPEN and pos.filled_qty == pytest.approx(round(q1 * 0.5, 3))
    assert pos.original_qty == q1  # the remainder is what is left of the re-sized order, not of the original one
    rest = await working(acc)
    assert len(rest) == 1 and rest[0]["qty"] - rest[0]["filled_qty"] == pytest.approx(q1 - pos.filled_qty)
    await move(eng, acc, clock, 100.9)  # the remainder fills as well
    assert pos.qty == pytest.approx(q1) and pos.filled_qty == pytest.approx(q1)
    assert pos.risk_amount == pytest.approx(stop_risk(q1, price, 98.0)) and pos.risk_amount <= BUDGET + 1e-9
    assert await working(acc) == []
    stops = [a for a in await acc.open_algo_orders(SYM) if a["kind"] == "STOP_MARKET"]
    assert len(stops) == 1 and stops[0]["close_position"]


async def test_partially_filled_entry_is_not_requoted_and_its_remainder_expires_after_two_bars():
    eng, acc, clock = make(acc=PartialPaper(name="p", initial_balance=10_000.0))
    acc._clock = clock
    pos = await enter(eng, acc, decision(limit=100.0, stop=98.0))
    q0 = pos.original_qty
    acc.partial = 0.4
    await move(eng, acc, clock, 99.99)  # the ask touches 100: 40% fills -> OPEN with a resting remainder
    filled = pos.filled_qty
    assert pos.status is PositionStatus.OPEN and filled == pytest.approx(round(q0 * 0.4, 3))
    await move(eng, acc, clock, 101.0, minutes=2)
    assert pos.requotes == 0 and await working(acc) == []  # the remainder was cancelled, not chased
    assert pos.original_qty == pytest.approx(filled) and pos.qty == pytest.approx(filled)
    assert pos.risk_amount == pytest.approx(stop_risk(filled, 100.0, 98.0)) and pos.risk_amount < BUDGET


async def test_requote_size_counts_the_filled_part_against_the_budget():
    eng, acc, clock = make(acc=PartialPaper(name="p", initial_balance=10_000.0))
    acc._clock = clock
    pos = await enter(eng, acc, decision(limit=100.0, stop=98.0))
    q0 = pos.original_qty
    acc.partial = 0.4
    await move(eng, acc, clock, 99.99)
    assert pos.status is PositionStatus.OPEN
    qty, risk = await eng._requote_size(pos, 101.0)
    assert 0 < qty < q0 - pos.filled_qty
    assert risk == pytest.approx(stop_risk(qty, 101.0, 98.0))
    total = pos.risk_amount + risk
    assert total <= BUDGET + 1e-9 < pos.risk_amount + stop_risk(qty + 0.001, 101.0, 98.0)
    # a price closer to the stop never yields more than the unfilled remainder
    qty, _ = await eng._requote_size(pos, 99.5)
    assert qty == pytest.approx(q0 - pos.filled_qty)


# --- signal entry TTL ------------------------------------------------------------------------------------------

async def test_ttl_entry_rests_unchanged_and_expires_after_its_bars():
    eng, acc, clock = make()
    pos = await enter(eng, acc, decision(limit=99.0, stop=97.0, ttl=2, tf="1h"))  # retest limit below the market
    assert pos.entry_ttl_bars == 2 and pos.status is PositionStatus.PENDING
    oid, cid, qty = pos.entry_order_id, pos.entry_client_id, pos.original_qty
    await move(eng, acc, clock, 101.0, minutes=119)  # 2 ATR away: the default policy would have abandoned it
    assert pos.status is PositionStatus.PENDING and pos.requotes == 0
    assert (pos.entry_order_id, pos.entry_client_id, pos.original_qty, pos.limit_price) == (oid, cid, qty, 99.0)
    rest = await working(acc)
    assert [o["order_id"] for o in rest] == [oid] and rest[0]["price"] == 99.0
    await move(eng, acc, clock, 101.0)  # 120 bars of 1m = 2 bars of 1h after placement
    assert SYM not in eng.positions and pos.status is PositionStatus.CANCELLED
    assert "유효 기간" in events(eng, "entry_cancelled")[-1]["reason"] and await working(acc) == []


async def test_ttl_entry_fills_on_the_retest_within_its_window():
    eng, acc, clock = make()
    pos = await enter(eng, acc, decision(limit=99.0, stop=97.0, ttl=2, tf="1h"))
    await move(eng, acc, clock, 101.0, minutes=90)
    await move(eng, acc, clock, 98.95)  # the retest trades through 99
    assert pos.status is PositionStatus.OPEN and pos.entry_price == 99.0 and pos.qty == pos.original_qty
    assert pos.risk_amount == pytest.approx(stop_risk(pos.qty, 99.0, 97.0)) and pos.risk_amount <= BUDGET


async def test_ttl_none_keeps_the_default_two_requotes_then_cancel():
    eng, acc, clock = make()
    pos = await enter(eng, acc, decision(limit=100.0, stop=98.0, ttl=None))
    assert pos.entry_ttl_bars is None
    await move(eng, acc, clock, 100.5, minutes=1)
    assert pos.requotes == 0
    await move(eng, acc, clock, 100.5, minutes=1)
    assert pos.requotes == 1 and pos.limit_price == pytest.approx(100.48)
    await move(eng, acc, clock, 100.5, minutes=2)
    assert pos.requotes == 2 and pos.status is PositionStatus.PENDING
    await move(eng, acc, clock, 100.5, minutes=2)
    assert SYM not in eng.positions and pos.exit_reason == "진입가 미도달로 주문 취소" and await working(acc) == []


async def test_ttl_is_ignored_for_market_entries():
    eng, acc, clock = make()
    d = decision(limit=100.0, stop=98.0, ttl=3, style=EntryStyle.MARKET)
    pos = await enter(eng, acc, d)
    assert pos.status is PositionStatus.OPEN and pos.entry_ttl_bars is None


async def test_ttl_partial_fill_keeps_the_remainder_working_until_the_deadline():
    eng, acc, clock = make(acc=PartialPaper(name="p", initial_balance=10_000.0))
    acc._clock = clock
    pos = await enter(eng, acc, decision(limit=99.0, stop=97.0, ttl=2, tf="1h"))
    q0 = pos.original_qty
    acc.partial = 0.5
    await move(eng, acc, clock, 98.99)  # half fills -> OPEN
    filled = pos.filled_qty
    assert pos.status is PositionStatus.OPEN and 0 < filled < q0
    await move(eng, acc, clock, 101.0, minutes=60)  # far past the default two bars: the remainder still rests
    rest = await working(acc)
    assert len(rest) == 1 and rest[0]["price"] == 99.0 and pos.original_qty == q0
    await move(eng, acc, clock, 101.0, minutes=58)  # 119 minutes since placement
    assert len(await working(acc)) == 1
    await move(eng, acc, clock, 101.0)  # the deadline: the remainder is cancelled, the filled part stays
    assert await working(acc) == [] and pos.status is PositionStatus.OPEN
    assert pos.original_qty == pytest.approx(filled) and pos.qty == pytest.approx(filled)


async def test_ttl_entry_whose_order_vanished_is_settled_at_once():
    eng, acc, clock = make()
    pos = await enter(eng, acc, decision(limit=99.0, stop=97.0, ttl=4, tf="1h"))
    acc.orders.clear()  # e.g. a paper restart: the simulator no longer holds the order
    await move(eng, acc, clock, 100.0)
    assert SYM not in eng.positions and pos.status is PositionStatus.CANCELLED and "UNKNOWN" in pos.exit_reason


async def test_ttl_entry_survives_a_restart_and_keeps_its_deadline(tmp_path):
    store = Store(tmp_path / "h.db")
    eng, acc, clock = make(store=store)
    pos = await enter(eng, acc, decision(limit=99.0, stop=97.0, ttl=2, tf="1h"))
    placed = pos.pending_since
    await move(eng, acc, clock, 101.0, minutes=60)
    # the process dies; a new engine restores the row (the simulator kept the order, like an exchange would)
    acc._fill_handlers.clear()
    acc._algo_handlers.clear()
    eng2, _, _ = make(store=store, acc=acc, clock=clock)
    await eng2.start()
    pos2 = eng2.positions[SYM]
    assert pos2 is not pos and pos2.status is PositionStatus.PENDING
    assert pos2.entry_ttl_bars == 2 and pos2.pending_since == placed and pos2.exit_on_regime_change is False
    await move(eng2, acc, clock, 101.0, minutes=59)
    assert pos2.status is PositionStatus.PENDING and len(await working(acc)) == 1
    await move(eng2, acc, clock, 101.0)
    assert SYM not in eng2.positions and await working(acc) == []
    store.close()


# --- regime-flip exit opt-in -----------------------------------------------------------------------------------

def flip_against_long() -> Cursor:
    return Cursor(ok=True, adx=40.0, slope20=-0.5)


async def opened(eng, acc, clock, d: Decision, price: float = 100.0) -> Position:
    pos = await enter(eng, acc, d, price)
    if pos.status is PositionStatus.PENDING:
        await move(eng, acc, clock, price - 0.05)
    assert pos.status is PositionStatus.OPEN
    return pos


@pytest.mark.parametrize("alpha", ["mean_reversion", "htf_trend"])
async def test_regime_flip_exit_follows_the_signal_flag_not_the_alpha_name(alpha):
    eng, acc, clock = make()
    pos = await opened(eng, acc, clock, decision(regime_exit=True, alpha=alpha))
    assert pos.exit_on_regime_change is True
    await move(eng, acc, clock, 99.5, minutes=4 - pos.bars_held % 5, m15=flip_against_long())
    assert pos.status is PositionStatus.OPEN and pos.bars_held % 5 == 4  # checked every 5th bar held
    await move(eng, acc, clock, 99.5, m15=flip_against_long())
    assert SYM not in eng.positions and eng.closed[-1].exit_reason == "레짐 전환(강한 역추세) 청산"


@pytest.mark.parametrize("alpha", ["mean_reversion", "funding_fade", "htf_trend"])
async def test_without_the_flag_no_regime_flip_exit(alpha):
    eng, acc, clock = make()
    pos = await opened(eng, acc, clock, decision(regime_exit=False, alpha=alpha))
    assert pos.exit_on_regime_change is False
    await move(eng, acc, clock, 99.5, minutes=15, m15=flip_against_long())
    assert pos.status is PositionStatus.OPEN and SYM in eng.positions


async def test_regime_flip_exit_needs_a_losing_trade_and_a_trend_against_it():
    eng, acc, clock = make()
    pos = await opened(eng, acc, clock, decision(regime_exit=True))
    await move(eng, acc, clock, 100.5, minutes=5, m15=flip_against_long())  # in profit: keep
    assert pos.status is PositionStatus.OPEN
    await move(eng, acc, clock, 99.5, minutes=5, m15=Cursor(ok=True, adx=40.0, slope20=0.5))  # trend with us
    assert pos.status is PositionStatus.OPEN


async def test_regime_flag_survives_persistence_and_restart(tmp_path):
    store = Store(tmp_path / "h.db")
    eng, acc, clock = make(store=store)
    pos = await opened(eng, acc, clock, decision(regime_exit=True, alpha="htf_trend"))
    row = store.load_open_positions("paper")[0]
    assert row.exit_on_regime_change is True and row.alpha == "htf_trend"
    acc._fill_handlers.clear()
    acc._algo_handlers.clear()
    eng2, _, _ = make(store=store, acc=acc, clock=clock)
    await eng2.start()
    pos2 = eng2.positions[SYM]
    assert pos2 is not pos and pos2.exit_on_regime_change is True
    await move(eng2, acc, clock, 99.5, minutes=5, m15=flip_against_long())
    assert SYM not in eng2.positions and eng2.closed[-1].exit_reason == "레짐 전환(강한 역추세) 청산"
    store.close()


async def test_live_restart_reconcile_keeps_flags_and_the_resting_ttl_order(tmp_path):
    """Live path: rows restored from the store go through reconcile; a resting TTL entry keeps its order (not an
    orphan), an open opted-in position keeps its flag and its stop, and both behave as before the restart."""
    store = Store(tmp_path / "h.db")
    acc = FakeLive()
    clock = Clock()
    syms = synth_symbols([SYM, "ETHUSDT"])
    acc.set_symbols(syms)
    # what the previous process left behind: a resting TTL entry on SYM and an open opted-in position on ETHUSDT
    cid = "HLELIVE00000001"
    oid = acc._new_order(SYM, "BUY", 1.0, cid, "NEW")
    pending = Position(id="P1", engine="live", symbol=SYM, side=Side.LONG, qty=0.0, entry_price=99.0, entry_time=T0,
                       stop=97.0, take_profit=None, tp1=None, initial_stop=97.0, alpha="htf_trend",
                       alphas=["htf_trend"], reason="t", confidence=0.7, regime="RANGE", risk_amount=2.13, r_unit=2.0,
                       notional=99.0, leverage=5, params_version="v", atr=2.0, timeframe="1h", entry_client_id=cid,
                       entry_order_id=oid, original_qty=1.0, entry_style="LIMIT", limit_price=99.0, pending_since=T0,
                       entry_ttl_bars=3, extra={"risk_pct": 0.5, "size_mult": 1.0})
    stop_id = await acc.place_stop("ETHUSDT", "SELL", 97.0, close_position=True)
    acc.positions["ETHUSDT"] = PositionSnapshot("ETHUSDT", 2.0, 100.0, mark=100.0)
    open_pos = Position(id="P2", engine="live", symbol="ETHUSDT", side=Side.LONG, qty=2.0, entry_price=100.0,
                        entry_time=T0, stop=97.0, take_profit=None, tp1=None, initial_stop=97.0, alpha="htf_trend",
                        alphas=["htf_trend"], reason="t", confidence=0.7, regime="RANGE", risk_amount=6.26,
                        r_unit=3.0, notional=200.0, leverage=5, params_version="v", atr=2.0, timeframe="1h",
                        status=PositionStatus.OPEN, filled_qty=2.0, original_qty=2.0, sl_algo_id=stop_id,
                        exit_on_regime_change=True, extra={"tp1_frac": 0.0})
    store.save_position(pending)
    store.save_position(open_pos)
    eng2 = TradingEngine("live", acc, StrategyParams.default(), Settings(_env_file=None), syms, store=store,
                         bus=EventBus(), clock=clock, persist=True)
    await eng2.start()  # live: restores the rows, then reconciles them against the exchange
    p1, p2 = eng2.positions[SYM], eng2.positions["ETHUSDT"]
    assert p1.status is PositionStatus.PENDING and p1.entry_ttl_bars == 3 and p1.exit_on_regime_change is False
    assert acc.orders[oid]["status"] == "NEW"  # the working TTL entry is ours, not an orphan
    assert p2.status is PositionStatus.OPEN and p2.exit_on_regime_change is True and p2.sl_algo_id == stop_id
    assert len(acc.open_stops("ETHUSDT")) == 1
    acc.calls.clear()
    # one bar later: the TTL entry is checked, not re-quoted; the opted-in position is not yet at a 5th bar
    clock.advance(MS_MINUTE)
    tk = Ticker(SYM, bid=100.99, ask=101.01, mark=101.0, ts=clock())
    await eng2.on_bar(View(101.0), ctx(eng2, tk))
    assert "limit_order" not in acc.names() and p1.status is PositionStatus.PENDING
    for i in range(5):
        clock.advance(MS_MINUTE)
        acc.price = 99.0
        tk = Ticker("ETHUSDT", bid=98.99, ask=99.01, mark=99.0, ts=clock())
        c = Context(symbol="ETHUSDT", info=syms["ETHUSDT"], ticker=tk, regime=Regime.RANGE, now=clock())
        await eng2.on_bar(View(99.0, flip_against_long()), c)
    assert "ETHUSDT" not in eng2.positions and eng2.closed[-1].exit_reason == "레짐 전환(강한 역추세) 청산"
    # the TTL entry rests 2 ATR away from the touch until 3 x 1h after its placement, then expires
    clock.t = T0 + 3 * MS_HOUR - MS_MINUTE
    tk = Ticker(SYM, bid=100.99, ask=101.01, mark=101.0, ts=clock())
    await eng2.on_bar(View(101.0), ctx(eng2, tk))
    assert p1.status is PositionStatus.PENDING and acc.orders[oid]["status"] == "NEW" and p1.requotes == 0
    clock.t = T0 + 3 * MS_HOUR
    tk = Ticker(SYM, bid=100.99, ask=101.01, mark=101.0, ts=clock())
    await eng2.on_bar(View(101.0), ctx(eng2, tk))
    assert SYM not in eng2.positions and acc.orders[oid]["status"] == "CANCELED" and "유효 기간" in p1.exit_reason
    assert "limit_order" not in acc.names()
    store.close()
