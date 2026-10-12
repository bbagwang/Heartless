"""E1 execution opt-ins (Signal.time_exit / allow_renewal / exit_on_opposite / cooldown_bars / avoid_funding), driven
through the paper simulator. Every opt-in defaults to the previous behaviour, which each test also pins as the control.
"""
from __future__ import annotations

import json

import pytest
from synth import synth_symbols

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import Decision, EntryStyle, Position, PositionStatus, Regime, Side, Signal, Ticker
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.strategy.base import Alpha, Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_HOUR, MS_MINUTE

SYM = "BTCUSDT"
T0 = 1_700_000_000_000
ALPHA = "htf_trend"  # a registered parameter set; the alpha logic itself is replaced by Scripted below


class Clock:
    def __init__(self, t: int = T0):
        self.t = t

    def __call__(self) -> int:
        return self.t


class Cursor:
    ok = False

    def v(self, name: str, k: int = 0) -> float:
        return float("nan")


class View:
    """Enough MarketView for management and a scripted decision: `closed` lists the timeframes closing on this bar."""
    frames: dict = {}

    def __init__(self, price: float, closed: tuple[str, ...] = ()):
        self.price = price
        self._closed = set(closed)

    def tf(self, name: str) -> Cursor:
        return Cursor()

    def ready(self) -> bool:
        return True

    def closed(self, tf: str) -> bool:
        return tf in self._closed


class Scripted(Alpha):
    """An alpha whose next signal is set by the test (None = no signal)."""
    name = ALPHA
    timeframe = "1h"
    param_specs = []

    def __init__(self, name: str = ALPHA):
        self.name = name
        self.next: Signal | None = None
        self.calls = 0

    def evaluate(self, view, ctx, p):
        self.calls += 1
        return self.next


def signal(side: Side = Side.LONG, price: float = 100.0, hold: int = 5, alpha: str = ALPHA, **optins) -> Signal:
    stop = price - side.sign * 2.0
    return Signal(alpha=alpha, symbol=SYM, side=side, confidence=0.9, reason="t", stop=stop, take_profit=None, tp1=None,
                  entry_style=EntryStyle.MARKET, max_hold_bars=hold, atr=2.0, timeframe="1h",
                  tags={"ref_price": price}, **optins)


def make(settings: Settings | None = None):
    clock = Clock()
    acc = PaperAccount(name="p", initial_balance=10_000.0, clock=clock)
    syms = synth_symbols([SYM])
    acc.set_symbols(syms)
    params = StrategyParams.default()
    params.enabled[ALPHA] = True
    params.enabled["other_alpha"] = True
    eng = TradingEngine("paper", acc, params, settings or Settings(_env_file=None), syms, bus=EventBus(), clock=clock,
                        persist=False)
    alpha = Scripted()
    eng.ensemble.alphas = [alpha]
    return eng, acc, clock, alpha


def book(price: float, ts: int) -> Ticker:
    return Ticker(SYM, bid=price * (1 - 1e-4), ask=price * (1 + 1e-4), mark=price, last=price, ts=ts)


def context(eng, tk: Ticker, mtf: float = 999.0, funding: float = 0.0) -> Context:
    return Context(symbol=SYM, info=eng.symbols[SYM], ticker=tk, regime=Regime.RANGE, now=tk.ts,
                   funding_rate=funding, minutes_to_funding=mtf)


async def enter(eng, acc, sig: Signal, price: float = 100.0) -> Position:
    tk = book(price, eng.clock())
    await acc.on_ticker(tk)
    d = Decision(symbol=SYM, side=sig.side, score=0.9, confidence=0.9, alphas=[sig.alpha], primary=sig, reason="t",
                 regime=Regime.RANGE)
    await eng._open(d, View(price), context(eng, tk), await eng.equity())
    pos = eng.positions[SYM]
    assert pos.status is PositionStatus.OPEN
    return pos


async def bar(eng, acc, clock, price: float, closed: tuple[str, ...] = (), **ctx_kw) -> None:
    clock.t += MS_MINUTE
    tk = book(price, clock())
    await acc.on_ticker(tk)
    await eng.on_bar(View(price, closed), context(eng, tk, **ctx_kw))


# --- time_exit -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("time_exit", [False, True])
async def test_time_exit_closes_a_winner_at_the_deadline_instead_of_trailing_it(time_exit):
    eng, acc, clock, _ = make()
    pos = await enter(eng, acc, signal(hold=3, time_exit=time_exit))
    assert pos.time_exit is time_exit
    for _ in range(3):
        await bar(eng, acc, clock, 101.5)  # +0.75R at the deadline
    if time_exit:
        assert SYM not in eng.positions and eng.closed[-1].exit_reason.startswith("보유 기간 만료")
    else:  # default: a working trade past its deadline keeps running at breakeven under a trail
        assert eng.positions[SYM] is pos and pos.be_moved and pos.trail_atr_mult == 1.5


# --- allow_renewal -------------------------------------------------------------------------------------------------

async def test_renewal_re_arms_the_time_stop_on_a_same_side_signal_of_the_same_alpha():
    eng, acc, clock, alpha = make()
    pos = await enter(eng, acc, signal(hold=5, allow_renewal=True))
    await bar(eng, acc, clock, 100.0)
    await bar(eng, acc, clock, 100.0)
    alpha.next = signal(hold=5)  # the rule still holds at the alpha's next decision bar (bar 3)
    await bar(eng, acc, clock, 100.0, closed=("1h",))
    assert pos.max_hold_bars == 3 + 5 and pos.extra["renewals"] == 1
    alpha.next = None
    for _ in range(4):  # bars 4..7: past the original deadline (5), not working (r < 0.5), still held
        await bar(eng, acc, clock, 100.0)
    assert eng.positions.get(SYM) is pos
    await bar(eng, acc, clock, 100.0)  # bar 8: the renewed deadline
    assert SYM not in eng.positions and eng.closed[-1].exit_reason == "시간 초과 청산(time stop)"


async def test_without_the_opt_in_a_same_side_signal_neither_renews_nor_is_evaluated():
    eng, acc, clock, alpha = make()
    pos = await enter(eng, acc, signal(hold=5))
    alpha.next = signal(hold=5)
    for i in range(5):
        await bar(eng, acc, clock, 100.0, closed=("1h",) if i == 2 else ())
    assert alpha.calls == 0  # the alpha is not asked while its position is open (previous behaviour)
    assert SYM not in eng.positions and pos.max_hold_bars == 5


async def test_renewal_needs_the_alpha_decision_bar_and_its_own_alpha():
    eng, acc, clock, alpha = make()
    pos = await enter(eng, acc, signal(hold=5, allow_renewal=True))
    alpha.next = signal(hold=5)
    await bar(eng, acc, clock, 100.0)  # the 1h frame did not close: the alpha is not asked
    assert alpha.calls == 0 and pos.max_hold_bars == 5
    other = Scripted("other_alpha")
    other.next = signal(hold=50, alpha="other_alpha")
    eng.ensemble.alphas = [other, alpha]
    alpha.next = None
    await bar(eng, acc, clock, 100.0, closed=("1h",))  # another alpha's signal never renews this position
    assert other.calls == 0 and alpha.calls == 1 and pos.max_hold_bars == 5


async def test_no_renewal_while_new_entries_are_blocked():
    eng, acc, clock, alpha = make()
    pos = await enter(eng, acc, signal(hold=3, allow_renewal=True))
    eng.risk.pause("owner")
    alpha.next = signal(hold=5)
    await bar(eng, acc, clock, 100.0, closed=("1h",))
    assert pos.max_hold_bars == 3 and "renewals" not in pos.extra
    await bar(eng, acc, clock, 100.0)
    await bar(eng, acc, clock, 100.0)
    assert SYM not in eng.positions


# --- exit_on_opposite ----------------------------------------------------------------------------------------------

async def test_opposite_signal_closes_and_the_same_bar_may_reverse_with_no_cooldown():
    eng, acc, clock, alpha = make()
    pos = await enter(eng, acc, signal(hold=50, exit_on_opposite=True, cooldown_bars=0))
    await bar(eng, acc, clock, 99.0)
    alpha.next = signal(Side.SHORT, price=99.0, hold=50)
    await bar(eng, acc, clock, 99.0, closed=("1h",))
    trade = eng.closed[-1]
    assert trade.exit_reason == "반대 신호 청산" and trade.pnl < 0 and pos.status is PositionStatus.CLOSED
    new = eng.positions.get(SYM)
    assert new is not None and new is not pos and new.side is Side.SHORT  # decided on the very bar of the exit
    assert eng.risk.state.symbol_cooldown.get(SYM, 0) <= clock()


async def test_opposite_exit_with_the_default_cooldown_blocks_the_reversal_for_20_minutes():
    eng, acc, clock, alpha = make()
    await enter(eng, acc, signal(hold=50, exit_on_opposite=True))
    alpha.next = signal(Side.SHORT, price=99.0, hold=50)
    await bar(eng, acc, clock, 99.0, closed=("1h",))
    assert eng.closed[-1].exit_reason == "반대 신호 청산" and SYM not in eng.positions
    assert eng.risk.state.symbol_cooldown[SYM] == clock() + 20 * MS_MINUTE


async def test_opposite_signal_is_ignored_without_the_opt_in():
    eng, acc, clock, alpha = make()
    pos = await enter(eng, acc, signal(hold=50, allow_renewal=True))
    alpha.next = signal(Side.SHORT, price=100.0, hold=50)
    await bar(eng, acc, clock, 100.0, closed=("1h",))
    assert eng.positions.get(SYM) is pos and pos.status is PositionStatus.OPEN


# --- cooldown_bars -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bars,expected", [(None, 20 * MS_MINUTE), (2, 2 * MS_HOUR), (0, None)])
async def test_loss_cooldown_in_bars_of_the_alpha_timeframe(bars, expected):
    eng, acc, clock, _ = make()
    pos = await enter(eng, acc, signal(hold=50, cooldown_bars=bars))
    assert pos.cooldown_ms == (None if bars is None else bars * MS_HOUR)
    await bar(eng, acc, clock, 97.5)  # through the stop at 98
    trade = eng.closed[-1]
    assert trade.pnl < 0
    until = eng.risk.state.symbol_cooldown.get(SYM)
    if expected is None:
        assert until is None
    else:
        assert until == trade.exit_time + expected


def test_cooldown_counts_bars_of_the_alpha_decision_timeframe_not_the_signal_one():
    eng, _, _, alpha = make()
    alpha.timeframe = "4h"  # e.g. vol_breakout_dbox: 4h decisions, daily-ATR signal timeframe
    assert eng._cooldown_ms(ALPHA, 1, "1d") == 4 * MS_HOUR
    assert eng._cooldown_ms("adopted", 1, "1d") == 24 * MS_HOUR  # unknown alpha: the signal's timeframe
    assert eng._cooldown_ms(ALPHA, None, "1d") is None


# --- avoid_funding -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("avoid", [True, False])
async def test_funding_avoidance_exit_is_an_opt_out(avoid):
    eng, acc, clock, _ = make()
    pos = await enter(eng, acc, signal(hold=500, avoid_funding=avoid))
    await bar(eng, acc, clock, 100.0, mtf=1.5, funding=0.001)  # a long about to pay 0.1%, not working
    if avoid:
        assert SYM not in eng.positions and eng.closed[-1].exit_reason.startswith("펀딩 회피 청산")
    else:
        assert eng.positions.get(SYM) is pos


# --- persistence ---------------------------------------------------------------------------------------------------

async def test_opt_ins_survive_persistence_and_legacy_rows_load_the_previous_behaviour():
    eng, acc, _, _ = make()
    pos = await enter(eng, acc, signal(hold=5, time_exit=True, allow_renewal=True, exit_on_opposite=True,
                                       cooldown_bars=3, avoid_funding=False))
    row = json.loads(json.dumps(pos.to_row(), default=str))
    back = Position.from_row(row)
    assert (back.time_exit, back.allow_renewal, back.exit_on_opposite, back.cooldown_ms, back.avoid_funding) == \
        (True, True, True, 3 * MS_HOUR, False)
    for k in ("time_exit", "allow_renewal", "exit_on_opposite", "cooldown_ms", "avoid_funding"):
        row.pop(k)
    legacy = Position.from_row(row)
    assert (legacy.time_exit, legacy.allow_renewal, legacy.exit_on_opposite, legacy.cooldown_ms, legacy.avoid_funding) == \
        (False, False, False, None, True)


def test_signal_defaults_are_the_previous_behaviour():
    s = signal()
    assert (s.time_exit, s.allow_renewal, s.exit_on_opposite, s.cooldown_bars, s.avoid_funding) == \
        (False, False, False, None, True)


def test_no_shipped_alpha_opts_in():
    """The evidence of every shipped alpha was measured on the engine without these opt-ins: none may set one (a new
    behaviour needs its own research run), checked on the alpha sources."""
    import inspect

    from heartless.strategy.alphas import ALL_ALPHAS

    for a in ALL_ALPHAS:
        src = inspect.getsource(type(a))
        for k in ("time_exit", "allow_renewal", "exit_on_opposite", "cooldown_bars", "avoid_funding"):
            assert f"{k}=" not in src, (a.name, k)
