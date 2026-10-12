"""E1 open sizing (P0b review): an entry is sized for the stop its order carries (tick-rounded) and, for a market entry,
for the expected fill beyond the decision touch (modelled slippage + MARKET_ENTRY_BUFFER_BPS + one tick), so the risk
measured at the fill (TradingEngine._on_entry_fill) stays within the per-trade budget. Before, a limit entry could
realize up to ~1.006x and a market entry (next open + slippage) up to ~1.04-1.07x of the budget."""
from __future__ import annotations

import pytest
from synth import synth_candles, synth_symbols

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import Decision, EntryStyle, PositionStatus, Regime, Side, Signal, SymbolInfo, Ticker
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.learning.backtester import Backtester
from heartless.strategy.base import Alpha, Context
from heartless.strategy.params import StrategyParams

SYM = "BTCUSDT"
T0 = 1_700_000_000_000
COST = 2 * 0.0005 + 0.0003
BUDGET = 10_000 * 0.5 / 100


class View:
    frames: dict = {}

    def __init__(self, price):
        self.price = price

    def ready(self):
        return True

    def closed(self, tf):
        return False


def make(info: SymbolInfo, slippage_bps: float = 1.5):
    acc = PaperAccount(name="p", initial_balance=10_000.0, clock=lambda: T0, slippage_bps=slippage_bps,
                       impact_bps_per_10k=0.0)
    acc.set_symbols({SYM: info})
    eng = TradingEngine("paper", acc, StrategyParams.default(), Settings(_env_file=None), {SYM: info}, bus=EventBus(),
                        clock=lambda: T0, persist=False)
    return eng, acc


def decision(stop: float, style: EntryStyle, limit: float | None = None) -> Decision:
    sig = Signal(alpha="htf_trend", symbol=SYM, side=Side.LONG, confidence=0.7, reason="t", stop=stop,
                 take_profit=None, tp1=None, entry_style=style, limit_price=limit, atr=1.0, timeframe="1h",
                 max_hold_bars=60, tags={"ref_price": 100.0})
    return Decision(symbol=SYM, side=Side.LONG, score=0.7, confidence=0.7, alphas=["htf_trend"], primary=sig, reason="t",
                    regime=Regime.RANGE)


def ctx(eng, tk):
    return Context(symbol=SYM, info=eng.symbols[SYM], ticker=tk, regime=Regime.RANGE, now=T0)


async def test_limit_entry_is_sized_for_the_tick_rounded_stop():
    # a coarse tick: the signal's stop 99.24 rounds to 99.2, 0.04 further away (5% of the 0.76 distance)
    info = SymbolInfo(SYM, "BTC", "USDT", tick_size=0.1, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=1, quantity_precision=3)
    eng, acc = make(info)
    tk = Ticker(SYM, bid=100.0, ask=100.1, mark=100.05, last=100.05, ts=T0)
    await acc.on_ticker(tk)
    await eng._open(decision(99.24, EntryStyle.LIMIT, limit=100.0), View(100.0), ctx(eng, tk), 10_000.0)
    pos = eng.positions[SYM]
    assert pos.stop == pytest.approx(99.2) and pos.extra["risk_budget"] == pytest.approx(BUDGET)
    q = pos.original_qty
    risk = q * (100.0 - 99.2 + 100.0 * COST)
    assert risk <= BUDGET + 1e-9 < (q + info.step_size) * (100.0 - 99.2 + 100.0 * COST)  # the largest qty that fits
    assert pos.risk_amount == pytest.approx(risk)


async def test_market_entry_realized_risk_at_the_fill_stays_within_the_budget():
    info = synth_symbols([SYM])[SYM]
    eng, acc = make(info, slippage_bps=1.5)
    tk = Ticker(SYM, bid=99.995, ask=100.005, mark=100.0, last=100.0, ts=T0)
    await acc.on_ticker(tk)
    await eng._open(decision(99.4, EntryStyle.MARKET), View(100.0), ctx(eng, tk), 10_000.0)
    pos = eng.positions[SYM]
    assert pos.status is PositionStatus.OPEN  # the plain paper account fills at the ask + slippage
    assert pos.entry_price == pytest.approx(100.005 * 1.00015)
    assert pos.risk_amount <= BUDGET + 1e-9
    # the former sizing (raw stop, decision touch) realized more than the budget on the same fill
    q_old = BUDGET / (100.005 - 99.4 + 100.005 * COST)
    realized_old = q_old * (pos.entry_price - 99.4 + pos.entry_price * COST)
    assert realized_old > BUDGET * 1.002


async def test_post_only_that_would_cross_is_resized_for_the_touch_it_takes():
    info = synth_symbols([SYM])[SYM]
    eng, acc = make(info)
    # the retest level 100.0 is already above the ask: the post-only buy would cross and becomes a market order at the
    # ask 99.9 (+slippage), which is closer to the stop; at 100.4 (above the level) the market order is never larger
    tk = Ticker(SYM, bid=99.89, ask=99.9, mark=99.895, last=99.895, ts=T0)
    await acc.on_ticker(tk)
    await eng._open(decision(98.0, EntryStyle.LIMIT, limit=100.0), View(99.9), ctx(eng, tk), 10_000.0)
    pos = eng.positions[SYM]
    assert pos.status is PositionStatus.OPEN and pos.risk_amount <= BUDGET + 1e-9
    assert pos.qty == pytest.approx(pos.original_qty)


class _MarketEveryDay(Alpha):
    """Long at every 1h close with a tight stop: most realized-risk overshoot came from tight-stop market entries."""
    name = "htf_trend"
    timeframe = "1h"
    param_specs = []

    def evaluate(self, view, ctx, p):
        h = view.tf("1h")
        if not h.ok:
            return None
        close, atr = h.v("close"), h.v("atr")
        if not (atr > 0):
            return None
        return Signal(alpha=self.name, symbol=ctx.symbol, side=Side.LONG, confidence=0.9, reason="t",
                      stop=close - 0.8 * atr, take_profit=close + 1.0 * atr, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=120, atr=atr, timeframe="1h", tags={"ref_price": close})


def test_backtest_market_entries_never_realize_more_than_their_budget(monkeypatch):
    """The bar backtester fills a market entry at the NEXT bar's open (+ half spread + slippage + size impact): every
    trade's risk at its fill stays within the budget it was sized for."""
    import heartless.strategy.ensemble as E

    monkeypatch.setattr(E, "ALL_ALPHAS", [_MarketEveryDay()])
    s = Settings(_env_file=None)
    bt = Backtester(s, synth_symbols([SYM]), {SYM: synth_candles(6000, seed=11, vol=0.0012)})
    ratios = []
    orig = TradingEngine._finalize

    async def fin(self, pos, exit_price, ts, reason):
        if pos.filled_qty > 0:
            ratios.append(pos.risk_amount / pos.extra["risk_budget"])
        return await orig(self, pos, exit_price, ts, reason)

    monkeypatch.setattr(TradingEngine, "_finalize", fin)
    params = StrategyParams.default()
    res = bt.run(params, only_alpha="htf_trend")
    assert res.stats["n"] >= 20 and len(ratios) == res.stats["n"]
    assert max(ratios) <= 1.0 + 1e-9, max(ratios)
    assert min(ratios) > 0.9  # the buffer costs a few % of size on tight stops, not more
