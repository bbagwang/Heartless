"""squeeze_breakout: 1h squeeze -> decisive Donchian breakout. Signal geometry and gating on stub views."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.squeeze_breakout import SqueezeBreakout
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

NAN = float("nan")
SYM = "ETHUSDT"


class StubCursor:
    """``v(name, k)`` reads series[k] (k bars back) or a scalar; missing -> nan."""

    ok = True

    def __init__(self, **series):
        self.series = series

    def v(self, name: str, k: int = 0) -> float:
        s = self.series.get(name)
        if s is None:
            return NAN
        if isinstance(s, (list, tuple)):
            return float(s[k]) if k < len(s) else NAN
        return float(s)


class StubView:
    def __init__(self, h1: StubCursor, closed: tuple = ("1m", "5m", "15m", "1h")):
        self._h1 = h1
        self._closed = closed

    def tf(self, tf: str) -> StubCursor:
        return self._h1 if tf == "1h" else StubCursor()

    def closed(self, tf: str) -> bool:
        return tf in self._closed


def long_bar(**over) -> StubCursor:
    # 6 bars of squeeze, prior 20-bar high 101, previous close 100.4 (inside), breakout bar 100.4 -> 102.0
    s = dict(close=[102.0, 100.4], open=100.5, high=102.2, low=100.3, atr=1.0, dc_hi20=[101.0, 101.0],
             dc_lo20=[97.0, 97.0], squeeze_bars=[0, 6], ema50=99.0, ema200=98.0, vol_z=1.5)
    s.update(over)
    return StubCursor(**s)


def short_bar(**over) -> StubCursor:
    s = dict(close=[98.0, 99.6], open=99.5, high=99.7, low=97.8, atr=1.0, dc_hi20=[103.0, 103.0],
             dc_lo20=[99.0, 99.0], squeeze_bars=[0, 6], ema50=101.0, ema200=102.0, vol_z=0.2)
    s.update(over)
    return StubCursor(**s)


def ctx(price: float = 100.0, regime: Regime = Regime.RANGE) -> Context:
    info = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)
    return Context(symbol=SYM, info=info, ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1),
                   regime=regime)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["squeeze_breakout"])
    p.update(over)
    return p


def test_registered_on_1h_with_bounded_defaults():
    a = ALPHA_BY_NAME["squeeze_breakout"]
    assert isinstance(a, SqueezeBreakout) and a.timeframe == "1h"
    assert len(a.param_specs) <= 8
    for s in a.param_specs:
        assert s.clip(s.default) == s.default


def test_long_geometry():
    p = params()
    sig = SqueezeBreakout().evaluate(StubView(long_bar()), ctx(102.0), p)
    assert sig is not None and sig.side is Side.LONG
    entry = sig.tags["ref_price"]
    assert entry == 102.0
    assert sig.stop < entry < sig.take_profit
    assert abs((entry - sig.stop) - p["sl_atr"] * 1.0) < 1e-9
    assert abs((sig.take_profit - entry) - p["tp_atr"] * 1.0) < 1e-9
    assert sig.tp1 is None  # no partial target: partials cut the winners this edge depends on
    assert sig.entry_style is EntryStyle.MARKET and sig.timeframe == "1h"
    assert sig.max_hold_bars == int(p["max_hold"]) * 60
    assert 0.55 <= sig.confidence <= 0.98 and sig.reason


def test_short_geometry_mirrors_long():
    p = params()
    sig = SqueezeBreakout().evaluate(StubView(short_bar()), ctx(98.0), p)
    assert sig is not None and sig.side is Side.SHORT
    entry = sig.tags["ref_price"]
    assert sig.take_profit < entry < sig.stop
    assert abs((sig.stop - entry) - p["sl_atr"] * 1.0) < 1e-9
    assert abs((entry - sig.take_profit) - p["tp_atr"] * 1.0) < 1e-9 and sig.tp1 is None


def test_trend_filter_blocks_counter_trend_breakouts():
    a = SqueezeBreakout()
    counter = long_bar(ema50=97.0, ema200=98.0)  # 1h EMA50 below EMA200: a long breakout against the trend
    assert a.evaluate(StubView(counter), ctx(), params(trend_align=1)) is None
    sig = a.evaluate(StubView(counter), ctx(), params(trend_align=0))
    assert sig is not None and sig.side is Side.LONG and sig.stop < sig.tags["ref_price"] < sig.take_profit
    assert a.evaluate(StubView(short_bar(ema50=103.0, ema200=102.0)), ctx(), params(trend_align=1)) is None


def test_gates_reject_non_setups():
    a, p = SqueezeBreakout(), params()
    assert a.evaluate(StubView(long_bar(squeeze_bars=[0, 2])), ctx(), p) is None  # no squeeze before the break
    assert a.evaluate(StubView(long_bar(close=[101.2, 100.4])), ctx(), p) is None  # move < 1 ATR
    assert a.evaluate(StubView(long_bar(high=103.5, low=99.0)), ctx(), p) is None  # wicky bar, body < 50%
    stale = long_bar(close=[103.0, 101.5], open=101.6, high=103.1, low=101.5)  # strong bar, but prev close already out
    assert a.evaluate(StubView(stale), ctx(), p) is None
    assert a.evaluate(StubView(long_bar(), closed=("1m", "5m", "15m")), ctx(), p) is None  # 1h bar not closed
    assert a.evaluate(StubView(long_bar(atr=NAN)), ctx(), p) is None  # warm-up
