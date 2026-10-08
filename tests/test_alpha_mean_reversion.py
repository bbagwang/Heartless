"""mean_reversion: multi-hour stretch + first 1h exhaustion bar. Signal geometry and gating on stub views."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.mean_reversion import MeanReversion
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


def closes(now: float, past: float) -> list:
    """close[k] for k = 0..8: the bar 8 hours back closed at `past` (default lookback 8)."""
    return [now] + [past] * 8


def long_bar(**over) -> StubCursor:
    # sold off from 105 to 100 in 8h (5 ATR), current bar closes up (open 99.5 -> close 100)
    s = dict(close=closes(100.0, 105.0), open=99.5, atr=1.0)
    s.update(over)
    return StubCursor(**s)


def short_bar(**over) -> StubCursor:
    # ripped from 95 to 100 in 8h (5 ATR), current bar closes down (open 100.5 -> close 100)
    s = dict(close=closes(100.0, 95.0), open=100.5, atr=1.0)
    s.update(over)
    return StubCursor(**s)


def ctx(price: float = 100.0, regime: Regime = Regime.TREND_DOWN) -> Context:
    info = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)
    return Context(symbol=SYM, info=info, ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price),
                   regime=regime)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["mean_reversion"])
    p.update(over)
    return p


def run(h1: StubCursor, p: dict | None = None, closed: tuple = ("1m", "5m", "15m", "1h"), regime=Regime.TREND_DOWN):
    return MeanReversion().evaluate(StubView(h1, closed), ctx(regime=regime), p or params())


def test_registered_with_defaults_and_disabled():
    a = ALPHA_BY_NAME["mean_reversion"]
    assert a.timeframe == "1h" and a.enabled_by_default is False
    p = params()
    assert p["lookback"] == 8 and p["stretch_atr"] == 4.5 and p["sl_atr"] == 2.0 and p["tp_atr"] == 2.0
    assert len(a.param_specs) <= 8


def test_long_geometry_after_selloff():
    sig = run(long_bar())
    assert sig is not None and sig.side is Side.LONG
    entry = sig.tags["ref_price"]
    assert entry == 100.0
    assert sig.stop < entry < sig.take_profit
    assert abs((entry - sig.stop) - 2.0) < 1e-9 and abs((sig.take_profit - entry) - 2.0) < 1e-9
    assert sig.tp1 is None and sig.entry_style is EntryStyle.MARKET and sig.timeframe == "1h"
    assert sig.max_hold_bars == 12 * 60 and sig.confidence >= 0.55
    assert sig.reason and "롱" in sig.reason


def test_short_geometry_after_rally():
    sig = run(short_bar())
    assert sig is not None and sig.side is Side.SHORT
    entry = sig.tags["ref_price"]
    assert sig.take_profit < entry < sig.stop
    assert abs((sig.stop - entry) - 2.0) < 1e-9 and abs((entry - sig.take_profit) - 2.0) < 1e-9
    assert sig.reason and "숏" in sig.reason


def test_maker_entry_quotes_the_touch_on_the_passive_side():
    sig = run(long_bar(), params(maker_entry=1))
    assert sig.entry_style is EntryStyle.LIMIT and sig.limit_price < 100.0 and sig.stop < sig.limit_price < sig.take_profit
    sig = run(short_bar(), params(maker_entry=1))
    assert sig.entry_style is EntryStyle.LIMIT and sig.limit_price > 100.0 and sig.take_profit < sig.limit_price < sig.stop


def test_requires_exhaustion_bar_and_enough_stretch():
    assert run(long_bar(open=100.5)) is None  # still falling: the bar closed down
    assert run(short_bar(open=99.5)) is None  # still rising
    assert run(long_bar(close=closes(100.0, 104.0))) is None  # only 4 ATR < 4.5
    assert run(short_bar(close=closes(100.0, 96.0))) is None


def test_gating_on_timeframe_and_missing_data():
    assert run(long_bar(), closed=("1m", "5m", "15m")) is None
    assert run(long_bar(atr=NAN)) is None
    assert run(long_bar(close=[100.0])) is None  # no history 8 bars back
    assert run(long_bar(atr=0.0)) is None


def test_fires_in_every_regime():
    for regime in Regime:
        sig = run(long_bar(), regime=regime)
        assert sig is not None and sig.confidence * MeanReversion.regime_affinity[regime] >= 0.55
