"""Regression tests for trend_pullback: NaN higher-timeframe EMAs must not be read as a downtrend.

Before the fix ``up_15 = m15.v("ema21") > m15.v("ema50")`` had no ``ok()`` guard. ``nan > nan`` is False, so while
the 15m EMA50 (50 bars = 12.5h) and 1h EMA50 (~50h) were still warming up the alpha saw a *bearish* alignment: it
could emit unconfirmed SHORTs and could never emit a LONG. MarketView.ready() only needs 30 completed 15m bars, so
the window is real for newly listed universe symbols and for the start of every backtest/research window.
"""
from __future__ import annotations

import math

from synth import synth_symbols

from heartless.core.models import Regime, Side, Ticker
from heartless.strategy.alphas.trend_pullback import TrendPullback
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

NAN = float("nan")
SYM = "ETHUSDT"


class StubCursor:
    """Minimal FrameCursor: ``v(name, k)`` reads series[k] (k bars back) or a scalar; missing -> nan."""

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
    def __init__(self, m5: StubCursor, m15: StubCursor, h1: StubCursor):
        self._tf = {"5m": m5, "15m": m15, "1h": h1}

    def tf(self, tf: str) -> StubCursor:
        return self._tf[tf]

    def closed(self, tf: str) -> bool:
        return tf == "5m"


# ---- textbook 5m setups ------------------------------------------------------------------------
# SHORT: 5m EMA21 < EMA50, pullback highs tag EMA21 zone, RSI bounced >= 55 then rolled over, red resumption bar.
def short_5m() -> StubCursor:
    return StubCursor(close=96.5, open=97.5, atr=1.0, ema9=97.0, ema21=97.8, ema50=100.0,
                      rsi14=[48, 56, 58, 52], high=[97.6, 98.2, 98.5, 98.0], low=[96.3, 96.8, 97.2, 97.0],
                      taker_ratio3=0.40)


# LONG: mirror image of the short setup around 100.
def long_5m() -> StubCursor:
    return StubCursor(close=103.5, open=102.5, atr=1.0, ema9=103.0, ema21=102.2, ema50=100.0,
                      rsi14=[52, 44, 42, 48], high=[103.7, 103.2, 102.8, 103.0], low=[102.4, 101.8, 101.5, 102.0],
                      taker_ratio3=0.60)


def m15(ema21: float, ema50: float, adx: float = 25.0) -> StubCursor:
    return StubCursor(adx=adx, ema21=ema21, ema50=ema50)


def h1(ema21: float, ema50: float) -> StubCursor:
    return StubCursor(ema21=ema21, ema50=ema50)


def ctx(price: float) -> Context:
    return Context(symbol=SYM, info=synth_symbols([SYM])[SYM],
                   ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1), regime=Regime.RANGE)


def params() -> dict:
    return StrategyParams.default().alphas["trend_pullback"]


def evaluate(m5: StubCursor, m15c: StubCursor, h1c: StubCursor):
    return TrendPullback().evaluate(StubView(m5, m15c, h1c), ctx(m5.v("close")), params())


# ---- sanity: the setups really do fire once the higher timeframes are aligned ---------------------
def test_short_setup_fires_with_bearish_htf():
    sig = evaluate(short_5m(), m15(98.0, 100.0), h1(98.0, 100.0))
    assert sig is not None and sig.side is Side.SHORT
    assert sig.stop > sig.limit_price > sig.take_profit  # protective stop above, TP below


def test_long_setup_fires_with_bullish_htf():
    sig = evaluate(long_5m(), m15(102.0, 100.0), h1(102.0, 100.0))
    assert sig is not None and sig.side is Side.LONG
    assert sig.stop < sig.limit_price < sig.take_profit


# ---- the defect: 15m/1h EMAs still NaN (warm-up) ------------------------------------------------
def test_short_setup_with_nan_15m_and_1h_emas_returns_none():
    # Pre-fix this returned Signal(SHORT, conf 0.60): nan > nan == False read as "15m downtrend".
    assert evaluate(short_5m(), m15(NAN, NAN), h1(NAN, NAN)) is None


def test_long_setup_with_nan_15m_and_1h_emas_returns_none():
    assert evaluate(long_5m(), m15(NAN, NAN), h1(NAN, NAN)) is None


def test_only_15m_ema50_nan_blocks_both_sides():
    # Real warm-up shape: EMA21 is available after 21 bars but EMA50 only after 50 (12.5h of 15m bars).
    assert evaluate(short_5m(), m15(98.0, NAN), h1(NAN, NAN)) is None
    assert evaluate(long_5m(), m15(102.0, NAN), h1(NAN, NAN)) is None


def test_nan_15m_does_not_create_long_short_asymmetry():
    # Both sides must be treated identically while the 15m trend is unknown.
    s = evaluate(short_5m(), m15(NAN, NAN), h1(98.0, 100.0))
    l = evaluate(long_5m(), m15(NAN, NAN), h1(102.0, 100.0))
    assert s is None and l is None


# ---- preserved behaviour: 1h NaN still falls back to the 15m alignment --------------------------
def test_nan_1h_falls_back_to_15m_alignment():
    long_sig = evaluate(long_5m(), m15(102.0, 100.0), h1(NAN, NAN))
    assert long_sig is not None and long_sig.side is Side.LONG
    short_sig = evaluate(short_5m(), m15(98.0, 100.0), h1(NAN, NAN))
    assert short_sig is not None and short_sig.side is Side.SHORT
    # but a 15m alignment against the setup still blocks it
    assert evaluate(short_5m(), m15(102.0, 100.0), h1(NAN, NAN)) is None
    assert evaluate(long_5m(), m15(98.0, 100.0), h1(NAN, NAN)) is None


def test_htf_disagreement_still_blocks():
    # 15m up but 1h down (or vice versa) must not trade either side.
    assert evaluate(long_5m(), m15(102.0, 100.0), h1(98.0, 100.0)) is None
    assert evaluate(short_5m(), m15(98.0, 100.0), h1(102.0, 100.0)) is None


def test_nan_guard_does_not_swallow_non_nan_values():
    # ok() guard must only trip on NaN/None, not on legitimate zero/negative-looking values.
    assert not math.isnan(m15(98.0, 100.0).v("ema50"))
    sig = evaluate(short_5m(), m15(0.0, 0.5), h1(0.0, 0.5))
    assert sig is not None and sig.side is Side.SHORT
