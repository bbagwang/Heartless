"""Regression tests for mean_reversion: the band-midline take-profit must lie on the favourable side of entry.

Before the fix the reward filter was ``if abs(tp - entry) < 0.6 * abs(entry - stop): return None``. The trigger only
requires the *wick* to pierce the Bollinger band (``l < lower`` / ``h > upper``) plus an RSI(2) extreme and a rejection
wick; nothing required ``close`` to be on the far side of ``bb_mid``. A long liquidation wick on a bar that still closes
above the midline (LONG) or below it (SHORT) therefore passed the abs() filter and emitted a bracket whose
``take_profit`` (and ``tp1``) sat on the *wrong* side of entry. Live: Binance rejects the TAKE_PROFIT_MARKET algo order
(-2021 "would immediately trigger") and the position runs with no TP. Paper/backtest: ``_check_triggers`` fires on the
very next tick and the trade is closed at once for fees + slippage, while expected_r/expected_profit were computed as
if the TP were favourable.
"""
from __future__ import annotations

from synth import synth_symbols

from heartless.core.models import Regime, Side, Ticker
from heartless.strategy.alphas.mean_reversion import MeanReversion
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

NAN = float("nan")
SYM = "ETHUSDT"
MID, SD, ATR = 100.0, 1.0, 1.0  # bb_k default 2.4 -> bands at 97.6 / 102.4


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


def m5(close: float, o: float, h: float, l: float, rsi2: float) -> StubCursor:
    return StubCursor(close=close, open=o, high=h, low=l, atr=ATR, bb_mid=MID, bb_sd=SD, rsi2=rsi2)


# Range regime: 15m ADX well under adx_max (20), CHOP >= 61 (+0.08 conf). 1h EMAs aligned so the knife guard stays off.
def m15() -> StubCursor:
    return StubCursor(adx=15.0, chop=62.0)


def h1_up() -> StubCursor:
    return StubCursor(adx=20.0, ema21=101.0, ema50=100.0)


def h1_down() -> StubCursor:
    return StubCursor(adx=20.0, ema21=99.0, ema50=100.0)


def ctx(price: float) -> Context:
    return Context(symbol=SYM, info=synth_symbols([SYM])[SYM],
                   ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1), regime=Regime.RANGE)


def params() -> dict:
    return StrategyParams.default().alphas["mean_reversion"]


def evaluate(m5c: StubCursor, h1c: StubCursor):
    return MeanReversion().evaluate(StubView(m5c, m15(), h1c), ctx(m5c.v("close")), params())


# ---- the defect: wick pierces the band but the bar closes on the wrong side of the midline -------
def test_long_closing_above_midline_is_rejected():
    # Low 97.0 < lower band 97.6, RSI(2)=3, rejection wick 4.0/4.5 -> trigger fires; but close 101 > mid 100.
    # Pre-fix: stop = 99.6, |tp - entry| = 1.0 >= 0.6 * 1.4 = 0.84 -> LONG emitted with take_profit 100 < entry 101.
    sig = evaluate(m5(close=101.0, o=101.2, h=101.5, l=97.0, rsi2=3.0), h1_up())
    assert sig is None


def test_short_closing_below_midline_is_rejected():
    # Mirror: high 103.0 > upper band 102.4, RSI(2)=97, rejection wick; but close 99 < mid 100.
    # Pre-fix: SHORT emitted with take_profit 100 > entry 99.
    sig = evaluate(m5(close=99.0, o=98.8, h=103.0, l=98.5, rsi2=97.0), h1_down())
    assert sig is None


def test_close_exactly_on_midline_is_rejected():
    # Zero reward: tp == entry must never produce a bracket on either side.
    assert evaluate(m5(close=100.0, o=100.2, h=100.5, l=97.0, rsi2=3.0), h1_up()) is None
    assert evaluate(m5(close=100.0, o=99.8, h=103.0, l=99.5, rsi2=97.0), h1_down()) is None


# ---- preserved behaviour: textbook setups still fire with a correctly oriented bracket -----------
def test_valid_long_fires_with_tp_above_entry():
    # Close 98 below mid 100; low 97 < 97.6; wick 1.0/1.4. Structure stop 96.8 is tighter than ATR stop 96.6.
    sig = evaluate(m5(close=98.0, o=98.3, h=98.4, l=97.0, rsi2=3.0), h1_up())
    assert sig is not None and sig.side is Side.LONG
    assert sig.stop < sig.limit_price < sig.tp1 < sig.take_profit
    assert sig.take_profit == MID
    assert (sig.take_profit - 98.0) * sig.side.sign > 0 and (sig.tp1 - 98.0) * sig.side.sign > 0
    assert (sig.stop - 98.0) * sig.side.sign < 0


def test_valid_short_fires_with_tp_below_entry():
    sig = evaluate(m5(close=102.0, o=101.7, h=103.0, l=101.6, rsi2=97.0), h1_down())
    assert sig is not None and sig.side is Side.SHORT
    assert sig.stop > sig.limit_price > sig.tp1 > sig.take_profit
    assert sig.take_profit == MID
    assert (sig.take_profit - 102.0) * sig.side.sign > 0 and (sig.tp1 - 102.0) * sig.side.sign > 0
    assert (sig.stop - 102.0) * sig.side.sign < 0


def test_small_but_favourable_reward_is_still_filtered():
    # Close 99.5 is on the right side of mid but reward 0.5 < 0.6 * 1.4 = 0.84 -> the reward filter must still reject.
    assert evaluate(m5(close=99.5, o=99.7, h=99.8, l=97.0, rsi2=3.0), h1_up()) is None
    assert evaluate(m5(close=100.5, o=100.3, h=103.0, l=100.2, rsi2=97.0), h1_down()) is None


def test_emitted_brackets_never_self_trigger_across_a_sweep():
    # Property check: whatever the bar shape, an emitted signal has TP/tp1 strictly favourable and stop strictly adverse.
    for close in (96.5, 97.0, 97.5, 98.0, 98.5, 99.0, 99.5, 100.0, 100.5, 101.0, 101.5):
        sig = evaluate(m5(close=close, o=close + 0.2, h=close + 0.5, l=96.0, rsi2=3.0), h1_up())
        if sig is not None:
            assert sig.side is Side.LONG
            assert sig.take_profit > close and sig.tp1 > close and sig.stop < close
    for close in (103.5, 103.0, 102.5, 102.0, 101.5, 101.0, 100.5, 100.0, 99.5, 99.0, 98.5):
        sig = evaluate(m5(close=close, o=close - 0.2, h=104.0, l=close - 0.5, rsi2=97.0), h1_down())
        if sig is not None:
            assert sig.side is Side.SHORT
            assert sig.take_profit < close and sig.tp1 < close and sig.stop > close
