"""trend_pullback signal geometry: 1h EMA21/50/200 trend, pullback into the EMA21 zone, resumption bar.

Checks the protective stop sits on the correct side of the entry for longs and shorts, that both targets lie beyond
the entry in the trade direction (partial inside the final target), that the stop is never tighter than sl_atr * ATR,
and that warm-up NaNs on the slow EMAs block both sides symmetrically (nan comparisons must not read as a trend).
"""
from __future__ import annotations

from synth import synth_symbols

from heartless.core.models import EntryStyle, Regime, Side, Ticker
from heartless.strategy.alphas.trend_pullback import TF, TrendPullback
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
    def __init__(self, cur: StubCursor, closed: bool = True):
        self.cur = cur
        self._closed = closed

    def tf(self, tf: str) -> StubCursor:
        return self.cur

    def closed(self, tf: str) -> bool:
        return self._closed and tf == TF


def long_setup(**over) -> StubCursor:
    # EMA21 > EMA50 > EMA200, the last bars dipped to the EMA21 zone, green bar closing above EMA9
    d = dict(close=101.0, open=100.2, atr=1.0, ema9=100.8, ema21=100.0, ema50=97.0, ema200=90.0, adx=22.0,
             low=[99.9, 100.1, 100.6, 101.2], high=[101.2, 101.0, 101.5, 102.0])
    d.update(over)
    return StubCursor(**d)


def short_setup(**over) -> StubCursor:
    d = dict(close=99.0, open=99.8, atr=1.0, ema9=99.2, ema21=100.0, ema50=103.0, ema200=110.0, adx=22.0,
             high=[100.1, 99.9, 99.4, 98.8], low=[98.8, 99.0, 98.5, 98.0])
    d.update(over)
    return StubCursor(**d)


def ctx(price: float) -> Context:
    return Context(symbol=SYM, info=synth_symbols([SYM])[SYM],
                   ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1), regime=Regime.RANGE)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["trend_pullback"])
    p.update(over)
    return p


def evaluate(cur: StubCursor, closed: bool = True, **over):
    return TrendPullback().evaluate(StubView(cur, closed), ctx(cur.v("close")), params(**over))


def test_long_geometry():
    p = params()
    sig = evaluate(long_setup())
    assert sig is not None and sig.side is Side.LONG
    entry = sig.limit_price
    assert sig.entry_style is EntryStyle.LIMIT and abs(entry - 101.0) < 0.05
    assert sig.stop < entry < sig.take_profit
    if sig.tp1 is not None:
        assert entry < sig.tp1 < sig.take_profit
    assert sig.tags["ref_price"] - sig.stop >= p["sl_atr"] * 1.0 - 1e-9  # never tighter than sl_atr * ATR (from the close)
    assert sig.stop < min(long_setup().series["low"])  # beyond the pullback swing
    assert sig.timeframe == TF and sig.trail_atr_mult == p["trail_atr"] and sig.max_hold_bars > 0
    assert sig.tags["ref_price"] == 101.0 and sig.reason


def test_short_geometry():
    p = params()
    sig = evaluate(short_setup())
    assert sig is not None and sig.side is Side.SHORT
    entry = sig.limit_price
    assert sig.take_profit < entry < sig.stop
    if sig.tp1 is not None:
        assert sig.take_profit < sig.tp1 < entry
    assert sig.stop - sig.tags["ref_price"] >= p["sl_atr"] * 1.0 - 1e-9
    assert sig.stop > max(short_setup().series["high"])


def test_wide_swing_sets_stop_beyond_swing():
    # a deep pullback swing (2.5 ATR below the close) must widen the stop past it, not cut through it
    cur = long_setup(low=[98.5, 100.1, 100.6, 101.2], ema21=98.7)
    sig = evaluate(cur)
    assert sig is not None and sig.stop < 98.5


def test_not_closed_returns_none():
    assert evaluate(long_setup(), closed=False) is None


def test_warmup_nan_slow_ema_blocks_both_sides():
    assert evaluate(long_setup(ema200=NAN)) is None
    assert evaluate(short_setup(ema200=NAN)) is None
    assert evaluate(long_setup(ema50=NAN)) is None
    assert evaluate(short_setup(ema50=NAN)) is None


def test_unaligned_or_no_resumption_returns_none():
    assert evaluate(long_setup(ema50=101.0)) is None  # EMA21 < EMA50: no aligned trend
    assert evaluate(long_setup(open=101.3)) is None  # red bar: no resumption
    assert evaluate(short_setup(open=98.7)) is None
    assert evaluate(long_setup(low=[100.9, 101.0, 101.1, 101.2])) is None  # never reached the EMA21 zone
