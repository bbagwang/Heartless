"""sweep_reversal signal geometry: a 15m bar sweeps the prior N-bar extreme and closes back inside the range.

Checks the protective stop sits beyond the sweep wick (on the correct side of the entry) for longs and shorts, that
both targets lie beyond the entry in the trade direction (partial inside the final target), that the final target is
capped at the opposite side of the range, and that warm-up / missing data and the optional trend filter block entries.
"""
from __future__ import annotations

import numpy as np
from synth import synth_symbols

from heartless.core.models import EntryStyle, Regime, Side, Ticker
from heartless.strategy.alphas.sweep_reversal import ATR_TF, TF, SweepReversal
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

NAN = float("nan")
SYM = "ETHUSDT"
N = 96  # default lookback


class StubCursor:
    """Minimal FrameCursor: ``v(name, k)`` reads k bars back from the newest value; ``arr(name, n)`` the last n values."""

    ok = True

    def __init__(self, **series):
        self.series = {k: (np.asarray(v, dtype=float) if isinstance(v, (list, tuple, np.ndarray)) else v)
                       for k, v in series.items()}

    def v(self, name: str, k: int = 0) -> float:
        s = self.series.get(name)
        if s is None:
            return NAN
        if isinstance(s, np.ndarray):
            return float(s[-1 - k]) if k < len(s) else NAN
        return float(s)

    def arr(self, name: str, n: int) -> np.ndarray:
        s = self.series.get(name)
        return s[-n:] if isinstance(s, np.ndarray) else np.array([])


class StubView:
    def __init__(self, m15: StubCursor, h1: StubCursor, closed: bool = True):
        self.frames = {TF: m15, ATR_TF: h1}
        self._closed = closed

    def tf(self, tf: str) -> StubCursor:
        return self.frames[tf]

    def closed(self, tf: str) -> bool:
        return self._closed and tf == TF


def bars(level_lo: float, level_hi: float, cur_low: float, cur_high: float, cur_close: float, n: int = N + 20):
    """n-1 range bars between level_lo and level_hi, then the current (sweep) bar."""
    lows = np.full(n, level_lo + 1.0)
    highs = np.full(n, level_hi - 1.0)
    lows[n // 2], highs[n // 3] = level_lo, level_hi  # the extremes inside the lookback window
    lows[-1], highs[-1] = cur_low, cur_high
    closes = (lows + highs) / 2
    closes[-1] = cur_close
    return StubCursor(low=lows, high=highs, close=closes, vol_z=1.0)


def long_setup(level_hi: float = 106.0, **over) -> StubCursor:
    # prior 24h low 100.0; the bar trades down to 99.5 and closes back at 100.4
    return bars(100.0, level_hi, 99.5, 100.6, 100.4, **over)


def short_setup(level_lo: float = 94.0, **over) -> StubCursor:
    # prior 24h high 100.0; the bar trades up to 100.5 and closes back at 99.6
    return bars(level_lo, 100.0, 99.4, 100.5, 99.6, **over)


def h1(atr: float = 1.0, ema200: float = 100.0) -> StubCursor:
    return StubCursor(atr=atr, ema200=ema200)


def ctx(price: float) -> Context:
    return Context(symbol=SYM, info=synth_symbols([SYM])[SYM],
                   ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1), regime=Regime.RANGE)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["sweep_reversal"])
    p.update(over)
    return p


def evaluate(m15: StubCursor, hour: StubCursor | None = None, closed: bool = True, **over):
    return SweepReversal().evaluate(StubView(m15, hour or h1(), closed), ctx(m15.v("close")), params(**over))


def test_long_geometry():
    p = params()
    sig = evaluate(long_setup())
    assert sig is not None and sig.side is Side.LONG
    entry = sig.limit_price
    assert sig.entry_style is EntryStyle.LIMIT and entry == 100.4 == sig.tags["ref_price"]
    assert sig.stop < 99.5 < entry  # beyond the sweep wick
    assert abs(sig.stop - (99.5 - p["sl_buffer_atr"] * 1.0)) < 1e-9
    assert entry < sig.tp1 < sig.take_profit
    assert abs(sig.take_profit - (entry + p["tp_r"] * (entry - sig.stop))) < 1e-9
    assert sig.timeframe == TF and sig.max_hold_bars == p["max_hold"] * 15 and sig.reason


def test_short_geometry():
    p = params()
    sig = evaluate(short_setup())
    assert sig is not None and sig.side is Side.SHORT
    entry = sig.limit_price
    assert entry < 100.5 < sig.stop  # beyond the sweep wick
    assert abs(sig.stop - (100.5 + p["sl_buffer_atr"] * 1.0)) < 1e-9
    assert sig.take_profit < sig.tp1 < entry


def test_target_capped_at_opposite_side_of_range():
    sig = evaluate(long_setup(level_hi=103.0))  # 2R would be 104.2, beyond the prior high
    assert sig is not None and sig.take_profit < 103.0 and sig.tp1 < sig.take_profit
    assert evaluate(long_setup(level_hi=102.0)) is None  # too little room to the other side of the range


def test_no_sweep_or_no_reclaim_returns_none():
    assert evaluate(bars(100.0, 106.0, 100.05, 100.6, 100.4)) is None  # low never trades through the level
    assert evaluate(bars(100.0, 106.0, 99.5, 100.6, 100.05)) is None  # closes back inside by less than rej_atr


def test_not_closed_missing_atr_and_warmup_return_none():
    assert evaluate(long_setup(), closed=False) is None
    assert evaluate(long_setup(), hour=h1(atr=NAN)) is None
    assert evaluate(long_setup(n=N)) is None  # needs N prior bars plus the current one


def test_trend_filter_blocks_counter_trend_side():
    assert evaluate(long_setup(), hour=h1(ema200=105.0), trend_filter=1) is None
    assert evaluate(long_setup(), hour=h1(ema200=95.0), trend_filter=1) is not None
    assert evaluate(short_setup(), hour=h1(ema200=95.0), trend_filter=1) is None


def test_shipped_disabled_no_edge():
    # no edge on real data (TRAIN -0.07R n=2082, VALID -0.05R n=798): must stay off for trading by default
    assert SweepReversal.enabled_by_default is False
    assert StrategyParams.default().enabled["sweep_reversal"] is False
