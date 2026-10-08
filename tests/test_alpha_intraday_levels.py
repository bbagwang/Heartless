"""intraday_levels signal geometry: at the 13:00 UTC 1h close, a move of >= k_adr x ADR away from the 00:00 UTC open in a
volatile market is joined in its direction.

Checks the stop sits on the protective side of the entry (below for longs, above for shorts) at sl_atr x 1h ATR, that
the partial and final targets lie beyond the entry in the trade direction (partial inside the final target), the
post-only entry at the bar close, and that the wrong hour, a quiet market, a small move, a missing 00:00 bar and an
unclosed bar block entries. A last test runs the alpha on a real MarketView built from 1m candles.
"""
from __future__ import annotations

import numpy as np
from synth import synth_symbols

from heartless.core.models import Candle, EntryStyle, Regime, Side, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.strategy.alphas.intraday_levels import ADR_DAYS, TF, IntradayLevels, day_levels
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

NAN = float("nan")
SYM = "SOLUSDT"
H = 3_600_000
D = 86_400_000
DAY0 = 1_768_435_200_000  # 2026-01-15 00:00 UTC


class StubFrame:
    def __init__(self, open_time: np.ndarray):
        self.open_time = open_time
        self.close_time = open_time + H - 1


class StubCursor:
    """Minimal 1h FrameCursor positioned at the newest bar."""

    ok = True

    def __init__(self, open_time, **series):
        self.frame = StubFrame(np.asarray(open_time, dtype=np.int64))
        self.idx = len(open_time) - 1
        self.series = {k: np.asarray(v, dtype=float) for k, v in series.items()}

    @property
    def close_time(self) -> int:
        return int(self.frame.close_time[self.idx])

    def v(self, name: str, k: int = 0) -> float:
        s = self.series.get(name)
        if s is None or k >= len(s):
            return NAN
        return float(s[-1 - k])

    def arr(self, name: str, n: int) -> np.ndarray:
        return self.series[name][-n:]


class StubView:
    def __init__(self, h1: StubCursor, closed: bool = True):
        self.h1 = h1
        self._closed = closed

    def tf(self, tf: str) -> StubCursor:
        assert tf == TF
        return self.h1

    def closed(self, tf: str) -> bool:
        return self._closed and tf == TF


def hourly(direction: int = 1, move: float = 3.0, atr: float = 2.0, hours_today: int = 13, prior_range: float = 4.0,
           drop_midnight: bool = False) -> StubCursor:
    """ADR_DAYS+2 quiet days oscillating around 100 (daily range `prior_range`), then today: open 100 at 00:00 UTC and
    a steady move of `move` (sign = direction) until the close of the last bar."""
    days = ADR_DAYS + 2
    ot, o, h, l, c = [], [], [], [], []
    for d in range(days, 0, -1):
        for hr in range(24):
            t = DAY0 - d * D + hr * H
            mid = 100.0 + (prior_range / 2 - 0.25) * np.sin(hr / 24 * 2 * np.pi)
            ot.append(t)
            o.append(mid)
            c.append(mid)
            h.append(mid + 0.25)
            l.append(mid - 0.25)
    for hr in range(hours_today):
        if drop_midnight and hr == 0:
            continue
        p0 = 100.0 + direction * move * hr / hours_today
        p1 = 100.0 + direction * move * (hr + 1) / hours_today
        ot.append(DAY0 + hr * H)
        o.append(p0)
        c.append(p1)
        h.append(max(p0, p1) + 0.05)
        l.append(min(p0, p1) - 0.05)
    n = len(ot)
    return StubCursor(ot, open=o, high=h, low=l, close=c, atr=np.full(n, atr))


def ctx(price: float) -> Context:
    return Context(symbol=SYM, info=synth_symbols([SYM])[SYM],
                   ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1), regime=Regime.RANGE)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["intraday_levels"])
    p.update(over)
    return p


def evaluate(h1: StubCursor, closed: bool = True, **over):
    return IntradayLevels().evaluate(StubView(h1, closed), ctx(h1.v("close")), params(**over))


def test_long_geometry():
    p = params()
    sig = evaluate(hourly(direction=1))
    assert sig is not None and sig.side is Side.LONG
    entry = sig.limit_price
    assert sig.entry_style is EntryStyle.LIMIT and entry == sig.tags["ref_price"] == 103.0
    assert sig.stop < entry and abs((entry - sig.stop) - p["sl_atr"] * 2.0) < 1e-9
    assert entry < sig.tp1 < sig.take_profit
    assert abs(sig.take_profit - (entry + p["tp_r"] * (entry - sig.stop))) < 1e-9
    assert sig.timeframe == TF and sig.max_hold_bars == p["hold_h"] * 60 and sig.atr == 2.0
    assert sig.tags["day_open"] == 100.0 and abs(sig.tags["adr"] - 4.0) < 1e-9 and sig.reason


def test_short_geometry():
    p = params()
    sig = evaluate(hourly(direction=-1))
    assert sig is not None and sig.side is Side.SHORT
    entry = sig.limit_price
    assert entry == 97.0 and sig.stop > entry and abs((sig.stop - entry) - p["sl_atr"] * 2.0) < 1e-9
    assert sig.take_profit < sig.tp1 < entry


def test_no_partial_when_tp1_frac_zero():
    sig = evaluate(hourly(), tp1_frac=0.0)
    assert sig is not None and sig.tp1 is None and sig.take_profit > sig.limit_price


def test_filters_block_entries():
    assert evaluate(hourly(hours_today=12)) is None  # 12:00 UTC close is not the decision hour
    assert evaluate(hourly(hours_today=12), entry_hour=12) is not None
    assert evaluate(hourly(atr=0.5)) is None  # 1h ATR 50bp < min_atr_bp: quiet market
    assert evaluate(hourly(move=0.8)) is None  # 0.8 < 0.3 x ADR(4.0)
    assert evaluate(hourly(drop_midnight=True)) is None  # no 00:00 UTC bar -> no reliable daily open
    assert evaluate(hourly(), closed=False) is None
    assert evaluate(hourly(atr=NAN)) is None


def test_day_levels_uses_only_today_and_complete_prior_days():
    cur = hourly()
    lv = day_levels(cur.frame.open_time, cur.arr("high", 10_000), cur.arr("low", 10_000), cur.arr("open", 10_000),
                    cur.close_time)
    d_open, d_hi, d_lo, adr = lv
    assert d_open == 100.0 and abs(d_hi - 103.05) < 1e-9 and abs(d_lo - 99.95) < 1e-9 and abs(adr - 4.0) < 1e-9
    short = 3 * 24 + 13  # too little history for the ADR
    assert day_levels(cur.frame.open_time[-short:], cur.arr("high", short), cur.arr("low", short),
                      cur.arr("open", short), cur.close_time) is None


def test_runs_on_a_real_market_view():
    """1m candles: quiet sawtooth days (hourly swings keep the 1h ATR high), then a steady rally from 00:00 UTC."""
    days = ADR_DAYS + 2
    start = DAY0 - days * D
    n = days * 1440 + 13 * 60
    ca = CandleArrays("1m", capacity=n + 16)
    for i in range(n):
        t = start + i * 60_000
        minute = i % 60
        base = 100.0 if t < DAY0 else 100.0 + 4.0 * (t - DAY0) / (13 * H)
        swing = 1.2 * (1 - abs(minute - 30) / 30)  # 0 at the top of the hour, 1.2 at half past
        o = base + swing
        c = base + 1.2 * (1 - abs(minute + 1 - 30) / 30)
        ca.append(Candle(t, o, max(o, c) + 0.02, min(o, c) - 0.02, c, 10.0, 10.0 * c, 10, 5.0, t + 59_999))
    view = MarketView(SYM)
    view.rebuild(ca, live=False)
    view.seek(DAY0 + 13 * H - 1)
    assert view.closed(TF)
    sig = IntradayLevels().evaluate(view, ctx(view.price), params())
    assert sig is not None and sig.side is Side.LONG
    assert sig.stop < sig.limit_price < sig.tp1 < sig.take_profit
    view.seek(DAY0 + 12 * H - 1)  # an hour earlier: not the decision hour
    assert IntradayLevels().evaluate(view, ctx(view.price), params()) is None
