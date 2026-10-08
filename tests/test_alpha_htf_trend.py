"""htf_trend: 1-week trend + 2-day pullback on 1h. Signal geometry and gating on stub views."""
from __future__ import annotations

import math

from heartless.core.models import Candle, EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.htf_trend import HtfTrend, vol_z
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

NAN = float("nan")
SYM = "ETHUSDT"


class StubCursor:
    """``v(name, k)``: a scalar, or a {k: value} dict for series looked up k bars back; missing -> nan."""

    ok = True

    def __init__(self, **series):
        self.series = series

    def v(self, name: str, k: int = 0) -> float:
        s = self.series.get(name)
        if s is None:
            return NAN
        if isinstance(s, dict):
            return float(s.get(k, NAN))
        return float(s) if k == 0 else NAN


class StubView:
    def __init__(self, h1: StubCursor, closed: tuple = ("1m", "5m", "15m", "1h")):
        self._h1 = h1
        self._closed = closed

    def tf(self, tf: str) -> StubCursor:
        return self._h1 if tf == "1h" else StubCursor()

    def closed(self, tf: str) -> bool:
        return tf in self._closed


def bar(close_week: float, close_pull: float, close: float = 100.0, atr: float = 1.0) -> StubCursor:
    return StubCursor(close={0: close, 48: close_pull, 168: close_week}, atr=atr)


def ctx(price: float = 100.0, regime: Regime = Regime.RANGE) -> Context:
    info = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)
    return Context(symbol=SYM, info=info, ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1),
                   regime=regime)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["htf_trend"])
    p.update(over)
    return p


def test_registered_on_1h_with_bounded_defaults():
    a = ALPHA_BY_NAME["htf_trend"]
    assert isinstance(a, HtfTrend) and a.timeframe == "1h"
    assert len(a.param_specs) <= 8
    for s in a.param_specs:
        assert s.clip(s.default) == s.default
    assert set(a.regime_affinity) == set(Regime)


def test_vol_z_scales_by_atr_and_sqrt_bars():
    # +10% over 100 bars with ATR = 1% of price -> 0.10 / (0.01 * 10) = 1.0
    assert math.isclose(vol_z(110.0, 100.0, 1.1, 100), 1.0, rel_tol=1e-9)
    assert math.isnan(vol_z(100.0, NAN, 1.0, 48)) and math.isnan(vol_z(100.0, 90.0, 0.0, 48))


def test_long_geometry():
    # week up (90 -> 100: z ~ +0.86), last 48h down (105 -> 100: z ~ -0.69)
    p = params()
    s = HtfTrend().evaluate(StubView(bar(90.0, 105.0)), ctx(), p)
    assert s is not None and s.side is Side.LONG
    entry = s.limit_price
    assert s.entry_style is EntryStyle.LIMIT and entry == ctx().ticker.bid
    assert s.stop < entry < s.take_profit
    assert math.isclose(s.tags["ref_price"] - s.stop, p["sl_atr"] * 1.0)
    assert math.isclose(s.take_profit - s.tags["ref_price"], p["tp_atr"] * 1.0)
    assert s.tp1 is None and s.timeframe == "1h" and s.trail_atr_mult == p["trail_atr"]
    assert s.max_hold_bars == int(p["max_hold"]) * 60
    assert s.tags["z_week"] > 0 > s.tags["z_pull"]
    assert s.reason and "상승" in s.reason
    assert s.confidence * HtfTrend.regime_affinity[Regime.RANGE] >= 0.55


def test_short_geometry():
    # week down (111 -> 100: z ~ -0.76), last 48h up (95.5 -> 100: z ~ +0.68)
    p = params()
    s = HtfTrend().evaluate(StubView(bar(111.0, 95.5)), ctx(), p)
    assert s is not None and s.side is Side.SHORT
    entry = s.limit_price
    assert entry == ctx().ticker.ask
    assert s.take_profit < entry < s.stop
    assert math.isclose(s.stop - s.tags["ref_price"], p["sl_atr"] * 1.0)
    assert s.tags["z_week"] < 0 < s.tags["z_pull"]
    assert "하락" in s.reason


def test_no_signal_without_pullback_weak_trend_or_closed_bar():
    a, p = HtfTrend(), params()
    assert a.evaluate(StubView(bar(90.0, 100.0)), ctx(), p) is None  # no 48h counter-move
    assert a.evaluate(StubView(bar(90.0, 97.0)), ctx(), p) is None  # 48h move WITH the trend is no pullback
    assert a.evaluate(StubView(bar(99.0, 105.0)), ctx(), p) is None  # weekly z ~ +0.08 < trend_z
    assert a.evaluate(StubView(bar(90.0, 105.0), closed=("1m", "5m", "15m")), ctx(), p) is None  # 1h still open
    warm = StubCursor(close={0: 100.0, 48: 105.0}, atr=1.0)  # no 168-bar history yet
    assert a.evaluate(StubView(warm), ctx(), p) is None
    assert a.evaluate(StubView(bar(90.0, 105.0, atr=NAN)), ctx(), p) is None


def test_far_target_dropped_when_not_beyond_stop():
    s = HtfTrend().evaluate(StubView(bar(90.0, 105.0)), ctx(), params(tp_atr=3.0, sl_atr=3.0))
    assert s is not None and s.take_profit is None and s.stop < s.limit_price


# --- audit: causality ---------------------------------------------------------------------------------------------
class RecordingCursor(StubCursor):
    """StubCursor that records every (name, k) the alpha reads."""

    def __init__(self, reads: list, **series):
        super().__init__(**series)
        self.reads = reads

    def v(self, name: str, k: int = 0) -> float:
        self.reads.append((name, k))
        return super().v(name, k)


def test_reads_only_completed_1h_bars():
    reads: list = []
    cur = RecordingCursor(reads, close={0: 100.0, 48: 105.0, 168: 90.0}, atr=1.0)
    asked: list = []

    class View(StubView):
        def tf(self, tf: str):
            asked.append(tf)
            return super().tf(tf)

    s = HtfTrend().evaluate(View(cur), ctx(), params())
    assert s is not None
    assert reads and all(k >= 0 for _, k in reads)  # never a negative (future) offset
    assert set(asked) == {"1h"}


def _candles(n_up: int, n_down: int, n_future: int) -> list[Candle]:
    """1m bars: steady rise, a 2-day pullback, then a violent 'future' move that must not affect earlier decisions."""
    t0 = 1_767_225_600_000  # 2026-01-01 00:00 UTC, hour aligned
    path = []
    for i in range(n_up + n_down + n_future):
        if i < n_up:
            base = 100.0 + 25.0 * i / n_up
        elif i < n_up + n_down:
            base = 125.0 - 7.0 * (i - n_up) / n_down
        else:
            base = 118.0 * (1 - 0.3 * (i - n_up - n_down) / n_future)  # crash after the decision bar
        path.append(base * (1 + 0.003 * math.sin(2 * math.pi * i / 37)))
    out, prev = [], path[0]
    for i, c in enumerate(path):
        o = prev
        ot = t0 + i * 60_000
        out.append(Candle(ot, o, max(o, c) * 1.0005, min(o, c) * 0.9995, c, 1.0, c, 1, 0.5, ot + 59_999))
        prev = c
    return out


def test_decision_identical_with_and_without_future_data():
    n_up, n_down, n_future = 192 * 60, 48 * 60, 24 * 60
    bars = _candles(n_up, n_down, n_future)
    cut = n_up + n_down  # decide at the close of the last pullback hour
    t = bars[cut - 1].close_time
    sigs = []
    for rows in (bars[:cut], bars):  # history only vs history + the future crash
        ca = CandleArrays("1m", capacity=len(rows) + 16)
        ca.extend(rows)
        view = MarketView(SYM)
        view.rebuild(ca, live=False)
        view.seek(t)
        assert view.closed("1h")
        sigs.append(HtfTrend().evaluate(view, ctx(price=bars[cut - 1].close), params()))
    a, b = sigs
    assert a is not None and b is not None and a.side is Side.LONG and b.side is Side.LONG
    assert math.isclose(a.stop, b.stop) and math.isclose(a.take_profit, b.take_profit)
    assert math.isclose(a.tags["z_week"], b.tags["z_week"]) and math.isclose(a.tags["z_pull"], b.tags["z_pull"])
