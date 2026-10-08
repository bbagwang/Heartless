"""momentum_burst signal geometry: fade a high-volume 1h burst bar one hour later (OI rising, burst with the 1h trend).

Checks the protective stop sits on the correct side of the entry for longs and shorts, that the target lies beyond
the entry in the trade direction, that the stop distance is sl_atr * ATR, and that the burst / volume / trend /
open-interest filters and warm-up NaNs block entries as designed.
"""
from __future__ import annotations

from synth import synth_symbols

from heartless.core.models import EntryStyle, Regime, Side, Ticker
from heartless.strategy.alphas.momentum_burst import TF, MomentumBurst
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

NAN = float("nan")
SYM = "SOLUSDT"


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


def down_burst(**over) -> StubCursor:
    # downtrend (EMA50 < EMA200); bar k=1 fell 2.5 ATR on volume z=4; decision bar k=0 closed at 99 -> fade LONG
    d = dict(close=[99.0, 98.0, 100.5], atr=1.0, vol_z=[1.0, 4.0, 0.5], ema50=101.0, ema200=104.0)
    d.update(over)
    return StubCursor(**d)


def up_burst(**over) -> StubCursor:
    # uptrend (EMA50 > EMA200); bar k=1 rose 2.5 ATR on volume z=4; decision bar k=0 closed at 101 -> fade SHORT
    d = dict(close=[101.0, 102.0, 99.5], atr=1.0, vol_z=[1.0, 4.0, 0.5], ema50=99.0, ema200=96.0)
    d.update(over)
    return StubCursor(**d)


def ctx(price: float, extras: dict | None = None) -> Context:
    return Context(symbol=SYM, info=synth_symbols([SYM])[SYM],
                   ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1), regime=Regime.VOLATILE,
                   extras={"oi_chg_4h": 0.012} if extras is None else extras)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["momentum_burst"])
    p.update(over)
    return p


def evaluate(cur: StubCursor, closed: bool = True, extras: dict | None = None, **over):
    return MomentumBurst().evaluate(StubView(cur, closed), ctx(cur.v("close"), extras), params(**over))


def test_long_geometry_fades_down_burst():
    p = params()
    sig = evaluate(down_burst())
    assert sig is not None and sig.side is Side.LONG
    entry = sig.tags["ref_price"]
    assert entry == 99.0 and sig.entry_style is EntryStyle.MARKET
    assert sig.stop < entry < sig.take_profit
    assert abs((entry - sig.stop) - p["sl_atr"] * 1.0) < 1e-9
    assert abs((sig.take_profit - entry) - p["tp_r"] * (entry - sig.stop)) < 1e-9
    assert sig.tp1 is None and sig.timeframe == TF and sig.max_hold_bars == 60 * int(p["max_hold"])
    assert sig.reason and sig.confidence >= p["min_conf"]


def test_short_geometry_fades_up_burst():
    p = params()
    sig = evaluate(up_burst())
    assert sig is not None and sig.side is Side.SHORT
    entry = sig.tags["ref_price"]
    assert sig.take_profit < entry < sig.stop
    assert abs((sig.stop - entry) - p["sl_atr"] * 1.0) < 1e-9
    assert abs((entry - sig.take_profit) - p["tp_r"] * (sig.stop - entry)) < 1e-9


def test_not_closed_returns_none():
    assert evaluate(down_burst(), closed=False) is None


def test_small_or_quiet_burst_returns_none():
    assert evaluate(down_burst(close=[99.0, 98.0, 99.0])) is None  # only 1 ATR
    assert evaluate(down_burst(vol_z=[1.0, 1.5, 0.5])) is None  # ordinary volume
    assert evaluate(up_burst(close=[101.0, 102.0, 101.5])) is None


def test_trend_filter():
    # a burst against the 1h trend is not a climax: no fade unless the filter is switched off
    assert evaluate(down_burst(ema50=104.0, ema200=101.0)) is None
    assert evaluate(up_burst(ema50=96.0, ema200=99.0)) is None
    sig = evaluate(down_burst(ema50=104.0, ema200=101.0), trend_align=0)
    assert sig is not None and sig.side is Side.LONG


def test_stall_filter():
    # the decision bar kept running 1.5 ATR further in the burst direction: no climax yet, no fade
    assert evaluate(down_burst(close=[96.5, 98.0, 100.5])) is None
    assert evaluate(up_burst(close=[103.5, 102.0, 99.5])) is None
    sig = evaluate(up_burst(close=[102.3, 102.0, 99.5]))  # small further push (0.3 ATR) is still a stall
    assert sig is not None and sig.side is Side.SHORT and sig.stop > 102.3 > sig.take_profit


def test_open_interest_filter_is_an_optional_knob():
    # shipped default: off, so the alpha also trades without positioning data
    assert params()["oi_min"] <= -1
    assert evaluate(down_burst(), extras={}) is not None
    # switched on: only fade when open interest grew (late chasers); missing data blocks the trade
    assert evaluate(down_burst(), extras={"oi_chg_4h": -0.01}, oi_min=0.0) is None
    assert evaluate(down_burst(), extras={}, oi_min=0.0) is None
    assert evaluate(down_burst(), extras={"stale": True}, oi_min=0.0) is None
    sig = evaluate(down_burst(), extras={"oi_chg_4h": 0.012}, oi_min=0.0)
    assert sig is not None and sig.side is Side.LONG


def test_warmup_nan_blocks_both_sides():
    assert evaluate(down_burst(ema200=NAN)) is None
    assert evaluate(up_burst(ema200=NAN)) is None
    assert evaluate(down_burst(atr=NAN)) is None
    assert evaluate(up_burst(close=[101.0, 102.0])) is None  # bar before the burst missing


def test_shipped_enabled_with_research_defaults():
    p = params()
    assert MomentumBurst.enabled_by_default is True and StrategyParams.default().enabled["momentum_burst"] is True
    assert (p["burst_atr"], p["vol_z_min"], p["trend_align"], p["ft_max"], p["sl_atr"], p["tp_r"], p["max_hold"]) == \
        (2.0, 3.0, 1, 0.5, 1.5, 2.5, 8)
