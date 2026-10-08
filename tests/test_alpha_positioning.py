"""positioning: leveraged-spike fade (1h spike + open-interest burst + long/short ratios chasing it, then a 15m close
back through EMA9). Signal geometry and gating on stub views."""
from __future__ import annotations

import math

from heartless.core.models import EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.positioning import Positioning
from heartless.strategy.base import Context
from heartless.strategy.params import COMMON, StrategyParams

NAN = float("nan")
SYM = "ETHUSDT"
BAR = 15 * 60_000
T0 = 1_768_262_400_000


class StubCursor:
    ok = True

    def __init__(self, **series):
        self.series = series  # name -> scalar or list indexed by bars back (k)

    def v(self, name: str, k: int = 0) -> float:
        x = self.series.get(name, NAN)
        if isinstance(x, (list, tuple)):
            return float(x[k]) if k < len(x) else NAN
        return float(x)


class StubView:
    def __init__(self, m15: dict, h1: dict, closed: tuple = ("1m", "5m", "15m", "1h")):
        self._c = {"15m": StubCursor(**m15), "1h": StubCursor(**h1)}
        self._closed = closed

    def tf(self, tf: str) -> StubCursor:
        return self._c.get(tf, StubCursor())

    def closed(self, tf: str) -> bool:
        return tf in self._closed


def ctx(now: int, extras: dict | None, price: float = 100.0, regime: Regime = Regime.TREND_UP) -> Context:
    info = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)
    return Context(symbol=SYM, info=info, ticker=Ticker(SYM, bid=price * 0.99995, ask=price * 1.00005, mark=price),
                   regime=regime, now=now, extras=extras if extras is not None else {})


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["positioning"])
    p.update(over)
    return p


def crowded(sign: float, oiz: float = 2.5) -> dict:
    """OI burst and both positioning groups adding in the direction `sign` of the spike."""
    return {"oi_chg_1h_z": oiz, "ls_acc_chg_4h": 0.05 * sign, "top_ls_pos_chg_4h": 0.04 * sign, "age_min": 1.0}


# 1h ATR = 1.0; up-spike: close 103 vs 100 four 15m bars ago (+3 ATR), spike high 103.5
UP_SPIKE = StubView({"close": [103.0, 102.0, 101.0, 100.5, 100.0], "high": [103.5, 102.2, 101.2, 100.8],
                     "low": [102.0, 101.0, 100.4, 100.0], "ema9": 101.5}, {"atr": 1.0})
UP_ROLLOVER = StubView({"close": [102.0, 103.0], "high": [103.2], "low": [101.9], "ema9": 102.3}, {"atr": 1.0})
DN_SPIKE = StubView({"close": [97.0, 98.0, 99.0, 99.5, 100.0], "high": [98.0, 99.0, 99.6, 100.0],
                     "low": [96.5, 97.8, 98.8, 99.2], "ema9": 98.5}, {"atr": 1.0})
DN_ROLLOVER = StubView({"close": [98.0, 97.0], "high": [98.1], "low": [96.8], "ema9": 97.7}, {"atr": 1.0})


def test_registered_with_common_params_and_bounded_specs():
    a = ALPHA_BY_NAME["positioning"]
    assert isinstance(a, Positioning) and a.timeframe == "15m"
    p = StrategyParams.default().alphas["positioning"]
    assert {s.name for s in COMMON} <= set(p)
    assert len(a.param_specs) <= 8
    for s in a.param_specs:
        assert s.lo <= s.default <= s.hi
    assert all(0.62 * w >= 0.55 for w in a.regime_affinity.values())


def test_short_after_up_spike_rolls_over():
    a = Positioning()
    assert a.evaluate(UP_SPIKE, ctx(T0, crowded(+1)), params()) is None  # armed, not yet rolled over
    sig = a.evaluate(UP_ROLLOVER, ctx(T0 + 3 * BAR, {"oi_chg_1h_z": 0.0}, price=102.0), params())
    assert sig is not None and sig.side is Side.SHORT
    entry = sig.tags["ref_price"]
    assert entry == 102.0
    assert sig.stop > entry and sig.stop > 103.5  # beyond the spike high (extended to 103.5 by the setup bar)
    assert math.isclose(sig.stop, 103.5 + 0.3, rel_tol=1e-9)
    assert sig.take_profit < entry and math.isclose(entry - sig.take_profit, 3.0 * (sig.stop - entry), rel_tol=1e-9)
    assert sig.tp1 is None and sig.entry_style is EntryStyle.LIMIT and sig.limit_price >= entry
    assert sig.max_hold_bars == 12 * 60 and sig.timeframe == "1h"
    assert sig.confidence >= 0.55 and "숏" in sig.reason
    # one trade per setup
    assert a.evaluate(UP_ROLLOVER, ctx(T0 + 4 * BAR, {"oi_chg_1h_z": 0.0}, price=102.0), params()) is None


def test_long_after_down_spike_rolls_over():
    a = Positioning()
    assert a.evaluate(DN_SPIKE, ctx(T0, crowded(-1), regime=Regime.TREND_DOWN), params()) is None
    sig = a.evaluate(DN_ROLLOVER, ctx(T0 + 2 * BAR, {}, price=98.0, regime=Regime.TREND_DOWN), params())
    assert sig is not None and sig.side is Side.LONG
    entry = sig.tags["ref_price"]
    assert sig.stop < entry and sig.stop < 96.5
    assert sig.take_profit > entry and sig.limit_price <= entry
    assert "롱" in sig.reason


def test_min_stop_distance_applies():
    a = Positioning()
    a.evaluate(UP_SPIKE, ctx(T0, crowded(+1)), params())
    # rolled over right at the spike high: the structure stop would be too tight -> sl_min ATR
    view = StubView({"close": [103.4], "high": [103.6], "low": [103.3], "ema9": 103.45}, {"atr": 1.0})
    sig = a.evaluate(view, ctx(T0 + BAR, {}, price=103.4), params(sl_buf=0.0, sl_min=1.5))
    assert sig is not None and sig.side is Side.SHORT
    assert math.isclose(sig.stop - 103.4, 1.5, rel_tol=1e-9)


def test_no_signal_without_positioning_data_or_oi_burst():
    for extras in ({}, {"stale": True, "age_min": 45.0}, crowded(+1, oiz=0.2), {"oi_chg_1h_z": NAN}):
        a = Positioning()
        assert a.evaluate(UP_SPIKE, ctx(T0, extras), params()) is None
        assert a.evaluate(UP_ROLLOVER, ctx(T0 + BAR, {}, price=102.0), params()) is None


def test_chase_filter_and_expiry():
    a = Positioning()
    against = {"oi_chg_1h_z": 2.5, "ls_acc_chg_4h": -0.05, "top_ls_pos_chg_4h": -0.04}
    a.evaluate(UP_SPIKE, ctx(T0, against), params())
    assert a.evaluate(UP_ROLLOVER, ctx(T0 + BAR, {}, price=102.0), params()) is None  # nobody chased -> no setup
    b = Positioning()
    b.evaluate(UP_SPIKE, ctx(T0, against), params(chase=0))
    assert b.evaluate(UP_ROLLOVER, ctx(T0 + BAR, {}, price=102.0), params(chase=0)) is not None
    c = Positioning()
    c.evaluate(UP_SPIKE, ctx(T0, crowded(+1)), params())
    late = T0 + (int(params()["wait"]) + 1) * BAR
    assert c.evaluate(UP_ROLLOVER, ctx(late, {}, price=102.0), params()) is None  # setup expired


def test_small_move_or_unclosed_bar_is_ignored():
    a = Positioning()
    small = StubView({"close": [100.5, 100.4, 100.3, 100.2, 100.0], "high": [100.6] * 4, "low": [100.0] * 4,
                      "ema9": 100.3}, {"atr": 1.0})
    assert a.evaluate(small, ctx(T0, crowded(+1)), params()) is None
    assert a.evaluate(UP_ROLLOVER, ctx(T0 + BAR, {}, price=102.0), params()) is None
    b = Positioning()
    unclosed = StubView(UP_SPIKE._c["15m"].series, {"atr": 1.0}, closed=("1m",))
    assert b.evaluate(unclosed, ctx(T0, crowded(+1)), params()) is None
