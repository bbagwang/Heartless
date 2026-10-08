"""funding_fade: crowding fade (top-trader position z + OI build-up + funding z, near settlement). Signal geometry and
gating on stub views with a synthetic positioning history."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.funding_fade import FundingFade
from heartless.strategy.base import Context
from heartless.strategy.params import COMMON, StrategyParams

NAN = float("nan")
SYM = "ETHUSDT"
BAR = 15 * 60_000
T0 = 1_768_262_400_000


class StubCursor:
    ok = True

    def __init__(self, **series):
        self.series = series

    def v(self, name: str, k: int = 0) -> float:
        return float(self.series.get(name, NAN))


class StubView:
    def __init__(self, m15: dict, m5: dict, h1: dict, closed: tuple = ("1m", "5m", "15m", "1h")):
        self._c = {"15m": StubCursor(**m15), "5m": StubCursor(**m5), "1h": StubCursor(**h1)}
        self._closed = closed

    def tf(self, tf: str) -> StubCursor:
        return self._c.get(tf, StubCursor())

    def closed(self, tf: str) -> bool:
        return tf in self._closed


def ctx(now: int, tlp: float, fr: float, oi24: float = 0.0, mtf: float = 30.0, regime: Regime = Regime.RANGE,
        price: float = 100.0) -> Context:
    info = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)
    return Context(symbol=SYM, info=info, ticker=Ticker(SYM, bid=price * 0.99995, ask=price * 1.00005, mark=price),
                   regime=regime, now=now, funding_rate=fr, minutes_to_funding=mtf,
                   extras={"top_ls_pos": tlp, "oi_chg_24h": oi24, "age_min": 1.0})


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["funding_fade"])
    p.update(over)
    return p


NEUTRAL = StubView({"close": 100.0, "ema9": 100.0}, {"st_dir": 1.0}, {"atr": 1.0})


def warmed(days: float = 4.0) -> tuple[FundingFade, int]:
    """An alpha whose history holds `days` of calm positioning (no fresh leverage, so it never trades)."""
    a = FundingFade()
    n = int(days * 96)
    for i in range(n):
        tlp = 1.45 if i % 2 else 1.55
        fr = 0.00009 if i % 3 else 0.00011
        assert a.evaluate(NEUTRAL, ctx(T0 + i * BAR, tlp, fr), params()) is None
    return a, T0 + n * BAR


def short_setup(a: FundingFade, now: int, **over):
    """Longs crowded (top-trader ratio and funding spike, OI +5%/24h), price rolled over below the 15m EMA9."""
    view = StubView({"close": 100.0, "ema9": 101.0}, {"st_dir": -1.0}, {"atr": 1.0})
    c = dict(tlp=2.5, fr=0.0003, oi24=0.05, mtf=30.0, regime=Regime.RANGE)
    c.update(over)
    return a.evaluate(view, ctx(now, **c), params())


def long_setup(a: FundingFade, now: int, **over):
    """Shorts crowded (top-trader ratio collapses, funding deeply negative, OI +5%/24h), price turned up."""
    view = StubView({"close": 100.0, "ema9": 99.0}, {"st_dir": 1.0}, {"atr": 1.0})
    c = dict(tlp=0.9, fr=-0.0003, oi24=0.05, mtf=450.0, regime=Regime.RANGE)
    c.update(over)
    return a.evaluate(view, ctx(now, **c), params())


def test_registered_with_defaults_and_disabled():
    a = ALPHA_BY_NAME["funding_fade"]
    assert isinstance(a, FundingFade) and a.timeframe == "15m" and a.enabled_by_default is False
    assert StrategyParams.default().enabled["funding_fade"] is False
    own = [s.name for s in a.param_specs]
    assert len(own) <= 8 and not set(own) & {s.name for s in COMMON}
    assert {"tp_r", "tp1_r"} <= set(own)


def test_short_geometry_fades_crowded_longs():
    a, now = warmed()
    s = short_setup(a, now)
    assert s is not None and s.side is Side.SHORT
    entry = s.tags["ref_price"]
    assert entry == 100.0 and s.entry_style is EntryStyle.LIMIT and s.limit_price > entry * 0.999
    assert s.stop > entry and abs(s.stop - (entry + 1.8)) < 1e-9  # 1.8 x 1h ATR above
    assert s.take_profit < entry and abs((entry - s.take_profit) - 3.0 * (s.stop - entry)) < 1e-9
    assert s.tp1 is None  # tp1_r = 0 by default
    assert s.max_hold_bars == 24 * 60 and s.timeframe == "1h" and s.atr == 1.0
    assert s.tags["top_pos_z"] > 1.5 and s.tags["funding_z"] > 0.5 and s.tags["crowd"] == "LONG"
    assert 0.55 <= s.confidence <= 0.98 and "숏" in s.reason


def test_long_geometry_fades_crowded_shorts():
    a, now = warmed()
    s = long_setup(a, now)
    assert s is not None and s.side is Side.LONG
    entry = s.tags["ref_price"]
    assert s.stop < entry < s.take_profit
    assert abs((s.take_profit - entry) - 3.0 * (entry - s.stop)) < 1e-9
    assert s.limit_price < entry * 1.001 and s.tags["crowd"] == "SHORT" and "롱" in s.reason


def test_partial_target_lies_between_entry_and_final_target():
    a, now = warmed()
    view = StubView({"close": 100.0, "ema9": 101.0}, {"st_dir": -1.0}, {"atr": 1.0})
    s = a.evaluate(view, ctx(now, 2.5, 0.0003, 0.05), params(tp1_r=1.5))
    assert s is not None and s.take_profit < s.tp1 < s.tags["ref_price"] < s.stop


def test_gating():
    a, now = warmed()
    # outside the settlement window (4h before the next settlement)
    assert short_setup(a, now, mtf=240.0) is None
    # no fresh leverage
    assert short_setup(a, now + BAR, oi24=0.0) is None
    # funding does not lean with the crowd
    assert short_setup(a, now + 2 * BAR, fr=0.0001) is None
    # never fade into a running trend
    assert short_setup(a, now + 3 * BAR, regime=Regime.TREND_UP) is None
    assert long_setup(a, now + 4 * BAR, regime=Regime.TREND_DOWN) is None
    # price has not rolled over yet
    view = StubView({"close": 100.0, "ema9": 99.0}, {"st_dir": -1.0}, {"atr": 1.0})
    assert a.evaluate(view, ctx(now + 5 * BAR, 2.5, 0.0003, 0.05), params()) is None
    # 15m bar not closed
    closed_5m = StubView({"close": 100.0, "ema9": 101.0}, {"st_dir": -1.0}, {"atr": 1.0}, closed=("1m", "5m"))
    assert a.evaluate(closed_5m, ctx(now + 6 * BAR, 2.5, 0.0003, 0.05), params()) is None


def test_needs_history_and_resets_when_time_goes_back():
    a, now = warmed(days=2.0)  # less than the 3-day minimum span
    assert short_setup(a, now) is None
    a, now = warmed()
    assert short_setup(a, now) is not None
    # a new backtest (time jumps back) starts a fresh history -> silent again
    assert short_setup(a, T0 - 10 * BAR) is None
    # stale positioning snapshots are ignored
    b = FundingFade()
    c = ctx(T0, 1.5, 0.0001)
    c.extras = {"stale": True, "age_min": 45.0}
    assert b.evaluate(NEUTRAL, c, params()) is None and SYM not in b._hist


def test_missing_or_nan_positioning_never_crashes_or_pollutes_history():
    a, now = warmed()
    n0 = len(a._hist[SYM].ts)
    view = StubView({"close": 100.0, "ema9": 101.0}, {"st_dir": -1.0}, {"atr": 1.0})
    for i, ex in enumerate(({}, None, {"top_ls_pos": NAN, "oi_chg_24h": 0.05}, {"top_ls_pos": 0.0, "oi_chg_24h": 0.05})):
        c = ctx(now + i * BAR, 2.5, 0.0003, 0.05)
        c.extras = ex
        assert a.evaluate(view, c, params()) is None
    assert len(a._hist[SYM].ts) == n0  # unusable snapshots are not sampled
    # usable ratio but unknown OI change / funding: sampled or skipped, never a trade
    c = ctx(now + 5 * BAR, 2.5, 0.0003)
    c.extras = {"top_ls_pos": 2.5, "oi_chg_24h": NAN}
    assert a.evaluate(view, c, params()) is None
    assert a.evaluate(view, ctx(now + 6 * BAR, 2.5, NAN, 0.05), params()) is None
    nan_view = StubView({"close": NAN, "ema9": 101.0}, {"st_dir": -1.0}, {"atr": NAN})
    assert a.evaluate(nan_view, ctx(now + 7 * BAR, 2.5, 0.0003, 0.05), params()) is None
