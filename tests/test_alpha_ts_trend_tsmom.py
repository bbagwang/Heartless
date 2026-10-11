"""ts_trend_tsmom: daily time-series momentum. Signal geometry and gating on stub views, causality on a real 1d frame,
sanity on synthetic trends and one engine round trip (market entry at 00:00 UTC, time stop after max_hold days)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle, EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.learning.backtester import Backtester
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.ts_trend_tsmom import TsTrendTsmom, trend_z
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_MINUTE

NAN = float("nan")
SYM = "ETHUSDT"
DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC


class StubCursor:
    """``v(name, k)``: a scalar, or a {k: value} dict for series looked up k bars back; missing -> nan."""

    ok = True

    def __init__(self, reads: list | None = None, **series):
        self.series = series
        self.reads = reads

    def v(self, name: str, k: int = 0) -> float:
        if self.reads is not None:
            self.reads.append((name, k))
        s = self.series.get(name)
        if s is None:
            return NAN
        if isinstance(s, dict):
            return float(s.get(k, NAN))
        return float(s) if k == 0 else NAN


class StubView:
    def __init__(self, d1: StubCursor, closed: tuple = ("1m", "5m", "15m", "1h", "4h", "1d")):
        self._d1 = d1
        self._closed = closed
        self.asked: list = []

    def tf(self, tf: str) -> StubCursor:
        self.asked.append(tf)
        return self._d1 if tf == "1d" else StubCursor()

    def closed(self, tf: str) -> bool:
        return tf in self._closed


def day(close: float = 100.0, past: float = 85.0, atr: float = 3.0, lookback: int = 28) -> StubCursor:
    # +17.6% over 28 days with ATR = 3% -> z = 0.176 / (0.03 * sqrt(28)) ~ +1.11
    return StubCursor(close={0: close, lookback: past}, atr=atr)


def ctx(price: float = 100.0, funding: float = 0.0001, market: dict | None = None) -> Context:
    info = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)
    c = Context(symbol=SYM, info=info, ticker=Ticker(SYM, bid=price * 0.9999, ask=price * 1.0001, mark=price, ts=1),
                regime=Regime.VOLATILE, funding_rate=funding)
    if market is not None:
        c.market = market
    return c


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["ts_trend_tsmom"])
    p.update(over)
    return p


def test_registered_daily_off_by_default_with_consistent_history_and_hold_bounds():
    a = ALPHA_BY_NAME["ts_trend_tsmom"]
    assert isinstance(a, TsTrendTsmom) and a.timeframe == "1d" and a.enabled_by_default is False
    assert StrategyParams.default().enabled["ts_trend_tsmom"] is False
    specs = {s.name: s for s in a.param_specs}
    for s in a.param_specs:
        assert s.clip(s.default) == s.default
    # the declared history covers the longest look-back (+1 bar for the past close) and the ATR14 seed
    assert a.lookback_days >= specs["lookback_d"].hi + 1 and a.lookback_days >= 14
    # chunked lab runs hand off positions held up to the engine's cap of 3x the longest time stop
    assert a.max_hold_days >= 3 * specs["max_hold"].hi
    assert set(a.regime_affinity) == set(Regime) and all(w == 1.0 for w in a.regime_affinity.values())


def test_trend_z_scales_by_atr_and_sqrt_days():
    # +10% over 25 days with ATR = 2% of price -> 0.10 / (0.02 * 5) = 1.0
    assert math.isclose(trend_z(110.0, 100.0, 2.2, 25), 1.0, rel_tol=1e-9)
    for bad in ((100.0, NAN, 1.0, 28), (100.0, 90.0, 0.0, 28), (100.0, 0.0, 1.0, 28), (100.0, 90.0, 1.0, 0)):
        assert math.isnan(trend_z(*bad))


def test_long_geometry_market_entry_no_target_daily_time_stop():
    s = TsTrendTsmom().evaluate(StubView(day()), ctx(), params())
    assert s is not None and s.side is Side.LONG and s.alpha == "ts_trend_tsmom"
    assert s.entry_style is EntryStyle.MARKET and s.limit_price is None
    assert math.isclose(s.stop, 100.0 - 3.0 * 3.0) and s.take_profit is None and s.tp1 is None
    assert s.timeframe == "1d" and s.atr == 3.0 and s.trail_atr_mult == 0.0
    assert s.max_hold_bars == 10 * 1440 - 1 and s.tags["ref_price"] == 100.0  # ends before the daily close
    assert math.isclose(s.tags["z"], (100 / 85 - 1) / (0.03 * math.sqrt(28)))
    assert not s.exit_on_regime_change and s.entry_ttl_bars is None


def test_short_mirrors_long():
    s = TsTrendTsmom().evaluate(StubView(day(close=85.0, past=100.0, atr=2.55)), ctx(85.0, funding=-0.0001), params())
    assert s is not None and s.side is Side.SHORT
    assert math.isclose(s.stop, 85.0 + 3.0 * 2.55) and s.take_profit is None


def test_weak_trend_warm_up_and_unclosed_bar_give_nothing():
    a = TsTrendTsmom()
    assert a.evaluate(StubView(day(past=99.0)), ctx(), params()) is None  # z ~ +0.06 < 0.5
    assert a.evaluate(StubView(StubCursor(close={0: 100.0}, atr=3.0)), ctx(), params()) is None  # no past close yet
    assert a.evaluate(StubView(day(), closed=("1m", "5m", "15m", "1h")), ctx(), params()) is None


def test_side_mode_restricts_the_direction():
    a = TsTrendTsmom()
    up, dn = day(), day(close=85.0, past=100.0, atr=2.55)
    assert a.evaluate(StubView(up), ctx(), params(side_mode=-1)) is None
    assert a.evaluate(StubView(dn), ctx(85.0), params(side_mode=1)) is None
    assert a.evaluate(StubView(up), ctx(), params(side_mode=1)).side is Side.LONG
    assert a.evaluate(StubView(dn), ctx(85.0), params(side_mode=-1)).side is Side.SHORT


def test_funding_filter_skips_the_crowded_side_only():
    a = TsTrendTsmom()
    assert a.evaluate(StubView(day()), ctx(funding=0.0005), params()) is None  # longs would pay 0.05% / 8h
    assert a.evaluate(StubView(day()), ctx(funding=0.0005), params(fund_max=0.0)) is not None  # filter off
    assert a.evaluate(StubView(day()), ctx(funding=-0.0005), params()) is not None  # longs are paid
    dn = day(close=85.0, past=100.0, atr=2.55)
    assert a.evaluate(StubView(dn), ctx(85.0, funding=-0.0005), params()) is None
    assert a.evaluate(StubView(dn), ctx(85.0, funding=0.0005), params()) is not None


def test_btc_filter_needs_a_known_btc_week_in_the_trade_direction():
    a = TsTrendTsmom()
    p = params(btc_filter=1)
    assert a.evaluate(StubView(day()), ctx(), p) is None  # no market context: unknown
    assert a.evaluate(StubView(day()), ctx(market={"btc_ret_168h": NAN}), p) is None
    assert a.evaluate(StubView(day()), ctx(market={"btc_ret_168h": -0.02}), p) is None
    assert a.evaluate(StubView(day()), ctx(market={"btc_ret_168h": 0.02}), p).side is Side.LONG
    dn = day(close=85.0, past=100.0, atr=2.55)
    assert a.evaluate(StubView(dn), ctx(85.0, market={"btc_ret_168h": -0.02}), p).side is Side.SHORT


def test_reads_only_completed_daily_bars():
    reads: list = []
    v = StubView(StubCursor(reads, close={0: 100.0, 28: 85.0}, atr=3.0))
    assert TsTrendTsmom().evaluate(v, ctx(), params()) is not None
    assert reads and all(k >= 0 for _, k in reads)  # never a negative (future) offset
    assert set(v.asked) == {"1d"}


# --- real 1d frames on synthetic 1m data --------------------------------------------------------------------------
def _candles(daily_drift: list[float], seed: int = 0, noise: float = 0.0002, wave: float = 0.0,
             start: int = DAY0) -> CandleArrays:
    """1m bars whose log price drifts by daily_drift[d] over UTC day d, with Gaussian minute noise and an optional
    intraday sine of amplitude `wave` (one period per UTC day, so it adds range but leaves the daily closes alone)."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(np.asarray(daily_drift, dtype=float) / 1440.0, 1440)
    close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, noise, len(drift))))
    close = close * (1.0 + wave * np.sin(2 * np.pi * (np.arange(len(close)) + 1) / 1440.0))
    ca = CandleArrays("1m", capacity=len(close) + 16)
    prev = float(close[0])
    for i, c in enumerate(close.tolist()):
        t = start + i * MS_MINUTE
        ca.append(Candle(t, prev, max(prev, c) * 1.0002, min(prev, c) * 0.9998, c, 10.0, 10.0 * c, 10, 5.0, t + 59_999))
        prev = c
    return ca


def _decide(ca: CandleArrays, day_index: int, **over):
    """The alpha's decision at the close of UTC day `day_index` (the minute its daily bar closed)."""
    view = MarketView(SYM)
    view.rebuild(ca, live=False)
    t = DAY0 + (day_index + 1) * MS_DAY - 1
    view.seek(t)
    assert view.closed("1d") and view.tf("1d").close_time == t
    px = view.tf("1d").v("close")
    return TsTrendTsmom().evaluate(view, ctx(px, funding=0.0), params(**over))


def test_decision_identical_with_and_without_future_data():
    drift = [0.006] * 40 + [-0.08] * 10  # steady rise, then a crash after the decision day
    full = _candles(drift, seed=4)
    decide_on = 35
    head = CandleArrays("1m", capacity=(decide_on + 1) * 1440 + 16)
    head.extend([full.candle_at(i) for i in range((decide_on + 1) * 1440)])
    a, b = _decide(head, decide_on), _decide(full, decide_on)
    assert a is not None and b is not None and a.side is b.side is Side.LONG
    assert a.stop == b.stop and a.atr == b.atr and a.tags["z"] == b.tags["z"] and a.tags["ref_price"] == b.tags["ref_price"]


def test_synthetic_trends_give_the_trend_side_and_a_flat_market_nothing():
    up = _candles([0.008] * 40, seed=1)
    dn = _candles([-0.008] * 40, seed=2)
    flat = _candles([0.0] * 40, seed=3, noise=0.0, wave=0.02)  # identical daily closes, 4% daily range
    s_up, s_dn = _decide(up, 35), _decide(dn, 35)
    assert s_up is not None and s_up.side is Side.LONG and s_up.stop < s_up.tags["ref_price"]
    assert s_dn is not None and s_dn.side is Side.SHORT and s_dn.stop > s_dn.tags["ref_price"]
    assert _decide(flat, 35) is None
    assert _decide(up, 20) is None  # fewer than 29 daily bars: warm-up


def test_engine_round_trip_enters_after_the_daily_close_and_exits_on_the_time_stop():
    # 29 days of rise (the first decision with a 28-day look-back is at the day-28 close), then a flat market: the
    # long opens at the next 00:00 UTC open, is closed by the time stop a minute before its 10th daily close (not
    # 0.5R up), and that same close renews it because the 28-day trend still holds
    s = Settings(_env_file=None, DAILY_LOSS_LIMIT_PCT=1000.0, WEEKLY_LOSS_LIMIT_PCT=1000.0, MAX_DRAWDOWN_HALT_PCT=1000.0)
    ca = _candles([0.01] * 29 + [0.0] * 16, seed=7)
    info = {SYM: SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                            price_precision=2, quantity_precision=3)}
    bt = Backtester(s, info, {SYM: ca})
    res = bt.run(StrategyParams.default(), only_alpha="ts_trend_tsmom")
    trades = [t for t in res.trades if t["alpha"] == "ts_trend_tsmom"]
    assert trades, res.stats
    t0 = trades[0]
    assert t0["side"] == "LONG" and t0["entry_time"] == DAY0 + 29 * MS_DAY  # filled at the open after the day-28 close
    assert "time stop" in t0["exit_reason"]
    assert t0["exit_time"] == t0["entry_time"] + 10 * MS_DAY - MS_MINUTE - 1  # close of the 23:58 bar
    assert math.isfinite(t0["r_multiple"]) and abs(t0["r_multiple"]) < 1.0
    assert len(trades) >= 2 and trades[1]["side"] == "LONG"
    assert trades[1]["entry_time"] == t0["entry_time"] + 10 * MS_DAY  # renewed at the same daily close, not a day late


@pytest.mark.parametrize("field", ["lookback_d", "max_hold"])
def test_integer_params_stay_integer(field):
    spec = {s.name: s for s in TsTrendTsmom.param_specs}[field]
    assert isinstance(spec.clip(spec.default + 0.4), int)
