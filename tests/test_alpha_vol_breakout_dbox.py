"""vol_breakout_dbox: daily compression -> first 4h close beyond the daily box. Helper semantics, gating on a real
MarketView over synthetic 1m data (a quiet month then a breakout), causality (the decision at a 4h close does not
change when later bars are appended) and one engine round trip (market entry after the 4h close, 72 h time stop)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import FeatureFrame, FrameCursor, MarketView
from heartless.learning.backtester import Backtester
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.vol_breakout_dbox import RANK_WINDOW, VolBreakoutDbox, box, width_rank
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

NAN = float("nan")
SYM = "ETHUSDT"
DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
LOUD, QUIET = 65, 30  # volatile days, then compressed days; the breakout comes on day LOUD + QUIET
BREAK_DAY = LOUD + QUIET
INFO = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                  price_precision=2, quantity_precision=3)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["vol_breakout_dbox"])
    p.update(over)
    return p


def ctx(price: float, t: int, btc: float = 0.03, funding: float = 0.0) -> Context:
    return Context(symbol=SYM, info=INFO, ticker=Ticker(SYM, bid=price, ask=price, mark=price, last=price, ts=t),
                   regime=Regime.VOLATILE, now=t, funding_rate=funding, market={"btc_ret_168h": btc})


def _candles(logret: np.ndarray, start: int = DAY0) -> CandleArrays:
    """1m bars from per-minute log returns (open = previous close, a 1bp wick each side)."""
    close = 100.0 * np.exp(np.cumsum(logret))
    n = len(close)
    ca = CandleArrays("1m", capacity=n + 16)
    o = np.r_[100.0, close[:-1]]
    ca.open_time[:n] = start + np.arange(n, dtype=np.int64) * MS_MINUTE
    ca.close_time[:n] = ca.open_time[:n] + MS_MINUTE - 1
    ca.open[:n], ca.close[:n] = o, close
    ca.high[:n], ca.low[:n] = np.maximum(o, close) * 1.0001, np.minimum(o, close) * 0.9999
    ca.volume[:n], ca.quote_volume[:n], ca.trades[:n], ca.taker_buy[:n] = 10.0, 10.0 * close, 10, 5.0
    ca.n = n
    return ca


def _path(jump: float, after: int = 6, quiet: bool = True, seed: int = 1) -> np.ndarray:
    """LOUD volatile days (+-2.5% daily drifts and minute noise), then QUIET days of the same 0.3% intraday sine
    (identical daily closes: the bandwidth collapses and no close leaves the box), then `jump` (log) spread over the
    08:00-11:59 UTC bar of BREAK_DAY and `after` more sine days. quiet=False: instead daily swings that grow every day
    up to BREAK_DAY, so the bandwidth is at a 60-day high every day (never compressed)."""
    rng = np.random.default_rng(seed)
    days = LOUD + QUIET + 1 + after
    m = np.arange(days * 1440) % 1440
    sine = 0.003 * (np.sin(2 * np.pi * (m + 1) / 1440) - np.sin(2 * np.pi * m / 1440))
    if quiet:
        drift = np.r_[rng.choice([-0.025, 0.025], LOUD), np.zeros(days - LOUD)]
        r = np.r_[rng.normal(0.0, 0.0004, LOUD * 1440), sine[LOUD * 1440:]]
    else:
        amp = np.linspace(0.002, 0.03, BREAK_DAY)
        # alternating, ending on a down day: BREAK_DAY opens at the bottom of the box
        drift = np.r_[amp * np.where(np.arange(BREAK_DAY) % 2 == 1, 1.0, -1.0), np.zeros(days - BREAK_DAY)]
        r = sine.copy()
    r = r + np.repeat(drift / 1440.0, 1440)
    s = BREAK_DAY * 1440 + 8 * 60
    r[s:s + 240] += jump / 240.0
    return r


def _view(ca: CandleArrays) -> MarketView:
    v = MarketView(SYM)
    v.rebuild(ca, live=False)
    return v


def _decide(view: MarketView, day: int, hour_close: int, **over):
    """The decision at the 4h close ending at hour_close:00 UTC of `day` (the bar closing a millisecond before)."""
    t = DAY0 + day * MS_DAY + hour_close * MS_HOUR - 1
    view.seek(t)
    assert view.closed("4h") and view.tf("4h").close_time == t
    px = view.tf("4h").v("close")
    return VolBreakoutDbox().evaluate(view, ctx(px, t, **{k: over.pop(k) for k in ("btc", "funding") if k in over}),
                                      params(**over))


@pytest.fixture(scope="module")
def up_view() -> MarketView:
    return _view(_candles(_path(+0.04)))


# --- registration and helpers -------------------------------------------------------------------------------------
def test_registered_4h_off_by_default_with_consistent_history_and_hold_bounds():
    a = ALPHA_BY_NAME["vol_breakout_dbox"]
    assert isinstance(a, VolBreakoutDbox) and a.timeframe == "4h" and a.enabled_by_default is False
    assert StrategyParams.default().enabled["vol_breakout_dbox"] is False
    specs = {s.name: s for s in a.param_specs}
    for s in a.param_specs:
        assert s.clip(s.default) == s.default
    # ranks over RANK_WINDOW widths (20 closes each) at up to comp_days + 1 offsets, and the longest box / SMA50
    assert a.lookback_days >= 20 + RANK_WINDOW + specs["comp_days"].hi + 1
    assert a.lookback_days >= max(specs["box_d"].choices) + 1
    assert a.max_hold_days >= 3 * specs["max_hold_h"].hi / 24
    assert set(a.regime_affinity) == set(Regime) and all(w == 1.0 for w in a.regime_affinity.values())


def test_width_rank_counts_the_window_and_refuses_gaps():
    w = np.arange(1.0, 101.0)  # rising widths: the newest is the largest of any window
    assert width_rank(w, 0) == 1.0 and width_rank(w, 40) == 1.0
    assert math.isnan(width_rank(w, 41)) and math.isnan(width_rank(w, 40, window=61))  # not enough history
    w2 = w[::-1].copy()  # falling: the newest is the smallest -> 1 / window
    assert width_rank(w2, 0) == pytest.approx(1 / RANK_WINDOW)
    w2[70] = NAN
    assert math.isnan(width_rank(w2, 0)) and width_rank(w2, 30) == pytest.approx(1 / RANK_WINDOW)


def test_box_reads_the_days_before_the_breakout_day_from_the_last_closed_daily_bar():
    from heartless.data.candles import resample

    ca = _candles(np.r_[np.full(30 * 1440, 0.00001), np.full(5 * 1440, -0.00002)])
    fr = FeatureFrame("1d", resample(ca, "1d"))
    d = FrameCursor(fr, 30)  # day 30 closed
    day30 = (DAY0 + 30 * MS_DAY) // MS_DAY
    hi_excl, lo_excl = box(d, day30, 20)  # the bar's own day: days 10..29
    hi_incl, lo_incl = box(d, day30 + 1, 20)  # the next day: days 11..30
    assert hi_excl == float(fr.f["high"][10:30].max()) and lo_excl == float(fr.f["low"][10:30].min())
    assert hi_incl == float(fr.f["high"][11:31].max()) and lo_incl == float(fr.f["low"][11:31].min())
    assert all(math.isnan(x) for x in box(d, day30 + 2, 20) + box(d, day30 - 1, 20))


# --- synthetic data -----------------------------------------------------------------------------------------------
def test_a_breakout_from_a_quiet_month_goes_long_at_the_first_4h_close_beyond_the_box(up_view):
    assert _decide(up_view, BREAK_DAY, 8) is None  # 04:00-07:59: still inside the box
    s = _decide(up_view, BREAK_DAY, 12)
    assert s is not None and s.side is Side.LONG and s.alpha == "vol_breakout_dbox"
    ref, d = s.tags["ref_price"], up_view.tf("1d")
    assert ref > s.tags["level"] and s.tags["comp_rank"] <= 0.25
    assert s.atr == d.v("atr") and math.isclose(s.stop, ref - 2.0 * d.v("atr"))
    assert s.entry_style is EntryStyle.MARKET and s.take_profit is None and s.tp1 is None
    assert s.timeframe == "1d" and s.max_hold_bars == 72 * 60 - 1 and s.trail_atr_mult == 0.0
    assert not s.exit_on_regime_change and s.entry_ttl_bars is None
    assert _decide(up_view, BREAK_DAY, 16) is None  # no longer fresh: the previous 4h close was already beyond


def test_a_breakdown_goes_short_and_mirrors_the_geometry():
    view = _view(_candles(_path(-0.04, seed=2)))
    s = _decide(view, BREAK_DAY, 12, btc=-0.03)
    assert s is not None and s.side is Side.SHORT
    assert s.tags["ref_price"] < s.tags["level"] and s.stop > s.tags["ref_price"]
    assert _decide(view, BREAK_DAY, 12, btc=0.03) is None  # against BTC's week


def test_filters_btc_funding_side_and_trend(up_view):
    assert _decide(up_view, BREAK_DAY, 12, btc=-0.01) is None
    assert _decide(up_view, BREAK_DAY, 12, btc=NAN) is None  # unknown BTC week
    assert _decide(up_view, BREAK_DAY, 12, btc=-0.01, btc_filter=0) is not None
    assert _decide(up_view, BREAK_DAY, 12, funding=0.0005) is None  # longs would pay 0.05% / 8h
    assert _decide(up_view, BREAK_DAY, 12, funding=-0.0005) is not None
    assert _decide(up_view, BREAK_DAY, 12, funding=0.0005, fund_max=0.0) is not None
    assert _decide(up_view, BREAK_DAY, 12, side_mode=-1) is None
    assert _decide(up_view, BREAK_DAY, 12, side_mode=1) is not None
    d = up_view.tf("1d")
    sma50 = float(d.arr("close", 50).mean())
    want = d.v("bb_mid") > sma50
    assert (_decide(up_view, BREAK_DAY, 12, trend_filter=1) is not None) == want


def test_no_compression_no_trade_unless_the_filter_is_off():
    with np.errstate(over="ignore"):  # RSI of the loss-free synthetic swings overflows harmlessly (unused here)
        view = _view(_candles(_path(+0.12, quiet=False, seed=3)))
    assert _decide(view, BREAK_DAY, 12) is None  # bandwidth at a 60-day high: not compressed
    s = _decide(view, BREAK_DAY, 12, comp_q=1.0)
    assert s is not None and s.side is Side.LONG and math.isnan(s.tags["comp_rank"])


def test_warm_up_gives_nothing():
    ca = _candles(_path(+0.04))
    short = CandleArrays("1m", capacity=40 * 1440 + 16)
    short.extend([ca.candle_at(i) for i in range(LOUD * 1440 - 30 * 1440, LOUD * 1440 + 10 * 1440)])
    view = _view(short)
    for day in range(1, 39):
        for hour in (4, 8, 12, 16, 20, 24):
            t = short.open_time[0] + day * MS_DAY + hour * MS_HOUR - 1
            view.seek(int(t))
            px = view.tf("4h").v("close")
            assert VolBreakoutDbox().evaluate(view, ctx(px, int(t)), params(comp_q=1.0)) is None or day >= 21
            assert VolBreakoutDbox().evaluate(view, ctx(px, int(t)), params()) is None  # ranks need ~80 days


def test_decision_identical_with_and_without_future_data():
    full = _candles(_path(+0.04, after=6) + np.r_[np.zeros((BREAK_DAY + 1) * 1440), np.full(6 * 1440, -0.0001)])
    cut = (BREAK_DAY * 1440) + 12 * 60  # through the 11:59 bar
    head = CandleArrays("1m", capacity=cut + 16)
    head.extend([full.candle_at(i) for i in range(cut)])
    a, b = _decide(_view(head), BREAK_DAY, 12), _decide(_view(full), BREAK_DAY, 12)
    assert a is not None and b is not None and a.side is b.side is Side.LONG
    assert a.stop == b.stop and a.atr == b.atr and a.tags == b.tags and a.reason == b.reason


def test_only_closed_bars_are_read(up_view):
    reads: list = []
    a = VolBreakoutDbox()
    t = DAY0 + BREAK_DAY * MS_DAY + 12 * MS_HOUR - 1
    up_view.seek(t)
    orig_v, orig_arr = FrameCursor.v, FrameCursor.arr

    def v(self, name, k=0):
        reads.append((self.frame.tf, name, k, self.close_time))
        return orig_v(self, name, k)

    def arr(self, name, n):
        reads.append((self.frame.tf, name, 0, self.close_time))
        return orig_arr(self, name, n)

    FrameCursor.v, FrameCursor.arr = v, arr
    try:
        assert a.evaluate(up_view, ctx(up_view.tf("4h").v("close"), t), params(trend_filter=0)) is not None
    finally:
        FrameCursor.v, FrameCursor.arr = orig_v, orig_arr
    assert reads and all(k >= 0 and ct <= t for _, _, k, ct in reads)
    assert {tf for tf, *_ in reads} == {"4h", "1d"}


def test_engine_round_trip_enters_after_the_4h_close_and_exits_on_the_time_stop():
    # the breakout lifts the price 4% inside the 08:00 bar and it then stays flat: the long fills at the 12:00 open and
    # the time stop closes it a minute before the 72nd hour (not 0.5R up: the stop is two daily ATRs away)
    s = Settings(_env_file=None, DAILY_LOSS_LIMIT_PCT=1000.0, WEEKLY_LOSS_LIMIT_PCT=1000.0, MAX_DRAWDOWN_HALT_PCT=1000.0)
    ca = _candles(_path(+0.04))
    bt = Backtester(s, {SYM: INFO}, {SYM: ca})
    p = StrategyParams.default()
    p.alphas["vol_breakout_dbox"]["btc_filter"] = 0  # no market context in this backtest
    res = bt.run(p, only_alpha="vol_breakout_dbox")
    trades = [t for t in res.trades if t["alpha"] == "vol_breakout_dbox"]
    assert len(trades) == 1, res.stats
    t0 = trades[0]
    assert t0["side"] == "LONG" and t0["entry_time"] == DAY0 + BREAK_DAY * MS_DAY + 12 * MS_HOUR
    assert "time stop" in t0["exit_reason"]
    assert t0["exit_time"] == t0["entry_time"] + (72 * 60 - 1) * MS_MINUTE - 1
    assert math.isfinite(t0["r_multiple"]) and abs(t0["r_multiple"]) < 0.5


@pytest.mark.parametrize("field", ["comp_days", "max_hold_h", "btc_filter", "trend_filter", "side_mode"])
def test_integer_params_stay_integer(field):
    spec = {s.name: s for s in VolBreakoutDbox.param_specs}[field]
    assert isinstance(spec.clip(spec.default + 0.4), int)
