"""positioning_flow_oimom: follow a large 24h move made while open interest built up and the global account ratio
leaned against it, at the 4h closes. Signal geometry and gating on stub views, causality on a real MarketView, sanity
on synthetic data, and engine round trips through the backtester with a synthetic positioning series (entry at the
next open, time stop a minute before the decision bar 48h later, no trade without the OI build-up, with retail
accounts chasing the move or against an expensive funding rate)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle, EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.extras import MetricsSeries
from heartless.data.features import MarketView
from heartless.learning.backtester import Backtester
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.positioning_flow_oimom import PositioningFlowOiMom, move_in_atr
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

NAN = float("nan")
NAME = "positioning_flow_oimom"
SYM = "ETHUSDT"
DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
T4 = DAY0 + 20 * MS_DAY + 12 * MS_HOUR - 1  # 11:59:59.999 UTC on day 20: a 4h close
SETTLE = 8 * MS_HOUR


def info() -> SymbolInfo:
    return SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)


class StubCursor:
    ok = True

    def __init__(self, reads: list | None = None, past: float = NAN, **vals):
        self.vals = vals
        self.past = past  # the value 24 bars back (1h close a day earlier)
        self.reads = reads

    def v(self, name: str, k: int = 0) -> float:
        if self.reads is not None:
            self.reads.append((name, k))
        if k == 24 and name == "close":
            return self.past
        return float(self.vals.get(name, NAN)) if k == 0 else NAN


class StubView:
    def __init__(self, h1: StubCursor | None = None, d1: StubCursor | None = None, q: StubCursor | None = None,
                 closed=("1m", "5m", "15m", "1h", "4h")):
        self.h1 = h1 if h1 is not None else StubCursor(close=110.0, past=100.0)  # +10% over 24h
        self.d1 = d1 if d1 is not None else StubCursor(atr=4.0, close=104.0)  # daily ATR 4 (~3.6% of 110)
        self.q = q if q is not None else StubCursor(atr=1.5, close=110.0)
        self._closed = closed
        self.asked: list = []

    def tf(self, tf: str) -> StubCursor:
        self.asked.append(tf)
        return {"1h": self.h1, "1d": self.d1, "4h": self.q}.get(tf, StubCursor())

    def closed(self, tf: str) -> bool:
        return tf in self._closed


def extras(oi: float = 0.15, retail: float = -0.05, **more) -> dict:
    """Positioning snapshot; `retail` is the 4h change of the global account long/short ratio (ls_acc_chg_4h)."""
    d = {"age_min": 1.0, "oi": 1000.0, "oi_chg_1h": 0.01, "oi_chg_4h": 0.04, "oi_chg_24h": oi, "ls_acc": 1.5,
         "ls_acc_chg_4h": retail}
    d.update(more)
    return d


def ctx(now: int = T4, ex: dict | None = None, price: float = 110.0, funding: float = 0.0001) -> Context:
    return Context(symbol=SYM, info=info(), ticker=Ticker(SYM, bid=price * 0.99995, ask=price * 1.00005, mark=price,
                                                          ts=now),
                   regime=Regime.VOLATILE, now=now, funding_rate=funding, extras=extras() if ex is None else ex)


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas[NAME])
    p.update(over)
    return p


def test_registered_off_by_default_on_the_4h_closes_with_history_and_hold_bounds():
    a = ALPHA_BY_NAME[NAME]
    assert isinstance(a, PositioningFlowOiMom) and a.timeframe == "4h" and a.enabled_by_default is False
    assert StrategyParams.default().enabled[NAME] is False
    specs = {s.name: s for s in a.param_specs}
    for s in a.param_specs:
        assert s.clip(s.default) == s.default
    assert a.lookback_days >= 15  # daily ATR14 on the 1d frame
    assert a.max_hold_days >= 3 * specs["max_hold"].hi / 24  # the engine's 3x cap on the longest time stop
    assert specs["max_hold"].lo % 4 == 0 and specs["max_hold"].step % 4 == 0  # every hold ends next to a 4h close
    assert set(a.regime_affinity) == set(Regime) and all(w == 1.0 for w in a.regime_affinity.values())
    for f in ("max_hold", "side_mode", "retail_filter"):
        assert isinstance(specs[f].clip(specs[f].default + 0.4), int)


def test_move_in_atr():
    assert math.isclose(move_in_atr(110.0, 100.0, 4.0), 0.10 / (4.0 / 110.0))
    assert move_in_atr(90.0, 100.0, 4.0) < 0
    for bad in ((NAN, 100.0, 4.0), (110.0, NAN, 4.0), (110.0, 100.0, NAN), (110.0, 100.0, 0.0), (110.0, 0.0, 4.0)):
        assert math.isnan(move_in_atr(*bad))


def test_long_with_a_leveraged_rally_geometry():
    s = PositioningFlowOiMom().evaluate(StubView(), ctx(), params())
    assert s is not None and s.side is Side.LONG and s.alpha == NAME and s.symbol == SYM
    assert s.entry_style is EntryStyle.MARKET and s.limit_price is None
    assert math.isclose(s.stop, 110.0 - 2.5 * 4.0) and s.take_profit is None and s.tp1 is None
    assert s.timeframe == "4h" and s.atr == 1.5 and s.trail_atr_mult == 0.0
    assert s.max_hold_bars == 48 * 60 - 1  # a minute before the decision bar 48h later
    assert s.tags["ref_price"] == 110.0 and math.isclose(s.tags["move_atr"], 0.10 * 110.0 / 4.0)
    assert s.tags["oi_chg_24h"] == 0.15 and s.tags["daily_atr"] == 4.0 and s.tags["ls_acc_chg_4h"] == -0.05
    assert not s.exit_on_regime_change and s.entry_ttl_bars is None
    assert 0.0 < s.confidence <= 1.0 and s.reason


def test_short_with_a_leveraged_drop_mirrors_long():
    v = StubView(h1=StubCursor(close=90.0, past=100.0), q=StubCursor(atr=1.5, close=90.0))
    s = PositioningFlowOiMom().evaluate(v, ctx(price=90.0, ex=extras(retail=0.05)), params())
    assert s is not None and s.side is Side.SHORT
    assert math.isclose(s.stop, 90.0 + 2.5 * 4.0) and s.take_profit is None
    assert s.tags["move_atr"] < -1.0


def test_needs_a_4h_close_fresh_positioning_an_oi_build_up_and_a_large_move():
    a = PositioningFlowOiMom()
    assert a.evaluate(StubView(closed=("1m", "5m", "15m", "1h")), ctx(), params()) is None  # not a 4h close
    assert a.evaluate(StubView(), ctx(ex={}), params()) is None  # no positioning data
    assert a.evaluate(StubView(), ctx(ex={"stale": True, "age_min": 45.0}), params()) is None  # feed gap
    assert a.evaluate(StubView(), ctx(ex=extras(oi=NAN)), params()) is None  # OI change unknown
    assert a.evaluate(StubView(), ctx(ex=extras(oi=0.05)), params()) is None  # +5% OI: no build-up
    assert a.evaluate(StubView(), ctx(ex=extras(oi=-0.15)), params()) is None  # deleveraging: not this setup
    assert a.evaluate(StubView(), ctx(ex=extras(oi=0.05)), params(oi_min=0.05)) is not None
    small = StubView(h1=StubCursor(close=103.0, past=100.0))  # +3% = 0.77 daily ATR
    assert a.evaluate(small, ctx(price=103.0), params()) is None
    assert a.evaluate(small, ctx(price=103.0), params(m_min=0.75)) is not None
    assert a.evaluate(StubView(d1=StubCursor(atr=NAN, close=104.0)), ctx(), params()) is None  # daily ATR warm-up
    assert a.evaluate(StubView(h1=StubCursor(close=110.0)), ctx(), params()) is None  # no close a day back


def test_retail_filter_needs_the_account_ratio_leaning_against_the_move():
    a = PositioningFlowOiMom()
    drop = StubView(h1=StubCursor(close=90.0, past=100.0))
    assert a.evaluate(StubView(), ctx(ex=extras(retail=0.05)), params()) is None  # retail accounts chased the rally
    assert a.evaluate(StubView(), ctx(ex=extras(retail=0.0)), params()) is None  # ratio unchanged: no opposition
    assert a.evaluate(StubView(), ctx(ex=extras(retail=NAN)), params()) is None  # unknown
    assert a.evaluate(StubView(), ctx(ex=extras(retail=0.05)), params(retail_filter=0)) is not None  # filter off
    assert a.evaluate(StubView(), ctx(ex=extras(retail=NAN)), params(retail_filter=0)) is not None
    assert a.evaluate(drop, ctx(price=90.0, ex=extras(retail=-0.05)), params()) is None  # retail chased the drop
    assert a.evaluate(drop, ctx(price=90.0, ex=extras(retail=0.05)), params()).side is Side.SHORT


def test_funding_filter_and_side_mode():
    a = PositioningFlowOiMom()
    assert a.evaluate(StubView(), ctx(funding=0.0005), params()) is None  # longs already pay 0.05%/8h
    assert a.evaluate(StubView(), ctx(funding=0.0005), params(fund_max=0.0)) is not None  # filter off
    assert a.evaluate(StubView(), ctx(funding=-0.0005), params()) is not None  # longs are paid
    drop = StubView(h1=StubCursor(close=90.0, past=100.0))
    short_ex = extras(retail=0.05)
    assert a.evaluate(drop, ctx(price=90.0, funding=-0.0005, ex=short_ex), params()) is None  # shorts pay
    assert a.evaluate(drop, ctx(price=90.0, funding=0.0005, ex=short_ex), params()) is not None
    assert a.evaluate(StubView(), ctx(), params(side_mode=-1)) is None
    assert a.evaluate(drop, ctx(price=90.0, ex=short_ex), params(side_mode=1)) is None
    assert a.evaluate(StubView(), ctx(), params(side_mode=1)).side is Side.LONG
    assert a.evaluate(drop, ctx(price=90.0, ex=short_ex), params(side_mode=-1)).side is Side.SHORT


def test_reads_only_completed_bars():
    reads: list = []
    v = StubView(h1=StubCursor(reads, close=110.0, past=100.0), d1=StubCursor(reads, atr=4.0, close=104.0),
                 q=StubCursor(reads, atr=1.5, close=110.0))
    assert PositioningFlowOiMom().evaluate(v, ctx(), params()) is not None
    assert reads and all(k >= 0 for _, k in reads)  # the last closed bar or earlier, never a future offset
    assert set(v.asked) <= {"1h", "1d", "4h"}


# --- real frames on synthetic 1m data -----------------------------------------------------------------------------
def _candles(days: int, seed: int = 0, noise: float = 0.0006, jump_day: int | None = None, jump: float = 0.0,
             start: int = DAY0) -> CandleArrays:
    """Random walk with an optional steady move of `jump` (log) spread over the 24 hours of day `jump_day`."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    steps = rng.normal(0.0, noise, n)
    if jump_day is not None:
        steps[jump_day * 1440:(jump_day + 1) * 1440] += jump / 1440.0
    close = 100.0 * np.exp(np.cumsum(steps))
    ca = CandleArrays("1m", capacity=n + 16)
    prev = float(close[0])
    for i, c in enumerate(close.tolist()):
        t = start + i * MS_MINUTE
        ca.append(Candle(t, prev, max(prev, c) * 1.0001, min(prev, c) * 0.9999, c, 10.0, 10.0 * c, 10, 5.0, t + 59_999))
        prev = c
    return ca


def _decide(ca: CandleArrays, t: int, ex: dict | None = None, funding: float = 0.0001, **over):
    view = MarketView(SYM)
    view.rebuild(ca, live=False)
    view.seek(t)
    assert view.closed("4h") and view.tf("4h").close_time == t and view.tf("1h").close_time == t
    return PositioningFlowOiMom().evaluate(view, ctx(now=t, ex=ex, price=view.tf("1h").v("close"), funding=funding),
                                           params(**over))


def test_decision_identical_with_and_without_future_data():
    T = DAY0 + 21 * MS_DAY - 1  # 23:59:59.999 on day 20, after a +12% day
    full = _candles(26, seed=4, jump_day=20, jump=0.12)
    n_head = (T + 1 - DAY0) // MS_MINUTE  # bars up to and including the decision bar
    head = CandleArrays("1m", capacity=n_head + 16)
    head.extend([full.candle_at(i) for i in range(n_head)])
    a, b = _decide(head, T), _decide(full, T)
    assert a is not None and b is not None and a.side is b.side is Side.LONG
    assert a.stop == b.stop and a.atr == b.atr and a.tags == b.tags
    assert a.tags["ref_price"] == float(full.close[n_head - 1])  # the decision bar's close, nothing later
    # every 4h decision of the head is unchanged by the appended days
    for k in range(1, 6 * 20):
        t = DAY0 + k * 4 * MS_HOUR - 1
        if t > T:
            break
        sa, sb = _decide(head, t), _decide(full, t)
        assert (sa is None) == (sb is None), t
        if sa is not None:
            assert sa.side is sb.side and sa.stop == sb.stop and sa.tags == sb.tags


def test_synthetic_rally_with_new_positions_is_followed_and_a_calm_day_or_deleveraging_is_not():
    T = DAY0 + 21 * MS_DAY - 1
    up = _candles(22, seed=2, jump_day=20, jump=0.12)
    assert _decide(up, T).side is Side.LONG
    assert _decide(up, T, ex=extras(oi=-0.12)) is None  # the same rally while OI collapsed (short covering)
    assert _decide(up, T, funding=0.0008) is None  # longs crowded and paying
    assert _decide(up, T, ex=extras(retail=0.05)) is None  # retail accounts chased it
    down = _candles(22, seed=2, jump_day=20, jump=-0.12)
    assert _decide(down, T, ex=extras(retail=0.05)).side is Side.SHORT
    calm = _candles(22, seed=2)
    assert _decide(calm, T) is None  # OI build-up alone, without a large move
    early = DAY0 + 10 * MS_DAY + 4 * MS_HOUR - 1  # 10 daily bars: no daily ATR14 yet
    assert _decide(_candles(12, seed=2, jump_day=9, jump=0.12), early) is None


# --- engine round trips ---------------------------------------------------------------------------------------------
def _metrics(days: int, build_day: int | None, build: float, retail: float = 0.0) -> MetricsSeries:
    """5-minute positioning rows: OI flat, then growing by `build` (fraction) over the last 4 hours of `build_day`
    (so both conditions first hold at that day's 23:59 close, after the move); the global account ratio changes by
    `retail` (absolute) over the same 4 hours."""
    ts = DAY0 + np.arange(days * 288, dtype=np.int64) * 300_000
    oi = np.full(len(ts), 1000.0)
    if build_day is not None:
        b = (build_day + 1) * 288
        a = b - 48
        oi[a:b] = 1000.0 * (1.0 + build * np.arange(1, b - a + 1) / (b - a))
        oi[b:] = 1000.0 * (1.0 + build)
    one = np.ones(len(ts))
    acc = np.full(len(ts), 1.5)
    if build_day is not None:
        acc[a:b] = 1.5 + retail * np.arange(1, b - a + 1) / (b - a)
        acc[b:] = 1.5 + retail
    return MetricsSeries(ts, oi, oi * 100.0, one, one, acc, one)


def _round_trip(jump: float, build: float, rate: float = 0.0001, days: int = 26, retail: float | None = None, **over):
    s = Settings(_env_file=None, DAILY_LOSS_LIMIT_PCT=1000.0, WEEKLY_LOSS_LIMIT_PCT=1000.0, MAX_DRAWDOWN_HALT_PCT=1000.0)
    ca = _candles(days, seed=7, noise=0.0002, jump_day=20, jump=jump)
    funding = {SYM: [(DAY0 + i * SETTLE, rate) for i in range(days * 3 + 1)]}
    retail = -0.1 * (1 if jump >= 0 else -1) if retail is None else retail  # by default retail accounts fade the move
    bt = Backtester(s, {SYM: info()}, {SYM: ca}, funding=funding, metrics={SYM: _metrics(days, 20, build, retail)})
    params_ = StrategyParams.default()
    params_.alphas[NAME].update(over)
    res = bt.run(params_, start=DAY0 + 17 * MS_DAY, only_alpha=NAME)
    return [t for t in res.trades if t["alpha"] == NAME]


def test_engine_round_trip_follows_the_leveraged_rally_and_ends_a_minute_before_the_decision_bar():
    trades = _round_trip(0.10, 0.20)
    assert len(trades) == 1, trades
    t0 = trades[0]
    assert t0["side"] == "LONG"
    assert t0["entry_time"] == DAY0 + 21 * MS_DAY  # filled at the open after the 23:59 close (OI built by then)
    assert "time stop" in t0["exit_reason"]
    assert t0["exit_time"] == t0["entry_time"] + 48 * MS_HOUR - MS_MINUTE - 1  # the bar before the decision bar
    assert t0["r_multiple"] == pytest.approx(t0["pnl"] / t0["risk_amount"], rel=1e-9)


def test_engine_round_trip_short_side_and_no_trade_without_build_up_or_against_funding():
    trades = _round_trip(-0.10, 0.20)
    assert trades and trades[0]["side"] == "SHORT"
    assert _round_trip(0.10, -0.20) == []  # rally while OI collapsed
    assert _round_trip(0.10, 0.20, rate=0.0006) == []  # longs pay 0.06%/8h
    assert _round_trip(0.10, 0.05) == []  # the rally with only +5% open interest
    assert _round_trip(0.10, 0.20, retail=0.1) == []  # retail accounts chased the rally
    assert _round_trip(0.10, 0.20, retail=0.1, retail_filter=0)  # ... which only the retail filter rejects
