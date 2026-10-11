"""funding_carry_xs: fade a settled funding rate far from the cross-sectional median right after a settlement.
Signal geometry and gating on stub views, causality on a real MarketView, sanity on synthetic data, and engine round
trips that check the settlements held over are paid / received inside the trade's pnl and R."""
from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle, EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.learning.backtester import Backtester
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.funding_carry_xs import FundingCarryXs, decision_hour
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

NAN = float("nan")
SYM = "ETHUSDT"
DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
SETTLE = 8 * MS_HOUR
T_DEC = DAY0 + 16 * MS_DAY + 8 * MS_HOUR + MS_HOUR - 1  # 08:59:59.999 UTC on day 16: first hour close after 08:00


def info() -> SymbolInfo:
    return SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                      price_precision=2, quantity_precision=3)


class StubCursor:
    ok = True

    def __init__(self, reads: list | None = None, **vals):
        self.vals = vals
        self.reads = reads

    def v(self, name: str, k: int = 0) -> float:
        if self.reads is not None:
            self.reads.append((name, k))
        return float(self.vals.get(name, NAN)) if k == 0 else NAN


class StubView:
    def __init__(self, d1: StubCursor | None = None, h1: StubCursor | None = None, closed=("1m", "5m", "15m", "1h")):
        self.d1 = d1 if d1 is not None else StubCursor(atr=4.0, close=99.0)
        self.h1 = h1 if h1 is not None else StubCursor(close=100.0)
        self._closed = closed
        self.asked: list = []

    def tf(self, tf: str) -> StubCursor:
        self.asked.append(tf)
        return {"1d": self.d1, "1h": self.h1}.get(tf, StubCursor())

    def closed(self, tf: str) -> bool:
        return tf in self._closed


def market(ts: int = T_DEC, funding: float = -0.0008, median: float = 0.0001, oi: float = 0.03, **more):
    d = {"ts": ts, "funding": funding, "funding_median": median, "oi_chg_24h": oi, "funding_rank": 0.0}
    d.update(more)
    return MappingProxyType(d)


def ctx(now: int = T_DEC, mkt=None, price: float = 100.0) -> Context:
    c = Context(symbol=SYM, info=info(), ticker=Ticker(SYM, bid=price * 0.99995, ask=price * 1.00005, mark=price, ts=now),
                regime=Regime.VOLATILE, now=now)
    c.market = market(ts=now) if mkt is None else mkt
    return c


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["funding_carry_xs"])
    p.update(over)
    return p


def test_registered_off_by_default_hourly_with_history_and_hold_bounds():
    a = ALPHA_BY_NAME["funding_carry_xs"]
    assert isinstance(a, FundingCarryXs) and a.timeframe == "1h" and a.enabled_by_default is False
    assert StrategyParams.default().enabled["funding_carry_xs"] is False
    specs = {s.name: s for s in a.param_specs}
    for s in a.param_specs:
        assert s.clip(s.default) == s.default
    assert a.lookback_days >= 15  # daily ATR14 on the 1d frame
    assert a.max_hold_days >= 3 * specs["hold_k"].hi * 8 / 24  # the engine's 3x cap on the longest time stop
    assert set(a.regime_affinity) == set(Regime) and all(w == 1.0 for w in a.regime_affinity.values())
    for f in ("hold_k", "oi_filter", "side_mode"):
        assert isinstance(specs[f].clip(specs[f].default + 0.4), int)


def test_decision_hour_is_the_first_hour_close_after_each_settlement():
    for h in (0, 8, 16):
        assert decision_hour(DAY0 + h * MS_HOUR + MS_HOUR - 1)
    for t in (DAY0 + MS_HOUR * 2 - 1, DAY0 + 8 * MS_HOUR - 1, DAY0 + 8 * MS_HOUR, DAY0 + 8 * MS_HOUR + MS_HOUR,
              DAY0 + 23 * MS_HOUR + MS_HOUR - 1):
        assert not decision_hour(t)


def test_long_against_paying_shorts_geometry():
    s = FundingCarryXs().evaluate(StubView(), ctx(), params())
    assert s is not None and s.side is Side.LONG and s.alpha == "funding_carry_xs"
    assert s.entry_style is EntryStyle.MARKET and s.limit_price is None
    assert math.isclose(s.stop, 100.0 - 2.5 * 4.0) and s.take_profit is None and s.tp1 is None
    assert s.timeframe == "1d" and s.atr == 4.0 and s.trail_atr_mult == 0.0
    assert s.max_hold_bars == 9 * 480 - 1  # a minute before the decision bar of the 9th next settlement
    assert s.tags["ref_price"] == 100.0 and math.isclose(s.tags["funding_dev"], -0.0009)
    assert not s.exit_on_regime_change and s.entry_ttl_bars is None


def test_short_against_paying_longs_mirrors_long():
    s = FundingCarryXs().evaluate(StubView(), ctx(mkt=market(funding=0.0006, median=0.0001)), params())
    assert s is not None and s.side is Side.SHORT
    assert math.isclose(s.stop, 100.0 + 2.5 * 4.0) and s.take_profit is None


def test_needs_a_settlement_hour_a_fresh_snapshot_and_a_large_enough_deviation():
    a = FundingCarryXs()
    assert a.evaluate(StubView(), ctx(now=T_DEC + MS_HOUR), params()) is None  # not right after a settlement
    assert a.evaluate(StubView(closed=("1m", "5m")), ctx(), params()) is None  # hour not closed
    assert a.evaluate(StubView(), ctx(mkt=MappingProxyType({})), params()) is None  # no market context
    assert a.evaluate(StubView(), ctx(mkt=market(ts=T_DEC - MS_HOUR)), params()) is None  # stale snapshot
    assert a.evaluate(StubView(), ctx(mkt=market(funding=NAN)), params()) is None  # symbol not in the cross-section
    assert a.evaluate(StubView(), ctx(mkt=market(funding=-0.0001)), params()) is None  # 2bp below: not crowded
    assert a.evaluate(StubView(), ctx(mkt=market(funding=0.0003)), params()) is None  # 2bp above: not crowded
    assert a.evaluate(StubView(), ctx(mkt=market(funding=-0.0001)), params(d_min=0.0002)) is not None
    assert a.evaluate(StubView(d1=StubCursor(atr=NAN, close=99.0)), ctx(), params()) is None  # daily ATR warm-up


def test_open_interest_filter_and_side_mode():
    a = FundingCarryXs()
    assert a.evaluate(StubView(), ctx(mkt=market(oi=-0.02)), params()) is None  # leverage leaving: no crowd build-up
    assert a.evaluate(StubView(), ctx(mkt=market(oi=NAN)), params()) is None  # unknown
    assert a.evaluate(StubView(), ctx(mkt=market(oi=-0.02)), params(oi_filter=0)) is not None
    assert a.evaluate(StubView(), ctx(mkt=market(oi=0.03)), params(oi_min=0.05)) is None
    short = market(funding=0.0006)
    assert a.evaluate(StubView(), ctx(), params(side_mode=-1)) is None
    assert a.evaluate(StubView(), ctx(mkt=short), params(side_mode=1)) is None
    assert a.evaluate(StubView(), ctx(), params(side_mode=1)).side is Side.LONG
    assert a.evaluate(StubView(), ctx(mkt=short), params(side_mode=-1)).side is Side.SHORT


def test_reads_only_completed_bars():
    reads: list = []
    v = StubView(d1=StubCursor(reads, atr=4.0, close=99.0), h1=StubCursor(reads, close=100.0))
    assert FundingCarryXs().evaluate(v, ctx(), params()) is not None
    assert reads and all(k == 0 for _, k in reads)  # the last closed bar only, never a future offset
    assert set(v.asked) <= {"1d", "1h"}


# --- real frames on synthetic 1m data -----------------------------------------------------------------------------
def _candles(days: int, seed: int = 0, noise: float = 0.0006, drift: float = 0.0, start: int = DAY0) -> CandleArrays:
    rng = np.random.default_rng(seed)
    n = days * 1440
    close = 100.0 * np.exp(np.cumsum(drift / 1440.0 + rng.normal(0.0, noise, n)))
    ca = CandleArrays("1m", capacity=n + 16)
    prev = float(close[0])
    for i, c in enumerate(close.tolist()):
        t = start + i * MS_MINUTE
        ca.append(Candle(t, prev, max(prev, c) * 1.0001, min(prev, c) * 0.9999, c, 10.0, 10.0 * c, 10, 5.0, t + 59_999))
        prev = c
    return ca


def _decide(ca: CandleArrays, t: int, mkt=None, **over):
    view = MarketView(SYM)
    view.rebuild(ca, live=False)
    view.seek(t)
    assert view.closed("1h") and view.tf("1h").close_time == t
    return FundingCarryXs().evaluate(view, ctx(now=t, mkt=mkt, price=view.tf("1h").v("close")), params(**over))


def test_decision_identical_with_and_without_future_data():
    full = _candles(20, seed=4)
    n_head = (T_DEC + 1 - DAY0) // MS_MINUTE  # bars up to and including the decision bar
    head = CandleArrays("1m", capacity=n_head + 16)
    head.extend([full.candle_at(i) for i in range(n_head)])
    a, b = _decide(head, T_DEC), _decide(full, T_DEC)
    assert a is not None and b is not None and a.side is b.side is Side.LONG
    assert a.stop == b.stop and a.atr == b.atr and a.tags["ref_price"] == b.tags["ref_price"]
    assert a.tags["ref_price"] == float(full.close[n_head - 1])  # the decision bar's close, nothing later


def test_synthetic_crowding_gives_the_contrarian_side_and_a_calm_cross_section_nothing():
    ca = _candles(20, seed=2)
    assert _decide(ca, T_DEC, market(funding=-0.0010)).side is Side.LONG  # shorts pay 0.1%/8h
    assert _decide(ca, T_DEC, market(funding=0.0009, median=0.0001)).side is Side.SHORT
    assert _decide(ca, T_DEC, market(funding=0.0001, median=0.0001)) is None  # everyone at the 0.01% default
    early = DAY0 + 10 * MS_DAY + MS_HOUR - 1  # 10 daily bars: no daily ATR14 yet
    assert _decide(ca, early, market(ts=early)) is None


class StubMarket:
    """ctx.market source for the backtester: the snapshot of the last hour close at or before t (as ContextTable)."""

    def __init__(self, funding: float, median: float = 0.0001, oi: float = 0.05):
        self.funding, self.median, self.oi = funding, median, oi

    def view(self, symbol: str, t: int):
        T = (t + 1) // MS_HOUR * MS_HOUR - 1
        return market(ts=T, funding=self.funding, median=self.median, oi=self.oi)


def _round_trip(rate: float, days: int = 24, seed: int = 7, **over):
    s = Settings(_env_file=None, DAILY_LOSS_LIMIT_PCT=1000.0, WEEKLY_LOSS_LIMIT_PCT=1000.0, MAX_DRAWDOWN_HALT_PCT=1000.0)
    ca = _candles(days, seed=seed, noise=0.0002)
    funding = {SYM: [(DAY0 + i * SETTLE, rate) for i in range(days * 3 + 1)]}
    bt = Backtester(s, {SYM: info()}, {SYM: ca}, funding=funding, market=StubMarket(rate))
    params_ = StrategyParams.default()
    params_.alphas["funding_carry_xs"].update(over)
    res = bt.run(params_, only_alpha="funding_carry_xs")
    return [t for t in res.trades if t["alpha"] == "funding_carry_xs"]


def test_engine_round_trip_collects_the_funding_and_renews_after_the_time_stop():
    # shorts pay 0.1% every 8h: long right after a settlement, hold 9 settlements, renewed on the next decision bar
    trades = _round_trip(-0.001)
    assert len(trades) >= 2, trades
    t0 = trades[0]
    assert t0["side"] == "LONG"
    assert (t0["entry_time"] - DAY0) % SETTLE == MS_HOUR  # filled at the open after the 00:59 / 08:59 / 16:59 close
    assert "time stop" in t0["exit_reason"]
    assert t0["exit_time"] == t0["entry_time"] + 9 * SETTLE - MS_MINUTE - 1  # close of the bar before the decision bar
    assert trades[1]["entry_time"] == t0["entry_time"] + 9 * SETTLE  # renewed on that decision bar
    # all nine settlements held over were received and are part of the trade's pnl and R
    notional = t0["qty"] * t0["entry_price"]
    assert t0["funding"] == pytest.approx(9 * 0.001 * notional, rel=0.03)
    assert t0["pnl"] == pytest.approx(t0["gross"] - t0["fees"] + t0["funding"], rel=1e-9, abs=1e-9)
    assert t0["r_multiple"] == pytest.approx(t0["pnl"] / t0["risk_amount"], rel=1e-9)


def test_engine_round_trip_short_side_receives_positive_funding():
    trades = _round_trip(0.0009, oi_filter=0)
    assert trades and trades[0]["side"] == "SHORT"
    t0 = trades[0]
    assert t0["funding"] == pytest.approx(9 * 0.0009 * t0["qty"] * t0["entry_price"], rel=0.03)
    assert t0["funding"] > 0 and t0["r_multiple"] == pytest.approx(t0["pnl"] / t0["risk_amount"], rel=1e-9)


def test_no_trade_when_the_funding_sits_at_the_default():
    assert _round_trip(0.0001) == []
