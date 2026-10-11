"""xs_rank_mom: daily cross-sectional momentum. Signal geometry and gating on stub views, then on a real 1d frame with
market context computed by the offline builder (data/context.py) from synthetic data: causality (the decision at a
daily close is the same when later bars exist), sanity (the strongest symbol goes long, the weakest short, a middling
one stays out) and one engine round trip (market entry at 00:00 UTC, time stop a minute before the 3rd daily close,
renewed by that close)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle, EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.core.store import Store
from heartless.data import context as C
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.learning.backtester import Backtester
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.xs_rank_mom import MIN_N, XsRankMom
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_MINUTE

NAN = float("nan")
SYM = "ETHUSDT"
DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC


class StubCursor:
    ok = True

    def __init__(self, reads: list | None = None, **series):
        self.series = series
        self.reads = reads

    def v(self, name: str, k: int = 0) -> float:
        if self.reads is not None:
            self.reads.append((name, k))
        return float(self.series.get(name, NAN)) if k == 0 else NAN


class StubView:
    def __init__(self, d1: StubCursor | None = None, closed: tuple = ("1m", "5m", "15m", "1h", "4h", "1d")):
        self._d1 = d1 if d1 is not None else StubCursor(close=100.0, atr=4.0)
        self._closed = closed
        self.asked: list = []

    def tf(self, tf: str) -> StubCursor:
        self.asked.append(tf)
        return self._d1 if tf == "1d" else StubCursor()

    def closed(self, tf: str) -> bool:
        return tf in self._closed


INFO = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                  price_precision=2, quantity_precision=3)


def ctx(market: dict | None = None, funding: float = 0.0001, price: float = 100.0, symbol: str = SYM) -> Context:
    c = Context(symbol=symbol, info=INFO, ticker=Ticker(symbol, bid=price * 0.9999, ask=price * 1.0001, mark=price,
                                                         ts=1), regime=Regime.VOLATILE, funding_rate=funding)
    if market is not None:
        c.market = market
    return c


def mkt(rank: float, n: float = 20.0, key: str = "ret_168h_rank", **more) -> dict:
    return {key: rank, "n": n, **more}


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["xs_rank_mom"])
    p.update(over)
    return p


def test_registered_daily_off_by_default_with_consistent_history_and_hold_bounds():
    a = ALPHA_BY_NAME["xs_rank_mom"]
    assert isinstance(a, XsRankMom) and a.timeframe == "1d" and a.enabled_by_default is False
    assert StrategyParams.default().enabled["xs_rank_mom"] is False
    specs = {s.name: s for s in a.param_specs}
    for s in a.param_specs:
        assert s.clip(s.default) == s.default
    # every rankable look-back exists in the market context, and the daily ATR14 seed fits in the declared history
    assert all(f"ret_{h}h_rank" in C.SYMBOL_FEATURES for h in specs["lookback_h"].choices)
    assert a.lookback_days >= 14 + 1
    # chunked lab runs hand off positions held up to the engine's cap of 3x the longest time stop
    assert a.max_hold_days * 24 >= 3 * max(specs["hold_h"].choices)
    assert set(a.regime_affinity) == set(Regime) and all(w == 1.0 for w in a.regime_affinity.values())


def test_long_geometry_market_entry_no_target_time_stop_before_the_daily_close():
    s = XsRankMom().evaluate(StubView(), ctx(mkt(0.95)), params())
    assert s is not None and s.side is Side.LONG and s.alpha == "xs_rank_mom"
    assert s.entry_style is EntryStyle.MARKET and s.limit_price is None
    assert math.isclose(s.stop, 100.0 - 3.0 * 4.0) and s.take_profit is None and s.tp1 is None
    assert s.timeframe == "1d" and s.atr == 4.0 and s.trail_atr_mult == 0.0
    assert s.max_hold_bars == 72 * 60 - 1 and s.tags["ref_price"] == 100.0  # ends a minute before the daily close
    assert s.tags["rank"] == 0.95 and s.tags["lookback_h"] == 168 and s.tags["n"] == 20
    assert not s.exit_on_regime_change and s.entry_ttl_bars is None and 0 < s.confidence <= 1


def test_short_mirrors_long():
    s = XsRankMom().evaluate(StubView(), ctx(mkt(0.05), funding=-0.0001), params())
    assert s is not None and s.side is Side.SHORT
    assert math.isclose(s.stop, 100.0 + 3.0 * 4.0) and s.take_profit is None


def test_rank_bands_follow_q():
    a = XsRankMom()
    side = lambda r, **o: (lambda s: s and s.side)(a.evaluate(StubView(), ctx(mkt(r)), params(**o)))  # noqa: E731
    # q = 0.2 on 20 symbols (ranks i / 19): the top four long (>= 16/19), the bottom four short (<= 3/19)
    assert side(16 / 19) is Side.LONG and side(15 / 19) is None
    assert side(3 / 19) is Side.SHORT and side(4 / 19) is None
    assert side(0.5) is None
    # q = 0.1: only the top / bottom two
    assert side(18 / 19, q=0.1) is Side.LONG and side(17 / 19, q=0.1) is None
    assert side(1 / 19, q=0.1) is Side.SHORT and side(2 / 19, q=0.1) is None


def test_lookback_selects_the_ranked_return():
    a = XsRankMom()
    m = {"ret_24h_rank": 0.0, "ret_72h_rank": 0.5, "ret_168h_rank": 1.0, "n": 20.0}
    assert a.evaluate(StubView(), ctx(m), params()).side is Side.LONG
    assert a.evaluate(StubView(), ctx(m), params(lookback_h=72)) is None
    assert a.evaluate(StubView(), ctx(m, funding=-0.0001), params(lookback_h=24)).side is Side.SHORT


def test_missing_context_small_cross_section_warm_up_and_unclosed_bar_give_nothing():
    a = XsRankMom()
    assert a.evaluate(StubView(), ctx(), params()) is None  # no market context at all
    assert a.evaluate(StubView(), ctx({"n": 20.0}), params()) is None  # symbol outside the hour's cross-section
    assert a.evaluate(StubView(), ctx(mkt(NAN)), params()) is None
    assert a.evaluate(StubView(), ctx(mkt(1.0, n=MIN_N - 1)), params()) is None
    assert a.evaluate(StubView(), ctx(mkt(1.0, n=NAN)), params()) is None
    assert a.evaluate(StubView(), ctx(mkt(1.0, n=MIN_N)), params()) is not None
    assert a.evaluate(StubView(StubCursor(close=100.0)), ctx(mkt(1.0)), params()) is None  # no daily ATR yet
    assert a.evaluate(StubView(closed=("1m", "5m", "15m", "1h", "4h")), ctx(mkt(1.0)), params()) is None


def test_funding_filter_skips_the_side_that_pays():
    a = XsRankMom()
    assert a.evaluate(StubView(), ctx(mkt(1.0), funding=0.0005), params()) is None  # a long would pay 0.05% / 8h
    assert a.evaluate(StubView(), ctx(mkt(1.0), funding=0.0005), params(fund_max=0.0)) is not None  # filter off
    assert a.evaluate(StubView(), ctx(mkt(1.0), funding=-0.0005), params()) is not None  # longs are paid
    assert a.evaluate(StubView(), ctx(mkt(0.0), funding=-0.0005), params()) is None
    assert a.evaluate(StubView(), ctx(mkt(0.0), funding=0.0005), params()) is not None


def test_volume_share_filter():
    a = XsRankMom()
    p = params(vol_rank_min=0.5)
    assert a.evaluate(StubView(), ctx(mkt(1.0)), p) is None  # unknown volume share
    assert a.evaluate(StubView(), ctx(mkt(1.0, vol_share_rank=0.5)), p) is None
    assert a.evaluate(StubView(), ctx(mkt(1.0, vol_share_rank=0.6)), p).side is Side.LONG
    assert a.evaluate(StubView(), ctx(mkt(0.0, vol_share_rank=0.9), funding=0.0), p).side is Side.SHORT
    assert a.evaluate(StubView(), ctx(mkt(1.0, vol_share_rank=0.1)), params()) is not None  # off by default


def test_reads_only_completed_daily_bars():
    reads: list = []
    v = StubView(StubCursor(reads, close=100.0, atr=4.0))
    assert XsRankMom().evaluate(v, ctx(mkt(1.0)), params()) is not None
    assert reads and all(k >= 0 for _, k in reads)  # never a negative (future) offset
    assert set(v.asked) == {"1d"}


@pytest.mark.parametrize("field", ["lookback_h", "hold_h"])
def test_choice_params_snap_to_their_choices(field):
    spec = {s.name: s for s in XsRankMom.param_specs}[field]
    assert spec.clip(spec.default + 7) in spec.choices and spec.clip(1000) == max(spec.choices)


# --- real 1d frames and market context on synthetic data ----------------------------------------------------------
UNIVERSE = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
            "BCHUSDT")
UP, MID, DOWN = "SOLUSDT", "ETHUSDT", "DOGEUSDT"
DRIFT = {UP: 0.012, DOWN: -0.012, MID: 0.0005}  # log drift per day; the others -0.5% .. +0.4% a day
DAYS = 21


def _bars(sym: str, days: int, every: int = 1) -> list[Candle]:
    """1m bars (or one bar every `every` minutes, enough for the context of a symbol that is never traded):
    UP / DOWN trend by 1.2% a day, MID and the rest drift slowly in between."""
    k = UNIVERSE.index(sym)
    rng = np.random.default_rng(100 + k)
    drift = DRIFT.get(sym, 0.001 * (k - 5))
    n = days * 1440
    noise = 0.0003 if sym in (UP, DOWN) else 0.00005  # the rest stay well inside the two extremes
    close = (50.0 + 10 * k) * np.exp(np.cumsum(np.full(n, drift / 1440.0) + rng.normal(0.0, noise, n)))
    if sym in (UP, DOWN):  # a 2% intraday swing (zero at every daily close): a daily ATR of ~4-5% of price
        close = close * (1.0 + 0.02 * np.sin(2 * np.pi * (np.arange(n) + 1) / 1440.0))
    out = []
    prev = float(close[0])
    for i in range(every - 1, n, every):
        c = float(close[i])
        t = DAY0 + i * MS_MINUTE
        out.append(Candle(t, prev, max(prev, c) * 1.0003, min(prev, c) * 0.9997, c, 10.0, 10.0 * c, 10, 5.0,
                          t + 59_999))
        prev = c
    return out


@pytest.fixture(scope="module")
def bars() -> dict[str, list[Candle]]:
    return {s: _bars(s, DAYS, 1 if s in DRIFT else 60) for s in UNIVERSE}


def _table(bars: dict[str, list[Candle]], until: int, path) -> C.ContextTable:
    """Market context of every hour close up to `until`, computed by the offline builder from a store that holds only
    the bars closed by `until`."""
    st = Store(path)
    for s, bs in bars.items():
        st.save_candles(s, [b for b in bs if b.close_time <= until])
    snaps = C.compute_unit(st, UNIVERSE, DAY0, until)
    st.close()
    market = {sn.ts: sn.mkt for sn in snaps}
    per = {s: {sn.ts: sn.sym[j] for sn in snaps if sn.present[j]} for j, s in enumerate(snaps[0].symbols)}
    return C.ContextTable(market, per)


@pytest.fixture(scope="module")
def full_table(bars, tmp_path_factory):
    return _table(bars, DAY0 + DAYS * MS_DAY - 1, tmp_path_factory.mktemp("xs") / "full.db")


def _arrays(bs: list[Candle], until: int | None = None) -> CandleArrays:
    keep = [b for b in bs if until is None or b.close_time <= until]
    ca = CandleArrays("1m", capacity=len(keep) + 16)
    ca.extend(keep)
    return ca


def _decide(ca: CandleArrays, table: C.ContextTable, sym: str, day_index: int):
    view = MarketView(sym)
    view.rebuild(ca, live=False)
    t = DAY0 + (day_index + 1) * MS_DAY - 1
    view.seek(t)
    assert view.closed("1d") and view.tf("1d").close_time == t
    m = table.view(sym, t)
    assert m["ts"] == t  # the snapshot of the decision hour close itself
    return XsRankMom().evaluate(view, ctx(m, funding=0.0, price=view.tf("1d").v("close"), symbol=sym), params())


def test_decision_identical_with_and_without_future_data(bars, full_table, tmp_path):
    decide_on = 15
    t = DAY0 + (decide_on + 1) * MS_DAY - 1
    head_table = _table(bars, t, tmp_path / "head.db")
    for sym in (UP, DOWN):
        a = _decide(_arrays(bars[sym], t), head_table, sym, decide_on)
        b = _decide(_arrays(bars[sym]), full_table, sym, decide_on)
        assert a is not None and b is not None and a.side is b.side
        assert (a.stop, a.atr, a.tags["rank"], a.tags["n"], a.tags["ref_price"]) == \
               (b.stop, b.atr, b.tags["rank"], b.tags["n"], b.tags["ref_price"])


def test_strongest_symbol_goes_long_weakest_short_middle_stays_out(bars, full_table):
    s_up = _decide(_arrays(bars[UP]), full_table, UP, 15)
    s_dn = _decide(_arrays(bars[DOWN]), full_table, DOWN, 15)
    assert s_up is not None and s_up.side is Side.LONG and s_up.tags["rank"] == 1.0 and s_up.tags["n"] == 10
    assert s_up.stop < s_up.tags["ref_price"]
    assert s_dn is not None and s_dn.side is Side.SHORT and s_dn.tags["rank"] == 0.0
    assert s_dn.stop > s_dn.tags["ref_price"]
    assert _decide(_arrays(bars[MID]), full_table, MID, 15) is None
    assert _decide(_arrays(bars[UP]), full_table, UP, 5) is None  # no 168h return (and no daily ATR) yet


def test_engine_round_trip_enters_after_the_daily_close_and_renews_at_the_time_stop(bars, full_table):
    # the first decision with a daily ATR14 is the day-13 close: the long opens at the next 00:00 UTC open, is closed
    # by the time stop on the 23:58 bar before its 3rd daily close (3.6% in three days is ~0.45R against the
    # 3-ATR stop: not the 0.5R that would let it run), and that same close renews it because the symbol still ranks first
    s = Settings(_env_file=None, DAILY_LOSS_LIMIT_PCT=1000.0, WEEKLY_LOSS_LIMIT_PCT=1000.0, MAX_DRAWDOWN_HALT_PCT=1000.0)
    info = {UP: SymbolInfo(UP, "SOL", "USDT", tick_size=0.001, step_size=0.01, min_qty=0.01, min_notional=5.0,
                           price_precision=3, quantity_precision=2)}
    bt = Backtester(s, info, {UP: _arrays(bars[UP])}, market=full_table)
    res = bt.run(StrategyParams.default(), only_alpha="xs_rank_mom")
    trades = [t for t in res.trades if t["alpha"] == "xs_rank_mom"]
    assert trades, res.stats
    t0 = trades[0]
    assert t0["side"] == "LONG" and t0["entry_time"] == DAY0 + 14 * MS_DAY  # filled at the open after the day-13 close
    assert "time stop" in t0["exit_reason"]
    assert t0["exit_time"] == t0["entry_time"] + 3 * MS_DAY - MS_MINUTE - 1  # close of the 23:58 bar
    assert math.isfinite(t0["r_multiple"]) and abs(t0["r_multiple"]) < 1.0
    assert len(trades) >= 2 and trades[1]["side"] == "LONG"
    assert trades[1]["entry_time"] == t0["entry_time"] + 3 * MS_DAY  # renewed at the same daily close
