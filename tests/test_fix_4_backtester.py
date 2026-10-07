"""Regression tests for the backtester fixes: upcoming funding rate in the Context, per-bar risk observation,
empty window after the last candle, and market entries filling at the next bar's open."""
import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import EntryStyle, Side, Signal
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.learning.backtester import FUNDING_INTERVAL, Backtester, BarPaperAccount
from heartless.strategy.base import Alpha
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from heartless.strategy.regime import REGIME_AFFINITY
from heartless.util.timeutil import MS_DAY, MS_MINUTE
from synth import synth_candles, synth_symbols

START = 1_700_006_400_000  # synth_candles default start: a funding boundary and 00:00 UTC


def _bt(n=3000, syms=("BTCUSDT",), funding=None):
    s = Settings(_env_file=None)
    candles = {name: synth_candles(n, seed=i, price=100 * (i + 1)) for i, name in enumerate(syms)}
    return s, Backtester(s, synth_symbols(list(syms)), candles, funding), candles


# --- 3) start after the last candle -------------------------------------------------------------------
def test_start_after_last_candle_returns_empty_result():
    s, bt, _ = _bt(400)
    last = int(bt._timeline[-1])
    res = bt.run(StrategyParams.default(), start=last + 1)
    assert res.stats["n"] == 0 and res.trades == [] and res.equity_curve == []
    assert res.final_equity == 10_000 and res.stats["final_equity"] == 10_000 and res.stats["return_pct"] == 0.0
    # end before start is just as empty
    res = bt.run(StrategyParams.default(), start=last, end=last - MS_MINUTE)
    assert res.stats["n"] == 0 and res.final_equity == 10_000


# --- 1) Context sees the upcoming (next-settlement) rate, apply_funding charges the settled one -----------
def test_context_funding_rate_is_upcoming_and_settlement_charges_it(monkeypatch):
    assert START % FUNDING_INTERVAL == 0
    t1, t2 = START + FUNDING_INTERVAL, START + 2 * FUNDING_INTERVAL
    rows = [(START, 0.0001), (t1, 0.0002), (t2, 0.0003)]
    s, bt, candles = _bt(1000, funding={"BTCUSDT": rows})  # 1000 bars = 16h40m: covers both settlements
    seen: dict[int, tuple[float, float]] = {}
    charged: list[tuple[str, float, int]] = []
    orig_on_bar = TradingEngine.on_bar
    orig_apply = PaperAccount.apply_funding

    async def rec_bar(self, view, ctx):
        seen[ctx.now] = (ctx.funding_rate, ctx.minutes_to_funding)
        return await orig_on_bar(self, view, ctx)

    async def rec_funding(self, symbol, rate, mark, ts):
        charged.append((symbol, rate, ts))
        return await orig_apply(self, symbol, rate, mark, ts)

    monkeypatch.setattr(TradingEngine, "on_bar", rec_bar)
    monkeypatch.setattr(PaperAccount, "apply_funding", rec_funding)
    bt.run(StrategyParams.default())

    # one minute before T2 the engine (funding-avoidance exit, funding_fade) sees ~the rate charged at T2 ...
    fr, mtf = seen[t2 - 1]
    assert mtf <= 2 and fr == pytest.approx(0.0003, abs=1e-6)
    # ... which is exactly what the account is then charged on the first bar after T2 (and 0.0002 after T1)
    assert (t2 + 59_999) in [c[2] for c in charged]
    assert {c[1] for c in charged if c[2] == t2 + 59_999} == {0.0003}
    assert {c[1] for c in charged if c[2] == t1 + 59_999} == {0.0002}
    # no oracle lookahead: right after T1 the estimate is still the rate settled at T1, converging towards T2's
    fr_start, _ = seen[t1 + 59_999]
    assert fr_start == pytest.approx(0.0002, abs=1e-6)
    fr_mid, _ = seen[t1 + FUNDING_INTERVAL // 2 - 1]
    assert 0.0002 < fr_mid < 0.0003 and fr_mid == pytest.approx(0.00025, abs=1e-6)
    # the settled lookup used by apply_funding is unchanged
    assert bt._funding_rate("BTCUSDT", t2 + 59_999) == 0.0003
    assert bt._funding_rate("BTCUSDT", t2 - 1) == 0.0002


def test_upcoming_funding_rate_falls_back_to_settled_rate_without_next_row():
    rows = [(START, 0.0001), (START + FUNDING_INTERVAL, 0.0002), (START + 3 * FUNDING_INTERVAL, 0.0009)]
    s, bt, _ = _bt(400, funding={"BTCUSDT": rows})
    t = START + FUNDING_INTERVAL + 4 * 3_600_000  # mid second bucket, next row is a whole interval late (gap)
    assert bt._upcoming_funding_rate("BTCUSDT", t) == 0.0002
    t = START + 3 * FUNDING_INTERVAL + 60_000 - 1  # last bucket: no next row at all
    assert bt._upcoming_funding_rate("BTCUSDT", t) == 0.0009
    assert bt._upcoming_funding_rate("ETHUSDT", t) == 0.0  # no funding history


# --- 2) risk limits are observed every bar, not once an hour ------------------------------------------
def test_daily_loss_halt_is_set_on_the_breaching_bar(monkeypatch):
    s, bt, _ = _bt(600)
    target = START + 90 * MS_MINUTE + 59_999  # close of bar 90: 01:30 UTC, mid-hour (90 % 60 == 30)
    observed: dict[str, int] = {}
    orig_on_bar = TradingEngine.on_bar

    async def rec_bar(self, view, ctx):
        if ctx.now == target:  # lose 5% (> daily 3% limit, < 15% peak halt) in the middle of the hour
            self.account.wallet *= 0.95
        if ctx.now == target + MS_MINUTE:
            observed["halted_until"] = self.risk.state.halted_until
            observed["day_start"] = self.risk.state.day_start_equity
        return await orig_on_bar(self, view, ctx)

    monkeypatch.setattr(TradingEngine, "on_bar", rec_bar)
    res = bt.run(StrategyParams.default())
    # the halt is in force on the very next bar (the hourly sample would only have caught it at bar 120)
    assert observed["halted_until"] == START + MS_DAY and observed["halted_until"] > target + MS_MINUTE
    assert observed["day_start"] == pytest.approx(10_000, rel=0.01)
    # the per-bar drawdown sees the intra-hour loss although the stored curve stays hourly
    assert res.stats["max_dd_pct"] >= 4.5
    assert len(res.equity_curve) == len(range(0, 600, 60))
    assert all(t not in (target,) for t, _ in res.equity_curve)


# --- 4) market entries fill at the NEXT bar's open ----------------------------------------------------
class OneShotLong(Alpha):
    name = "trend_pullback"
    timeframe = "1m"

    def __init__(self, candles, bars):
        self.c = candles
        self.bars = bars
        self.signalled: list[int] = []

    def evaluate(self, view, ctx, p):
        if not view.closed("1m"):
            return None
        ca = self.c[ctx.symbol]
        i = ca.index_at_or_before(view.cursor_time)
        if i not in self.bars:
            return None
        self.signalled.append(i)
        entry = float(ca.close[i])
        atr = entry * 0.004
        return Signal(self.name, ctx.symbol, Side.LONG, 0.9, "oneshot", entry - 1.5 * atr, entry + 3 * atr,
                      None, EntryStyle.MARKET, None, 40, 0.0, atr, "1m", {"ref_price": entry})


def _with_alpha(monkeypatch, alpha):
    """Make `alpha` the only alpha and let its signal clear the ensemble threshold in every regime."""
    orig = Ensemble.__init__

    def patched(self, params, bandit, alphas=None, only_alpha=None):
        orig(self, params, bandit, [alpha], only_alpha)

    monkeypatch.setattr(Ensemble, "__init__", patched)
    monkeypatch.setitem(REGIME_AFFINITY, alpha.name, {})  # -> default affinity 0.8; with only_alpha the weight is 1


def test_market_entry_fills_at_next_bar_open(monkeypatch):
    s, bt, candles = _bt(3000)
    ca = candles["BTCUSDT"]
    # synthetic bars are gapless (open == previous close): open a 3 bp up-gap on the candidate bars so a fill at
    # the next open is distinguishable from one at the signal bar's close
    for k in range(1501, 1701):
        ca.open[k] = ca.close[k - 1] * 1.0003
        ca.high[k] = max(ca.high[k], ca.open[k])
    alpha = OneShotLong(candles, set(range(1500, 1700)))
    _with_alpha(monkeypatch, alpha)
    res = bt.run(StrategyParams.default(), only_alpha=alpha.name)
    assert res.trades, "the oracle signal must have produced a trade"
    tr = min(res.trades, key=lambda t: t["entry_time"])
    # the entry is stamped at an OPEN time (not the close that produced the signal) ...
    assert (tr["entry_time"] - START) % 60_000 == 0
    j = int(np.searchsorted(ca.view("open_time"), tr["entry_time"]))
    assert int(ca.open_time[j]) == tr["entry_time"] and (j - 1) in alpha.signalled
    assert tr["entry_time"] > int(ca.close_time[j - 1]) >= int(ca.close_time[min(alpha.signalled)])
    # ... and executes at that bar's open + half spread + slippage (taker)
    half, qty = 1.0 / 2e4, tr["qty"]
    ref = float(ca.open[j]) * (1 + half)
    slip = min(1.5 + 0.3 * (qty * ref / 10_000.0), 25.0) / 1e4
    assert tr["entry_price"] == pytest.approx(ref * (1 + slip), rel=1e-9)
    assert tr["entry_price"] != pytest.approx(float(ca.close[j - 1]) * (1 + half) * (1 + slip), rel=1e-7)
    # pnl still reconciles with the account
    assert abs((res.final_equity - 10_000) - sum(t["pnl"] for t in res.trades)) < 1e-3


def test_entry_signalled_on_last_bar_is_not_filled_at_its_close(monkeypatch):
    s, bt, candles = _bt(3000)
    n = candles["BTCUSDT"].n
    alpha = OneShotLong(candles, {n - 1})
    _with_alpha(monkeypatch, alpha)
    sent: list[tuple[str, bool]] = []
    orig = BarPaperAccount.market_order

    async def rec(self, symbol, side, qty, reduce_only=False, client_id=""):
        sent.append((side, reduce_only))
        return await orig(self, symbol, side, qty, reduce_only=reduce_only, client_id=client_id)

    monkeypatch.setattr(BarPaperAccount, "market_order", rec)
    res = bt.run(StrategyParams.default(), only_alpha=alpha.name)
    assert ("BUY", False) in sent  # the entry was sent on the last bar ...
    assert res.stats["n"] == 0 and res.final_equity == 10_000  # ... but no bar followed, so nothing filled


async def test_bar_account_reduce_only_market_orders_stay_synchronous():
    acc = BarPaperAccount(initial_balance=1000, slippage_bps=0, impact_bps_per_10k=0, spread_bps=0)
    from heartless.core.models import Candle

    def bar(o, h, l, c, t0):
        return Candle(t0, o, h, l, c, 1, 1, 1, 1, t0 + 59_999)

    await acc.on_bar("X", bar(100, 100, 100, 100, 0))
    r = await acc.market_order("X", "BUY", 1)
    assert r.status == "NEW" and acc.positions.get("X") is None
    q = await acc.query_order("X", r.order_id)
    assert q.status == "NEW"
    await acc.on_bar("X", bar(101, 102, 100, 101.5, 60_000))  # fills at the open (101), not the signal close (100)
    assert acc.positions["X"].qty == pytest.approx(1.0) and acc.positions["X"].entry == pytest.approx(101.0)
    assert (await acc.query_order("X", r.order_id)).status == "FILLED"
    r2 = await acc.market_order("X", "SELL", 1, reduce_only=True)
    assert r2.status == "FILLED" and acc.positions["X"].qty == 0
    # a resting entry that is cancelled never fills
    r3 = await acc.market_order("X", "BUY", 1)
    assert await acc.cancel_order("X", r3.order_id)
    await acc.on_bar("X", bar(101, 102, 100, 101, 120_000))
    assert acc.positions["X"].qty == 0
