"""End-to-end: engine + paper account + alphas on synthetic data, plus an oracle alpha that must profit."""
import numpy as np

from heartless.config import Settings
from heartless.core.models import EntryStyle, Side, Signal
from heartless.learning.backtester import Backtester
from heartless.strategy.base import Alpha
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from synth import synth_candles, synth_symbols


def _bt(n=8000, syms=("BTCUSDT", "ETHUSDT")):
    s = Settings(_env_file=None)
    candles = {name: synth_candles(n, seed=i, price=100 * (i + 1)) for i, name in enumerate(syms)}
    return s, Backtester(s, synth_symbols(list(syms)), candles), candles


def test_backtest_runs_and_accounts_consistently():
    s, bt, _ = _bt()
    res = bt.run(StrategyParams.default())
    st = res.stats
    assert st["n"] >= 1
    # closed-trade pnl must reconcile with the account equity change
    assert abs((res.final_equity - 10_000) - sum(t["pnl"] for t in res.trades)) < 1e-3
    for t in res.trades:
        assert t["exit_time"] >= t["entry_time"]
        assert abs(t["r_multiple"]) < 10
        assert t["alpha"] in StrategyParams.default().alphas or t["alpha"] == "adopted"


def test_stop_losses_are_close_to_one_r():
    s, bt, _ = _bt(12000, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    res = bt.run(StrategyParams.default())
    sl = [t["r_multiple"] for t in res.trades if t["exit_reason"].startswith("손절")]
    if len(sl) >= 5:
        assert -2.0 < float(np.mean(sl)) < -0.6


class Oracle(Alpha):
    name = "trend_pullback"
    timeframe = "1m"

    def __init__(self, candles):
        self.c = candles

    def evaluate(self, view, ctx, p):
        if not view.closed("1m"):
            return None
        ca = self.c[ctx.symbol]
        i = ca.index_at_or_before(view.cursor_time)
        if i < 0 or i + 30 >= ca.n:
            return None
        fut = ca.close[i + 30] / ca.close[i] - 1
        atr = view.tf("5m").v("atr")
        if not atr or atr != atr or abs(fut) < 0.004:
            return None
        side = Side.LONG if fut > 0 else Side.SHORT
        entry = ca.close[i]
        return Signal(self.name, ctx.symbol, side, 0.9, "oracle", entry - side.sign * 1.5 * atr, entry + side.sign * 3 * atr,
                      None, EntryStyle.MARKET, None, 40, 0.0, atr, "1m", {"ref_price": entry})


def test_execution_plumbing_profits_with_perfect_foresight(monkeypatch):
    s, bt, candles = _bt(6000)
    oracle = Oracle(candles)
    orig = Ensemble.__init__

    def patched(self, params, bandit, alphas=None, only_alpha=None):
        orig(self, params, bandit, [oracle], only_alpha)

    monkeypatch.setattr(Ensemble, "__init__", patched)
    res = bt.run(StrategyParams.default())
    assert res.stats["n"] > 20
    assert res.stats["net"] > 0 and res.stats["win_rate"] > 55
