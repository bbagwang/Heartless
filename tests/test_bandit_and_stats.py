from heartless.core.models import Regime
from heartless.execution.stats import objective, sparkline, summarize
from heartless.learning.bandit import AlphaBandit


def test_bandit_learns_and_forgets():
    b = AlphaBandit(["good", "bad"], seed=1)
    assert b.weight("good", Regime.RANGE, explore=False) == 1.0
    for i in range(30):
        b.update("good", Regime.RANGE, 2.0 if i % 5 else -1.0)
        b.update("bad", Regime.RANGE, -1.0 if i % 4 else 1.5)
    assert b.weight("good", Regime.RANGE, explore=False) > 1.2
    assert b.weight("bad", Regime.RANGE, explore=False) < 0.85
    # regime-specific evidence blends with the overall arm
    assert 0.9 < b.weight("good", Regime.VOLATILE, explore=False) <= 1.5


def test_summarize_and_objective():
    trades = [{"pnl": 10, "r_multiple": 1.0, "entry_time": 0, "exit_time": 60000, "fees": 1, "funding": 0}] * 10 + \
             [{"pnl": -5, "r_multiple": -0.5, "entry_time": 0, "exit_time": 60000, "fees": 1, "funding": 0}] * 10
    st = summarize(trades)
    assert st["n"] == 20 and st["win_rate"] == 50 and abs(st["profit_factor"] - 2.0) < 1e-9
    assert objective(st) > 0
    assert objective(summarize(trades[:5])) < -1e8  # not enough trades
    assert len(sparkline([1, 2, 3, 2, 1])) == 5
