"""P4 statistics helpers (heartless/execution/stats.py): day-clustered t, daily R, paired daily difference, R drawdown,
R profit factor, lower CUSUM, fill slippage and the execution-quality aggregate. Synthetic data only."""
import math
import random

import pytest

from heartless.execution import stats as S
from heartless.learning.lab import day_clustered_t as lab_t_day

DAY = 86_400_000


def _t(r: float, exit_time: int) -> dict:
    return {"r_multiple": r, "exit_time": exit_time, "pnl": r * 10.0}


def test_day_clustered_t_is_the_lab_estimator():
    rng = random.Random(3)
    for _ in range(20):
        trades = [_t(rng.gauss(0.05, 1.0), rng.randrange(0, 60) * DAY + rng.randrange(0, DAY)) for _ in range(rng.randrange(3, 200))]
        assert S.day_clustered_t(trades) == pytest.approx(lab_t_day(trades), rel=1e-9, abs=1e-12)


def test_day_clustered_t_counts_days_not_trades():
    # ten identical trades per day: per-trade t would treat them as ten observations, the day-clustered t does not
    days = [0.5, -0.2, 0.4, 0.1, -0.3, 0.6, 0.2, -0.1]
    one = [_t(r, d * DAY + 1000) for d, r in enumerate(days)]
    ten = [_t(r, d * DAY + 1000 + k) for d, r in enumerate(days) for k in range(10)]
    assert S.day_clustered_t(ten) == pytest.approx(S.day_clustered_t(one), rel=1e-9)
    assert S.summarize(ten)["t_stat"] > 2.5 * S.summarize(one)["t_stat"]
    assert S.day_clustered_t(one[:4]) == 0.0  # fewer than 5 exit days: no verdict
    assert S.day_clustered_t(one[:2]) == 0.0


def test_daily_r_and_exit_days_use_the_utc_exit_day():
    trades = [_t(1.0, DAY - 1), _t(-0.5, DAY), _t(0.25, DAY + 5), _t(2.0, 3 * DAY)]
    assert S.daily_r(trades) == {0: 1.0, 1: -0.25, 3: 2.0}
    assert S.exit_days(trades) == 3 and S.exit_day(trades[1]) == 1


def test_paired_daily_difference_over_the_calendar_span():
    a = [_t(1.0, 0 * DAY + 10), _t(1.0, 1 * DAY + 10), _t(2.0, 3 * DAY + 10)]
    b = [_t(0.5, 0 * DAY + 20), _t(-1.0, 2 * DAY + 20)]
    # active days only: day 0 +0.5, day 1 +1.0, day 2 +1.0, day 3 +2.0
    pd = S.paired_daily_diff(a, b)
    assert pd["days"] == 4 and pd["mean"] == pytest.approx(1.125)
    diffs = [0.5, 1.0, 1.0, 2.0]
    sd = math.sqrt(sum((x - 1.125) ** 2 for x in diffs) / 3)
    assert pd["t"] == pytest.approx(1.125 / (sd / 2.0))
    assert pd["sum_a"] == pytest.approx(4.0) and pd["sum_b"] == pytest.approx(-0.5)
    # over a calendar span both engines ran: a day without any trade is a real 0R day for both
    span = S.paired_daily_diff(a, b, 0, 5 * DAY + 1)
    assert span["days"] == 6 and span["mean"] == pytest.approx(4.5 / 6)
    assert S.paired_daily_diff([], [])["t"] == 0.0


def test_r_drawdown_profit_factor_and_record():
    rs = [1.0, -1.0, -1.0, 2.0, -0.5, -0.5, -0.5, 3.0]
    trades = [_t(r, i * DAY) for i, r in enumerate(rs)]
    assert S.max_drawdown_r(trades) == pytest.approx(2.0)  # peak 1.0 -> -1.0, later peak 1.0 -> -0.5
    assert S.max_drawdown_r(list(reversed(trades))) == pytest.approx(2.0)  # ordered by exit time, not list order
    assert S.profit_factor_r(trades) == pytest.approx(6.0 / 3.5)
    assert S.profit_factor_r([_t(1.0, 0)]) == float("inf") and S.profit_factor_r([_t(-1.0, 0)]) == 0.0
    rec = S.r_stats(trades)
    assert rec["n"] == 8 and rec["days"] == 8 and rec["sum_r"] == pytest.approx(2.5)
    assert rec["avg_r"] == pytest.approx(2.5 / 8) and rec["t_day"] == pytest.approx(S.day_clustered_t(trades))


def test_lower_cusum():
    # k = 0.25, expected 0: a -1R trade adds -0.75, a +0.5R trade +0.75, the sum never goes above 0
    s, low = S.cusum_lower([-1.0, -1.0, 0.5, -1.0], 0.0, 0.25)
    assert s == pytest.approx(-1.5) and low == pytest.approx(-1.5)
    s, low = S.cusum_lower([-1.0, 3.0, -1.0], 0.0, 0.25)
    assert low == pytest.approx(-0.75) and s == pytest.approx(-0.75)
    # trades exactly k below expectation do not accumulate
    assert S.cusum_lower([0.05] * 100, 0.3, 0.25)[0] == pytest.approx(0.0)
    assert S.cusum_lower([], 0.0, 0.25) == (0.0, 0.0)


def test_objective_can_use_the_day_clustered_t():
    st = {"n": 30, "t_stat": 4.0, "t_day": 1.0, "profit_factor": 1.5, "avg_r": 0.2, "max_dd": 10.0, "net": 100.0}
    assert S.objective(st, 10) - S.objective(st, 10, t_key="t_day") == pytest.approx(3.0)
    del st["t_day"]
    assert S.objective(st, 10, t_key="t_day") == S.objective(st, 10)  # falls back to the per-trade t


def test_slippage_sign_and_execution_quality():
    assert S.slippage_bps("BUY", 100.02, 100.0) == pytest.approx(2.0)
    assert S.slippage_bps("SELL", 99.98, 100.0) == pytest.approx(2.0)
    assert S.slippage_bps("SELL", 100.03, 100.0) == pytest.approx(-3.0)  # price improvement is negative cost
    assert S.slippage_bps("BUY", 100.0, 0.0) == 0.0

    def row(slip, maker=False, notional=1000.0, fee_rate=0.0005):
        return {"slip_bps": slip, "maker": maker, "notional": notional, "fee": notional * fee_rate,
                "model_fee": notional * (0.0002 if maker else 0.0005)}

    rows = [row(4.0) for _ in range(49)] + [row(0.0, maker=True, fee_rate=0.0002)]
    q = S.exec_quality(rows, 1.5, 50)
    assert q["fills"] == 50 and q["taker_fills"] == 49 and q["slip_bps"] == pytest.approx(4.0)
    assert q["suggest"] is False  # 49 taker fills: below the 50-fill minimum
    q = S.exec_quality(rows + [row(4.0)], 1.5, 50)
    assert q["suggest"] is True and q["fee_vs_model"] == pytest.approx(1.0)
    assert S.exec_quality([row(2.9) for _ in range(60)], 1.5, 50)["suggest"] is False  # not above 2 x 1.5bp
    assert S.exec_quality([], 1.5)["fills"] == 0
