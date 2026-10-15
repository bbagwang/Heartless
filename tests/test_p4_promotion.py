"""P4 promotion / retirement on day-clustered statistics (heartless/learning/research.py judge_challenger and
ResearchManager.evaluate_challengers). Synthetic trade records only. C2-B: promotion is judged at the checkpoints of the
group-sequential design (tests/test_c2b_promotion.py covers the schedule itself); these tests keep checking the P4
conditions of each path at a checkpoint."""
import asyncio

import pytest

from heartless.config import Settings
from heartless.core.models import TradeRecord
from heartless.core.store import Store
from heartless.execution.stats import r_stats
from heartless.learning import research as R
from heartless.learning.research import ChallengerSlot, ResearchManager, judge_challenger
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, now_ms

S = Settings(_env_file=None)  # PROMOTION_MIN_TRADES 25, PROMOTION_MIN_DAYS 14
T0 = 1_780_000_000_000 // MS_DAY * MS_DAY  # a UTC midnight


def _trades(days: list[list[float]], engine: str = "challenger-1", start: int = T0) -> list[dict]:
    """One list of R multiples per UTC exit day (an empty list = a day without a closed trade)."""
    out = []
    for d, rs in enumerate(days):
        for k, r in enumerate(rs):
            out.append({"position_id": f"{engine}-{d}-{k}", "engine": engine, "r_multiple": r, "pnl": r * 50.0,
                        "exit_time": start + d * MS_DAY + (k + 1) * MS_HOUR, "entry_time": start + d * MS_DAY,
                        "alpha": "squeeze_breakout"})
    return out


GOOD_16 = ([[1.0, -0.4]] * 3 + [[-1.0, 0.5]]) * 4  # 32 trades, 16 days, mean +0.1625R, t_day ~2.6
GOOD_32 = GOOD_16 * 2  # 64 trades, 32 days, t_day ~3.8
GOOD_64 = GOOD_32 * 2  # 128 trades, 64 days, t_day ~5.3
# C2-B: promotion is judged at the checkpoints only (research.PROMOTION_CHECKPOINTS, 14/30/60/90 days after the start;
# FAILED_CHECKPOINTS 30/60/90 for a failed-alpha re-enable), on the record up to the checkpoint and with that
# checkpoint's calibrated t threshold. The P4 conditions themselves are kept; these tests judge at the 60-day checkpoint
# (not the last, so a record that misses the bar is held rather than retired at the end of its trial).
AT60 = 60.05  # days after the start: just past the 60-day checkpoint
# E2: the path statistics are HAC t values (Newey-West over 5 calendar days). A strictly periodic record (GOOD_16 repeats
# every 4 days) is strong evidence for it; the records that must land BETWEEN two thresholds are taken from a fixed
# irregular 60-day pattern shifted by the smallest step that lands there (_between), so they follow a recalibration
MID_60 = ([[0.5, -0.5], [1.5, 1.5], [-0.5, -1.0], [-0.5, 0.5], [0.5, -0.5], [1.5, 1.0], [-1.0, 1.0], [1.0, -1.0],
           [1.0, -0.5], [0.5, -0.5], [1.5, 0.5], [-1.0, -0.5], [-1.0, -1.0], [1.0, 1.0], [-0.5, -1.0], [1.5, 1.0],
           [1.5, -0.5], [0.5, -1.0], [1.0, -0.5], [0.5, 1.0], [1.0, 1.5], [1.0, 0.5], [-1.0, 1.0], [0.5, 1.0],
           [-0.5, 1.0], [-1.0, 1.0], [-0.5, -1.0], [0.5, -1.0], [-0.5, -0.5], [0.5, 0.5]] * 2)
Z_60 = [1 if a + b > 0.4 else -1 for a, b in MID_60]  # an irregular +-1 day sequence
G, B = [0.8, -0.3], [0.0, -0.3]
# +0.5R/day better than CHAMP (below) on 42 days, 0.3R/day worse on 18, interleaved (no runs of good or bad days)
NOISY = ([G] * 2 + [B] + [G] * 3 + [B] + [G] * 2 + [B]) * 6
LOSING_30 = ([[-1.0, 0.3]] * 2 + [[1.0, -0.2]]) * 10  # 60 trades, 30 days, mean -0.1R, losing without long runs


def _between(make, stat, lo: float, hi: float) -> list:
    """The first record make(c), c = 0, 0.005, 0.01, ..., whose statistic lies in [lo, hi)."""
    for i in range(600):
        days = make(i / 200)
        if lo <= stat(days) < hi:
            return days
    raise AssertionError(f"no record with a statistic in [{lo}, {hi})")


def _judge(ch_days, champ_days=(), new_alphas=("discovered",), now_day=None):
    ch, champ = _trades(ch_days), _trades(list(champ_days), "paper")
    now = T0 + int((now_day if now_day is not None else max(len(ch_days), len(champ_days))) * MS_DAY)
    return judge_challenger(ch, champ, S, T0, now, list(new_alphas))


# --- absolute path ------------------------------------------------------------------------------------------------
def test_absolute_path_promotes_a_challenger_that_clears_the_bar():
    rec = r_stats(_trades(GOOD_64[:60]))
    assert rec["n"] == 120 and rec["days"] == 60 and rec["t_hac"] >= R.ABS_T_AT[60] and rec["pf_r"] >= 1.15
    d = _judge(GOOD_64, now_day=AT60)  # the champion closed nothing: e.g. no alpha enabled
    assert d["path"] == "absolute" and d["verdict"] == "promote" and all(d["checks"].values())
    assert d["bar"] == {"min_trades": 30, "min_days": 14, "min_avg_r": 0.0, "min_t_hac": R.ABS_T_AT[60],
                        "min_t_day": R.ABS_T_AT[60], "min_pf_r": 1.15, "max_dd_r": 10.0, "checkpoint": 60,
                        "checkpoints": [14, 30, 60, 90], "hac_lag": R.PROMOTION_HAC_LAG}
    assert d["challenger"]["n"] == 120 and d["champion"]["n"] == 0 and d["failed_alphas"] == []
    assert d["checkpoint"] == 60 and d["next_checkpoint"] == 90
    text = R.describe_decision(d)
    assert "절대 경로" in text and "n=120" in text and f"t_day≥{R.ABS_T_AT[60]:g}" in text and "60일 판정점" in text


def test_absolute_path_is_used_while_the_champion_trades_too_little():
    d = _judge(GOOD_64, champ_days=[[0.5]] * 24, now_day=AT60)  # 24 champion trades < PROMOTION_MIN_TRADES
    assert d["path"] == "absolute" and d["verdict"] == "promote"


@pytest.mark.parametrize("days, failing", [
    ([[1.0, -0.4, 1.0, -0.4, -1.0, 0.5, 1.0, -0.4]] * 4 + [[1.0]] * 9 + [[]] * 50, "days >= 14"),  # 41 trades, 13 days
    ([[0.4, -0.3]] * 30 + [[-0.6, 0.5]] * 30, "HAC t_day >="),  # mean +0.0R... weak
    (GOOD_16 * 3 + [[-6.0, -6.0, -6.0, -6.0]] + GOOD_16[:11], "DD <="),  # a 24R hole: over 10R x sqrt(122/30)
])
def test_absolute_path_holds_when_any_condition_fails(days, failing):
    d = _judge(days, now_day=AT60)
    assert d["path"] == "absolute" and d["verdict"] == "hold", d["checks"]
    key = next(k for k in d["checks"] if k.startswith(failing))
    assert d["checks"][key] is False


def test_absolute_path_profit_factor_and_mean_gates():
    low_pf = [[1.0, -0.95]] * 60  # PF 1.05 < 1.15 although positive every day
    d = _judge(low_pf, now_day=AT60)
    assert d["verdict"] == "hold" and d["checks"]["PF >= 1.15"] is False and d["checks"]["avgR > 0"] is True
    negative = [[-0.2, -0.1]] * 16  # 32 trades, a third of the 2x trade minimum: not retired either
    d = _judge(negative, now_day=16)
    assert d["checks"]["avgR > 0"] is False and d["verdict"] == "hold"


def test_failed_out_of_sample_alpha_needs_its_own_schedule_and_bar():
    d = _judge(GOOD_16, new_alphas=["squeeze_breakout"])  # failed in the P1 judgement: nothing before day 30
    assert d["failed_alphas"] == ["squeeze_breakout"] and d["bar"]["min_days"] == 30 and d["checkpoint"] is None
    assert d["bar"]["min_t_day"] == R.FAILED_ABS_T_AT[30] and d["checkpoints"] == [30, 60, 90]
    assert d["verdict"] == "hold" and "OOS 탈락 알파 squeeze_breakout" in R.describe_decision(d)
    d = _judge(GOOD_64, new_alphas=["squeeze_breakout", "discovered"], now_day=AT60)
    assert d["verdict"] == "promote" and d["failed_alphas"] == ["squeeze_breakout"]
    assert d["bar"]["min_t_day"] == R.FAILED_ABS_T_AT[60]
    # HAC t_day between the two 60-day thresholds: enough for an untested alpha, not for a failed one
    mid = _between(lambda c: [[round(a + c, 3), round(b + c, 3)] for a, b in MID_60],
                   lambda days: r_stats(_trades(days))["t_hac"], R.ABS_T_AT[60], R.FAILED_ABS_T_AT[60])
    rec = r_stats(_trades(mid))
    assert rec["days"] == 60 and R.ABS_T_AT[60] <= rec["t_hac"] < R.FAILED_ABS_T_AT[60] and rec["pf_r"] >= 1.15, rec
    assert _judge(mid, new_alphas=["discovered"], now_day=AT60)["verdict"] == "promote"
    assert _judge(mid, new_alphas=["momentum_burst"], now_day=AT60)["verdict"] == "hold"


# --- relative path ------------------------------------------------------------------------------------------------
CHAMP = [[0.3, -0.3]] * 60  # 120 trades over 60 days, mean 0


def test_relative_path_promotes_on_the_paired_daily_difference():
    # +0.5R/day better on 42 days, 0.3R/day worse on 18: the paired t over the 60 calendar days of the checkpoint
    noisy = NOISY
    d = _judge(noisy, CHAMP, now_day=AT60)
    assert d["path"] == "relative" and d["verdict"] == "promote", d["checks"]
    assert d["paired"]["days"] == 60 and d["paired"]["t"] >= R.REL_T_AT[60]
    assert d["paired"]["mean"] == pytest.approx(15.6 / 60, abs=1e-4)
    assert d["challenger"]["avg_r"] > d["champion"]["avg_r"]
    assert "상대 경로" in R.describe_decision(d)
    # no dispersion at all (the same edge every single day) gives no t (like the lab's t_day): degenerate, held
    flat = _trades([[0.8, -0.3]] * 61)
    champ = _trades(CHAMP + [[0.3, -0.3]], "paper")
    assert judge_challenger(flat, champ, S, T0, T0 + int(AT60 * MS_DAY), [])["paired"]["t"] == 0.0


def test_relative_path_holds_on_weak_difference_mean_or_drawdown():
    # paired t below the bar: better on 9 days out of 60, identical otherwise ... plus worse days
    weak = ([[0.3, -0.3]] * 14 + [[1.3, -0.3]] * 3 + [[-0.5, -0.3]] * 3) * 3
    d = _judge(weak, CHAMP, now_day=AT60)
    assert d["path"] == "relative" and d["paired"]["t"] < R.REL_T_AT[60] and d["verdict"] == "hold"
    # better daily but a deep losing streak: R drawdown over 1.2 x champion + 2R
    streak = [[0.9, -0.3]] * 56 + [[-1.0, -1.0]] * 2 + [[0.9, -0.3]] * 2
    d = _judge(streak, CHAMP, now_day=AT60)
    cap = 1.2 * d["champion"]["max_dd_r"] + 2.0
    assert d["challenger"]["max_dd_r"] > cap and d["verdict"] == "hold"
    assert d["checks"][f"DD <= {cap:.2f}R"] is False


def test_relative_path_also_raises_the_bar_for_an_alpha_that_failed_out_of_sample():
    # once the champion trades, the relative path is the usual one: a challenger re-enabling an alpha that lost over
    # three years out of sample faces the failed-alpha schedule and thresholds there too
    noisy = NOISY
    assert _judge(noisy, CHAMP, now_day=AT60)["verdict"] == "promote"
    d = _judge(noisy[:16], CHAMP[:16], new_alphas=["squeeze_breakout"])  # day 16: before its first checkpoint
    assert d["path"] == "relative" and d["failed_alphas"] == ["squeeze_breakout"] and d["verdict"] == "hold"
    assert d["bar"]["min_days"] == 30 and d["bar"]["min_paired_t"] == R.FAILED_REL_T_AT[30] and d["checkpoint"] is None
    assert "OOS 탈락 알파 squeeze_breakout" in R.describe_decision(d)
    assert f"차이 t≥{R.FAILED_REL_T_AT[30]:g}" in R.describe_decision(d)
    # 60 days, paired HAC t between the two 60-day thresholds: the champion's trades plus one irregular extra trade
    mid = _between(lambda c: [[0.3, -0.3, round(c + 0.3 * z, 3)] for z in Z_60],
                   lambda days: _judge(days, CHAMP, now_day=AT60)["paired"]["t"], R.REL_T_AT[60], R.FAILED_REL_T_AT[60])
    dm = _judge(mid, CHAMP, now_day=AT60)
    assert R.REL_T_AT[60] <= dm["paired"]["t"] < R.FAILED_REL_T_AT[60] and dm["verdict"] == "promote"
    d = _judge(mid, CHAMP, new_alphas=["squeeze_breakout"], now_day=AT60)
    assert d["verdict"] == "hold" and d["checks"][f"paired t >= {R.FAILED_REL_T_AT[60]:g}"] is False
    strong = NOISY  # paired t well above both
    d = _judge(strong, CHAMP, new_alphas=["squeeze_breakout"], now_day=AT60)
    assert d["verdict"] == "promote" and d["paired"]["t"] >= R.FAILED_REL_T_AT[60]


def test_relative_path_counts_calendar_days_without_trades_as_zero():
    sparse = [[2.0]] + [[]] * 13 + [[2.0]] * 6 + [[0.5]] * 40  # few days, but they are the challenger's real record
    d = _judge(sparse, CHAMP, now_day=AT60)
    assert d["paired"]["days"] == 60  # every calendar day from the challenger's start through the checkpoint


# --- retirement ---------------------------------------------------------------------------------------------------
def test_early_drop_keeps_working():
    d = _judge([[-0.8, -0.6]] * 8)
    assert d["verdict"] == "retire" and d["rule"].startswith("early drop")
    assert _judge([[-0.8, -0.6]] * 7)["verdict"] == "hold"  # 14 trades: not enough for the early drop


def test_absolute_retirement_after_twice_the_minimum_trades_and_days():
    losing = LOSING_30  # 60 trades, 30 days, mean -0.1R, HAC t_day far below -1
    rec = r_stats(_trades(losing))
    assert rec["t_hac"] <= -1.0 and rec["avg_r"] > -0.5
    d = _judge(losing)
    assert d["verdict"] == "retire" and d["path"] == "absolute" and "t_day <= -1" in d["rule"]
    assert _judge(losing[:27])["verdict"] == "hold"  # 54 trades, 27 days: under 2 x 30 trades / 2 x 14 days
    # with a failed alpha the minimum is 30 days, so retirement waits for 60
    assert _judge(losing, new_alphas=["htf_trend"])["verdict"] == "hold"


def test_relative_retirement_on_the_paired_difference():
    worse = [[-0.2, -0.3]] * 30  # 60 trades on 30 days, worse than the champion every day
    champ = CHAMP[:30]
    d = _judge(worse, champ)
    assert d["path"] == "relative" and d["verdict"] == "retire" and "paired t" in d["rule"]
    assert _judge(worse[:25], champ)["verdict"] == "hold"  # 50 trades but 25 < 28 days


# --- the manager acts on the verdicts and records their inputs ------------------------------------------------------
class _App:
    def __init__(self, tmp_path):
        self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
        self.store = Store(tmp_path / "h.db")
        self.params = StrategyParams.default()
        self.engines = {"paper": object()}
        self.challengers: list[ChallengerSlot] = []
        self.symbols, self.universe, self.mode = {}, [], "paper"
        self.promoted, self.retired, self.events = [], [], []

    async def emit(self, kind, payload):
        self.events.append((kind, payload))

    async def promote(self, slot, decision):
        self.promoted.append((slot, decision))
        self.challengers.remove(slot)

    async def retire_challenger(self, slot, reason, silent=False, decision=None):
        self.retired.append((slot, reason, decision))
        self.challengers.remove(slot)


def _store_trades(app, rows):
    for t in rows:
        app.store.save_trade(TradeRecord(
            position_id=t["position_id"], engine=t["engine"], symbol="BTCUSDT", side="LONG", alpha=t["alpha"],
            alphas=[t["alpha"]], regime="RANGE", entry_time=t["entry_time"], exit_time=t["exit_time"], entry_price=1.0,
            exit_price=1.0, qty=1.0, notional=1.0, pnl=t["pnl"], gross=t["pnl"], fees=0.0, funding=0.0,
            r_multiple=t["r_multiple"], risk_amount=50.0, exit_reason="x", confidence=0.6, params_version="v",
            bars_held=1, max_fav_r=0.0, max_adv_r=0.0))


def _slot(app, name, enable: list[str], start: int) -> ChallengerSlot:
    p = app.params.clone(version=f"v{name[-1]}", note=name)
    for a in enable:
        p.enabled[a] = True
    slot = ChallengerSlot(name=name, params=p, started=start, source=",".join(enable))
    app.challengers.append(slot)
    app.engines[name] = object()
    return slot


def test_manager_promotes_on_the_absolute_path_and_records_the_decision(tmp_path):
    app = _App(tmp_path)
    start = now_ms() // MS_DAY * MS_DAY - 65 * MS_DAY
    s1 = _slot(app, "challenger-1", ["squeeze_breakout"], start)  # failed OOS: the 30/60/90-day schedule
    _store_trades(app, _trades(GOOD_64, "challenger-1", start))
    rm = ResearchManager(app)
    assert rm.new_alphas(s1.params) == ["squeeze_breakout"] and rm.trial_days(s1) == 90
    asyncio.run(rm.evaluate_challengers(force=True))
    (slot, d), = app.promoted
    assert slot is s1 and d["verdict"] == "promote" and d["path"] == "absolute" and d["failed_alphas"] == ["squeeze_breakout"]
    assert d["checkpoint"] == 60 and d["bar"]["min_t_day"] == R.FAILED_ABS_T_AT[60]
    assert d["slot"] == "challenger-1" and d["version"] == "v1" and d["champion_version"] == app.params.version
    assert rm.last_decisions["challenger-1"] is d


def test_manager_only_counts_alphas_beyond_the_champion_and_retires_with_inputs(tmp_path):
    app = _App(tmp_path)
    app.params.enabled["squeeze_breakout"] = True  # the champion already runs it: not a re-enable
    start = now_ms() // MS_DAY * MS_DAY - 65 * MS_DAY
    s1 = _slot(app, "challenger-1", ["squeeze_breakout"], start)
    s2 = _slot(app, "challenger-2", ["discovered"], start)
    _store_trades(app, _trades(GOOD_64, "challenger-1", start))  # the normal schedule: promoted at its 60-day checkpoint
    _store_trades(app, _trades(LOSING_30, "challenger-2", start))
    rm = ResearchManager(app)
    assert rm.new_alphas(s1.params) == [] and rm.slot_new_alphas(s1) == [] and rm.trial_days(s1) == 90
    asyncio.run(rm.evaluate_challengers(force=True))
    assert [s for s, _ in app.promoted] == [s1]
    assert app.retired == []  # one promotion per pass: the rest is judged against the new champion next time
    asyncio.run(rm.evaluate_challengers(force=True))
    (slot, reason, d), = app.retired
    assert slot is s2 and d["verdict"] == "retire" and "절대 기준 열위" in reason and "t_day" in reason
    assert d["challenger"]["n"] == 60 and d["checks"]


def test_manager_compares_trades_opened_since_the_challenger_started(tmp_path):
    # champion positions opened before the challenger existed and closed after it started are not part of the
    # comparison: here they would have made it a relative-path judgement on trades the challenger could not take
    app = _App(tmp_path)
    app.params.enabled["squeeze_breakout"] = True
    start = now_ms() // MS_DAY * MS_DAY - 20 * MS_DAY
    _slot(app, "challenger-1", ["discovered"], start)
    early = _trades([[0.5, 0.5, 0.5]] * 10, "paper", start)
    for t in early:
        t["entry_time"] = start - MS_HOUR
    _store_trades(app, early)  # 30 champion trades closed after the start, all opened before it
    _store_trades(app, _trades([[0.2]] * 3, "paper", start + 10 * MS_DAY))  # 3 opened after the start
    rm = ResearchManager(app)
    asyncio.run(rm.evaluate_challengers(force=True))
    d = rm.last_decisions["challenger-1"]
    assert d["champion"]["n"] == 3 and d["path"] == "absolute"


def test_manager_waits_an_hour_between_evaluations(tmp_path):
    app = _App(tmp_path)
    rm = ResearchManager(app)
    _slot(app, "challenger-1", ["discovered"], now_ms() - MS_DAY)
    asyncio.run(rm.evaluate_challengers())
    first = rm.last_decisions["challenger-1"]
    asyncio.run(rm.evaluate_challengers())
    assert rm.last_decisions["challenger-1"] is first and first["verdict"] == "hold"
