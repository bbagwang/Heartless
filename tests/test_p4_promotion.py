"""P4 promotion / retirement on day-clustered statistics (heartless/learning/research.py judge_challenger and
ResearchManager.evaluate_challengers). Synthetic trade records only."""
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


def _judge(ch_days, champ_days=(), new_alphas=("discovered",), now_day=None):
    ch, champ = _trades(ch_days), _trades(list(champ_days), "paper")
    now = T0 + (now_day if now_day is not None else max(len(ch_days), len(champ_days))) * MS_DAY
    return judge_challenger(ch, champ, S, T0, now, list(new_alphas))


# --- absolute path ------------------------------------------------------------------------------------------------
def test_absolute_path_promotes_a_challenger_that_clears_the_bar():
    rec = r_stats(_trades(GOOD_16))
    assert rec["n"] == 32 and rec["days"] == 16 and rec["t_day"] >= 2.0 and rec["pf_r"] >= 1.15 and rec["max_dd_r"] <= 10
    d = _judge(GOOD_16)  # the champion closed nothing: e.g. no alpha enabled
    assert d["path"] == "absolute" and d["verdict"] == "promote" and all(d["checks"].values())
    assert d["bar"] == {"min_trades": 30, "min_days": 14, "min_avg_r": 0.0, "min_t_day": 1.5, "min_pf_r": 1.15,
                        "max_dd_r": 10.0}
    assert d["challenger"]["n"] == 32 and d["champion"]["n"] == 0 and d["failed_alphas"] == []
    text = R.describe_decision(d)
    assert "절대 경로" in text and "n=32" in text and "t_day≥1.5" in text


def test_absolute_path_is_used_while_the_champion_trades_too_little():
    d = _judge(GOOD_16, champ_days=[[0.5]] * 24)  # 24 champion trades < PROMOTION_MIN_TRADES
    assert d["path"] == "absolute" and d["verdict"] == "promote"


@pytest.mark.parametrize("days, failing", [
    (GOOD_16[:14] + [[1.0]], "n >= 30"),  # 29 trades
    ([[1.0, -0.4, 1.0, -0.4, -1.0, 0.5, 1.0, -0.4]] * 4 + [[1.0]] * 9, "days >= 14"),  # 41 trades on 13 days
    ([[0.4, -0.3]] * 8 + [[-0.6, 0.5]] * 8, "t_day >= 1.5"),  # mean +0.0R... weak
    ([[0.4] * 3] * 16 + [[-12.0]], "DD <= 10R"),  # one 12R loss
])
def test_absolute_path_holds_when_any_condition_fails(days, failing):
    d = _judge(days)
    assert d["path"] == "absolute" and d["verdict"] == "hold"
    assert not d["checks"][failing] if failing in d["checks"] else any(not v for v in d["checks"].values())


def test_absolute_path_profit_factor_and_mean_gates():
    low_pf = [[1.0, -0.95]] * 16  # PF 1.05 < 1.15 although positive every day
    d = _judge(low_pf)
    assert d["verdict"] == "hold" and d["checks"]["PF >= 1.15"] is False and d["checks"]["avgR > 0"] is True
    negative = [[-0.2, -0.1]] * 16
    d = _judge(negative)
    assert d["checks"]["avgR > 0"] is False and d["verdict"] == "hold"


def test_failed_out_of_sample_alpha_needs_30_days_and_t_2():
    d = _judge(GOOD_16, new_alphas=["squeeze_breakout"])  # failed in the P1 judgement
    assert d["failed_alphas"] == ["squeeze_breakout"] and d["bar"]["min_days"] == 30 and d["bar"]["min_t_day"] == 2.0
    assert d["verdict"] == "hold" and d["checks"]["days >= 30"] is False
    assert "OOS 탈락 알파 squeeze_breakout" in R.describe_decision(d)
    d = _judge(GOOD_32, new_alphas=["squeeze_breakout", "discovered"])
    assert d["verdict"] == "promote" and d["failed_alphas"] == ["squeeze_breakout"]
    # 30 days but t_day between 1.5 and 2.0: enough for an untested alpha, not for a failed one
    mid = [[0.5, -0.5], [1.5, 1.5], [-0.5, -1.0], [-0.5, 0.5], [0.5, -0.5], [1.5, 1.0], [-1.0, 1.0], [1.0, 1.0],
           [1.0, -0.5], [0.5, -0.5], [1.5, 0.5], [-1.0, -0.5], [-1.0, -1.0], [1.0, 1.0], [-0.5, -1.0], [1.5, 1.0],
           [1.5, -0.5], [0.5, -1.0], [1.0, -0.5], [0.5, 1.0], [1.0, 1.5], [1.0, 1.5], [-1.0, 1.0], [0.5, 1.0],
           [-0.5, 1.0], [-1.0, 1.0], [-0.5, -1.0], [0.5, -1.0], [-0.5, -0.5], [0.5, 0.5], [0.5, -1.0]]
    rec = r_stats(_trades(mid))
    assert rec["days"] == 31 and 1.5 <= rec["t_day"] < 2.0 and rec["pf_r"] >= 1.15, rec
    assert _judge(mid, new_alphas=["discovered"])["verdict"] == "promote"
    assert _judge(mid, new_alphas=["momentum_burst"])["verdict"] == "hold"


# --- relative path ------------------------------------------------------------------------------------------------
CHAMP = [[0.3, -0.3]] * 20  # 40 trades over 20 days, mean 0


def test_relative_path_promotes_on_the_paired_daily_difference():
    # +0.5R/day better on 14 days, 0.3R/day worse on 6: mean +0.26R per calendar day (the current day counts as a
    # 0R day for both: the span runs from the challenger's start through today)
    noisy = [[0.8, -0.3]] * 14 + [[0.0, -0.3]] * 6
    d = _judge(noisy, CHAMP)
    assert d["path"] == "relative" and d["verdict"] == "promote", d["checks"]
    assert d["paired"]["days"] == 21 and d["paired"]["t"] >= 1.0 and d["paired"]["mean"] == pytest.approx(5.2 / 21, abs=1e-4)
    assert d["challenger"]["avg_r"] > d["champion"]["avg_r"]
    assert "상대 경로" in R.describe_decision(d)
    # no dispersion at all (the same edge every single day) gives no t (like the lab's t_day): degenerate, held
    flat = _trades([[0.8, -0.3]] * 20)
    champ = _trades(CHAMP, "paper")
    assert judge_challenger(flat, champ, S, T0, T0 + 19 * MS_DAY + 5 * MS_HOUR, [])["paired"]["t"] == 0.0


def test_relative_path_holds_on_weak_difference_mean_or_drawdown():
    # paired t below 1: better on 3 days out of 20, identical otherwise ... plus worse days
    weak = [[0.3, -0.3]] * 14 + [[1.3, -0.3]] * 3 + [[-0.5, -0.3]] * 3
    d = _judge(weak, CHAMP)
    assert d["path"] == "relative" and d["paired"]["t"] < 1.0 and d["verdict"] == "hold"
    # better daily but a deep losing streak: R drawdown over 1.2 x champion + 2R
    streak = [[0.9, -0.3]] * 16 + [[-1.0, -1.0]] * 2 + [[0.9, -0.3]] * 2
    d = _judge(streak, CHAMP)
    cap = 1.2 * d["champion"]["max_dd_r"] + 2.0
    assert d["challenger"]["max_dd_r"] > cap and d["verdict"] == "hold"
    assert d["checks"][f"DD <= {cap:.2f}R"] is False


def test_relative_path_counts_calendar_days_without_trades_as_zero():
    sparse = [[2.0]] + [[]] * 13 + [[2.0]] * 6 + [[0.5]] * 20  # few days, but they are the challenger's real record
    d = _judge(sparse, CHAMP)
    assert d["paired"]["days"] == 41  # every calendar day from the challenger's start through today


# --- retirement ---------------------------------------------------------------------------------------------------
def test_early_drop_keeps_working():
    d = _judge([[-0.8, -0.6]] * 8)
    assert d["verdict"] == "retire" and d["rule"].startswith("early drop")
    assert _judge([[-0.8, -0.6]] * 7)["verdict"] == "hold"  # 14 trades: not enough for the early drop


def test_absolute_retirement_after_twice_the_minimum_trades_and_days():
    losing = [[-1.0, 0.3]] * 20 + [[1.0, -0.2]] * 10  # 60 trades, 30 days, mean -0.1R, t_day ~ -1.5
    rec = r_stats(_trades(losing))
    assert rec["t_day"] <= -1.0 and rec["avg_r"] > -0.5
    d = _judge(losing)
    assert d["verdict"] == "retire" and d["path"] == "absolute" and "t_day <= -1" in d["rule"]
    assert _judge(losing[:27])["verdict"] == "hold"  # 54 trades, 27 days: under 2 x 30 trades / 2 x 14 days
    # with a failed alpha the minimum is 30 days, so retirement waits for 60
    assert _judge(losing, new_alphas=["htf_trend"])["verdict"] == "hold"


def test_relative_retirement_on_the_paired_difference():
    worse = [[-0.2, -0.3]] * 30  # 60 trades on 30 days, worse than the champion every day
    champ = CHAMP + [[0.3, -0.3]] * 10
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
    start = now_ms() // MS_DAY * MS_DAY - 40 * MS_DAY
    s1 = _slot(app, "challenger-1", ["squeeze_breakout"], start)  # failed OOS: 30-day bar
    _store_trades(app, _trades(GOOD_32, "challenger-1", start))
    rm = ResearchManager(app)
    assert rm.new_alphas(s1.params) == ["squeeze_breakout"] and rm.trial_days(s1) == 30
    asyncio.run(rm.evaluate_challengers(force=True))
    (slot, d), = app.promoted
    assert slot is s1 and d["verdict"] == "promote" and d["path"] == "absolute" and d["failed_alphas"] == ["squeeze_breakout"]
    assert d["slot"] == "challenger-1" and d["version"] == "v1" and d["champion_version"] == app.params.version
    assert rm.last_decisions["challenger-1"] is d


def test_manager_only_counts_alphas_beyond_the_champion_and_retires_with_inputs(tmp_path):
    app = _App(tmp_path)
    app.params.enabled["squeeze_breakout"] = True  # the champion already runs it: not a re-enable
    start = now_ms() // MS_DAY * MS_DAY - 40 * MS_DAY
    s1 = _slot(app, "challenger-1", ["squeeze_breakout"], start)
    s2 = _slot(app, "challenger-2", ["discovered"], start)
    _store_trades(app, _trades(GOOD_16, "challenger-1", start))  # 16 days: enough without the failed-alpha bar
    _store_trades(app, _trades([[-1.0, 0.3]] * 20 + [[1.0, -0.2]] * 10, "challenger-2", start))
    rm = ResearchManager(app)
    assert rm.new_alphas(s1.params) == [] and rm.trial_days(s1) == 14
    asyncio.run(rm.evaluate_challengers(force=True))
    assert [s for s, _ in app.promoted] == [s1]
    assert app.retired == []  # one promotion per pass: the rest is judged against the new champion next time
    asyncio.run(rm.evaluate_challengers(force=True))
    (slot, reason, d), = app.retired
    assert slot is s2 and d["verdict"] == "retire" and "절대 기준 열위" in reason and "t_day" in reason
    assert d["challenger"]["n"] == 60 and d["checks"]


def test_manager_waits_an_hour_between_evaluations(tmp_path):
    app = _App(tmp_path)
    rm = ResearchManager(app)
    _slot(app, "challenger-1", ["discovered"], now_ms() - MS_DAY)
    asyncio.run(rm.evaluate_challengers())
    first = rm.last_decisions["challenger-1"]
    asyncio.run(rm.evaluate_challengers())
    assert rm.last_decisions["challenger-1"] is first and first["verdict"] == "hold"
