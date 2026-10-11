"""P4 per-alpha kill switch (heartless/learning/research.py AlphaKillSwitch) on a stub orchestrator: CUSUM and rolling
window triggers, the evidence-based expectation, epochs, challengers, and state derived from the store (restart)."""
import asyncio
import json
import random

import pytest

from heartless.config import Settings
from heartless.core.models import TradeRecord
from heartless.core.store import Store
from heartless.execution.stats import cusum_lower
from heartless.learning.research import AlphaKillSwitch, ChallengerSlot, describe_kill
from heartless.strategy import evidence as E
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_HOUR, now_ms


class _App:
    def __init__(self, tmp_path, **settings):
        self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), **settings)
        self.store = Store(tmp_path / "h.db")
        self.params = StrategyParams.default()
        self.params.enabled["htf_trend"] = True
        self.params.created = now_ms() - 100 * MS_HOUR
        self.engines = {"paper": object()}
        self.challengers: list[ChallengerSlot] = []
        self.killed, self.retired = [], []

    async def kill_alpha(self, alpha, decision):
        self.killed.append((alpha, decision))
        self.params.enabled[alpha] = False

    async def retire_challenger(self, slot, reason, silent=False, decision=None):
        self.retired.append((slot.name, reason, decision))
        self.challengers.remove(slot)
        self.engines.pop(slot.name, None)


_N = [0]


def _trade(engine: str, alpha: str, r: float, exit_time: int) -> TradeRecord:
    _N[0] += 1
    return TradeRecord(position_id=f"P{_N[0]}", engine=engine, symbol="BTCUSDT", side="LONG", alpha=alpha, alphas=[alpha],
                       regime="RANGE", entry_time=exit_time - MS_HOUR, exit_time=exit_time, entry_price=1.0,
                       exit_price=1.0, qty=1.0, notional=1.0, pnl=r * 50, gross=r * 50, fees=0.0, funding=0.0,
                       r_multiple=r, risk_amount=50.0, exit_reason="x", confidence=0.6, params_version="v",
                       bars_held=60, max_fav_r=0.0, max_adv_r=0.0)


def _close(app, ks, engine, alpha, r, t):
    """Store a closed trade and deliver the position_closed event like the engine does."""
    rec = _trade(engine, alpha, r, t)
    app.store.save_trade(rec)
    return asyncio.run(ks.on_trade({"engine": engine, "trade": rec}))


def test_cusum_trips_after_a_run_of_full_losses(tmp_path):
    app = _App(tmp_path)
    ks = AlphaKillSwitch(app)
    ks.sync_champion(app.params)
    t = now_ms() - 50 * MS_HOUR
    for i in range(10):  # identical losses: sd 0 -> the 1R floor; each -1R adds -(1 - 0.25): -7.5 after ten
        _close(app, ks, "paper", "htf_trend", -1.0, t + i * MS_HOUR)
    st = ks.state[("paper", "htf_trend")]
    assert app.killed == [] and st["cusum"] == pytest.approx(-7.5) and st["scale"] == 1.0
    _close(app, ks, "paper", "htf_trend", -1.0, t + 11 * MS_HOUR)  # -8.25 <= -8
    (alpha, d), = app.killed
    assert alpha == "htf_trend" and d["kind"] == "kill_switch" and d["engine"] == "paper"
    assert d["n"] == 11 and d["cusum"] == pytest.approx(-8.25) and d["expected"] == 0.0 and d["verdict"] == "failed"
    assert d["rule"].startswith("CUSUM") and d["tripped"] is True
    assert "CUSUM -8.25σ" in describe_kill(d) and "σ=1.00R" in describe_kill(d) and "OOS 탈락" in describe_kill(d)
    # the alpha is off now: later trades of it (open positions closing) do not act again
    _close(app, ks, "paper", "htf_trend", -1.0, t + 12 * MS_HOUR)
    assert len(app.killed) == 1


def _synthetic_r(n: int, mean: float, win_rate: float, seed: int) -> list[float]:
    """Stop-based trade R: 60% of losers at the stop (-1.0..-1.1R), the rest smaller; right-skewed winners sized so
    the expected R is `mean`."""
    rng = random.Random(seed)
    w = (mean - (1 - win_rate) * (0.6 * -1.05 + 0.4 * -0.4)) / win_rate
    out = []
    for _ in range(n):
        if rng.random() < win_rate:
            out.append(rng.gammavariate(2.0, w / 2.0))
        else:
            out.append(-1.0 - rng.uniform(0, 0.1) if rng.random() < 0.6 else -rng.uniform(0, 0.8))
    return out


def _first_trip(app, ks, rs: list[float], engine: str = "paper", alpha: str = "htf_trend") -> int | None:
    t = app.params.created + MS_HOUR  # every trade after the champion's epoch (created 100 h ago), 10 minutes apart
    for i, r in enumerate(rs):
        _close(app, ks, engine, alpha, r, t + i * 600_000)
        if app.killed:
            return i + 1
    return None


def test_an_alpha_with_a_real_edge_survives_ordinary_losing_streaks(tmp_path):
    # A trend follower earning +0.10R a trade at a 35% win rate loses 7+ trades in a row every few dozen trades. The
    # first P4 calibration (raw R, k 0.25, alarm -5R, last 30 < -8R) switched such an alpha off within 150 trades in
    # 9 of 10 synthetic runs; on (R - expected) / sd with an alarm at -8 sd it survives 300 trades in most runs.
    survived, old_rule_tripped = 0, 0
    for seed in range(10):
        rs = _synthetic_r(300, 0.10, 0.35, seed)
        (tmp_path / str(seed)).mkdir()
        app = _App(tmp_path / str(seed))
        ks = AlphaKillSwitch(app)
        survived += _first_trip(app, ks, rs) is None
        old_cusum = [cusum_lower(rs[:m], 0.0, 0.25)[0] for m in range(1, 151)]
        old_window = [sum(rs[max(0, m - 30):m]) for m in range(1, 151)]
        old_rule_tripped += any(c <= -5.0 for c in old_cusum) or any(w < -8.0 for w in old_window)
    assert old_rule_tripped >= 8 and survived >= 8


def test_a_broken_alpha_is_still_switched_off_within_a_few_dozen_trades(tmp_path):
    for seed in range(5):  # -0.5R a trade at a 30% win rate
        (tmp_path / str(seed)).mkdir()
        app = _App(tmp_path / str(seed))
        ks = AlphaKillSwitch(app)
        n = _first_trip(app, ks, _synthetic_r(120, -0.5, 0.30, 100 + seed))
        assert n is not None and n <= 70


def test_the_cusum_is_measured_in_standard_deviations_of_the_alphas_r(tmp_path):
    app = _App(tmp_path)
    ks = AlphaKillSwitch(app)
    t = now_ms() - 80 * MS_HOUR
    for i, r in enumerate([3.0, -3.0] * 10):  # a volatile record that nets 0: sd ~3.1R, clamped to 2R
        _close(app, ks, "paper", "htf_trend", r, t + i * MS_HOUR)
    for i in range(12):  # twelve full losses: -1 / 2 + 0.25 = -0.25 sd each, nowhere near -8 sd
        _close(app, ks, "paper", "htf_trend", -1.0, t + (20 + i) * MS_HOUR)
    st = ks.state[("paper", "htf_trend")]
    assert app.killed == [] and st["scale"] == 2.0 and st["sd"] > 2.0 and st["cusum"] > -8.0


def test_wins_pull_the_cusum_back_and_other_alphas_are_ignored(tmp_path):
    app = _App(tmp_path)
    ks = AlphaKillSwitch(app)
    t = now_ms() - 50 * MS_HOUR
    for i, r in enumerate([-1.0] * 5 + [2.0] + [-1.0] * 4):
        _close(app, ks, "paper", "htf_trend", r, t + i * MS_HOUR)
    st = ks.state[("paper", "htf_trend")]
    # -3.75 after five losses, the win pulls it to -1.5, four more losses: -4.5 (> -5: no trip)
    assert app.killed == [] and st["cusum"] == pytest.approx(-4.5) and st["cusum_low"] == pytest.approx(-4.5)
    # an alpha the champion does not have on is not tracked (its trades come from nowhere it could switch off)
    _close(app, ks, "paper", "momentum_burst", -5.0, t + 20 * MS_HOUR)
    assert ("paper", "momentum_burst") not in ks.state


def test_rolling_window_backstops_absolute_losses(tmp_path):
    # a volatile early record inflates the sd (to the 2R cap), so a later bleed moves the CUSUM slowly; the R sum of
    # the last 30 trades still catches -20R
    app = _App(tmp_path)
    ks = AlphaKillSwitch(app)
    t = now_ms() - 80 * MS_HOUR
    for i, r in enumerate([-3.0, 3.0] * 15):
        _close(app, ks, "paper", "htf_trend", r, t + i * MS_HOUR)
    for i in range(29):  # -0.7R each, CUSUM only -0.1 sd a trade; last 30: +3 - 20.3 = -17.3R
        _close(app, ks, "paper", "htf_trend", -0.7, t + (30 + i) * MS_HOUR)
    assert app.killed == [] and ks.state[("paper", "htf_trend")]["window_sum"] == pytest.approx(-17.3)
    _close(app, ks, "paper", "htf_trend", -0.7, t + 59 * MS_HOUR)  # last 30: -21R
    (alpha, d), = app.killed
    assert d["window_n"] == 30 and d["window_sum"] == pytest.approx(-21.0) and d["cusum"] == pytest.approx(-3.0)
    assert d["scale"] == 2.0 and d["rule"] == "last 30 sum < -20R"


def test_a_passed_alpha_is_held_to_its_out_of_sample_mean(tmp_path, monkeypatch):
    data = json.loads(json.dumps(E.registry()))
    data["alphas"]["htf_trend"].update(verdict="passed", avg_r=0.3)
    monkeypatch.setattr(E, "registry", lambda: data)
    app = _App(tmp_path)
    ks = AlphaKillSwitch(app)
    t = now_ms() - 50 * MS_HOUR
    for i in range(31):  # -0.2R vs an expected +0.3R (sd floor 1R): -0.25 sd a trade
        _close(app, ks, "paper", "htf_trend", -0.2, t + i * MS_HOUR)
    assert app.killed == []
    _close(app, ks, "paper", "htf_trend", -0.2, t + 31 * MS_HOUR)
    assert app.killed and app.killed[0][1]["expected"] == pytest.approx(0.3) and app.killed[0][1]["n"] == 32
    # the same record for an alpha that FAILED out of sample (expected 0) never accumulates: -0.2 + 0.25 > 0
    monkeypatch.undo()
    (tmp_path / "b").mkdir()
    app2 = _App(tmp_path / "b")
    ks2 = AlphaKillSwitch(app2)
    for i in range(32):
        _close(app2, ks2, "paper", "htf_trend", -0.2, t + i * MS_HOUR)
    assert app2.killed == [] and ks2.state[("paper", "htf_trend")]["cusum"] == 0.0


def test_epochs_reset_when_a_promotion_switches_an_alpha_on_or_retunes_it(tmp_path):
    app = _App(tmp_path)
    app.params.enabled["squeeze_breakout"] = True
    ks = AlphaKillSwitch(app)
    ks.sync_champion(app.params)
    assert ks.epochs == {"htf_trend": app.params.created, "squeeze_breakout": app.params.created}
    old = app.params
    new = old.with_alpha("squeeze_breakout", {"min_conf": 0.66})
    new.enabled["htf_trend"] = True
    new.enabled["momentum_burst"] = True
    now = now_ms()
    ks.on_new_champion(old, new, now)
    assert ks.epochs["htf_trend"] == old.created  # unchanged alpha keeps its live record
    assert ks.epochs["squeeze_breakout"] == now and ks.epochs["momentum_burst"] == now
    off = new.clone()
    off.enabled["squeeze_breakout"] = False
    ks.sync_champion(off)
    assert "squeeze_breakout" not in ks.epochs
    assert AlphaKillSwitch(app).epochs == ks.epochs  # persisted in the store


def test_trades_before_the_epoch_do_not_count(tmp_path):
    app = _App(tmp_path)
    t = now_ms() - 50 * MS_HOUR
    for i in range(12):  # an old losing streak from before the alpha was (re-)enabled
        app.store.save_trade(_trade("paper", "htf_trend", -1.0, t + i * MS_HOUR))
    app.params.created = t + 13 * MS_HOUR
    ks = AlphaKillSwitch(app)
    ks.sync_champion(app.params)
    assert asyncio.run(ks.check_all()) == [] and app.killed == []
    assert ks.state[("paper", "htf_trend")]["n"] == 0


def test_positions_opened_before_the_epoch_do_not_count(tmp_path):
    # positions opened under the previous settings that close after the epoch judged those settings, not these
    app = _App(tmp_path)
    epoch = now_ms() - 40 * MS_HOUR
    app.params.created = epoch
    for i in range(12):
        rec = _trade("paper", "htf_trend", -1.0, epoch + (i + 1) * 60_000)
        rec.entry_time = epoch - MS_HOUR  # opened an hour before the epoch, closed after it
        app.store.save_trade(rec)
    ks = AlphaKillSwitch(app)
    ks.sync_champion(app.params)
    assert asyncio.run(ks.check_all()) == [] and ks.state[("paper", "htf_trend")]["n"] == 0


def test_state_is_derived_from_the_store_after_a_restart(tmp_path):
    app = _App(tmp_path)
    ks = AlphaKillSwitch(app)
    ks.sync_champion(app.params)
    t = now_ms() - 50 * MS_HOUR
    for i in range(11):  # stored, but the process died before acting on the trip
        app.store.save_trade(_trade("paper", "htf_trend", -1.0, t + i * MS_HOUR))
    restarted = AlphaKillSwitch(app)
    trips = asyncio.run(restarted.check_all())
    assert [x["alpha"] for x in trips] == ["htf_trend"] and app.killed[0][1]["n"] == 11


def test_a_challenger_that_trips_is_retired(tmp_path):
    app = _App(tmp_path)
    p = StrategyParams.default()
    p.enabled["squeeze_breakout"] = True
    slot = ChallengerSlot("challenger-1", p, now_ms() - 60 * MS_HOUR, "squeeze_breakout")
    app.challengers.append(slot)
    app.engines["challenger-1"] = object()
    ks = AlphaKillSwitch(app)
    old = now_ms() - 70 * MS_HOUR
    app.store.save_trade(_trade("challenger-1", "squeeze_breakout", -9.0, old))  # before its start: ignored
    t = now_ms() - 50 * MS_HOUR
    for i in range(11):
        _close(app, ks, "challenger-1", "squeeze_breakout", -1.0, t + i * MS_HOUR)
    (name, reason, d), = app.retired
    assert name == "challenger-1" and "킬 스위치" in reason and d["n"] == 11 and app.killed == []


def test_disabled_kill_switch_never_acts(tmp_path):
    app = _App(tmp_path, KILL_SWITCH=False)
    ks = AlphaKillSwitch(app)
    t = now_ms() - 50 * MS_HOUR
    for i in range(15):
        _close(app, ks, "paper", "htf_trend", -1.0, t + i * MS_HOUR)
    assert app.killed == []
