"""P4 inside the P5 full-system replay (synthetic store): a losing alpha trips the kill switch in the champion and in a
challenger, the owner is told, the champion trades nothing afterwards, and every invariant holds. (A challenger that
clears the absolute bar being promoted inside a replay is test_p5_replay.test_learning_loop_runs_for_real_inside_a_replay.)"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from synth import synth_candles

from heartless.app import NO_ALPHA_TEXT
from heartless.core.store import Store
from heartless.data import features
from heartless.replay.runner import ReplayConfig, run_replay
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

START = 1_772_323_200_000  # 2026-03-01 00:00 UTC
SEED_DAYS = 14
SMALL_FRAMES = {"1m": 240, "5m": 240, "15m": 240, "1h": 260}
NO_LEARNING = {"RESEARCH_INTERVAL_MINUTES": 10**9, "DISCOVERY_INTERVAL_HOURS": 10**8, "HISTORY_DAYS": 12}


@pytest.fixture
def small_frames(monkeypatch):
    for tf, n in SMALL_FRAMES.items():
        monkeypatch.setitem(features.LIVE_WINDOW, tf, n)


def _source(path) -> str:
    st = Store(path)
    t0 = START - SEED_DAYS * MS_DAY
    n = (SEED_DAYS + 1) * 1440
    ca = synth_candles(n, seed=7, start_ms=t0, price=200.0, vol=0.0012)
    st.save_candles("ETHUSDT", [ca.candle_at(k) for k in range(ca.n)])
    st.save_funding("ETHUSDT", [(t, 0.0001, 0.0) for t in range(t0 + 8 * MS_HOUR - t0 % (8 * MS_HOUR), t0 + n * MS_MINUTE,
                                                              8 * MS_HOUR)])
    st.save_metrics("ETHUSDT", [SimpleNamespace(ts=t, oi=1e6, oi_value=1e8, top_ls_accounts=1.2, top_ls_positions=1.1,
                                                ls_accounts=1.3, taker_ls_vol=1.0)
                                for t in range(t0, t0 + n * MS_MINUTE, 5 * MS_MINUTE)])
    st.close()
    return str(path)


def _trade(engine: str, alpha: str, i: int, r: float, exit_time: int) -> dict:
    return {"position_id": f"seed-{engine}-{i}", "engine": engine, "symbol": "ETHUSDT", "side": "LONG", "alpha": alpha,
            "alphas": [alpha], "regime": "RANGE", "entry_time": exit_time - MS_HOUR, "exit_time": exit_time,
            "entry_price": 200.0, "exit_price": 198.0, "qty": 1.0, "notional": 200.0, "pnl": r * 50.0, "gross": r * 50.0,
            "fees": 0.0, "funding": 0.0, "r_multiple": r, "risk_amount": 50.0, "exit_reason": "seed", "confidence": 0.7,
            "params_version": "seed", "bars_held": 60, "max_fav_r": 0.0, "max_adv_r": 0.0}


def test_losing_alphas_trip_the_kill_switch_inside_a_replay(tmp_path, small_frames):
    champ = StrategyParams.default().clone(version="vchamp", note="htf_trend on")
    champ.enabled["htf_trend"] = True
    champ.created = START - 10 * MS_DAY
    challenger = StrategyParams.default().clone(version="vchal", note="squeeze_breakout on")
    challenger.enabled["squeeze_breakout"] = True
    started = START - 3 * MS_DAY
    trades = [_trade("paper", "htf_trend", i, -1.0, START - 5 * MS_DAY + i * MS_HOUR) for i in range(12)]
    trades += [_trade("challenger-1", "squeeze_breakout", i, -1.0, started + (i + 1) * MS_HOUR) for i in range(12)]
    cfg = ReplayConfig(source_db=_source(tmp_path / "src.db"), start_ms=START, end_ms=START + 2 * MS_HOUR - 1,
                       work_dir=str(tmp_path / "bot"), seed_days=SEED_DAYS, settings=NO_LEARNING, commands=[],
                       params=champ.to_dict(), seed_trades=trades,
                       seed_challengers=[("challenger-1", challenger.to_dict(), started, "squeeze_breakout")],
                       messages_out=str(tmp_path / "messages.jsonl"))
    s = run_replay(cfg)
    assert s["passed"], s["report"]
    learn = s["learning"]
    trips = {(t["engine"], t["alpha"]): t for t in learn["kill_switch_trips"]}
    assert set(trips) == {("paper", "htf_trend"), ("challenger-1", "squeeze_breakout")}
    assert trips[("paper", "htf_trend")]["n"] == 12 and trips[("paper", "htf_trend")]["cusum"] == pytest.approx(-9.0)
    assert learn["champion_enabled"] == [] and learn["champion_final"] != "vchamp"
    assert learn["challengers_retired"] and not learn["challengers_final"]
    assert s["engines"]["paper"]["params_version"] == learn["champion_final"]
    assert s["trades"]["opened"].get("paper", 0) == 0  # the champion never trades the switched-off alpha
    msgs = [json.loads(line) for line in (tmp_path / "messages.jsonl").read_text().splitlines()]
    kill = [m for m in msgs if m["topic"] == "kill_switch"]
    assert len(kill) == 1 and "htf_trend 끔" in kill[0]["text"] and "CUSUM -9.00σ" in kill[0]["text"]
    retired = [m for m in msgs if m["topic"] == "challenger_retired"]
    assert len(retired) == 1 and "킬 스위치" in retired[0]["text"] and "squeeze_breakout" in retired[0]["text"]
    startup = next(m for m in msgs if m["topic"] == "startup")
    assert NO_ALPHA_TEXT in startup["text"]  # the startup summary is written after the switch: truthful
