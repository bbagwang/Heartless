"""P5 full-system replay on a synthetic store (no network, no gitignored data).

The real orchestrator runs over generated 1m history on the virtual-time loop. To keep the suite fast the live feature
windows are shortened (fewer bars per indicator recomputation); everything else is the production path."""
from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from synth import synth_candles

from heartless.core.store import Store
from heartless.data import features
from heartless.replay.runner import ReplayConfig, run_replay
from heartless.replay.telegram import ScriptedCommand, parse_command
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

START = 1_772_323_200_000  # 2026-03-01 00:00 UTC (outside the sealed 2025 period, like the seed history before it)
SEED_DAYS = 14
SYMS = ["BTCUSDT", "ETHUSDT"]
SMALL_FRAMES = {"1m": 240, "5m": 240, "15m": 240, "1h": 260}
NO_LEARNING = {"RESEARCH_INTERVAL_MINUTES": 10**9, "DISCOVERY_INTERVAL_HOURS": 10**8, "HISTORY_DAYS": 12}
PAUSE, RESUME, CLOSE = START + 18 * MS_HOUR, START + 34 * MS_HOUR, START + 36 * MS_HOUR


def build_source(path, days: int, syms=SYMS) -> str:
    """A store with SEED_DAYS + `days` of synthetic 1m candles and 8-hourly funding per symbol."""
    st = Store(path)
    t0 = START - SEED_DAYS * MS_DAY
    n = (SEED_DAYS + days) * 1440
    for i, s in enumerate(syms):
        ca = synth_candles(n, seed=11 + i, start_ms=t0, price=100.0 * (i + 1), vol=0.0012)
        st.save_candles(s, [ca.candle_at(k) for k in range(ca.n)])
        st.save_funding(s, [(t, 0.0001 if (t // (8 * MS_HOUR)) % 2 else -0.00005, 0.0)
                            for t in range(t0 + 8 * MS_HOUR - t0 % (8 * MS_HOUR), t0 + n * MS_MINUTE, 8 * MS_HOUR)])
    st.close()
    return str(path)


def busy_params() -> dict:
    """Shipped defaults plus a few alphas that trade often on synthetic data (more trades in a short window)."""
    p = StrategyParams.default()
    for a in ("trend_pullback", "sweep_reversal", "squeeze_breakout", "mean_reversion"):
        p.enabled[a] = True
    return p.to_dict()


@pytest.fixture(scope="module")
def small_frames():
    with pytest.MonkeyPatch.context() as mp:
        for tf, n in SMALL_FRAMES.items():
            mp.setitem(features.LIVE_WINDOW, tf, n)
        yield


@pytest.fixture(scope="module")
def replay3d(tmp_path_factory, small_frames):
    d = tmp_path_factory.mktemp("replay3d")
    cmds = [ScriptedCommand(START + MS_HOUR, "/status"), ScriptedCommand(START + 2 * MS_HOUR, "/positions"),
            ScriptedCommand(PAUSE, "/pause"), ScriptedCommand(RESUME, "/resume"),
            ScriptedCommand(CLOSE, "/close *"), ScriptedCommand(START + 40 * MS_HOUR, "/pnl"),
            ScriptedCommand(START + 41 * MS_HOUR, "/pnl week"), ScriptedCommand(START + 42 * MS_HOUR, "/report"),
            ScriptedCommand(START + 43 * MS_HOUR, "/trades"), ScriptedCommand(START + 44 * MS_HOUR, "/challengers"),
            ScriptedCommand(START + 45 * MS_HOUR, "/close all"), ScriptedCommand(START + 45 * MS_HOUR + 1000, "@confirm")]
    cfg = ReplayConfig(source_db=build_source(d / "src.db", 3), start_ms=START, end_ms=START + 3 * MS_DAY - 1,
                       work_dir=str(d / "bot"), seed_days=SEED_DAYS, params=busy_params(), settings=NO_LEARNING,
                       commands=cmds, out_json=str(d / "summary.json"), messages_out=str(d / "messages.jsonl"))
    return run_replay(cfg), d


def test_three_day_replay_passes_every_invariant(replay3d):
    s, d = replay3d
    inv = s["invariants"]
    assert s["passed"] and inv["violations_total"] == 0, s["report"]
    checks = inv["checks"]
    for k in ("positions", "stops", "orphans", "ledger"):
        assert checks[k]["runs"] >= 3 * 1440  # every virtual minute (+ the final passes)
    assert checks["pnl_reports"]["runs"] >= 5 and checks["trades"]["runs"] >= 70
    assert s["market"]["bars"] == 3 * 1440 * len(SYMS) and s["market"]["gaps"] == 0
    assert s["virtual_seconds"] == pytest.approx(3 * 86400, abs=5) and s["speedup"] > 50
    assert not s["loop"]["deadlocked"] and s["market"]["rest_calls"].get("/fapi/v1/exchangeInfo") == 1
    assert (d / "summary.txt").exists() and "PASS" in s["report"]


def test_trades_happen_and_telegram_captures_opens_and_closes(replay3d):
    s, _ = replay3d
    champ = s["engines"]["paper"]
    assert champ["trades"] >= 5 and s["trades"]["opened"]["paper"] >= 5
    tg = s["telegram"]["by_type"]
    assert tg["position_opened"] == s["trades"]["opened"]["paper"]  # the champion notifies every open ...
    assert tg["position_closed"] == champ["trades"]  # ... and every close
    assert tg["daily_report"] >= 3 and tg["startup"] == 1
    for cmd in ("cmd:/status", "cmd:/positions", "cmd:/pnl", "cmd:/trades", "cmd:/challengers", "cmd:/close"):
        assert tg.get(cmd, 0) >= 1, cmd
    assert tg["control"] >= 2  # /pause and /resume answer through the control event
    assert s["before_final_flatten"]["paper"]["open_positions"] == 0 or champ["exit_reasons"].get(
        "리플레이 종료(replay end)", 0) >= 1


def test_scripted_pause_blocks_entries_and_resume_reenables_them(replay3d):
    s, d = replay3d
    import json

    msgs = [json.loads(line) for line in (d / "messages.jsonl").read_text().splitlines()]
    opens = [m["ts"] for m in msgs if m["topic"] == "position_opened"]
    assert any(t < PAUSE for t in opens), "no entry before the pause"
    assert not any(PAUSE <= t < RESUME for t in opens), "an entry opened while paused"
    assert any(t >= RESUME for t in opens), "no entry after the resume"
    skipped = s["before_final_flatten"]["paper"]["skipped"]
    assert skipped.get("paused", 0) >= 1  # signals arrived during the pause and were refused for it
    ctrl = [m for m in msgs if m["topic"] == "control"]
    assert ctrl[0]["ts"] == PAUSE and "일시정지" in ctrl[0]["text"] and ctrl[1]["ts"] == RESUME


def test_close_commands_go_through_the_control_path(replay3d):
    s, d = replay3d
    import json

    msgs = [json.loads(line) for line in (d / "messages.jsonl").read_text().splitlines()]
    reply = next(m for m in msgs if m["topic"] == "cmd:/close" and m["ts"] >= CLOSE)
    reasons = s["engines"]["paper"]["exit_reasons"]
    if "완료" in reply["text"]:  # a position was open at that instant: it is booked as a manual close
        assert reasons.get("수동 청산(owner)", 0) >= 1
    else:
        assert "포지션 없음" in reply["text"]
    confirm = [m for m in msgs if m["topic"] == "callback:cf"]
    assert confirm and "청산 요청 완료" in confirm[0]["text"]  # /close all -> confirmation button -> close_all


def test_fault_injection_is_detected_end_to_end(tmp_path, small_frames):
    """A replay with injected faults fails, and each fault is reported by its invariant at the minute it happened."""
    t1, t2, t3, t4 = START + 30 * MS_MINUTE, START + 60 * MS_MINUTE, START + 75 * MS_MINUTE, START + 90 * MS_MINUTE

    async def orphan(app):
        await app.paper_accounts["paper"].place_stop("ETHUSDT", "SELL", 1.0, close_position=True)

    def corrupt(app):
        app.paper_accounts["paper"].wallet += 3.0

    async def network(app):
        try:
            async with httpx.AsyncClient() as c:
                await c.get("https://api.telegram.org/botX/getMe")
        except Exception:  # noqa: BLE001 - recorded by the guard either way
            pass

    async def crash(app):
        async def boom():
            raise RuntimeError("injected task failure")

        asyncio.get_running_loop().create_task(boom())
        await asyncio.sleep(0)
        logging.getLogger("heartless.app").error("injected error log")

    cfg = ReplayConfig(source_db=build_source(tmp_path / "src.db", 1, ["ETHUSDT"]), start_ms=START,
                       end_ms=START + 2 * MS_HOUR - 1, work_dir=str(tmp_path / "bot"), seed_days=SEED_DAYS,
                       settings=NO_LEARNING, commands=[],
                       inject=[(t1, orphan), (t2, corrupt), (t3, network), (t4, crash)])
    s = run_replay(cfg)
    assert not s["passed"]
    checks = s["invariants"]["checks"]
    assert checks["orphans"]["violations"] >= 1 and checks["ledger"]["violations"] >= 1
    assert checks["network"]["violations"] == 1 and checks["exceptions"]["violations"] >= 2
    first = {}
    for v in s["invariants"]["violations"]:
        first.setdefault(v["check"], v)
    assert first["orphans"]["ts"] == t1 + 500  # caught by the minute check right after the injection
    assert first["ledger"]["ts"] == t2 + 500 and "wallet change +3.0" in first["ledger"]["detail"]
    assert "api.telegram.org" in first["network"]["detail"]
    details = " | ".join(v["detail"] for v in s["invariants"]["violations"] if v["check"] == "exceptions")
    assert "injected task failure" in details and "injected error log" in details


def test_command_specs_parse_offsets_and_dates():
    assert parse_command("+2d=/pause", START).at_ms == START + 2 * MS_DAY
    assert parse_command("+1d6h30m=/close BTCUSDT", START) == ScriptedCommand(START + 30 * MS_HOUR + 30 * MS_MINUTE,
                                                                             "/close BTCUSDT")
    assert parse_command("2026-03-02T06:00=/resume", START).at_ms == START + 30 * MS_HOUR
    with pytest.raises(ValueError):
        parse_command("/pause", START)


def test_a_window_touching_the_sealed_period_is_refused(tmp_path):
    from heartless.learning.seal import SealedError

    src = build_source(tmp_path / "src.db", 1, ["ETHUSDT"])
    with pytest.raises(SealedError):  # the seed history (the 60 days before start) reaches into 2025
        run_replay(ReplayConfig(source_db=src, start_ms=START, end_ms=START + MS_DAY - 1))


def _trade(engine: str, i: int, r: float, exit_time: int) -> dict:
    risk = 50.0
    return {"position_id": f"seed-{engine}-{i}", "engine": engine, "symbol": "ETHUSDT", "side": "LONG",
            "alpha": "htf_trend", "alphas": ["htf_trend"], "regime": "RANGE", "entry_time": exit_time - MS_HOUR,
            "exit_time": exit_time, "entry_price": 200.0, "exit_price": 200.0 + r, "qty": 1.0, "notional": 200.0,
            "pnl": r * risk, "gross": r * risk, "fees": 0.0, "funding": 0.0, "r_multiple": r, "risk_amount": risk,
            "exit_reason": "seed", "confidence": 0.7, "params_version": "seed", "bars_held": 60, "max_fav_r": 0.0,
            "max_adv_r": 0.0}


def test_learning_loop_runs_for_real_inside_a_replay(tmp_path, small_frames):
    """Research and discovery cycles run in their worker process on the bot's own store, and a challenger evaluation
    compares records and hot-swaps the champion, all inside the replay and with every invariant holding."""
    seed_days = 20
    src = tmp_path / "src.db"
    st = Store(src)
    t0 = START - seed_days * MS_DAY
    for i, sym in enumerate(SYMS):
        ca = synth_candles((seed_days + 1) * 1440, seed=21 + i, start_ms=t0, price=100.0 * (i + 1), vol=0.0012)
        st.save_candles(sym, [ca.candle_at(k) for k in range(ca.n)])
    st.close()
    started = START - 3 * MS_DAY
    challenger = StrategyParams.from_dict(busy_params()).clone(version="vchallenger", note="seeded challenger")
    trades = [_trade("paper", i, 0.5 if i % 2 else -0.6, started + (i + 1) * 3 * MS_HOUR) for i in range(12)]
    trades += [_trade("challenger-1", i, 1.0 if i % 5 else -0.5, started + (i + 1) * 2 * MS_HOUR) for i in range(26)]
    cfg = ReplayConfig(source_db=str(src), start_ms=START, end_ms=START + 2 * MS_HOUR - 1, work_dir=str(tmp_path / "bot"),
                       seed_days=seed_days, commands=[ScriptedCommand(START + MS_HOUR, "/research")],
                       settings={"RESEARCH_CANDIDATES": 1, "RESEARCH_FOLDS": 1, "RESEARCH_LOOKBACK_DAYS": 5,
                                 "DISCOVERY_MIN_DAYS": 10, "HISTORY_DAYS": 12},
                       seed_challengers=[("challenger-1", challenger.to_dict(), started, "seeded")], seed_trades=trades)
    s = run_replay(cfg)
    assert s["passed"], s["report"]
    learn = s["learning"]
    assert len(learn["research_cycles"]) == 1 and learn["research_cycles"][0]["error"] is None
    assert learn["research_cycles"][0]["alphas"]  # every alpha with parameters was walk-forward evaluated
    assert len(learn["discovery_cycles"]) == 1 and learn["discovery_cycles"][0]["tested"] > 0
    assert learn["challenger_evaluations"]["compared"] >= 1 and len(learn["promotions"]) == 1
    assert learn["champion_final"] == learn["promotions"][0]["version"] != "vchallenger"
    assert s["engines"]["paper"]["params_version"] == learn["champion_final"]  # the champion engine hot-swapped
    tg = s["telegram"]["by_type"]
    assert tg["research_started"] >= 2 and tg["research_done"] >= 2 and tg["promotion"] == 1
    assert tg["cmd:/research"] == 1
