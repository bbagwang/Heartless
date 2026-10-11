"""P5 full-system replay on a synthetic store (no network, no gitignored data).

The real orchestrator runs over generated 1m history on the virtual-time loop. To keep the suite fast the live feature
windows are shortened (fewer bars per indicator recomputation); everything else is the production path."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from types import SimpleNamespace

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
        st.save_metrics(s, [SimpleNamespace(ts=t, oi=1e6 + k, oi_value=1e8, top_ls_accounts=1.2, top_ls_positions=1.1,
                                            ls_accounts=1.3, taker_ls_vol=1.0)
                            for k, t in enumerate(range(t0, t0 + n * MS_MINUTE, 5 * MS_MINUTE))])
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


def test_the_replayed_bot_keeps_collecting_funding_and_metrics(replay3d):
    """The bot's store gains every funding settlement and every closed 5m metrics bucket of the window (research reads
    both from there); nothing from after the end leaks in."""
    s, d = replay3d
    con = sqlite3.connect(str(d / "bot" / "heartless.db"))
    try:
        for sym in SYMS:
            f = [t for (t,) in con.execute("SELECT funding_time FROM funding WHERE symbol=? AND funding_time>=?",
                                           (sym, START))]
            assert f[0] == START and max(f) == START + 2 * MS_DAY + 16 * MS_HOUR, sym
            lo, hi, n = con.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM metrics WHERE symbol=?", (sym,)).fetchone()
            assert lo <= START and hi <= START + 3 * MS_DAY - 5 * MS_MINUTE and hi >= START + 3 * MS_DAY - 15 * MS_MINUTE
            assert n >= 3 * 288
    finally:
        con.close()


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


def test_a_check_that_raises_once_fails_the_run_and_checking_goes_on(tmp_path, small_frames, monkeypatch):
    """The per-minute checks run in a task nobody awaits: one exception inside a check used to end that task silently,
    and the replay then passed with no invariant checked for the rest of the window."""
    from heartless.replay.invariants import InvariantChecker

    orig = InvariantChecker._orphans

    def break_once(app):
        def boom(self, *a, **k):
            monkeypatch.setattr(InvariantChecker, "_orphans", orig)  # a transient failure
            raise KeyError("checker bug")

        monkeypatch.setattr(InvariantChecker, "_orphans", boom)

    cfg = ReplayConfig(source_db=build_source(tmp_path / "src.db", 1, ["ETHUSDT"]), start_ms=START,
                       end_ms=START + 2 * MS_HOUR - 1, work_dir=str(tmp_path / "bot"), seed_days=SEED_DAYS,
                       settings=NO_LEARNING, commands=[], inject=[(START + 20 * MS_MINUTE, break_once)])
    s = run_replay(cfg)
    assert not s["passed"]
    (v,) = s["invariants"]["violations"]
    assert v["check"] == "exceptions" and "invariant check failed: KeyError" in v["detail"]
    assert v["ts"] == START + 20 * MS_MINUTE + 500
    checks = s["invariants"]["checks"]
    assert checks["orphans"]["runs"] >= 110 and checks["ledger"]["runs"] >= 110  # every later minute still checked


def test_a_retired_challengers_close_out_is_checked(tmp_path, small_frames):
    """Retiring a challenger closes its books after it left app.engines, out of reach of the minute checks (and a new
    challenger in that slot deletes its trades): the replay checks the retired engine once more right away."""
    async def stray_then_retire(app):
        slot = app.challengers[0]
        await app.paper_accounts[slot.name].place_stop("ETHUSDT", "SELL", 1.0, close_position=True)
        await app.retire_challenger(slot, "test")

    challenger = StrategyParams.from_dict(busy_params()).clone(version="vretire", note="seeded challenger")
    at = START + 30 * MS_MINUTE
    cfg = ReplayConfig(source_db=build_source(tmp_path / "src.db", 1, ["ETHUSDT"]), start_ms=START,
                       end_ms=START + MS_HOUR - 1, work_dir=str(tmp_path / "bot"), seed_days=SEED_DAYS,
                       settings=NO_LEARNING, commands=[], inject=[(at, stray_then_retire)],
                       seed_challengers=[("challenger-1", challenger.to_dict(), START - MS_DAY, "seeded")])
    s = run_replay(cfg)
    assert s["learning"]["challengers_retired"] and not s["learning"]["challengers_final"]
    (v,) = s["invariants"]["violations"]
    assert v["check"] == "orphans" and v["engine"] == "challenger-1" and v["ts"] == at


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


def test_a_work_dir_holding_the_source_is_refused(tmp_path):
    """`overwrite` wipes the work dir; pointed at the data dir (the default source lives there) it deleted the very
    history it was about to replay."""
    src = build_source(tmp_path / "src.db", 1, ["ETHUSDT"])
    with pytest.raises(ValueError, match="contains the replay source"):
        run_replay(ReplayConfig(source_db=src, start_ms=START, end_ms=START + MS_HOUR - 1, work_dir=str(tmp_path),
                                seed_days=SEED_DAYS, overwrite=True))
    assert (tmp_path / "src.db").exists()


def test_a_window_ending_at_midnight_flattens_into_the_new_day(tmp_path, small_frames):
    """The final flatten runs just after the window end, i.e. after midnight for a date-bounded window, and closed the
    positions through the engines directly, before any tick or scheduler pass had rolled the day: their close messages
    reported the previous day's counter as "today" and a consistent bot failed pnl_reports."""
    from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal
    from heartless.util.timeutil import now_ms

    async def close_one_keep_one(app):
        eng = app.engines["paper"]
        for sym in SYMS:
            ctx = app._context(sym, now_ms())
            px = ctx.ticker.mark
            sig = Signal(alpha="htf_trend", symbol=sym, side=Side.LONG, confidence=0.7, reason="t", stop=px * 0.97,
                         take_profit=None, tp1=None, entry_style=EntryStyle.MARKET, atr=px * 0.01, timeframe="1h")
            await eng._open(Decision(symbol=sym, side=Side.LONG, score=0.7, confidence=0.7, alphas=["htf_trend"],
                                     primary=sig, reason="t", regime=Regime.RANGE), app.views[sym], ctx,
                            await eng.equity())
        assert all(eng.positions[s].status.value == "OPEN" for s in SYMS)
        await app.close_symbol(SYMS[0])  # one trade on the last day; the other position is still open at the end

    quiet = StrategyParams.default()
    quiet.enabled = {a: False for a in quiet.enabled}  # no organic entries: only the two above
    start = START + 23 * MS_HOUR
    s = run_replay(ReplayConfig(source_db=build_source(tmp_path / "src.db", 2), start_ms=start,
                                end_ms=START + MS_DAY - 1, work_dir=str(tmp_path / "bot"), seed_days=SEED_DAYS,
                                settings=NO_LEARNING, commands=[], params=quiet.to_dict(),
                                inject=[(start + 30 * MS_MINUTE, close_one_keep_one)]))
    assert s["engines"]["paper"]["exit_reasons"] == {"수동 청산(owner)": 1, "리플레이 종료(replay end)": 1}
    assert s["passed"] and s["invariants"]["checks"]["pnl_reports"]["runs"] == 2, s["report"]


def test_output_files_may_go_to_new_folders(tmp_path, small_frames):
    """The messages / trades dumps are written when the run is over: a missing folder raised then and lost the run."""
    out = tmp_path / "runs" / "a"
    s = run_replay(ReplayConfig(source_db=build_source(tmp_path / "src.db", 1, ["ETHUSDT"]), start_ms=START,
                                end_ms=START + 30 * MS_MINUTE - 1, work_dir=str(tmp_path / "bot"), seed_days=SEED_DAYS,
                                settings=NO_LEARNING, commands=[], messages_out=str(out / "m.jsonl"),
                                trades_out=str(out / "t.json"), out_report=str(tmp_path / "rep" / "r.txt")))
    assert s["passed"] and (out / "m.jsonl").exists() and (out / "t.json").exists()
    assert (tmp_path / "rep" / "r.txt").read_text().startswith("Heartless replay")


def test_the_replay_serves_nothing_older_than_the_seal_checked_seed_window(tmp_path, small_frames):
    """The seal check covers [start - seed_days, end]; the replayed bot backfills HISTORY_DAYS (here longer than the
    seed) from the replay REST, which used to answer from the whole source store, sealed 2025 included."""
    start = 1_768_435_200_000  # 2026-01-15 00:00 UTC
    t0 = start - 22 * MS_DAY  # 2025-12-24: the source reaches into the sealed year
    st = Store(tmp_path / "src.db")
    ca = synth_candles(22 * 1440 + 180, seed=5, start_ms=t0, price=200.0, vol=0.0012)
    st.save_candles("ETHUSDT", [ca.candle_at(k) for k in range(ca.n)])
    st.save_funding("ETHUSDT", [(t, 0.0001, 0.0) for t in range(t0, start + 3 * MS_HOUR, 8 * MS_HOUR)])
    st.close()
    s = run_replay(ReplayConfig(source_db=str(tmp_path / "src.db"), start_ms=start, end_ms=start + 2 * MS_HOUR - 1,
                                work_dir=str(tmp_path / "bot"), seed_days=10, commands=[],
                                settings={**NO_LEARNING, "HISTORY_DAYS": 21}))
    assert s["market"]["bars"] == 120
    con = sqlite3.connect(str(tmp_path / "bot" / "heartless.db"))
    try:
        floor = start - 10 * MS_DAY
        assert con.execute("SELECT MIN(open_time) FROM candles").fetchone()[0] >= floor
        assert con.execute("SELECT MIN(funding_time) FROM funding").fetchone()[0] >= floor
    finally:
        con.close()


def test_the_replay_serves_nothing_after_the_window_end(tmp_path, small_frames):
    """A window that ends right before the sealed year passes the seal check, but the stream's funding estimate is
    blended toward the NEXT settlement: in the last hours it showed the first sealed rate."""
    end = 1_735_689_600_000 - 1  # 2024-12-31 23:59:59.999, the last instant before the sealed 2025
    start = end + 1 - 2 * MS_HOUR
    t0 = start - 12 * MS_DAY
    st = Store(tmp_path / "src.db")
    ca = synth_candles(12 * 1440 + 180, seed=6, start_ms=t0, price=200.0, vol=0.0012)  # runs into 2025
    st.save_candles("ETHUSDT", [ca.candle_at(k) for k in range(ca.n)])
    eight = 8 * MS_HOUR
    st.save_funding("ETHUSDT", [(t, 0.0001, 0.0) for t in range(t0 - t0 % eight + eight, end, eight)]
                    + [(end + 1, 0.0123, 0.0)])  # the first settlement of the sealed year
    st.close()
    seen: dict[str, float] = {}

    def peek(app):
        seen["r"] = app.tickers["ETHUSDT"].funding_rate

    s = run_replay(ReplayConfig(source_db=str(tmp_path / "src.db"), start_ms=start, end_ms=end,
                                work_dir=str(tmp_path / "bot"), seed_days=10, commands=[], settings=NO_LEARNING,
                                inject=[(end + 1 - 90_000, peek)]))
    assert s["market"]["bars"] == 120 and seen["r"] == pytest.approx(0.0001)
    con = sqlite3.connect(str(tmp_path / "bot" / "heartless.db"))
    try:
        assert con.execute("SELECT MAX(close_time) FROM candles").fetchone()[0] <= end
        assert con.execute("SELECT MAX(funding_time) FROM funding").fetchone()[0] <= end
    finally:
        con.close()


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
    # P4: the champion closed 12 trades (< PROMOTION_MIN_TRADES) since the challenger started, so the challenger is
    # judged on the absolute path; it switches on alphas that failed out of sample, hence the 30-day / t >= 2 bar
    started = START - 35 * MS_DAY
    challenger = StrategyParams.from_dict(busy_params()).clone(version="vchallenger", note="seeded challenger")
    trades = [_trade("paper", i, 0.5 if i % 2 else -0.6, started + (i + 1) * 3 * MS_HOUR) for i in range(12)]
    trades += [_trade("challenger-1", i, 1.0 if i % 5 else -0.5, started + (i + 1) * 20 * MS_HOUR) for i in range(40)]
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
    assert learn["challenger_evaluations"]["by_path"].get("absolute", 0) >= 1 and len(learn["promotions"]) == 1
    (d,) = [x for x in learn["challenger_decisions"] if x["verdict"] == "promote"]
    # C2-B: judged at its 30-day checkpoint (the failed-alpha schedule) on the trades of its first 30 days
    from heartless.learning import research as R

    assert d["path"] == "absolute" and d["failed_alphas"] and d["bar"]["min_days"] == 30
    assert d["bar"]["checkpoint"] == 30 and d["bar"]["min_t_day"] == R.FAILED_ABS_T_AT[30]
    assert d["challenger"]["n"] >= 35 and d["challenger"]["days"] >= 30 and d["champion_n"] == 12
    assert learn["champion_final"] == learn["promotions"][0]["version"] != "vchallenger"
    assert s["engines"]["paper"]["params_version"] == learn["champion_final"]  # the champion engine hot-swapped
    tg = s["telegram"]["by_type"]
    assert tg["research_started"] >= 2 and tg["research_done"] >= 2 and tg["promotion"] == 1
    assert tg["cmd:/research"] == 1
