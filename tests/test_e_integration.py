"""Wave E integration: the packages read each other's outputs. The engine's decision record (E3) sees the portfolio
allocator's batch decisions (E1); /status, the daily report, the start message and the research page state the
allocator; the operator views name the evidence 'stage' (E2) and the judge's HAC statistic (E2); the research
schedule shown is the one ResearchManager.due() keeps (E2)."""
from __future__ import annotations

import json

import pytest

from heartless.execution.allocator import SAME_REASON, TOTAL_REASON
from heartless.learning import research as R
from heartless.notify import formatter as F
from heartless.strategy import evidence
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE, now_ms
from heartless.web.app import WebServer
from tests.test_e1_allocator import _bar_all, _engine
from tests.test_e3_operator_ux import _bot, _challenger, _make
from tests.test_p4_app import _boot


# --- E1 x E3: the allocator's decisions in the engine's decision record ----------------------------------------------
async def test_allocator_batch_decisions_reach_the_decision_record():
    # 0.5% risk per trade, total budget 0.9%: the best-ranked candidate opens, the other three do not fit the budget
    eng, acc = _engine(True, {"DDDUSDT": 0.95, "CCCUSDT": 0.9}, MAX_TOTAL_RISK_PCT=0.9)
    eng.decision_log = {}  # what a persisted (app) engine keeps
    await _bar_all(eng, acc)
    assert eng.decision_log == {}  # queued: nothing is decided before the whole close is seen
    assert await eng.flush_entries() == 1
    d = eng.decision_log["DDDUSDT"][-1]
    assert d["stage"] == "opened" and d["why"] == "MARKET"
    assert d["allocation"]["rank"] == 1 and d["allocation"]["of"] == 4 and d["allocation"]["by"] == "bandit"
    assert d["signals"][0]["alpha"] == "htf_trend" and d["decision"]["side"] == "LONG"
    c = eng.decision_log["CCCUSDT"][-1]
    assert c["stage"] == "allocator" and c["why"] == TOTAL_REASON and c["allocation"]["rank"] == 2
    a = c["allocation"]
    opened = eng.positions["DDDUSDT"].risk_amount / 10_000.0 * 100.0  # % of the 10,000 USDT account
    assert a["max_total_pct"] == 0.9 and a["total_pct"] == pytest.approx(opened, abs=1e-3)
    assert a["long_pct"] == pytest.approx(opened, abs=1e-3)  # one long, beta 1.0 without history
    assert a["add_pct"] > 0 and a["total_pct"] + a["add_pct"] > 0.9  # why it did not fit
    assert {eng.decision_log[s][-1]["allocation"]["rank"] for s in ("AAAUSDT", "BBBUSDT")} == {3, 4}
    assert eng.stats.skipped.get(TOTAL_REASON) == 3  # the counters are unchanged by the record
    json.dumps({s: list(v) for s, v in eng.decision_log.items()})  # plain values (served as JSON)
    text = F.fmt_decision(c, "UTC")
    assert "거절(할당기): 총 위험 예산 초과" in text and "할당 2/4위(밴딧 점수" in text and f"열린 위험 총 {a['total_pct']:.2f}%/0.9%" in text
    assert "거절(할당기): 같은 방향 위험 예산 초과(BTC 베타 가중)" in F.decision_reason(
        {"stage": "allocator", "why": SAME_REASON})


async def test_allocator_count_caps_and_disabled_entries_are_recorded_with_the_batch_place():
    eng, acc = _engine(True, {"DDDUSDT": 0.95, "CCCUSDT": 0.9})  # MAX_POSITIONS=2, budgets not binding
    eng.decision_log = {}
    await _bar_all(eng, acc)
    assert await eng.flush_entries() == 2
    assert [eng.decision_log[s][-1]["stage"] for s in ("DDDUSDT", "CCCUSDT")] == ["opened", "opened"]
    for s in ("AAAUSDT", "BBBUSDT"):
        rec = eng.decision_log[s][-1]
        assert rec["stage"] == "risk" and rec["why"] == "max positions" and rec["allocation"]["rank"] in (3, 4)
    # a batch queued before entries were switched off (live -> paper flatten): each candidate says why it did not open
    for p in list(eng.positions.values()):
        eng.positions.pop(p.symbol)
    await _bar_all(eng, acc)
    eng.entries_enabled = False
    assert await eng.flush_entries() == 0
    assert all(eng.decision_log[s][-1]["why"] == "entries disabled" for s in ("AAAUSDT", "DDDUSDT"))
    assert "모드 전환 중 진입 꺼짐" in F.decision_reason(eng.decision_log["AAAUSDT"][-1])


async def test_fifo_records_carry_no_batch_place():
    eng, acc = _engine(False, {"DDDUSDT": 0.95})
    eng.decision_log = {}
    await _bar_all(eng, acc)
    assert eng.decision_log["AAAUSDT"][-1]["stage"] == "opened" and "allocation" not in eng.decision_log["AAAUSDT"][-1]
    assert eng.decision_log["CCCUSDT"][-1]["stage"] == "risk"  # FIFO: the first two processed took the slots


# --- E1 x E3: /status, the daily report, the start message and the research page state the allocator -----------------
async def test_status_daily_report_start_message_and_research_page_state_the_allocator_off():
    app, candles = _make()
    await _boot(app, candles)
    st = app.status()
    assert st["portfolio"]["enabled"] is False and "exposure" not in st["portfolio"]
    assert "포트폴리오 할당기 꺼짐(<code>PORTFOLIO_ALLOCATOR=false</code>, 출하 기본)" in F.fmt_status(st, "UTC")
    assert "포트폴리오 할당기 꺼짐(출하 기본)" in app.startup_summary()
    bot, sent = _bot(app)
    await app.daily_report()
    text = next(m for m in sent if "일일 리포트" in m)
    assert "포트폴리오 할당기 꺼짐" in text
    html = WebServer(app).research_html()
    assert "포트폴리오 할당기 꺼짐(PORTFOLIO_ALLOCATOR=false, 출하 기본)" in html
    await app.stop()


async def test_status_shows_the_open_risk_against_the_budgets_with_the_allocator_on():
    app, candles = _make(PORTFOLIO_ALLOCATOR=True)
    await _boot(app, candles)
    st = app.status()
    p = st["portfolio"]
    assert p["enabled"] is True and p["exposure"] == {"total_pct": 0.0, "long_pct": 0.0, "short_pct": 0.0}
    assert p["rejected"] == {"total": 0, "same": 0}
    line = F.fmt_portfolio(st)
    assert line.startswith("포트폴리오 할당기 켜짐: 열린 위험 총 0.00%/2.5%") and "숏 0.00%/1.5%(BTC 베타 가중)" in line
    assert "포트폴리오 할당기 켜짐: 위험 예산 총 2.5%·같은 방향 1.5%" in app.startup_summary()
    await app.stop()


# --- E2 x E3: evidence stage, the judge's statistic, the research schedule and selection ------------------------------
def test_the_evidence_stage_field_names_a_train_gate_failure():
    # 'stage' wins over the blocks an entry carries (a TRAIN-gate failure never has a VALID verdict)
    assert F.evidence_stage({"verdict": "failed", "stage": "train", "valid": {"n": 1}}) == "P2 TRAIN"
    for a in ("xs_rank_mom", "funding_carry_xs"):
        e = evidence.entry(a)
        assert e["stage"] == "train" and evidence.stage(a) == "train" and F.evidence_stage(e) == "P2 TRAIN"
    assert F.fmt_evidence(evidence.entry("xs_rank_mom")).startswith("P2 TRAIN 탈락: TRAIN 2022~2024 n=3356 -0.004R t -0.46/3")
    assert F.evidence_stage(evidence.entry("vol_breakout_dbox")) == "2025 확인 시험"  # stage 'oos' + sealed2025 block


async def test_the_challenger_outlook_names_the_judges_hac_statistic():
    app, candles = _make()
    await _boot(app, candles)
    slot = await _challenger(app)  # switches on trend_pullback (failed P1): failed-alpha schedule
    out = app.challenger_outlook()[0]
    assert out["name"] == slot.name and out["hac_lag"] == R.PROMOTION_HAC_LAG
    assert out["need"] == R.FAILED_ABS_T_AT[30] and out["path"] == "absolute"
    line = F.fmt_outlook(out, "UTC")
    assert f"HAC({R.PROMOTION_HAC_LAG}일) t_day " in line and f"/{R.FAILED_ABS_T_AT[30]:g} →" in line
    assert "HAC(5일) t≥" not in app.startup_summary() and "일 단위 HAC(뉴이-웨스트 5일) t ≥ " in app.startup_summary()
    await app.stop()


async def test_the_research_schedule_and_selection_shown_are_the_ones_the_manager_keeps():
    app, candles = _make()
    await _boot(app, candles)
    rm = app.research
    now = now_ms()
    rm.last_cycle_ts, rm.last_cycle_ms = now - MS_HOUR, 0
    assert app.status()["research"]["next_due"] == rm.last_cycle_ts + 120 * MS_MINUTE  # RESEARCH_INTERVAL_MINUTES
    rm.last_cycle_ms = 3 * MS_HOUR  # a long cycle: due() waits twice its length
    assert rm.next_due_ts() == rm.last_cycle_ts + 6 * MS_HOUR == app.status()["research"]["next_due"]
    assert not rm.due(rm.last_cycle_ts + 5 * MS_HOUR) and rm.due(rm.last_cycle_ts + 6 * MS_HOUR)
    sel = rm.research_selection()
    assert sel["enabled"] == [] and sel["untested"] == [] and len(sel["failed"]) == 2 and sel["failed_total"] == 15
    summary = {"seconds": 1865.0, "symbols": ["BTCUSDT"], "window": {"since": now - 120 * MS_DAY, "until": now},
               "alphas": {}, "selection": sel["alphas"], "selection_detail": sel}
    rm.last_summary = summary
    app.store.save_research_run(now - MS_HOUR, "all", summary)
    ins = app.report_insight()
    assert ins["research"]["next_due"] == rm.next_due_ts()
    assert ins["research"]["selection"] == R.describe_selection(sel) and "탈락 알파 순환 2/15" in ins["research"]["selection"]
    assert "  조정 대상: 탈락 알파 순환 2/15" in "\n".join(F.fmt_insight(ins, app.status(), "UTC"))
    assert "탈락 알파 순환 2/15" in WebServer(app).research_html()
    await app.stop()


# --- E2 x C2-A: the context auto-build of lab / discover reads the candle database read-only -------------------------
def test_context_prepare_opens_the_candle_database_read_only(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    from heartless.data import context as C
    from heartless.learning import trials
    from tests.test_c2a_context import DAY, START, SYMS, _make_db

    monkeypatch.setattr(trials, "LEDGER", tmp_path / "p2_trials.jsonl")
    src = _make_db(tmp_path / "data" / "heartless.db", days=3)
    side = [Path(src + sfx) for sfx in ("-wal", "-shm", "-journal")]
    assert not any(p.exists() for p in side)  # the writer closed and checkpointed
    digest = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    import heartless.core.store as store_mod

    opened: list[tuple[str, bool]] = []

    class Spy(store_mod.Store):
        def __init__(self, path, readonly: bool = False):
            opened.append((str(path), readonly))
            super().__init__(path, readonly=readonly)

    monkeypatch.setattr(store_mod, "Store", Spy)
    path = C.prepare(src, {"universe": list(SYMS)}, START + DAY, START + 2 * DAY)  # builds the missing hours
    assert path == C.default_path(src)
    assert (src, True) in opened and (src, False) not in opened  # the source only ever opened read-only
    assert hashlib.sha256(Path(src).read_bytes()).hexdigest() == digest  # the source file is byte-identical
    assert not any(p.exists() for p in side)  # and no side file was left next to it


def test_status_challenger_lines_quote_the_judges_hac_statistic():
    from heartless.config import Settings

    bar = R.promotion_bar(Settings(_env_file=None), "absolute", [], 14)
    d = {"path": "absolute", "checkpoint": None, "next_checkpoint": 14, "verdict": "hold", "bar": bar,
         "challenger": {"n": 9, "days": 3, "avg_r": 0.8, "t_day": 3.1, "t_hac": 2.5}}
    st = {"learning": {"headline": "x"}, "challengers": [{"name": "challenger-1", "source": "discovered", "decision": d}]}
    line = F.fmt_learning(st)[1]
    assert f"HAC({R.PROMOTION_HAC_LAG}일) t_day +2.50/{R.ABS_T_AT[14]:g}" in line  # the HAC t, not the lag-0 3.10
    old = {**d, "bar": {k: v for k, v in bar.items() if k not in ("hac_lag", "min_t_hac")},
           "challenger": {"n": 9, "days": 3, "avg_r": 0.8, "t_day": 3.1}}  # a decision stored before E2
    assert "t_day +3.10/" in F.fmt_learning({**st, "challengers": [{**st["challengers"][0], "decision": old}]})[1]
