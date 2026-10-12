"""E3 operator UX (ROADMAP P6) on the real orchestrator (offline, fake Binance REST, synthetic bars): the engines' per-symbol
decision record, Telegram /why and /alphas, the daily report's learning section, the dashboard research page (owner-only,
server-rendered, no external resources) and that no secret is ever rendered."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import httpx
import pytest
from synth import synth_candles

from heartless.app import NO_ALPHA_TEXT, Heartless
from heartless.config import Settings
from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal
from heartless.execution.engine import DECISION_LOG_PER_SYMBOL, TradingEngine
from heartless.exchange.paper import PaperAccount
from heartless.learning import research as R
from heartless.learning import seal
from heartless.notify import formatter as F
from heartless.notify.telegram import COMMANDS, TelegramBot
from heartless.replay.invariants import _RE_DAILY_TODAY, _RE_DAILY_WEEK
from heartless.strategy import evidence
from heartless.util.timeutil import MS_DAY, MS_HOUR, now_ms
from heartless.web.app import WebServer, _evidence_cells, equity_svg
from tests.test_p4_app import N_HIST, SYMS, FakeRest, _boot, _rec

SECRETS = {"BINANCE_API_KEY": "AKtestKEY0123456789abcdef", "BINANCE_API_SECRET": "SECRETtest0123456789zyxw",
           "TELEGRAM_BOT_TOKEN": "424242:TGtestTOKENabcdefghijk", "ANTHROPIC_API_KEY": "sk-ant-testSECRET0123456789"}
INDEX = Path(__file__).resolve().parents[1] / "heartless" / "web" / "static" / "index.html"


def _make(tmp: Path | None = None, telegram: str = "", **settings):
    tmp = tmp or Path(tempfile.mkdtemp())
    start = (now_ms() - (N_HIST + 200) * 60_000) // 60_000 * 60_000
    candles = {s: synth_candles(N_HIST + 100, seed=i, start_ms=start, price=100 * (i + 1)) for i, s in enumerate(SYMS)}
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), HISTORY_DAYS=2, UNIVERSE_SIZE=3,
                 ALWAYS_INCLUDE=",".join(SYMS), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN=telegram, ARCHIVE_BACKFILL=False,
                 **settings)
    s.data_dir.mkdir(exist_ok=True)
    app = Heartless(s)
    app.rest = FakeRest({k: v.slice(0, N_HIST) for k, v in candles.items()})
    return app, candles


def _bot(app) -> tuple[TelegramBot, list[str]]:
    app.s.telegram_owner_id = 1
    bot = TelegramBot(app)
    sent: list[str] = []
    bot.send = lambda text, keyboard=None, chat_id=None: sent.append(text)
    return bot, sent


def _sig(sym: str, side: Side = Side.LONG, conf: float = 0.9, alpha: str = "htf_trend", price: float = 200.0) -> Signal:
    stop = price * (0.97 if side is Side.LONG else 1.03)
    tp = price * (1.06 if side is Side.LONG else 0.94)
    return Signal(alpha=alpha, symbol=sym, side=side, confidence=conf, reason=f"{alpha} 1주 추세 +2.1σ, 2일 조정",
                  stop=stop, take_profit=tp, tp1=None, entry_style=EntryStyle.MARKET, atr=price * 0.01, timeframe="1h",
                  tags={"ref_price": price, "z_week": 2.13, "z_pull": -1.2})


async def _feed(app, eng, sym: str, signals: list[Signal]) -> None:
    """One entry decision on `sym` with the alphas' output replaced by `signals` (the real ensemble, risk and _open)."""
    eng.ensemble.evaluate_signals = lambda view, ctx: eng.ensemble.last_signals.__setitem__(ctx.symbol, list(signals)) \
        or list(signals)
    await eng.on_bar(app.views[sym], app._context(sym, now_ms()))


async def _open(app, sym: str = "ETHUSDT", alpha: str = "htf_trend"):
    eng = app.engines["paper"]
    t = app.tickers[sym]
    sig = _sig(sym, alpha=alpha, price=t.ask)
    await eng._open(Decision(symbol=sym, side=Side.LONG, score=0.72, confidence=0.9, alphas=[alpha], primary=sig,
                             reason=sig.reason, regime=Regime.RANGE, expected_r=2.0), app.views[sym],
                    app._context(sym, now_ms()), await eng.equity())
    return eng.positions[sym]


def _scenario(app) -> dict:
    """A bot with history: htf_trend switched on with live trades, a challenger switching on trend_pullback, a
    kill-switch trip of intraday_levels, a research and a discovery cycle, an equity curve and a seal ledger."""
    now = now_ms()
    for i, r in enumerate([1.2, -1.0, 0.8, -0.4, 1.5, 0.3]):
        app.store.save_trade(_rec("paper", i, "htf_trend", r, now - (6 - i) * MS_DAY + MS_HOUR))
    app.store.save_params_version("vkill1", now - 3 * MS_DAY, "retired", "kill-switch", "kill switch: intraday_levels off",
                                  app.params.to_dict(), {"decisions": [{"kind": "kill_switch", "alpha": "intraday_levels",
                                                                        "engine": "paper", "rule": "CUSUM <= -8sd",
                                                                        "n": 11, "sum_r": -11.0}]})
    summary = {"seconds": 7833.0, "symbols": SYMS, "window": {"since": now - 120 * MS_DAY, "split": now - 30 * MS_DAY,
                                                              "until": now},
               "alphas": {"htf_trend": {"enabled": True, "n_folds": 3,
                                        "base": {"score": 0.1, "test": {"n": 40, "avg_r": 0.05}, "folds": []},
                                        "best": {"score": 0.4, "test": {"n": 38, "avg_r": 0.12}, "is_base": False}},
                          "trend_pullback": {"enabled": False, "n_folds": 3,
                                             "base": {"score": -0.2, "test": {"n": 50, "avg_r": -0.03}, "folds": []},
                                             "best": {"score": -0.2, "test": {"n": 50, "avg_r": -0.03}, "is_base": True}}}}
    app.research.last_summary, app.research.last_cycle_ts = summary, now - 2 * MS_HOUR
    app.store.save_research_run(now - 2 * MS_HOUR, "all", summary)
    app.store.log_event(now - 2 * MS_HOUR, "INFO", "research_done",
                        {"message": "• htf_trend: 파라미터 개선 (OOS n=38, avgR +0.12, 구간+ 2/3) → challenger-2", "summary": {},
                         "seconds": 7833.0})
    app.store.set("discovery.last", {"ts": now - 5 * MS_HOUR, "tested": 104871, "t_bar": 4.81, "passed": [],
                                     "seconds": 900.0, "rows": 0})
    app.store.set("discovery.last_cycle", now - 5 * MS_HOUR)
    app.store.log_event(now - 5 * MS_HOUR, "INFO", "research_done",
                        {"message": "알파 발굴: 104871개 규칙 검사, 다중검정·검증 구간을 통과한 규칙 없음 (과최적화 차단)"})
    for i in range(72):
        app.store.save_equity("paper", now - (72 - i) * MS_HOUR, 10_000 + 3 * i, 10_000 + 3 * i + (i % 5))
    seal.LEDGER.write_text(json.dumps({"utc": "2026-10-11T21:36:16Z", "period": "2025", "argv": [
        "heartless", "lab", "--alpha", "vol_breakout_dbox", "--unseal", "2025"], "commit": "6845f17f911aecd1"}) + "\n"
        + json.dumps({"utc": "2026-10-11T22:01:54Z", "period": "2025", "event": "spent", "argv": [],
                      "commit": "6845f17f911aecd1", "note": "The one-shot sealed-2025 confirmation test is complete"}) + "\n",
        encoding="utf-8")
    return summary


async def _challenger(app, alpha: str = "trend_pullback", r_list=(0.5, -0.2, 0.9)):
    p = app.params.clone(note=f"enable {alpha}")
    p.enabled[alpha] = True
    slot = await app.install_challenger(p, source=alpha)
    slot.started = now_ms() - 20 * MS_DAY
    for i, r in enumerate(r_list):
        app.store.save_trade(_rec(slot.name, i, alpha, r, slot.started + (i + 1) * MS_DAY))
    return slot


# --- engine decision record ----------------------------------------------------------------------------------------
async def test_the_engine_records_each_entry_decision_and_why_it_went_nowhere():
    app, candles = _make()
    await _boot(app, candles)
    eng, sym = app.engines["paper"], "ETHUSDT"
    await _feed(app, eng, sym, [_sig(sym, conf=0.2)])
    d = eng.decision_log[sym][-1]
    assert d["stage"] == "ensemble" and d["why"] == "threshold" and d["threshold"] == 0.5 and "decision" not in d
    assert d["signals"] == [{"alpha": "htf_trend", "side": "LONG", "confidence": 0.2,
                             "reason": "htf_trend 1주 추세 +2.1σ, 2일 조정"}]
    # recording never re-scores: the bandit weight is a Thompson draw, a second draw would shift the RNG stream
    rng = eng.ensemble.bandit.rng.getstate()
    eng._log_decision(app._context(sym, now_ms()), None, "ensemble", eng._ensemble_pass_reason(sym))
    assert eng.ensemble.bandit.rng.getstate() == rng
    await _feed(app, eng, sym, [_sig(sym, conf=0.9), _sig(sym, Side.SHORT, conf=0.85, alpha="vol_breakout_dbox")])
    assert eng.decision_log[sym][-1]["why"] == "conflict or threshold"  # Ensemble.decide: opposite side > 0.6x best
    eng.risk.state.symbol_cooldown[sym] = now_ms() + 600_000
    await _feed(app, eng, sym, [_sig(sym, conf=0.9, price=app.tickers[sym].ask)])
    d = eng.decision_log[sym][-1]
    assert d["stage"] == "risk" and d["why"] == "symbol cool-down after loss"
    assert d["decision"]["side"] == "LONG" and d["decision"]["alphas"] == ["htf_trend"] and d["decision"]["score"] >= 0.5
    eng.risk.state.symbol_cooldown.clear()
    app.tickers[sym].bid = app.tickers[sym].ask * 0.99  # a 100bp spread: _open skips it
    await _feed(app, eng, sym, [_sig(sym, conf=0.9, price=app.tickers[sym].ask)])
    assert eng.decision_log[sym][-1]["stage"] == "execution" and eng.decision_log[sym][-1]["why"] == "spread"
    app.tickers[sym].bid = app.tickers[sym].ask * 0.9999
    await _feed(app, eng, sym, [_sig(sym, conf=0.9, price=app.tickers[sym].ask)])
    d = eng.decision_log[sym][-1]
    assert d["stage"] == "opened" and d["why"] == "MARKET" and eng.positions[sym].alpha == "htf_trend"
    assert len(eng.decision_log[sym]) == DECISION_LOG_PER_SYMBOL  # bounded: the oldest of six records fell off
    assert [x["stage"] for x in eng.decision_log[sym]][0] == "ensemble"
    json.dumps(eng.decision_log[sym][-1])  # plain values (served as JSON)
    # a symbol with no signal records nothing; backtests (persist=False) keep no record at all
    await _feed(app, eng, "BTCUSDT", [])
    assert "BTCUSDT" not in eng.decision_log
    bt = TradingEngine("bt", PaperAccount(app.s, 10_000.0), app.params, app.s, app.symbols, persist=False)
    assert bt.decision_log is None
    await app.stop()


# --- /why ----------------------------------------------------------------------------------------------------------
async def test_why_without_a_position_explains_no_alpha_blockers_kill_switch_and_latest_decisions():
    app, candles = _make()
    await _boot(app, candles)
    bot, sent = _bot(app)
    await bot._handle_command("/why btc", 1)  # short form -> BTCUSDT
    text = sent[-1]
    assert "<b>BTCUSDT</b>" in text and "포지션 없음" in text
    assert "챔피언에 켜진 알파가 없어(0개)" in text and "최근 판단 기록 없음" in text
    # a cool-down, a kill-switch trip and recorded decisions of a challenger
    _scenario(app)
    slot = await _challenger(app)
    eng = app.engines[slot.name]
    await _feed(app, eng, "BTCUSDT", [_sig("BTCUSDT", conf=0.2, alpha="trend_pullback")])
    app.engines["paper"].risk.state.symbol_cooldown["BTCUSDT"] = now_ms() + 600_000
    await bot._handle_command("/why BTCUSDT", 1)
    text = sent[-1]
    assert "진입 차단: 손실 뒤 종목 쿨다운" in text
    assert "🛑 킬 스위치로 꺼진 알파 <code>intraday_levels</code>" in text and "vkill1" in text
    assert "<b>최근 판단</b>" in text and f"<code>{slot.name}</code>" in text and "trend_pullback LONG 0.20" in text
    assert "앙상블 점수 임계값 0.5 미달" in text
    # paused + unknown symbol + usage
    await app.pause()
    await bot._handle_command("/why ETHUSDT", 1)
    assert "진입 차단: 일시정지 중" in sent[-1]
    await bot._handle_command("/why NOPE123", 1)
    assert "거래소 심볼 목록에 없습니다" in sent[-1]
    await bot._handle_command("/why", 1)
    assert sent[-1].startswith("사용법: /why BTCUSDT")
    assert "why" in dict(COMMANDS)
    await app.stop()


async def test_why_with_an_open_position_shows_rationale_levels_risk_expected_r_and_evidence():
    app, candles = _make()
    app.params.enabled["htf_trend"] = True
    await _boot(app, candles)
    _scenario(app)
    pos = await _open(app)
    bot, sent = _bot(app)
    await bot._handle_command("/why ETHUSDT", 1)
    text = sent[-1]
    assert "<b>ETHUSDT</b> 왜 들고 있나" in text and "🟢 LONG" in text and "[OPEN]" in text
    assert "알파 <code>htf_trend</code>" in text and "근거: htf_trend 1주 추세 +2.1σ, 2일 조정" in text
    assert "태그: z_week=2.13, z_pull=-1.2" in text and "ref_price" not in text
    assert "진입 " in text and "손절 " in text and "목표 " in text
    assert f"리스크 {pos.risk_amount:,.2f} USDT" in text and "기대 +2.0R" in text
    assert re.search(r"지금 [+-]0\.0\dR \(미실현 [+-]\d+\.\d\d USDT\)", text)
    assert "증거: P1 OOS 탈락: 2022~2024 n=2064 -0.042R t -1.07" in text
    assert "운영 6건 평균 +0.40R" in text and "킬 스위치 CUSUM" in text
    w = app.why("ETHUSDT")
    assert w["position"]["expected_r"] == pytest.approx(2.0, abs=0.01) and w["position"]["stop_r"] == pytest.approx(-1.0)
    json.dumps(w, default=str)
    await app.stop()


def test_the_evidence_line_quotes_train_valid_and_the_sealed_2025_test(monkeypatch):
    data = json.loads(json.dumps(evidence.registry()))
    data["alphas"]["vol_breakout_dbox"]["sealed2025"] = {
        "window": "2025-01-01..2025-12-31", "n": 454, "avg_r": -0.0308, "t_day": -0.61, "stress_avg_r": -0.0373,
        "pooled_valid_2025": {"n": 765, "avg_r": 0.02293, "t_day": 0.3902}, "result": "fail"}
    monkeypatch.setattr(evidence, "registry", lambda: data)
    line = F.fmt_evidence(evidence.entry("vol_breakout_dbox"))
    assert line.startswith("2025 확인 시험 탈락: ")
    assert "TRAIN 2022~2024 n=1148 +0.128R t +3.21/3" in line and "VALID 2026 n=311 +0.101R t +0.82/1" in line
    assert "2025 n=454 -0.031R t -0.61 (VALID+2025 t +0.39/1.65)" in line
    assert F.fmt_evidence(evidence.entry("ts_trend_tsmom")).startswith("P2 VALID 탈락: ")
    assert F.fmt_evidence(None) == "OOS 미검증" and "재설계" in F.fmt_evidence(None, stale=True)


def test_a_train_gate_failure_is_not_called_a_valid_failure(monkeypatch):
    """A P2 family whose design failed the TRAIN gate never reached VALID: its entry carries a TRAIN block only
    (stage 'train'), and /why, /alphas and the research page must not present it as a VALID failure."""
    data = json.loads(json.dumps(evidence.registry()))
    data["alphas"]["funding_carry_xs"] = {
        "verdict": "failed", "stage": "train", "judgement": "p2", "rule": "p2-train-gate", "family": "funding_carry",
        "window": "2022-01-13..2024-12-31", "n": 563, "avg_r": 0.0375, "t_day": 1.36, "years_positive": 2, "years": 3,
        "stress_avg_r": 0.032, "train": {"window": "2022-01-13..2024-12-31", "n": 563, "avg_r": 0.0375, "t_day": 1.36,
                                         "years_positive": 2, "stress_avg_r": 0.032, "n_family": 7, "t_day_bar": 3.0},
        "source": "docs/RESEARCH_P2.md"}
    monkeypatch.setattr(evidence, "registry", lambda: data)
    e = evidence.entry("funding_carry_xs")
    assert F.evidence_stage(e) == "P2 TRAIN"
    assert F.fmt_evidence(e) == "P2 TRAIN 탈락: TRAIN 2022~2024 n=563 +0.037R t +1.36/3"
    assert _evidence_cells(e)[1:] == ["–", "–"]  # no VALID, no 2025 cell on the research page
    assert F.evidence_stage(evidence.entry("vol_breakout_dbox")) == "P2 VALID"
    assert F.evidence_stage(evidence.entry("htf_trend")) == "P1 OOS"


# --- /alphas -------------------------------------------------------------------------------------------------------
async def test_alphas_gives_one_line_per_alpha_with_challengers_evidence_live_stats_and_kill_switch():
    app, candles = _make()
    app.params.enabled["htf_trend"] = True
    await _boot(app, candles)
    _scenario(app)
    slot = await _challenger(app)
    bot, sent = _bot(app)
    await bot._handle_command("/alphas", 1)
    text = sent[-1]
    lines = {re.search(r"<code>(\w+)</code>", ln).group(1): ln for ln in text.split("\n")
             if ln[:1] in "✅🧪⛔" and "<code>" in ln}
    assert set(lines) == set(app.params.alphas)  # every registered alpha, one line each
    assert lines["htf_trend"].startswith("✅") and "P1 OOS 탈락" in lines["htf_trend"]
    assert "운영 6건 평균 +0.40R" in lines["htf_trend"] and "킬 스위치 CUSUM" in lines["htf_trend"]
    assert lines["trend_pullback"].startswith("🧪") and f"챌린저 {slot.name}(켬)" in lines["trend_pullback"]
    assert f"{slot.name}(거래)" in lines["htf_trend"]  # the challenger trades its base's alphas too
    assert "🛑 킬 스위치로 꺼짐 (vkill1, CUSUM &lt;= -8sd)" in lines["intraday_levels"]
    assert lines["discovered"].endswith("OOS 미검증 · 운영 0건")
    assert NO_ALPHA_TEXT not in text and "킬 스위치 켜짐" in text
    await app.stop()


async def test_alphas_explains_the_no_alpha_state():
    app, candles = _make()
    await _boot(app, candles)
    bot, sent = _bot(app)
    await bot._handle_command("/alphas", 1)
    text = sent[-1]
    assert NO_ALPHA_TEXT in text and "거래 시작 조건: 챌린저가 실시간 페이퍼에서 절대 기준을 넘으면" in text
    n = len([a for a in app.params.alphas if evidence.entry(a) is not None])  # every judged alpha failed (P1, P2)
    assert f"OOS 판정: {n}개 중 탈락 {n}, 통과 0" in text and "✅" not in text.split("\n", 2)[2]
    await app.stop()


# --- daily report --------------------------------------------------------------------------------------------------
async def test_the_daily_report_adds_alpha_performance_research_discovery_and_challenger_outlook():
    app, candles = _make()
    app.params.enabled["htf_trend"] = True
    await _boot(app, candles)
    _scenario(app)
    slot = await _challenger(app)
    await app.research.evaluate_challengers(force=True)
    bot, sent = _bot(app)
    await app.daily_report()
    text = next(m for m in sent if "일일 리포트" in m)
    assert _RE_DAILY_TODAY.search(text) and _RE_DAILY_WEEK.search(text)  # the replay checker's lines are intact
    assert "<b>알파별 성과</b> (paper, 7일 / 전체)" in text
    assert "✅ <code>htf_trend</code> 7일 6건" in text and "전체 6건 평균 +0.40R" in text
    assert "🔬 리서치 " in text and "131분" in text and "개선 후보 1(htf_trend)" in text and "파라미터 개선" in text
    assert "⛏ 발굴 " in text and "규칙 104,871개 검사" in text and "기준 t≥4.81, 통과 0" in text
    line = next(ln for ln in text.split("\n") if f"<code>{slot.name}</code>" in ln)
    # failed-alpha schedule (trend_pullback failed P1): 30/60/90, at day 20 the next one is 30 with its stricter bar
    bar = R.promotion_bar(app.s, "absolute", ["trend_pullback"], 30)
    assert f"절대 경로: 3/{bar['min_trades']}건" in line and "t_day " in line and f"/{bar['min_t_day']:g} →" in line
    assert "다음 판정점 30일" in line
    assert "판정점 30·60·90일" in line and "최근 판정 hold" in line and "OOS 탈락 알파 trend_pullback" in line
    assert "킬 스위치 <code>htf_trend</code>" in text
    assert "체결 품질(paper" in text or "체결 품질: 체결 기록 없음" in text
    await app.stop()


async def test_the_daily_report_explains_the_no_alpha_state_and_an_idle_loop():
    app, candles = _make()
    await _boot(app, candles)
    bot, sent = _bot(app)
    await app.daily_report()
    text = next(m for m in sent if "일일 리포트" in m)
    n = len([a for a in app.params.alphas if evidence.entry(a) is not None])
    assert NO_ALPHA_TEXT in text and "거래 시작 조건" in text and f"OOS 판정: {n}개 중 탈락 {n}" in text
    assert "알파별 성과: 거래 없음 (켜진 알파 0개라 챔피언은 거래하지 않음)" in text
    assert "🔬 리서치: 아직 사이클 없음" in text and "⛏ 발굴: 아직 실행 없음" in text and "챌린저: 없음" in text
    assert "체결 품질: 체결 기록 없음" in text
    await app.stop()


async def test_the_outlook_before_the_first_hourly_decision_is_the_judges_own_statistic(monkeypatch):
    """Before evaluate_challengers has judged a new challenger the outlook shows judge_challenger's retirement
    statistic, path and record, not a re-derivation that drifts from the statistic the bar is calibrated for."""
    app, candles = _make()
    await _boot(app, candles)
    slot = await _challenger(app, r_list=(0.8, 0.8, 0.8, -0.5, -0.5, 0.9, 0.9, -0.3, 0.7, 0.7))
    assert slot.name not in app.research.last_decisions
    ch = [t for t in app.store.load_trades(engine=slot.name, since=slot.started)
          if (t.get("entry_time") or 0) >= slot.started]
    d = R.judge_challenger(ch, [], app.s, slot.started, now_ms(), app.research.slot_new_alphas(slot))
    o = app.challenger_outlook()[0]
    assert (o["path"], o["stat"], o["current"]["n"]) == (d["retire_path"], d["retire_stat"], d["current"]["n"])
    real = R.judge_challenger
    monkeypatch.setattr(R, "judge_challenger", lambda *a, **k: {**real(*a, **k), "retire_stat": 9.99})
    assert app.challenger_outlook()[0]["stat"] == 9.99  # read from the judge, whatever its statistic is
    await app.stop()


async def test_a_failing_insight_never_drops_the_daily_report(monkeypatch):
    app, candles = _make()
    await _boot(app, candles)
    bot, sent = _bot(app)
    monkeypatch.setattr(app, "report_insight", lambda: 1 / 0)
    await app.daily_report()
    assert any(_RE_DAILY_TODAY.search(m) for m in sent)
    await app.stop()


# --- Telegram auth -------------------------------------------------------------------------------------------------
async def test_strangers_get_nothing_from_the_new_commands(monkeypatch):
    app, candles = _make()
    await _boot(app, candles)
    bot, sent = _bot(app)
    called = []
    monkeypatch.setattr(app, "why", lambda s: called.append(s) or {})
    monkeypatch.setattr(app, "alpha_overview", lambda: called.append("alphas") or {})
    for text in ("/why BTCUSDT", "/alphas"):
        await bot._handle_update({"message": {"from": {"id": 666}, "chat": {"id": 666, "type": "private"}, "text": text}})
    assert sent == [] and called == []
    await bot._handle_update({"message": {"from": {"id": 1}, "chat": {"id": 1, "type": "private"}, "text": "/why BTCUSDT"}})
    assert called == ["BTCUSDT"]
    await app.stop()


# --- web research page ---------------------------------------------------------------------------------------------
async def test_the_research_page_is_owner_only_and_renders_every_section():
    app, candles = _make()
    app.params.enabled["htf_trend"] = True
    await _boot(app, candles)
    _scenario(app)
    slot = await _challenger(app)
    web = WebServer(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.api), base_url="http://test") as c:
        r = await c.get("/research")
        assert r.status_code == 401 and 'action="/research"' in r.text and "연구" not in r.text
        assert (await c.get("/research", params={"token": "wrong"})).status_code == 401
        assert (await c.get("/api/learning")).status_code == 401
        assert (await c.get("/api/learning", headers={"x-token": "wrong"})).status_code == 401
        r = await c.get("/research", params={"token": app.web_token})
        assert r.status_code == 303 and r.headers["location"] == "/research"
        cookie = r.headers["set-cookie"]
        assert "hl_token=" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
        r = await c.get("/research", cookies={"hl_token": app.web_token})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert "default-src 'none'" in r.headers["content-security-policy"]
        page = r.text
        j = (await c.get("/api/learning", headers={"x-token": app.web_token})).json()
    assert "<h1>Heartless 연구·학습</h1>" in page
    for title in ("지금 상태", "자산 곡선 (paper, 최근 30일)", "알파 · 증거 기록 · 운영 성적",
                  "사전 등록 판정", "챌린저 판정점", "킬 스위치", "챔피언·챌린저 버전 이력", "리서치 사이클", "알파 발굴",
                  "봉인 원장"):
        assert f"<h2>{title}</h2>" in page, title
    assert '<svg class="eq"' in page and '<path class="line" d="M' in page and "자산 곡선: " in page
    assert "<code>htf_trend</code>" in page and "P1 OOS" in page and "<code>vol_breakout_dbox</code>" in page
    assert f"<code>{slot.name}</code>" in page and '<span class="chip next">30일</span>' in page
    assert "vkill1" in page and "intraday_levels" in page and "꺼짐 유지" in page
    assert "규칙 104,871개 검사" in page and "131분" in page
    assert "lab --alpha vol_breakout_dbox --unseal 2025" in page and "spent" in page and "6845f17f91" in page
    # nothing runs and nothing loads from elsewhere
    assert "<script" not in page.lower() and not re.search(r"(src|href)=\"(https?:)?//", page)
    assert "data/secrets.json" not in page and ".env" not in page
    assert set(j) >= {"overview", "judgements", "versions", "challengers", "kill_switch", "research", "discovery", "seal",
                      "equity"}
    assert j["seal"]["ledger"][-1]["event"] == "spent" and len(j["equity"]) >= 2
    assert [c["name"] for c in j["challengers"]] == [slot.name] and j["challengers"][0]["next_checkpoint"] == 30
    await app.stop()


async def test_the_research_page_escapes_stored_text_and_survives_a_damaged_ledger():
    app, candles = _make()
    await _boot(app, candles)
    evil = '<img src=x onerror="fetch(\'/api/kill\',{method:\'POST\'})">'
    app.store.save_params_version("v<b>x", now_ms(), "retired", evil, evil, app.params.to_dict(), {})
    app.store.log_event(now_ms(), "INFO", "research_done", {"message": evil, "summary": {}})
    seal.LEDGER.write_text("{not json\n", encoding="utf-8")
    web = WebServer(app)
    page = web.research_html()
    assert evil not in page and "&lt;img src=x" in page and "v&lt;b&gt;x" in page
    assert "원장을 읽지 못했습니다" in page and "<h2>봉인 원장</h2>" in page
    await app.stop()


def test_the_equity_svg_is_inline_and_handles_short_curves():
    assert "자산 기록이 아직 부족합니다" in equity_svg([], "UTC")
    svg = equity_svg([(1_700_000_000_000, 10_000.0), (1_700_003_600_000, 10_050.0), (1_700_007_200_000, 9_990.0)], "UTC")
    assert svg.startswith('<svg class="eq" viewBox="0 0 760 220"') and svg.endswith("</svg>")
    assert "10,000.00 → 9,990.00 USDT (-0.10%)" in svg and svg.count("<text") >= 8 and "http" not in svg
    flat = equity_svg([(1, 5.0), (2, 5.0)], "UTC")  # no range: padded, no division by zero
    assert "<path" in flat


def test_the_dashboard_links_to_the_research_page():
    html = INDEX.read_text(encoding="utf-8")
    assert 'href="/research"' in html


# --- secrets -------------------------------------------------------------------------------------------------------
async def test_no_secret_is_ever_rendered():
    app, candles = _make(telegram=SECRETS["TELEGRAM_BOT_TOKEN"], **{k: v for k, v in SECRETS.items()
                                                                    if k != "TELEGRAM_BOT_TOKEN"})
    app.params.enabled["htf_trend"] = True
    await _boot(app, candles)
    _scenario(app)
    await _challenger(app)
    await _open(app)
    bot, sent = _bot(app)
    for cmd in ("/why ETHUSDT", "/why BTCUSDT", "/alphas", "/challengers", "/status"):
        await bot._handle_command(cmd, 1)
    await app.daily_report()
    web = WebServer(app)
    rendered = sent + [web.research_html(), json.dumps(web.learning_data(), default=str)]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.api), base_url="http://test") as c:
        rendered.append((await c.get("/research", cookies={"hl_token": app.web_token})).text)
        rendered.append((await c.get("/api/learning", cookies={"hl_token": app.web_token})).text)
        rendered.append((await c.get("/research")).text)  # the login page
    secrets_ = list(SECRETS.values()) + [app.web_token, SECRETS["TELEGRAM_BOT_TOKEN"].split(":")[1]]
    assert len(rendered) >= 9
    for text in rendered:
        for s in secrets_:
            assert s not in text
    await app.stop()
