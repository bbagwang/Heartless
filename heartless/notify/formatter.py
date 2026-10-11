"""Telegram (HTML) message formatting."""
from __future__ import annotations

import html

from heartless.core.models import Position, Side
from heartless.execution.stats import sparkline
from heartless.util.mathutil import fmt_pct, fmt_price, fmt_usd
from heartless.util.timeutil import fmt_ts, human_duration, now_ms


def esc(s: object) -> str:
    return html.escape(str(s), quote=False)


def side_emoji(side: Side | str) -> str:
    s = side.value if isinstance(side, Side) else side
    return "🟢 LONG" if s == "LONG" else "🔴 SHORT"


def mode_tag(engine: str, paper: bool) -> str:
    if engine == "live":
        return "🔴 LIVE"
    return f"🧪 {engine.upper()}" if paper else engine


def fmt_position_opened(p: Position, tick: float | None, engine_label: str) -> str:
    rr = abs(p.take_profit - p.entry_price) / p.r_unit if (p.take_profit and p.r_unit) else 0.0
    lines = [
        f"<b>{engine_label} · 포지션 진입</b> {side_emoji(p.side)} <b>{esc(p.symbol)}</b>",
        f"알파: <code>{esc(p.alpha)}</code>" + (f" (+{esc(', '.join(p.alphas[1:]))})" if len(p.alphas) > 1 else ""),
        f"근거: {esc(p.reason)}",
        f"레짐: {esc(p.regime)} · 신뢰도 {p.confidence:.2f}",
        f"진입가 {fmt_price(p.entry_price, tick)} · 수량 {p.qty:g} · 명목 {p.notional:,.0f} USDT (x{p.leverage})",
        f"손절 {fmt_price(p.stop, tick)} ({fmt_pct((p.stop / p.entry_price - 1) * 100)}) · 리스크 {p.risk_amount:,.2f} USDT",
    ]
    if p.tp1:
        lines.append(f"1차 익절 {fmt_price(p.tp1, tick)} (50%) → 이후 본절 이동")
    if p.take_profit:
        lines.append(f"목표가 {fmt_price(p.take_profit, tick)} ({fmt_pct((p.take_profit / p.entry_price - 1) * 100)}, {rr:.1f}R) · 기대수익 ≈ {p.expected_profit:,.2f} USDT")
    elif p.trail_atr_mult:
        lines.append(f"목표: 트레일링 스탑 ({p.trail_atr_mult:g} ATR) · 기대수익 ≈ {p.expected_profit:,.2f} USDT")
    if p.max_hold_bars:
        lines.append(f"시간 제한 {human_duration(p.max_hold_bars * 60_000)}")
    return "\n".join(lines)


def fmt_position_closed(p: Position, trade, tick: float | None, engine_label: str, today_pnl: float, equity: float) -> str:
    pnl = trade.pnl
    icon = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
    dur = human_duration((trade.exit_time or now_ms()) - trade.entry_time)
    lines = [
        f"<b>{engine_label} · 포지션 종료</b> {icon} {side_emoji(p.side)} <b>{esc(p.symbol)}</b>",
        f"사유: {esc(trade.exit_reason)} · 알파 <code>{esc(p.alpha)}</code>",
        f"진입 {fmt_price(trade.entry_price, tick)} → 청산 {fmt_price(trade.exit_price, tick)} · 보유 {dur}",
        f"손익 <b>{fmt_usd(pnl)}</b> ({trade.r_multiple:+.2f}R, 수수료 {trade.fees:.2f}, 펀딩 {trade.funding:+.2f})",
        f"MFE {trade.max_fav_r:+.2f}R / MAE {-abs(trade.max_adv_r):.2f}R",
        f"오늘 실현손익 {fmt_usd(today_pnl)} · 자산 {equity:,.2f} USDT",
    ]
    return "\n".join(lines)


def fmt_partial(p: Position, price: float, qty: float, pnl: float, tick: float | None, engine_label: str) -> str:
    return (f"<b>{engine_label} · 1차 익절</b> {side_emoji(p.side)} <b>{esc(p.symbol)}</b>\n"
            f"{fmt_price(price, tick)} 에서 {qty:g} 청산, {fmt_usd(pnl)} 확보 · 잔량 {p.qty:g}\n"
            f"손절을 본절({fmt_price(p.stop, tick)})로 이동, 트레일링 시작")


def fmt_status(st: dict, tz: str) -> str:
    prim = st["engines"].get(st["primary"], {}) if st.get("primary") else {}
    today = st.get("today", {})
    h = st.get("health", {})
    lines = [
        f"<b>Heartless 상태</b> — {'🔴 LIVE' if st['mode'] == 'live' else '🟢 PAPER'}" + (" ⏸ 일시정지" if st.get("paused") else ""),
        f"자산 {prim.get('equity', 0):,.2f} USDT (시작 {prim.get('start_equity', 0):,.2f})",
        f"오늘: {today.get('n', 0)}건, 승률 {today.get('win_rate', 0):.0f}%, 실현 {fmt_usd(today.get('net', 0.0))}",
        f"포지션 {len(prim.get('positions', []))}개 · 챔피언 <code>{esc(st['params_version'])}</code> · 챌린저 {len(st.get('challengers', []))}",
        f"유니버스 {len(st.get('universe', []))}종목 · 스트림 {'OK' if h.get('market_stream') else '끊김'}"
        + (f" · 지연 {len(h.get('stale_symbols', []))}" if h.get("stale_symbols") else ""),
    ]
    if prim.get("halted"):
        lines.append(f"🚫 {esc(prim.get('halt_reason', ''))}")
    lines += fmt_learning(st)
    xq = fmt_exec_quality(st)
    if xq:
        lines.append(xq)
    if st.get("graduation"):
        g = st["graduation"]
        lines.append(f"졸업 조건: {g['n']}/{g['need']['trades']}건, PF {g['profit_factor']:.2f}/{g['need']['profit_factor']}, "
                     f"MDD {g['max_dd_pct']:.1f}%/{g['need']['max_dd_pct']}% {'✅' if g['ok'] else '⏳'}")
    r = st.get("research", {})
    if r.get("last_cycle"):
        lines.append(f"마지막 리서치 {fmt_ts(r['last_cycle'], tz, '%m-%d %H:%M')}" + (" (진행 중)" if r.get("running") else ""))
    return "\n".join(lines)


def fmt_positions(st: dict, ticks: dict[str, float]) -> str:
    prim = st["engines"].get(st["primary"], {}) if st.get("primary") else {}
    rows = prim.get("positions", [])
    if not rows:
        return "열린 포지션이 없습니다"
    out = [f"<b>포지션 ({len(rows)})</b>"]
    for r in rows:
        tick = ticks.get(r["symbol"])
        out.append(f"{side_emoji(r['side'])} <b>{esc(r['symbol'])}</b> [{r['status']}] <code>{esc(r['alpha'])}</code>\n"
                   f"  진입 {fmt_price(r['entry'], tick)} → 현재 {fmt_price(r['mark'], tick)} ({fmt_pct(r['pnl_pct'])}, {r['r']:+.2f}R) "
                   f"미실현 {fmt_usd(r['unrealized'])}\n"
                   f"  손절 {fmt_price(r['stop'], tick)}" + (f" · 목표 {fmt_price(r['tp'], tick)}" if r.get("tp") else "")
                   + (" · 1차익절 완료" if r.get("tp1_done") else "") + f" · {human_duration(r['bars_held'] * 60_000)}")
    return "\n".join(out)


def fmt_pnl(rep: dict, tz: str) -> str:
    st = rep.get("stats", {})
    labels = {"today": "오늘", "week": "최근 7일", "month": "최근 30일", "all": "전체"}
    lines = [f"<b>{labels.get(rep.get('period'), rep.get('period'))} 손익</b> ({esc(rep.get('engine', ''))})",
             f"실현 <b>{fmt_usd(st.get('net', 0.0))}</b> · 미실현 {fmt_usd(rep.get('unrealized', 0.0))} · 자산 {rep.get('equity_now', 0):,.2f}",
             f"{st.get('n', 0)}건 · 승률 {st.get('win_rate', 0):.1f}% · PF {min(st.get('profit_factor', 0), 99):.2f} · 평균 {st.get('avg_r', 0):+.2f}R",
             f"수수료 {st.get('fees', 0):.2f} · 펀딩 {st.get('funding', 0):+.2f} · 최대낙폭 {st.get('max_dd', 0):.2f} USDT",
             f"최고 {fmt_usd(st.get('best', 0))} · 최악 {fmt_usd(st.get('worst', 0))} · 평균 보유 {st.get('avg_hold_min', 0):.0f}분"]
    eq = rep.get("equity") or []
    if len(eq) >= 2:
        lines.append(f"자산곡선 <code>{sparkline([e[2] for e in eq])}</code>")
    ba = rep.get("by_alpha") or {}
    if ba:
        lines.append("알파별:")
        for a, s in sorted(ba.items(), key=lambda kv: -kv[1]["net"]):
            lines.append(f"  <code>{esc(a)}</code> {s['n']}건 {fmt_usd(s['net'])} ({s['avg_r']:+.2f}R, 승률 {s['win_rate']:.0f}%)")
    return "\n".join(lines)


def fmt_alphas(rows: list[dict], params, kill_switch: list[dict] | None = None) -> str:
    from heartless.strategy import evidence

    ks = {(k.get("alpha"), k.get("engine")): k for k in (kill_switch or [])}
    lines = ["<b>알파 신뢰도 (Thompson sampling)</b>"]
    for r in rows:
        en = params.enabled.get(r["alpha"], True)
        k = ks.get((r["alpha"], "live")) or ks.get((r["alpha"], "paper"))
        guard = (f" · 킬 스위치 CUSUM {k['cusum']:+.2f}/-{k['h']:g}σ" if k else "")
        lines.append(f"{'✅' if en else '⛔'} <code>{esc(r['alpha'])}</code> trust {r['trust']:.2f} · n={r['n']} · avgR {r['avg_r']:+.2f}\n"
                     f"   {esc(evidence.short(r['alpha']))}{guard}\n"
                     f"   추세↑ {r['trend_up']['mean']:.2f}({r['trend_up']['n']}) 추세↓ {r['trend_down']['mean']:.2f}({r['trend_down']['n']}) "
                     f"횡보 {r['range']['mean']:.2f}({r['range']['n']}) 변동 {r['volatile']['mean']:.2f}({r['volatile']['n']})")
    return "\n".join(lines)


def fmt_research(summary: dict, tz: str) -> str:
    if not summary:
        return "아직 리서치 사이클이 실행되지 않았습니다"
    w = summary.get("window", {})
    lines = [f"<b>마지막 리서치</b> ({summary.get('seconds', 0):.0f}s, {len(summary.get('symbols', []))}종목)",
             f"학습 {fmt_ts(w.get('since'), tz, '%m-%d')}~{fmt_ts(w.get('split'), tz, '%m-%d')} / 검증 ~{fmt_ts(w.get('until'), tz, '%m-%d')}"]
    for a, r in summary.get("alphas", {}).items():
        b, t = r["base"], r["best"]
        flag = "유지" if t.get("is_base") else f"개선 +{t['score'] - (b['score'] if b['score'] > -1e8 else -1):.2f}"
        lines.append(f"<code>{esc(a)}</code>: 현재 OOS n={b['test'].get('n', 0)} avgR {b['test'].get('avg_r', 0):+.2f} | "
                     f"최적 n={t['test'].get('n', 0)} avgR {t['test'].get('avg_r', 0):+.2f} → {flag}")
    return "\n".join(lines)


def fmt_daily(today: dict, week: dict, st: dict, tz: str, insight: dict | None = None) -> str:
    """The daily report. `insight` (Heartless.report_insight) adds per-alpha performance, the research and discovery
    cycles and the challengers' outlook; the PnL lines keep their layout (the replay checker parses them)."""
    t = today.get("stats", {})
    w = week.get("stats", {})
    prim = st["engines"].get(st["primary"], {}) if st.get("primary") else {}
    lines = [f"📊 <b>일일 리포트</b> {fmt_ts(now_ms(), tz, '%Y-%m-%d %H:%M')} — {'🔴 LIVE' if st['mode'] == 'live' else '🟢 PAPER'}",
             f"자산 {prim.get('equity', 0):,.2f} USDT",
             f"오늘 {t.get('n', 0)}건 · 실현 <b>{fmt_usd(t.get('net', 0.0))}</b> · 승률 {t.get('win_rate', 0):.0f}% · PF {min(t.get('profit_factor', 0), 99):.2f}",
             f"7일 {w.get('n', 0)}건 · 실현 {fmt_usd(w.get('net', 0.0))} · 승률 {w.get('win_rate', 0):.0f}% · 평균 {w.get('avg_r', 0):+.2f}R · MDD {w.get('max_dd', 0):.1f}"]
    eq = week.get("equity") or []
    if len(eq) >= 2:
        lines.append(f"7일 자산곡선 <code>{sparkline([e[2] for e in eq])}</code>")
    ba = today.get("by_alpha") or {}
    if ba:
        lines.append("오늘 알파별: " + ", ".join(f"{esc(a)} {fmt_usd(s['net'])}" for a, s in ba.items()))
    lines.append(f"열린 포지션 {len(prim.get('positions', []))} · 챔피언 <code>{esc(st['params_version'])}</code> · 챌린저 {len(st.get('challengers', []))}")
    if insight is None:
        lines += fmt_learning(st)
    else:
        ls = st.get("learning") or {}
        lines += fmt_learning(st, challengers=False, kill_switch=False)
        if ls.get("no_alpha") and ls.get("evidence"):
            lines.append(esc(ls["evidence"]))
        lines += fmt_insight(insight, st, tz)
        lines += fmt_learning(st, headline=False, challengers=False)
    xq = fmt_exec_quality(st, suggest=True)
    if xq:
        lines.append(xq)
    elif insight is not None:
        lines.append("체결 품질: 체결 기록 없음 (체결이 생기면 슬리피지·수수료를 모델값과 비교)")
    if st.get("graduation") and st["graduation"].get("ok"):
        lines.append("🎓 페이퍼 성과가 졸업 기준을 충족했습니다. /golive 로 실거래 전환 가능")
    return "\n".join(lines)


def fmt_learning(st: dict, headline: bool = True, challengers: bool = True, kill_switch: bool = True) -> list[str]:
    """What trades and what the learning loop is doing (truthful when no alpha is enabled), the challengers' standing
    against their promotion bar and the kill-switch state of the champion's alphas (each part can be left out)."""
    ls = st.get("learning") or {}
    if not ls:
        return []
    out = []
    if headline:
        out.append(esc(ls.get("headline", "")))
        if ls.get("no_alpha") and ls.get("criteria"):
            out.append("거래 시작 조건: " + esc(ls["criteria"]))
    for c in (st.get("challengers") or []) if challengers else ():
        d = c.get("decision")
        if not d:
            out.append(f"  <code>{esc(c['name'])}</code> [{esc(c.get('source', ''))}] 판정 대기")
            continue
        ch, bar = d.get("challenger") or {}, d.get("bar") or {}
        path = "상대" if d.get("path") == "relative" else "절대"
        extra = (f", 차이 t {d['paired']['t']:+.2f}/{bar.get('min_paired_t', 0):g}" if d.get("paired")
                 else f", t_day {ch.get('t_day', 0):+.2f}/{bar.get('min_t_day', 0):g}")
        cp, nxt = d.get("checkpoint"), d.get("next_checkpoint")
        when = (f"{cp}일 판정점" if cp is not None else "첫 판정점 전") + (f", 다음 {nxt}일" if nxt else "")
        out.append(f"  <code>{esc(c['name'])}</code> [{esc(c.get('source', ''))}] {path} 경로 ({when}): "
                   f"{ch.get('n', 0)}/{bar.get('min_trades', 0)}건, {ch.get('days', 0)}/{bar.get('min_days', 0)}일, "
                   f"평균 {ch.get('avg_r', 0):+.2f}R{extra} → {esc(d.get('verdict', ''))}")
    for k in (st.get("kill_switch") or []) if kill_switch else ():
        if k.get("engine") != "paper" and k.get("engine") != "live":
            continue
        out.append(f"  킬 스위치 <code>{esc(k['alpha'])}</code> ({esc(k['engine'])}): {k.get('n', 0)}건, CUSUM "
                   f"{k.get('cusum', 0):+.2f}/-{k.get('h', 0):g}σ (σ {k.get('scale', 1.0):.2f}R), 최근 {k.get('window_n', 0)}건 합 "
                   f"{k.get('window_sum', 0):+.2f}/{k.get('window_r', 0):g}R")
    return out


def fmt_exec_quality(st: dict, suggest: bool = False) -> str:
    """Rolling execution quality of the primary engine (realized slippage vs the decision reference, fees vs the fee
    model) and, with `suggest`, the BACKTEST_SLIPPAGE_BPS suggestion for any engine whose taker fills slip more than
    twice the model."""
    xq = st.get("exec_quality") or {}
    prim = st.get("primary")
    lines = []
    q = xq.get(prim) if prim else None
    if q:
        lines.append(f"체결 품질({esc(prim)}, 최근 {q['fills']}체결, 테이커 {q['taker_fills']}): 슬리피지 평균 "
                     f"{q['slip_bps']:+.1f}bp (p90 {q['slip_p90_bps']:+.1f}bp, 모델 {q['model_slip_bps']:g}bp) · 수수료 "
                     f"{q['fee_bps']:.1f}bp (모델 {q['model_fee_bps']:.1f}bp)")
    if suggest:
        for name, q in sorted(xq.items()):
            if q.get("suggest"):
                lines.append(f"⚠️ {esc(name)} 테이커 체결 {q['taker_fills']}건의 평균 슬리피지 {q['slip_bps']:.1f}bp가 "
                             f"BACKTEST_SLIPPAGE_BPS({q['model_slip_bps']:g}bp)의 2배를 넘습니다. 백테스트 슬리피지를 "
                             f"{q['slip_bps']:.1f}bp 정도로 올리는 것을 검토하세요 (자동 변경하지 않음)")
    return "\n".join(lines)


def fmt_discovery(last: dict | None, tz: str) -> str:
    if not last:
        return "<b>알파 발굴</b>: 아직 실행되지 않음 (/discover)"
    lines = [f"<b>알파 발굴</b> {fmt_ts(last.get('ts'), tz, '%m-%d %H:%M')} · 검사 {last.get('tested', 0)}개 규칙 · "
             f"다중검정 기준 t≥{last.get('t_bar', 0):.2f}"]
    passed = last.get("passed") or []
    if not passed:
        lines.append("검증 구간까지 통과한 규칙 없음 (과최적화 차단)")
    for r in passed[:5]:
        va = r.get("stats", {}).get("valid", {})
        conds = ", ".join(f"{c[0]} {c[1]} {c[2]:g}" for c in r.get("conds", []))
        lines.append(f"• {esc(r.get('id'))} {esc(r.get('side'))} [{esc(r.get('tf'))}] {esc(conds)} — 검증 n={va.get('n', 0)}, "
                     f"{va.get('avg_r', 0):+.2f}R")
    return "\n".join(lines)


# --- operator insight (ROADMAP P6): /why, /alphas, the daily report's learning section -----------------------------------
# bars of the pre-registered P2 protocol the evidence lines quote (docs/ROADMAP.md section 6): VALID t_day >= 1.0, and
# the sealed-2025 confirmation test's pooled VALID + 2025 t_day >= 1.65 (one-sided 5%)
P2_VALID_T_BAR = 1.0
SEALED2025_POOLED_T_BAR = 1.65
VERDICT_KO = {"failed": "탈락", "passed": "통과", "untested": "미검증"}
ROLE_KO = {"on": "켬", "off": "끔", "tuned": "조정", "trades": "거래"}
# why an entry decision went nowhere (TradingEngine.decision_log stage/why codes, RiskManager.can_open reasons)
RISK_KO = {"paused": "일시정지 중", "halted": "손실 한도로 신규 진입 중단", "market shock cool-off": "BTC 급변 뒤 진입 중단",
           "symbol cool-down after loss": "손실 뒤 종목 쿨다운", "already in position": "이미 포지션 보유",
           "max positions": "동시 포지션 한도", "max same-direction positions": "같은 방향 포지션 한도",
           "gross leverage cap": "총 레버리지 한도", "entries disabled": "모드 전환 중 진입 꺼짐"}
EXEC_KO = {"spread": "스프레드 과다(12bp 초과)", "meta veto": "메타 모델 거부", "size": "주문 수량 0(최소 수량 미달)",
           "stop": "손절가가 이미 현재가를 넘음", "entry not placed": "주문 미접수(거래소 거절 등)"}


def _span(window: str | None) -> str:
    """'2022-01-13..2024-12-31' -> '2022~2024', '2026-01-13..2026-09-30' -> '2026'."""
    a, _, b = str(window or "").partition("..")
    ya, yb = a[:4], b[:4]
    return ya if ya == yb or not yb else f"{ya}~{yb}"


def _num(v, fmt: str = "+.3f") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "-"


def evidence_stage(e: dict | None) -> str:
    """The last pre-registered test an alpha's evidence comes from."""
    if not e:
        return "미검증"
    if e.get("sealed2025"):
        return "2025 확인 시험"
    if e.get("judgement") == "p2" or e.get("train"):
        return "P2 VALID"
    return "P1 OOS"


def fmt_evidence(e: dict | None, stale: bool = False) -> str:
    """One-line Korean summary of an alpha's evidence-registry entry: stage, verdict and the numbers behind it (P1: the
    OOS window; P2: TRAIN against its multiple-testing bar, VALID, and the sealed-2025 test when it took one)."""
    if not e:
        return "OOS 미검증 (재설계로 이전 판정 무효)" if stale else "OOS 미검증"
    head = f"{evidence_stage(e)} {VERDICT_KO.get(e.get('verdict'), e.get('verdict'))}"
    tr, va, s25 = e.get("train") or {}, e.get("valid") or {}, e.get("sealed2025") or {}
    if tr or va:
        parts = []
        if tr:
            parts.append(f"TRAIN {_span(tr.get('window'))} n={tr.get('n', 0)} {_num(tr.get('avg_r'))}R "
                         f"t {_num(tr.get('t_day'), '+.2f')}/{_num(tr.get('t_day_bar'), 'g')}")
        if va:
            parts.append(f"VALID {_span(va.get('window'))} n={va.get('n', 0)} {_num(va.get('avg_r'))}R "
                         f"t {_num(va.get('t_day'), '+.2f')}/{P2_VALID_T_BAR:g}")
        if s25:
            pooled = s25.get("pooled_valid_2025") or {}
            parts.append(f"2025 n={s25.get('n', 0)} {_num(s25.get('avg_r'))}R t {_num(s25.get('t_day'), '+.2f')} "
                         f"(VALID+2025 t {_num(pooled.get('t_day'), '+.2f')}/{SEALED2025_POOLED_T_BAR:g})")
        return f"{head}: " + " · ".join(parts)
    return (f"{head}: {_span(e.get('oos_window'))} n={e.get('n', 0)} {_num(e.get('avg_r'))}R "
            f"t {_num(e.get('t_day'), '+.2f')}, 양수 연도 {e.get('years_positive', 0)}/{e.get('years', 0)}, "
            f"스트레스 {_num(e.get('stress_avg_r'))}R")


def _live(st: dict | None) -> str:
    st = st or {}
    if not st.get("n"):
        return "운영 0건"
    return f"운영 {st['n']}건 평균 {_num(st.get('avg_r'), '+.2f')}R t {_num(st.get('t_day'), '+.2f')}"


def _ks(k: dict | None) -> str:
    if not k:
        return ""
    return (f"킬 스위치 CUSUM {_num(k.get('cusum'), '+.2f')}/-{_num(k.get('h'), 'g')}σ, 최근 {k.get('window_n', 0)}건 합 "
            f"{_num(k.get('window_sum'), '+.1f')}/{_num(k.get('window_r'), 'g')}R" + (" 🛑" if k.get("tripped") else ""))


def _tags(tags: dict) -> str:
    out = []
    for k, v in list(tags.items())[:8]:
        if isinstance(v, float):
            v = f"{v:.4g}"
        elif isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v[:4])
        out.append(f"{k}={v}")
    return ", ".join(out)


def decision_reason(d: dict) -> str:
    """Korean outcome of one TradingEngine.decision_log record."""
    stage, why = d.get("stage"), str(d.get("why") or "")
    if stage == "opened":
        return "진입 주문" + (" (지정가)" if why == "LIMIT" else " (시장가)" if why == "MARKET" else "")
    if stage == "ensemble":
        thr = _num(d.get("threshold"), "g")
        return (f"반대 방향 신호와 충돌 또는 앙상블 점수 임계값 {thr} 미달" if why.startswith("conflict")
                else f"앙상블 점수 임계값 {thr} 미달")
    if stage == "risk":
        code, _, detail = why.partition(":")
        return "거절: " + RISK_KO.get(code, why) + (f" ({detail})" if detail else "")
    if stage == "execution":
        return "거절: " + EXEC_KO.get(why, why)
    return why


def fmt_decision(d: dict, tz: str) -> str:
    sigs = ", ".join(f"{s.get('alpha')} {s.get('side')} {_num(s.get('confidence'), '.2f')}" for s in d.get("signals") or [])
    dec = d.get("decision") or {}
    score = f" 점수 {_num(dec.get('score'), '.2f')}/{_num(d.get('threshold'), 'g')}" if dec else ""
    return (f"• {fmt_ts(d.get('ts'), tz, '%m-%d %H:%M')} <code>{esc(d.get('engine'))}</code> [{esc(d.get('regime'))}] "
            f"신호 {esc(sigs or '-')}{score} → {esc(decision_reason(d))}")


def fmt_why(w: dict, tz: str, tick: float | None = None) -> str:
    """/why <SYMBOL>: the open position's full rationale, or the latest decisions and what blocks entries."""
    sym, eng = w.get("symbol", "?"), w.get("engine") or "-"
    label = "🔴 LIVE" if eng == "live" else f"🧪 {str(eng).upper()}"
    p = w.get("position")
    lines: list[str] = []
    if p:
        q = f"보유 {human_duration(p.get('bars_held', 0) * 60_000)}"
        lines += [f"🔎 <b>{esc(sym)}</b> 왜 들고 있나 — {label} · {side_emoji(p['side'])} [{esc(p['status'])}]",
                  f"알파 <code>{esc(p['alpha'])}</code>" + (f" (+{esc(', '.join(p['alphas'][1:]))})" if len(p.get('alphas') or []) > 1 else "")
                  + f" · {esc(p.get('timeframe'))} · 레짐 {esc(p.get('regime'))} · 앙상블 점수 {_num(p.get('confidence'), '.2f')}",
                  f"근거: {esc(p.get('reason'))}"]
        if p.get("tags"):
            lines.append(f"태그: {esc(_tags(p['tags']))}")
        lines.append(f"진입 {fmt_price(p['entry'], tick)} → 현재 {fmt_price(p['mark'], tick)} · {q}")
        stop = f"손절 {fmt_price(p['stop'], tick)}"
        if p.get("initial_stop") and abs(p["initial_stop"] - p["stop"]) > 1e-12:
            stop += f" (최초 {fmt_price(p['initial_stop'], tick)}, 지금 손절 시 {_num(p.get('stop_r'), '+.2f')}R)"
        tgt = []
        if p.get("tp1"):
            tgt.append(f"1차 익절 {fmt_price(p['tp1'], tick)}" + (" ✓" if p.get("tp1_done") else ""))
        if p.get("tp"):
            tgt.append(f"목표 {fmt_price(p['tp'], tick)}")
        if p.get("trail_atr_mult"):
            tgt.append(f"트레일링 {p['trail_atr_mult']:g} ATR")
        if p.get("max_hold_bars"):
            tgt.append(f"시간 제한 {human_duration(p['max_hold_bars'] * 60_000)}")
        lines.append(" · ".join([stop] + tgt))
        lines.append(f"리스크 {p.get('risk', 0):,.2f} USDT (1R = {fmt_price(p.get('r_unit'), tick)}) · 지금 "
                     f"{_num(p.get('r_now'), '+.2f')}R (미실현 {fmt_usd(p.get('unrealized', 0.0))}"
                     + (f", 실현 {fmt_usd(p['realized'])}" if p.get("realized") else "") + ")"
                     + (f" · 기대 {_num(p['expected_r'], '+.1f')}R" if p.get("expected_r") is not None else ""))
        lines.append("증거: " + esc(fmt_evidence(p.get("evidence"), p.get("evidence_stale"))))
        live = p.get("live") or {}
        guard = _ks(p.get("kill_switch"))
        lines.append(f"{esc(_live(live))} ({esc(live.get('engine', eng))})" + (f" · {guard}" if guard else ""))
    else:
        lines.append(f"🔎 <b>{esc(sym)}</b> — {label} 포지션 없음")
        if not w.get("known"):
            lines.append("거래소 심볼 목록에 없습니다 (예: /why BTCUSDT)")
        elif not w.get("in_universe"):
            lines.append("지금 거래 유니버스 밖이라 신호를 평가하지 않습니다")
        if not w.get("active"):
            lines.append("⚠️ 챔피언에 켜진 알파가 없어(0개) 챔피언은 신호를 평가하지 않습니다. 켜는 길은 챌린저 승격뿐입니다 (/alphas)")
        else:
            lines.append(f"켜진 알파: {esc(', '.join(w['active']))}")
        for b in w.get("blockers") or []:
            code = b.get("code")
            if code == "cooldown":
                lines.append(f"진입 차단: 손실 뒤 종목 쿨다운 ~{fmt_ts(b.get('until'), tz, '%H:%M')}")
            elif code == "max positions":
                lines.append(f"진입 차단: 동시 포지션 한도 {esc(b.get('detail'))}개 도달")
            else:
                lines.append("진입 차단: " + esc(RISK_KO.get(code, code)) + (f" ({esc(b['detail'])})" if b.get("detail") else ""))
        for k in w.get("killed") or []:
            lines.append(f"🛑 킬 스위치로 꺼진 알파 <code>{esc(k['alpha'])}</code> ({fmt_ts(k.get('at'), tz, '%m-%d')}, "
                         f"{esc(k.get('version'))})")
        decs = w.get("decisions") or []
        if decs:
            lines.append("<b>최근 판단</b> (신호가 있었던 봉, 최신순)")
            lines += [fmt_decision(d, tz) for d in decs[:8]]
        else:
            lines.append("최근 판단 기록 없음 (재시작 뒤 이 종목에 신호가 없었음, 기록은 메모리에만 보관)")
    for o in w.get("others") or []:
        lines.append(f"다른 엔진: <code>{esc(o['engine'])}</code> {esc(o['side'])} [{esc(o['status'])}] "
                     f"<code>{esc(o['alpha'])}</code> {_num(o.get('r_now'), '+.2f')}R")
    return "\n".join(lines)


def fmt_alpha_overview(ov: dict) -> str:
    """/alphas: one line per alpha (champion switch, evidence stage/verdict/numbers, live record, kill switch,
    challengers), enabled ones first, and the explanation when the champion has none."""
    ls = ov.get("learning") or {}
    rows = sorted(ov.get("rows") or [], key=lambda r: (not r.get("enabled"), not r.get("challengers")))
    lines = [f"<b>알파 현황</b> — 챔피언 <code>{esc(ov.get('champion', ''))}</code>, 운영 성적은 {esc(ov.get('engine', ''))} 엔진",
             "✅ 챔피언에서 켜짐 · 🧪 챌린저가 바꾸거나 거래 · ⛔ 꺼짐"]
    for r in rows:
        icon = "✅" if r.get("enabled") else "🧪" if r.get("challengers") else "⛔"
        parts = [f"{icon} <code>{esc(r['alpha'])}</code>", esc(fmt_evidence(r.get("evidence"), r.get("evidence_stale"))),
                 esc(_live(r.get("live")))]
        if r.get("enabled"):
            if r.get("kill_switch"):
                parts.append(_ks(r["kill_switch"]))
            if r.get("trust") is not None:
                parts.append(f"신뢰 {_num(r['trust'], '.2f')}")
        elif r.get("killed"):
            k = r["killed"]
            parts.append(f"🛑 킬 스위치로 꺼짐 ({esc(k.get('version'))}, {esc(k.get('rule'))})")
        if r.get("challengers"):
            parts.append("챌린저 " + ", ".join(f"{esc(c['name'])}({ROLE_KO.get(c['role'], c['role'])})"
                                               for c in r["challengers"]))
        lines.append(" · ".join(parts))
    if ls.get("no_alpha"):
        lines += ["", esc(ls.get("headline", "")), "거래 시작 조건: " + esc(ls.get("criteria", ""))]
    if ls.get("evidence"):
        lines.append(esc(ls["evidence"]))
    if ls.get("kill_switch"):
        lines.append(esc(ls["kill_switch"]))
    return "\n".join(lines)


def fmt_outlook(c: dict, tz: str) -> str:
    """One challenger's standing: path, current record and statistic against the next checkpoint's bar."""
    cur = c.get("current") or {}
    path = "상대" if c.get("path") == "relative" else "절대"
    stat = "차이 t" if c.get("path") == "relative" else "t_day"
    nxt = c.get("next_checkpoint")
    when = (f"다음 판정점 {nxt}일({fmt_ts(c.get('next_at'), tz, '%m-%d')})" if nxt is not None
            else "판정점 모두 지남")
    done = "·".join(f"{x}{'✓' if x in (c.get('passed') or []) else ''}" for x in c.get("checkpoints") or [])
    verdict = f" · 최근 판정 {esc(c['verdict'])}" if c.get("verdict") else " · 판정 대기"
    return (f"  <code>{esc(c.get('name'))}</code> [{esc(c.get('source'))}] {path} 경로: {cur.get('n', 0)}/{c.get('min_trades', 0)}건, "
            f"청산일 {cur.get('days', 0)}/{c.get('min_days', 0)}일, 평균 {_num(cur.get('avg_r'), '+.2f')}R, "
            f"{stat} {_num(c.get('stat'), '+.2f')}/{_num(c.get('need'), 'g')} → {when} (판정점 {done}일){verdict}"
            + (f" · OOS 탈락 알파 {esc(', '.join(c['failed_alphas']))} 강화 기준" if c.get("failed_alphas") else ""))


def fmt_insight(ins: dict, st: dict, tz: str) -> list[str]:
    """The daily report's learning section: per-alpha performance, the research and discovery cycles and the
    challengers' outlook."""
    out: list[str] = []
    perf = [a for a in ins.get("alphas") or [] if a.get("all", {}).get("n") or a.get("enabled")]
    if perf:
        out.append(f"<b>알파별 성과</b> ({esc(ins.get('engine', ''))}, 7일 / 전체)")
        for a in perf:
            w, al = a.get("week") or {}, a.get("all") or {}
            out.append(f"  {'✅' if a.get('enabled') else '⛔'} <code>{esc(a['alpha'])}</code> 7일 {w.get('n', 0)}건 "
                       f"{fmt_usd(a.get('week_net', 0.0))} 평균 {_num(w.get('avg_r'), '+.2f')}R · 전체 {al.get('n', 0)}건 "
                       f"평균 {_num(al.get('avg_r'), '+.2f')}R t {_num(al.get('t_day'), '+.2f')}")
    else:
        out.append("알파별 성과: 거래 없음" + (" (켜진 알파 0개라 챔피언은 거래하지 않음)" if (st.get("learning") or {}).get("no_alpha") else ""))
    r = ins.get("research") or {}
    if r.get("last"):
        line = (f"🔬 리서치 {fmt_ts(r['last'], tz, '%m-%d %H:%M')}" + (" (진행 중)" if r.get("running") else "")
                + (f" · {r['seconds'] / 60:.0f}분" if r.get("seconds") else "")
                + f" · 알파 {r.get('evaluated', 0)}개 평가, 개선 후보 {len(r.get('improved') or [])}"
                + (f"({esc(', '.join(r['improved']))})" if r.get("improved") else "")
                + (f" · 다음 {fmt_ts(r['next_due'], tz, '%m-%d %H:%M')}" if r.get("next_due") else ""))
        out.append(line)
        if r.get("outcome"):
            out.append("  " + esc(str(r["outcome"]).split("\n")[0][:160]))
    else:
        out.append("🔬 리서치: 아직 사이클 없음")
    if r.get("failures"):
        out.append(f"  ⚠️ 리서치 실패 {r['failures']}회 (마지막: {esc(str(r.get('last_error', ''))[:120])})")
    dsc = ins.get("discovery") or {}
    if dsc.get("last"):
        out.append(f"⛏ 발굴 {fmt_ts(dsc['last'], tz, '%m-%d %H:%M')} · 규칙 {dsc.get('tested', 0):,}개 검사, 기준 "
                   f"t≥{_num(dsc.get('t_bar'), '.2f')}, 통과 {dsc.get('passed', 0)}"
                   + (f" · 다음 {fmt_ts(dsc['next_due'], tz, '%m-%d %H:%M')}" if dsc.get("next_due") else ""))
    else:
        out.append("⛏ 발굴: 아직 실행 없음 (저장 이력이 쌓이면 매일 자동)")
    chs = ins.get("challengers") or []
    if chs:
        out.append("<b>챌린저</b>")
        out += [fmt_outlook(c, tz) for c in chs]
    else:
        out.append("챌린저: 없음 (리서치·발굴이 후보를 찾으면 생성)")
    return out
