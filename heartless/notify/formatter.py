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


def fmt_daily(today: dict, week: dict, st: dict, tz: str) -> str:
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
    lines += fmt_learning(st)
    xq = fmt_exec_quality(st, suggest=True)
    if xq:
        lines.append(xq)
    if st.get("graduation") and st["graduation"].get("ok"):
        lines.append("🎓 페이퍼 성과가 졸업 기준을 충족했습니다. /golive 로 실거래 전환 가능")
    return "\n".join(lines)


def fmt_learning(st: dict) -> list[str]:
    """What trades and what the learning loop is doing (truthful when no alpha is enabled), the challengers' standing
    against their promotion bar and the kill-switch state of the champion's alphas."""
    ls = st.get("learning") or {}
    if not ls:
        return []
    out = [esc(ls.get("headline", ""))]
    if ls.get("no_alpha") and ls.get("criteria"):
        out.append("거래 시작 조건: " + esc(ls["criteria"]))
    for c in st.get("challengers", []):
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
    for k in st.get("kill_switch") or []:
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
