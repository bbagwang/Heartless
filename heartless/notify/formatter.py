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


def fmt_alphas(rows: list[dict], params) -> str:
    lines = ["<b>알파 신뢰도 (Thompson sampling)</b>"]
    for r in rows:
        en = params.enabled.get(r["alpha"], True)
        lines.append(f"{'✅' if en else '⛔'} <code>{esc(r['alpha'])}</code> trust {r['trust']:.2f} · n={r['n']} · avgR {r['avg_r']:+.2f}\n"
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
    if st.get("graduation") and st["graduation"].get("ok"):
        lines.append("🎓 페이퍼 성과가 졸업 기준을 충족했습니다. /golive 로 실거래 전환 가능")
    return "\n".join(lines)
