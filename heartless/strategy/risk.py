"""Position sizing and portfolio-level risk limits (never optimised, only configured)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from heartless.config import Settings
from heartless.core.models import Decision, Position, Side, SymbolInfo
from heartless.exchange.symbols import min_qty_for_notional, norm_qty
from heartless.util.timeutil import MS_DAY, MS_MINUTE, floor_ms


@dataclass
class RiskState:
    paused: bool = False
    halted_until: int = 0  # ms; entries blocked (daily/weekly loss limits)
    halt_reason: str = ""
    shock_until: int = 0  # market shock cool-off
    symbol_cooldown: dict[str, int] = field(default_factory=dict)
    day_start_equity: float = 0.0
    day_key: int = 0
    week_start_equity: float = 0.0
    week_key: int = 0
    peak_equity: float = 0.0
    risk_scale: float = 1.0  # 0.5 after a weekly limit breach, back to 1 next week


@dataclass
class SizeResult:
    qty: float
    notional: float
    risk_amount: float
    risk_pct: float
    leverage: int
    reason: str = ""


class RiskManager:
    def __init__(self, settings: Settings):
        self.s = settings
        self.state = RiskState()

    # --- equity bookkeeping --------------------------------------------------------------------
    def observe_equity(self, equity: float, now: int) -> list[str]:
        """Update daily/weekly anchors; returns a list of newly triggered risk events."""
        st = self.state
        events: list[str] = []
        day_key = floor_ms(now, MS_DAY)
        week_key = floor_ms(now - 3 * MS_DAY, 7 * MS_DAY)  # weeks start Monday 00:00 UTC
        if st.day_key != day_key:
            st.day_key = day_key
            st.day_start_equity = equity
            if st.halted_until and st.halted_until <= now:
                st.halted_until = 0
                st.halt_reason = ""
        if st.week_key != week_key:
            st.week_key = week_key
            st.week_start_equity = equity
            st.risk_scale = 1.0
        st.peak_equity = max(st.peak_equity, equity)
        if st.day_start_equity > 0:
            dd_day = (equity - st.day_start_equity) / st.day_start_equity * 100
            if dd_day <= -self.s.daily_loss_limit_pct and st.halted_until < day_key + MS_DAY:
                st.halted_until = day_key + MS_DAY
                st.halt_reason = f"일일 손실 한도 도달 ({dd_day:.2f}%). 다음 UTC 자정까지 신규 진입 중단"
                events.append(st.halt_reason)
        if st.week_start_equity > 0:
            dd_week = (equity - st.week_start_equity) / st.week_start_equity * 100
            if dd_week <= -self.s.weekly_loss_limit_pct and st.risk_scale > 0.5:
                st.risk_scale = 0.5
                events.append(f"주간 손실 {dd_week:.2f}% → 이번 주 리스크 50% 축소")
        if st.peak_equity > 0:
            dd_peak = (equity - st.peak_equity) / st.peak_equity * 100
            if dd_peak <= -self.s.max_drawdown_halt_pct and not st.paused:
                st.paused = True
                st.halt_reason = f"고점 대비 낙폭 {dd_peak:.1f}% → 봇 일시정지 (/resume 로 재개)"
                events.append(st.halt_reason)
        return events

    def daily_pnl_pct(self, equity: float) -> float:
        if not self.state.day_start_equity:
            return 0.0
        return (equity - self.state.day_start_equity) / self.state.day_start_equity * 100

    def register_market_shock(self, now: int, minutes: int = 15) -> None:
        self.state.shock_until = max(self.state.shock_until, now + minutes * MS_MINUTE)

    def register_loss(self, symbol: str, now: int, minutes: int = 20) -> None:
        self.state.symbol_cooldown[symbol] = now + minutes * MS_MINUTE

    # --- gating --------------------------------------------------------------------------------
    def can_open(self, decision: Decision, open_positions: list[Position], now: int,
                 equity: float) -> tuple[bool, str]:
        st = self.state
        if st.paused:
            return False, "paused"
        if st.halted_until > now:
            return False, "halted:" + st.halt_reason
        if st.shock_until > now:
            return False, "market shock cool-off"
        if st.symbol_cooldown.get(decision.symbol, 0) > now:
            return False, "symbol cool-down after loss"
        active = [p for p in open_positions if p.status.value in ("PENDING", "OPEN", "CLOSING")]
        if any(p.symbol == decision.symbol for p in active):
            return False, "already in position"
        if len(active) >= self.s.max_positions:
            return False, "max positions"
        same = [p for p in active if p.side is decision.side]
        if len(same) >= self.s.max_same_direction:
            return False, "max same-direction positions"
        gross = sum(p.notional for p in active)
        if equity > 0 and gross / equity >= self.s.max_gross_leverage:
            return False, "gross leverage cap"
        return True, ""

    # --- sizing --------------------------------------------------------------------------------
    def size(self, decision: Decision, equity: float, info: SymbolInfo, entry_price: float,
             open_notional: float = 0.0) -> SizeResult:
        sig = decision.primary
        dist = abs(entry_price - sig.stop)
        if dist <= 0 or equity <= 0 or entry_price <= 0:
            return SizeResult(0, 0, 0, 0, self.s.exchange_leverage, "invalid stop distance")
        risk_pct = self.s.risk_per_trade_pct * decision.size_mult * self.state.risk_scale
        risk_pct = max(0.1, min(self.s.max_risk_per_trade_pct, risk_pct))
        risk_amount = equity * risk_pct / 100
        # size so that a stop-out *including* round-trip fees and slippage costs exactly risk_amount
        cost_per_unit = entry_price * (2 * self.s.taker_fee + 0.0003)
        qty = risk_amount / (dist + cost_per_unit)
        max_notional = equity * self.s.max_position_leverage
        remaining_gross = equity * self.s.max_gross_leverage - open_notional
        max_notional = min(max_notional, max(remaining_gross, 0.0), equity * self.s.exchange_leverage * 0.8)
        if qty * entry_price > max_notional:
            qty = max_notional / entry_price
        q = norm_qty(info, qty)
        min_q = min_qty_for_notional(info, entry_price)
        if q < min_q:
            if min_q * entry_price <= max_notional and min_q * dist <= risk_amount * 2.5:
                q = min_q
            else:
                return SizeResult(0, 0, 0, risk_pct, self.s.exchange_leverage, "below min notional")
        notional = q * entry_price
        lev = int(min(self.s.exchange_leverage, max(1, math.ceil(notional / max(equity, 1e-9) * 1.5))))
        lev = max(lev, min(self.s.exchange_leverage, 3))
        return SizeResult(q, notional, q * (dist + cost_per_unit), risk_pct, lev)
