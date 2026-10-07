"""Position sizing and portfolio-level risk limits (never optimised, only configured)."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from heartless.config import Settings
from heartless.core.models import Decision, Position, Side, SymbolInfo
from heartless.exchange.symbols import min_qty_for_notional, norm_qty
from heartless.util.timeutil import MS_DAY, MS_MINUTE, floor_ms

if TYPE_CHECKING:  # pragma: no cover - type-only import (avoids a runtime dependency on the store)
    from heartless.core.store import Store

log = logging.getLogger(__name__)

# owner kill switch written by app.pause()/app.resume(); honoured when the state is loaded from the store
PAUSED_KEY = "paused"


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
    dd_halt_peak: float = 0.0  # peak_equity the max-drawdown pause last fired at (one-shot; re-arms on a new high)


@dataclass
class SizeResult:
    qty: float
    notional: float
    risk_amount: float
    risk_pct: float
    leverage: int
    reason: str = ""


class RiskManager:
    def __init__(self, settings: Settings, store: "Store | None" = None, name: str = ""):
        """`store`/`name` are optional: with a store attached the state is restored from the kv key
        "<name>.risk" right away and written back whenever it changes, so loss halts, the drawdown pause,
        the weekly risk scale, symbol cool-downs and the owner's pause survive a process restart.
        Pass no store for backtests (state stays memory-only, exactly as before)."""
        self.s = settings
        self.state = RiskState()
        self._store = store
        self._key = f"{name}.risk" if name else "risk"
        self._saved: dict | None = None  # last snapshot written to the store (change detection)
        if store is not None:
            self.load()

    # --- persistence ---------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self.state)

    dump = to_dict

    def restore(self, d: dict | None) -> None:
        """Apply a previously dumped state. Unknown keys are ignored, malformed values are skipped."""
        if not isinstance(d, dict):
            return
        defaults = RiskState()
        for k, v in d.items():
            if k not in RiskState.__dataclass_fields__:
                continue
            try:
                if k == "symbol_cooldown":
                    v = {str(sym): int(until) for sym, until in dict(v or {}).items()}
                else:
                    v = type(getattr(defaults, k))(v)
            except (TypeError, ValueError):
                continue
            setattr(self.state, k, v)

    def load(self, d: dict | None = None) -> None:
        """Restore from `d`, or (when `d` is None) from the attached store, including the owner's pause flag.
        Must run before the first observe_equity so a matching day/week key keeps the stored anchors and halts."""
        if d is not None:
            self.restore(d)
            return
        if self._store is None:
            return
        self.restore(self._store.get(self._key) or {})
        if self._store.get(PAUSED_KEY, False):
            self.state.paused = True
        self._saved = self.to_dict()

    def save(self, force: bool = False) -> None:
        """Write the state to the attached store when it differs from the last written snapshot."""
        if self._store is None:
            return
        d = self.to_dict()
        if not force and d == self._saved:
            return
        try:
            self._store.set(self._key, d)
            self._saved = d
        except Exception as e:  # noqa: BLE001 - persistence must never break risk evaluation
            log.warning("risk state save failed (%s): %s", self._key, e)

    def pause(self, reason: str = "") -> None:
        self.state.paused = True
        if reason:
            self.state.halt_reason = reason
        self.save()

    def resume(self, equity: float = 0.0) -> None:
        """Owner resume. With the current equity the drawdown peak is re-based to it, so the max-drawdown pause
        fires again only after a further max_drawdown_halt_pct decline; without it the pause stays latched
        until a new equity high. Either way the next equity tick does not immediately re-pause."""
        st = self.state
        st.paused = False
        st.halt_reason = ""
        if equity > 0:
            st.peak_equity = equity
            st.dd_halt_peak = 0.0
        else:
            st.dd_halt_peak = st.peak_equity
        self.save()

    # --- equity bookkeeping --------------------------------------------------------------------
    def observe_equity(self, equity: float, now: int) -> list[str]:
        """Update daily/weekly anchors; returns a list of newly triggered risk events."""
        st = self.state
        events: list[str] = []
        day_key = floor_ms(now, MS_DAY)
        # weeks start Monday 00:00 UTC: epoch day 0 is a Thursday, so shift by 4 days around the 7-day floor
        week_key = floor_ms(now - 4 * MS_DAY, 7 * MS_DAY) + 4 * MS_DAY
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
            # one-shot per peak: after the owner's /resume it must not re-fire on the next tick (peak unchanged)
            if dd_peak <= -self.s.max_drawdown_halt_pct and not st.paused and st.peak_equity > st.dd_halt_peak:
                st.paused = True
                st.dd_halt_peak = st.peak_equity
                st.halt_reason = f"고점 대비 낙폭 {dd_peak:.1f}% → 봇 일시정지 (/resume 로 재개)"
                events.append(st.halt_reason)
        self.save()
        return events

    def daily_pnl_pct(self, equity: float) -> float:
        if not self.state.day_start_equity:
            return 0.0
        return (equity - self.state.day_start_equity) / self.state.day_start_equity * 100

    def register_market_shock(self, now: int, minutes: int = 15) -> None:
        self.state.shock_until = max(self.state.shock_until, now + minutes * MS_MINUTE)
        self.save()

    def register_loss(self, symbol: str, now: int, minutes: int = 20) -> None:
        self.state.symbol_cooldown[symbol] = now + minutes * MS_MINUTE
        self.save()

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
            # lifting to the exchange minimum may exceed the per-trade budget, but never the configured hard cap
            # (max_risk_per_trade_pct) nor 2.5x the budget, both measured as the realised stop-out loss incl. costs
            hard_cap = equity * self.s.max_risk_per_trade_pct / 100
            min_loss = min_q * (dist + cost_per_unit)
            if min_q * entry_price <= max_notional and min_loss <= min(hard_cap, risk_amount * 2.5):
                q = min_q
                risk_pct = min_loss / equity * 100  # report the risk actually taken
            else:
                return SizeResult(0, 0, 0, risk_pct, self.s.exchange_leverage, "below min notional")
        notional = q * entry_price
        lev = int(min(self.s.exchange_leverage, max(1, math.ceil(notional / max(equity, 1e-9) * 1.5))))
        lev = max(lev, min(self.s.exchange_leverage, 3))
        return SizeResult(q, notional, q * (dist + cost_per_unit), risk_pct, lev)
