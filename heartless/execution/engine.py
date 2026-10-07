"""The trading engine: one instance per account/parameter-set (live, paper champion, paper challengers, backtest).

Every code path (entries, brackets, trailing, time stops, reconciliation) is identical across modes;
only the Account implementation differs.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import (Decision, EntryStyle, Fill, Position, PositionStatus, Side, SymbolInfo,
                                   TradeRecord)
from heartless.core.store import Store
from heartless.data.features import MarketView
from heartless.exchange.base import Account
from heartless.exchange.symbols import norm_price, norm_qty
from heartless.learning.bandit import AlphaBandit
from heartless.strategy.base import Context
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from heartless.strategy.risk import RiskManager
from heartless.util.ids import client_order_id, short_id
from heartless.util.timeutil import MS_MINUTE, now_ms

log = logging.getLogger(__name__)


@dataclass
class EngineStats:
    realized_today: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    last_trade_ts: int = 0
    total_trades: int = 0
    equity: float = 0.0
    balance: float = 0.0
    start_equity: float = 0.0
    peak_equity: float = 0.0
    last_equity_ts: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


class TradingEngine:
    def __init__(self, name: str, account: Account, params: StrategyParams, settings: Settings, symbols: dict[str, SymbolInfo],
                 store: Store | None = None, bus: EventBus | None = None, bandit: AlphaBandit | None = None,
                 only_alpha: str | None = None, clock: Callable[[], int] | None = None, persist: bool = True,
                 notify: bool = False):
        self.name = name
        self.account = account
        self.params = params
        self.s = settings
        self.symbols = symbols
        self.store = store
        self.bus = bus
        self.bandit = bandit or AlphaBandit([a for a in params.alphas])
        self.ensemble = Ensemble(params, self.bandit, only_alpha=only_alpha)
        self.risk = RiskManager(settings)
        self.clock = clock or now_ms
        self.persist = persist and store is not None
        self.notify = notify
        self.positions: dict[str, Position] = {}
        self.stats = EngineStats()
        self.closed: list[TradeRecord] = []  # in-memory (backtests + recent)
        self.account.on_fill(self.on_fill)
        self.last_views: dict[str, MarketView] = {}
        self._equity_cache: tuple[int, float] = (0, 0.0)
        self.entries_enabled = True
        self.tag = "".join(ch for ch in name.upper() if ch.isalnum())[:4]

    # --- lifecycle -----------------------------------------------------------------------------
    async def start(self) -> None:
        if self.persist:
            for p in self.store.load_open_positions(self.name):
                self.positions[p.symbol] = p
            if self.positions:
                log.info("[%s] restored %d open positions", self.name, len(self.positions))
        st = await self.account.get_state()
        self.stats.equity = self.stats.balance = st.equity
        self.stats.start_equity = self.stats.start_equity or st.equity
        self.stats.peak_equity = max(self.stats.peak_equity, st.equity)
        self.risk.observe_equity(st.equity, self.clock())
        self._equity_cache = (self.clock(), st.equity)
        if not self.account.is_paper:
            await self.reconcile()

    def set_params(self, params: StrategyParams) -> None:
        self.params = params
        self.ensemble.set_params(params)

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.status in (PositionStatus.PENDING, PositionStatus.OPEN, PositionStatus.CLOSING)]

    async def equity(self, max_age_ms: int = 5000) -> float:
        ts, eq = self._equity_cache
        if self.clock() - ts <= max_age_ms and eq > 0:
            return eq
        st = await self.account.get_state()
        self._equity_cache = (self.clock(), st.equity)
        self.stats.equity, self.stats.balance = st.equity, st.balance
        self.stats.peak_equity = max(self.stats.peak_equity, st.equity)
        return st.equity

    async def _emit(self, topic: str, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("engine", self.name)
        payload.setdefault("notify", self.notify)
        payload.setdefault("ts", self.clock())
        if self.bus is not None:
            await self.bus.publish(topic, payload)
        if self.persist and topic in ("position_opened", "position_closed", "risk_event", "reconcile", "error"):
            self.store.log_event(self.clock(), "INFO", topic, {k: v for k, v in payload.items() if k != "position"})

    def _save(self, p: Position) -> None:
        if self.persist:
            self.store.save_position(p)

    # --- bar handling --------------------------------------------------------------------------
    async def on_bar(self, view: MarketView, ctx: Context) -> None:
        """Called once per completed 1m bar per symbol (after the view has been sought to that bar)."""
        self.last_views[ctx.symbol] = view
        now = self.clock()
        pos = self.positions.get(ctx.symbol)
        if pos is not None and pos.status in (PositionStatus.PENDING, PositionStatus.OPEN, PositionStatus.CLOSING):
            await self._manage(pos, view, ctx)
            return
        if not view.ready() or not self.entries_enabled:
            return
        decision = self.ensemble.decide(view, ctx)
        if decision is None:
            return
        equity = await self.equity()
        allowed, why = self.risk.can_open(decision, self.open_positions(), now, equity)
        if not allowed:
            self.stats.skipped[why.split(":")[0]] = self.stats.skipped.get(why.split(":")[0], 0) + 1
            return
        await self._open(decision, view, ctx, equity)

    async def on_equity_tick(self) -> None:
        """Periodic: refresh equity, risk anchors, persist equity curve."""
        now = self.clock()
        try:
            eq = await self.equity(max_age_ms=0)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] equity refresh failed: %s", self.name, e)
            return
        for ev in self.risk.observe_equity(eq, now):
            await self._emit("risk_event", {"message": ev})
        if self.persist and now - self.stats.last_equity_ts >= 60_000:
            self.stats.last_equity_ts = now
            self.store.save_equity(self.name, now, self.stats.balance, eq)

    # --- entries -------------------------------------------------------------------------------
    async def _open(self, d: Decision, view: MarketView, ctx: Context, equity: float) -> None:
        sig = d.primary
        info = self.symbols.get(d.symbol)
        if info is None:
            return
        t = ctx.ticker
        if sig.entry_style is EntryStyle.LIMIT and sig.limit_price:
            entry_ref = sig.limit_price
        else:
            entry_ref = (t.ask if d.side is Side.LONG else t.bid) or t.ref or view.price
        if not entry_ref or entry_ref <= 0:
            return
        if t.spread_bps > 12:  # illiquid moment
            self.stats.skipped["spread"] = self.stats.skipped.get("spread", 0) + 1
            return
        open_notional = sum(p.notional for p in self.open_positions())
        sz = self.risk.size(d, equity, info, entry_ref, open_notional)
        if sz.qty <= 0:
            self.stats.skipped["size"] = self.stats.skipped.get("size", 0) + 1
            return
        stop = norm_price(info, sig.stop)
        tp = norm_price(info, sig.take_profit) if sig.take_profit else None
        tp1 = norm_price(info, sig.tp1) if sig.tp1 else None
        r_unit = abs(entry_ref - stop)
        if r_unit <= 0:
            return
        pid = short_id("P")
        cid = client_order_id(f"E{self.tag}")
        pos = Position(id=pid, engine=self.name, symbol=d.symbol, side=d.side, qty=0.0, entry_price=entry_ref,
                       entry_time=self.clock(), stop=stop, take_profit=tp, tp1=tp1, initial_stop=stop,
                       alpha=sig.alpha, alphas=list(d.alphas), reason=d.reason, confidence=d.score, regime=d.regime.value,
                       risk_amount=sz.risk_amount, r_unit=r_unit, notional=sz.notional, leverage=sz.leverage,
                       params_version=self.params.version, atr=sig.atr, trail_atr_mult=sig.trail_atr_mult,
                       max_hold_bars=sig.max_hold_bars, timeframe=sig.timeframe, status=PositionStatus.PENDING,
                       entry_client_id=cid, original_qty=sz.qty, entry_style=sig.entry_style.value,
                       limit_price=entry_ref if sig.entry_style is EntryStyle.LIMIT else None, pending_since=self.clock(),
                       expected_profit=(abs((tp or entry_ref + d.side.sign * d.expected_r * r_unit) - entry_ref) * sz.qty),
                       extra={"tp1_frac": self.params.alphas.get(sig.alpha, {}).get("tp1_frac", 0.5), "tags": sig.tags,
                              "size_mult": d.size_mult, "risk_pct": sz.risk_pct})
        self.positions[d.symbol] = pos
        self._save(pos)
        try:
            await self.account.prepare_symbol(d.symbol, sz.leverage)
            if sig.entry_style is EntryStyle.LIMIT:
                price = norm_price(info, entry_ref, d.side.order_side)
                res = await self.account.limit_order(d.symbol, d.side.order_side, sz.qty, price, post_only=True, client_id=cid)
                if res.status in ("EXPIRED", "REJECTED"):
                    # post-only would cross: take liquidity if the price is still within a fraction of the ATR
                    ref_now = (t.ask if d.side is Side.LONG else t.bid) or entry_ref
                    if abs(ref_now - entry_ref) <= 0.15 * max(sig.atr, r_unit):
                        res = await self.account.market_order(d.symbol, d.side.order_side, sz.qty, client_id=cid)
                    else:
                        res = None
            else:
                res = await self.account.market_order(d.symbol, d.side.order_side, sz.qty, client_id=cid)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] entry failed for %s", self.name, d.symbol)
            res = None
            await self._emit("error", {"message": f"{d.symbol} 진입 주문 실패: {e}"})
        if res is None or res.status == "REJECTED":
            if pos.status is PositionStatus.PENDING and pos.filled_qty <= 0:
                pos.status = PositionStatus.CANCELLED
                pos.exit_reason = "entry rejected"
                self._save(pos)
                self.positions.pop(d.symbol, None)
            return
        pos.entry_order_id = res.order_id
        if res.status == "FILLED" and pos.status is PositionStatus.PENDING and pos.filled_qty <= 0:
            # REST already reports the fill (live market order): synthesize if no stream fill arrived
            fill = Fill(symbol=d.symbol, order_side=d.side.order_side, qty=res.filled_qty or sz.qty,
                        price=res.avg_price or entry_ref, fee=(res.filled_qty or sz.qty) * (res.avg_price or entry_ref) * self.s.taker_fee,
                        ts=self.clock(), client_id=cid, order_id=res.order_id, kind="ENTRY")
            await self.on_fill(fill)
        self._save(pos)

    # --- fills ---------------------------------------------------------------------------------
    async def on_fill(self, fill: Fill) -> None:
        pos = self.positions.get(fill.symbol)
        if fill.kind == "FUNDING":
            if pos is not None and pos.status is PositionStatus.OPEN:
                pos.funding -= fill.fee  # fee holds the payment (positive = paid)
                self._save(pos)
            return
        if pos is None:
            if not fill.reduce_only and fill.kind in ("ENTRY", "UNKNOWN") and not self.account.is_paper:
                log.warning("[%s] fill for unknown position %s %s %s", self.name, fill.symbol, fill.order_side, fill.qty)
            return
        opens_side = fill.order_side == pos.side.order_side
        if opens_side and not fill.reduce_only:
            await self._on_entry_fill(pos, fill)
        else:
            await self._on_exit_fill(pos, fill)

    async def _on_entry_fill(self, pos: Position, fill: Fill) -> None:
        if fill.client_id and pos.entry_client_id and fill.client_id != pos.entry_client_id and fill.client_id.startswith("HLE"):
            return  # a different (stale) entry order
        if pos.status is PositionStatus.CLOSED:
            return
        # de-duplicate: a REST-synthesized fill and a stream fill for the same order
        key = f"{fill.order_id}:{fill.ts}:{fill.qty}"
        seen = pos.extra.setdefault("fills", [])
        if fill.order_id and any(k.startswith(f"{fill.order_id}:") for k in seen) and fill.client_id == pos.entry_client_id and pos.filled_qty >= pos.original_qty - 1e-12:
            return
        seen.append(key)
        total = pos.filled_qty + fill.qty
        pos.entry_price = (pos.entry_price * pos.filled_qty + fill.price * fill.qty) / total if pos.filled_qty > 0 else fill.price
        pos.filled_qty = total
        pos.qty = total
        pos.fees += fill.fee
        pos.notional = pos.qty * pos.entry_price
        if pos.status is PositionStatus.PENDING:
            pos.status = PositionStatus.OPEN
            pos.entry_time = fill.ts or self.clock()
            # recompute R geometry from the real fill
            pos.r_unit = abs(pos.entry_price - pos.stop) or pos.r_unit
            pos.risk_amount = pos.qty * (pos.r_unit + pos.entry_price * (2 * self.s.taker_fee + 0.0003))
            await self._place_brackets(pos)
            self._save(pos)
            await self._emit("position_opened", {"position": pos, "symbol": pos.symbol})
        else:
            # additional partial fill of a limit entry: brackets must cover the new quantity
            await self._refresh_brackets(pos)
            self._save(pos)

    async def _place_brackets(self, pos: Position) -> None:
        info = self.symbols[pos.symbol]
        close_side = pos.side.close_order_side
        try:
            pos.sl_algo_id = await self.account.place_stop(pos.symbol, close_side, pos.stop, close_position=True,
                                                           client_id=client_order_id(f"S{self.tag}"))
            pos.last_stop_update = self.clock()
        except Exception as e:  # noqa: BLE001
            log.error("[%s] failed to place stop for %s: %s", self.name, pos.symbol, e)
            await self._emit("error", {"message": f"{pos.symbol} 손절 주문 실패 ({e}). 봇 내부 백스톱으로 보호 중"})
        frac = float(pos.extra.get("tp1_frac", 0.5) or 0.0)
        if pos.tp1 and frac > 0:
            q1 = norm_qty(info, pos.qty * frac)
            rest = norm_qty(info, pos.qty - q1)
            if q1 > 0 and rest > 0:
                try:
                    pos.extra["tp1_algo_id"] = await self.account.place_take_profit(pos.symbol, close_side, pos.tp1, q1,
                                                                                    client_id=client_order_id(f"T{self.tag}"))
                    pos.extra["tp1_qty"] = q1
                except Exception as e:  # noqa: BLE001
                    log.warning("[%s] TP1 placement failed for %s: %s", self.name, pos.symbol, e)
            else:
                pos.tp1 = None
        if pos.take_profit:
            q_tp = norm_qty(info, pos.qty - float(pos.extra.get("tp1_qty", 0.0) or 0.0))
            if q_tp > 0:
                try:
                    pos.tp_algo_id = await self.account.place_take_profit(pos.symbol, close_side, pos.take_profit, q_tp,
                                                                          client_id=client_order_id(f"T{self.tag}"))
                except Exception as e:  # noqa: BLE001
                    log.warning("[%s] TP placement failed for %s: %s", self.name, pos.symbol, e)

    async def _refresh_brackets(self, pos: Position) -> None:
        await self._cancel_brackets(pos)
        await self._place_brackets(pos)

    async def _cancel_brackets(self, pos: Position) -> None:
        for aid in (pos.sl_algo_id, pos.tp_algo_id, pos.extra.get("tp1_algo_id", "")):
            if aid:
                try:
                    await self.account.cancel_algo(pos.symbol, aid)
                except Exception:  # noqa: BLE001
                    pass
        pos.sl_algo_id = pos.tp_algo_id = ""
        pos.extra.pop("tp1_algo_id", None)

    async def _on_exit_fill(self, pos: Position, fill: Fill) -> None:
        if pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING, PositionStatus.PENDING):
            return
        qty = min(fill.qty, pos.qty) if pos.qty > 0 else fill.qty
        if qty <= 0:
            return
        pnl = (fill.price - pos.entry_price) * qty * pos.side.sign
        pos.realized += pnl
        pos.fees += fill.fee
        pos.qty = max(pos.qty - qty, 0.0)
        kind = fill.kind
        if kind in ("UNKNOWN", "CLOSE") and fill.reduce_only:
            # infer from price relative to brackets
            if pos.tp1 and not pos.tp1_done and abs(fill.price - pos.tp1) <= abs(fill.price - pos.stop):
                kind = "TP"
            elif (pos.side is Side.LONG and fill.price <= pos.stop * 1.002) or (pos.side is Side.SHORT and fill.price >= pos.stop * 0.998):
                kind = "SL"
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        remaining = pos.qty > step * 0.5 + 1e-12
        if remaining and kind == "TP" and pos.tp1 and not pos.tp1_done:
            pos.tp1_done = True
            pos.extra.pop("tp1_algo_id", None)
            await self._move_stop_to_breakeven(pos)
            self._save(pos)
            await self._emit("partial_tp", {"position": pos, "symbol": pos.symbol, "price": fill.price, "qty": qty, "pnl": pnl})
            return
        if remaining:
            self._save(pos)
            return
        reason = {"SL": "손절(SL)" if not pos.be_moved else "본절 손절(BE)", "TP": "익절(TP)"}.get(kind, pos.exit_reason or "청산")
        if kind == "SL" and pos.be_moved and pos.stop != pos.initial_stop and (
                (pos.side is Side.LONG and pos.stop > pos.entry_price) or (pos.side is Side.SHORT and pos.stop < pos.entry_price)):
            reason = "트레일링 스탑"
        await self._finalize(pos, fill.price, fill.ts or self.clock(), reason)

    async def _move_stop_to_breakeven(self, pos: Position) -> None:
        info = self.symbols[pos.symbol]
        buffer = pos.entry_price * (self.s.taker_fee * 2 + 0.0002)
        be = pos.entry_price + pos.side.sign * buffer
        be = norm_price(info, be)
        better = be > pos.stop if pos.side is Side.LONG else be < pos.stop
        if better:
            await self._replace_stop(pos, be)
        pos.be_moved = True

    async def _replace_stop(self, pos: Position, new_stop: float) -> None:
        info = self.symbols[pos.symbol]
        new_stop = norm_price(info, new_stop)
        if abs(new_stop - pos.stop) < info.tick_size * 0.5:
            return
        old_id = pos.sl_algo_id
        try:
            new_id = await self.account.place_stop(pos.symbol, pos.side.close_order_side, new_stop, close_position=True,
                                                   client_id=client_order_id(f"S{self.tag}"))
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] replace stop failed on %s: %s", self.name, pos.symbol, e)
            return
        pos.sl_algo_id = new_id
        pos.stop = new_stop
        pos.last_stop_update = self.clock()
        if old_id:
            try:
                await self.account.cancel_algo(pos.symbol, old_id)
            except Exception:  # noqa: BLE001
                pass
        self._save(pos)

    # --- management ----------------------------------------------------------------------------
    async def _manage(self, pos: Position, view: MarketView, ctx: Context) -> None:
        now = self.clock()
        price = ctx.ticker.ref or view.price
        if pos.status is PositionStatus.PENDING:
            await self._manage_pending(pos, view, ctx)
            return
        if pos.status is not PositionStatus.OPEN or not price:
            return
        pos.bars_held += 1
        fav = (price - pos.entry_price) * pos.side.sign
        pos.max_fav = max(pos.max_fav, fav)
        pos.max_adv = max(pos.max_adv, -fav)
        r_now = pos.r_now(price)
        tf = view.tf(pos.timeframe if pos.timeframe in view.frames else "5m")
        atr = tf.v("atr") if tf.ok else pos.atr
        if not atr or math.isnan(atr):
            atr = pos.atr
        # 1) time stop
        if pos.max_hold_bars and pos.bars_held >= pos.max_hold_bars:
            if r_now < 0.5:
                await self.close_position(pos.symbol, "시간 초과 청산(time stop)")
                return
            if not pos.be_moved:
                await self._move_stop_to_breakeven(pos)
        # 2) funding avoidance: don't pay a large funding bill on a trade that is not working
        if ctx.minutes_to_funding <= 2 and r_now < 0.3:
            adverse = ctx.funding_rate * pos.side.sign  # long pays positive funding
            if adverse > 0.0003:
                await self.close_position(pos.symbol, f"펀딩 회피 청산 (펀딩 {ctx.funding_rate * 100:+.3f}%)")
                return
        # 3) breakeven once 1R in profit even without a partial TP
        if not pos.be_moved and r_now >= 1.0 and pos.trail_atr_mult > 0:
            await self._move_stop_to_breakeven(pos)
        # 4) trailing stop (chandelier) on the decision timeframe once in profit
        if pos.trail_atr_mult > 0 and atr and (pos.be_moved or r_now >= 0.8):
            close = tf.v("close") if tf.ok else price
            hh = tf.v("hh10") if tf.ok else price
            ll = tf.v("ll10") if tf.ok else price
            if pos.side is Side.LONG:
                anchor = max(hh if not math.isnan(hh) else close, close)
                cand = anchor - pos.trail_atr_mult * atr
                if cand > pos.stop + max(self.symbols[pos.symbol].tick_size, 0.05 * atr):
                    await self._replace_stop(pos, cand)
            else:
                anchor = min(ll if not math.isnan(ll) else close, close)
                cand = anchor + pos.trail_atr_mult * atr
                if cand < pos.stop - max(self.symbols[pos.symbol].tick_size, 0.05 * atr):
                    await self._replace_stop(pos, cand)
        # 5) mean reversion / fade trades: exit if the regime flips hard against us
        if pos.alpha in ("mean_reversion", "funding_fade") and pos.bars_held % 5 == 0:
            m15 = view.tf("15m")
            adx = m15.v("adx") if m15.ok else float("nan")
            slope = m15.v("slope20") if m15.ok else float("nan")
            if not math.isnan(adx) and not math.isnan(slope) and adx > 32 and (slope * pos.side.sign) < 0 and r_now < 0:
                await self.close_position(pos.symbol, "레짐 전환(강한 역추세) 청산")
                return
        self._save(pos)

    async def _manage_pending(self, pos: Position, view: MarketView, ctx: Context) -> None:
        """Re-quote or cancel a resting post-only entry."""
        now = self.clock()
        age_bars = (now - pos.pending_since) / MS_MINUTE
        if pos.entry_style != EntryStyle.LIMIT.value:
            if age_bars >= 3:  # market order never confirmed -> verify
                res = await self.account.query_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
                if res.status in ("FILLED",) and res.filled_qty > 0 and pos.filled_qty <= 0:
                    await self.on_fill(Fill(pos.symbol, pos.side.order_side, res.filled_qty, res.avg_price,
                                            res.filled_qty * res.avg_price * self.s.taker_fee, now, pos.entry_client_id,
                                            res.order_id, kind="ENTRY"))
                elif res.status in ("CANCELED", "EXPIRED", "REJECTED", "UNKNOWN"):
                    await self._cancel_pending(pos, f"entry {res.status.lower()}")
            return
        if pos.filled_qty > 0 and age_bars >= 2:
            # partially filled limit: convert the remainder to OPEN with what we have
            await self.account.cancel_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
            pos.original_qty = pos.filled_qty
            return
        if age_bars < 2:
            return
        if pos.requotes >= 2:
            await self._cancel_pending(pos, "진입가 미도달로 주문 취소")
            return
        # re-quote at the new touch if price has not run away (> 0.6 ATR)
        t = ctx.ticker
        touch = t.bid if pos.side is Side.LONG else t.ask
        if not touch:
            return
        info = self.symbols[pos.symbol]
        if abs(touch - pos.limit_price) > 0.6 * max(pos.atr, pos.r_unit):
            await self._cancel_pending(pos, "가격 이탈로 진입 포기")
            return
        await self.account.cancel_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
        res = await self.account.query_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
        if res.filled_qty > pos.filled_qty:
            await self.on_fill(Fill(pos.symbol, pos.side.order_side, res.filled_qty - pos.filled_qty, res.avg_price,
                                    (res.filled_qty - pos.filled_qty) * res.avg_price * self.s.maker_fee, now,
                                    pos.entry_client_id, res.order_id, kind="ENTRY"))
            if pos.status is not PositionStatus.PENDING:
                return
        price = norm_price(info, touch, pos.side.order_side)
        cid = client_order_id(f"E{self.tag}")
        pos.entry_client_id = cid
        pos.requotes += 1
        pos.pending_since = now
        pos.limit_price = price
        new_stop = pos.stop  # keep the structural stop; R shrinks/grows with the new price
        pos.r_unit = abs(price - new_stop)
        pos.entry_price = price
        res = await self.account.limit_order(pos.symbol, pos.side.order_side, pos.original_qty - pos.filled_qty, price,
                                             post_only=True, client_id=cid)
        if res.status in ("EXPIRED", "REJECTED"):
            res = await self.account.market_order(pos.symbol, pos.side.order_side, pos.original_qty - pos.filled_qty, client_id=cid)
            if res.status == "REJECTED":
                await self._cancel_pending(pos, "재주문 실패")
                return
        pos.entry_order_id = res.order_id
        if res.status == "FILLED" and pos.status is PositionStatus.PENDING and pos.filled_qty <= 0:
            await self.on_fill(Fill(pos.symbol, pos.side.order_side, res.filled_qty, res.avg_price,
                                    res.filled_qty * res.avg_price * self.s.taker_fee, now, cid, res.order_id, kind="ENTRY"))
        self._save(pos)

    async def _cancel_pending(self, pos: Position, reason: str) -> None:
        try:
            await self.account.cancel_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
        except Exception:  # noqa: BLE001
            pass
        if pos.filled_qty > 0:  # partially filled: manage what we have
            pos.original_qty = pos.filled_qty
            return
        pos.status = PositionStatus.CANCELLED
        pos.exit_reason = reason
        pos.exit_time = self.clock()
        self._save(pos)
        self.positions.pop(pos.symbol, None)
        await self._emit("entry_cancelled", {"position": pos, "symbol": pos.symbol, "reason": reason})

    async def on_tick(self, symbol: str, mark: float, bid: float, ask: float) -> None:
        """High-frequency backstop: enforce stops in software if the exchange stop is missing or late."""
        pos = self.positions.get(symbol)
        if pos is None or pos.status is not PositionStatus.OPEN or not mark:
            return
        fav = (mark - pos.entry_price) * pos.side.sign
        if fav > pos.max_fav:
            pos.max_fav = fav
        elif -fav > pos.max_adv:
            pos.max_adv = -fav
        through = (pos.side is Side.LONG and mark <= pos.stop * (1 - 0.0012)) or (pos.side is Side.SHORT and mark >= pos.stop * (1 + 0.0012))
        if through and self.clock() - pos.last_stop_update > 2500:
            await self.close_position(symbol, "소프트웨어 백스톱 손절")

    # --- closing -------------------------------------------------------------------------------
    async def close_position(self, symbol: str, reason: str) -> bool:
        pos = self.positions.get(symbol)
        if pos is None:
            return False
        if pos.status is PositionStatus.PENDING:
            await self._cancel_pending(pos, reason)
            return True
        if pos.status is not PositionStatus.OPEN:
            return False
        pos.status = PositionStatus.CLOSING
        pos.exit_reason = reason
        await self._cancel_brackets(pos)
        info = self.symbols[symbol]
        qty = norm_qty(info, pos.qty) or pos.qty
        res = await self.account.market_order(symbol, pos.side.close_order_side, qty, reduce_only=True,
                                              client_id=client_order_id(f"C{self.tag}"))
        if res.status == "REJECTED":
            # nothing left on the exchange -> treat as closed at mark
            snap = (await self.account.get_positions()).get(symbol)
            if snap is None or abs(snap.qty) <= 0:
                price = pos.extra.get("last_mark") or pos.entry_price
                await self._finalize(pos, price, self.clock(), reason + " (거래소 포지션 없음)")
                return True
            pos.status = PositionStatus.OPEN
            await self._place_brackets(pos)
            await self._emit("error", {"message": f"{symbol} 청산 주문 거부: {res.raw}"})
            return False
        if res.status == "FILLED" and pos.status is PositionStatus.CLOSING and res.filled_qty > 0 and not self.account.is_paper:
            await self.on_fill(Fill(symbol, pos.side.close_order_side, res.filled_qty, res.avg_price,
                                    res.filled_qty * res.avg_price * self.s.taker_fee, self.clock(), "", res.order_id,
                                    reduce_only=True, kind="CLOSE"))
        return True

    async def close_all(self, reason: str) -> int:
        n = 0
        for sym in list(self.positions):
            if await self.close_position(sym, reason):
                n += 1
        return n

    async def _finalize(self, pos: Position, exit_price: float, ts: int, reason: str) -> None:
        if pos.status is PositionStatus.CLOSED:
            return
        pos.status = PositionStatus.CLOSED
        pos.exit_price = exit_price
        pos.exit_time = ts
        pos.exit_reason = reason
        pos.qty = 0.0
        net = pos.net_pnl()
        pos.r_multiple = net / pos.risk_amount if pos.risk_amount else 0.0
        await self._cancel_brackets(pos)
        self._save(pos)
        self.positions.pop(pos.symbol, None)
        rec = TradeRecord(position_id=pos.id, engine=self.name, symbol=pos.symbol, side=pos.side.value, alpha=pos.alpha,
                          alphas=list(pos.alphas), regime=pos.regime, entry_time=pos.entry_time, exit_time=ts,
                          entry_price=pos.entry_price, exit_price=exit_price, qty=pos.filled_qty or pos.original_qty,
                          notional=pos.notional, pnl=net, gross=pos.realized, fees=pos.fees, funding=pos.funding,
                          r_multiple=pos.r_multiple, risk_amount=pos.risk_amount, exit_reason=reason,
                          confidence=pos.confidence, params_version=pos.params_version, bars_held=pos.bars_held,
                          max_fav_r=pos.max_fav / pos.r_unit if pos.r_unit else 0.0,
                          max_adv_r=pos.max_adv / pos.r_unit if pos.r_unit else 0.0, reason=pos.reason)
        self.closed.append(rec)
        if len(self.closed) > 5000:
            self.closed = self.closed[-2500:]
        if self.persist:
            self.store.save_trade(rec)
        self.stats.total_trades += 1
        self.stats.realized_today += net
        self.stats.trades_today += 1
        self.stats.wins_today += 1 if net > 0 else 0
        self.stats.last_trade_ts = ts
        if net < 0:
            self.risk.register_loss(pos.symbol, ts)
        self.bandit.update(pos.alpha, pos.regime, pos.r_multiple, live=not self.account.is_paper)
        self._equity_cache = (0, 0.0)
        await self._emit("position_closed", {"position": pos, "trade": rec, "symbol": pos.symbol})

    def reset_daily(self) -> None:
        self.stats.realized_today = 0.0
        self.stats.trades_today = 0
        self.stats.wins_today = 0

    # --- reconciliation (live) -----------------------------------------------------------------
    async def reconcile(self) -> None:
        if self.account.is_paper:
            return
        try:
            ex_positions = await self.account.get_positions()
            algos = await self.account.open_algo_orders()
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] reconcile skipped: %s", self.name, e)
            return
        algo_by_symbol: dict[str, list[dict]] = {}
        for a in algos:
            algo_by_symbol.setdefault(a.get("symbol", ""), []).append(a)
        now = self.clock()
        # positions we think are open
        for sym, pos in list(self.positions.items()):
            if pos.status is not PositionStatus.OPEN:
                continue
            snap = ex_positions.get(sym)
            if snap is None or abs(snap.qty) <= 0:
                mark = pos.extra.get("last_mark") or pos.stop
                await self._finalize(pos, mark, now, "거래소에서 청산 확인(reconcile)")
                await self._emit("reconcile", {"message": f"{sym} 포지션이 거래소에서 이미 종료되어 정리했습니다"})
                continue
            if (snap.qty > 0) != (pos.side is Side.LONG):
                await self._finalize(pos, snap.mark or pos.entry_price, now, "방향 불일치(reconcile)")
                continue
            if abs(abs(snap.qty) - pos.qty) > self.symbols[sym].step_size * 0.5:
                pos.qty = abs(snap.qty)
                pos.notional = pos.qty * pos.entry_price
            has_stop = any(a.get("orderType", a.get("type")) == "STOP_MARKET" for a in algo_by_symbol.get(sym, []))
            if not has_stop:
                log.warning("[%s] %s has no exchange stop, re-placing", self.name, sym)
                try:
                    pos.sl_algo_id = await self.account.place_stop(sym, pos.side.close_order_side, pos.stop, close_position=True,
                                                                   client_id=client_order_id(f"S{self.tag}"))
                    pos.last_stop_update = now
                except Exception as e:  # noqa: BLE001
                    await self._emit("error", {"message": f"{sym} 손절 재설정 실패: {e}"})
            self._save(pos)
        # positions on the exchange we do not know about -> adopt with a protective stop
        for sym, snap in ex_positions.items():
            if sym in self.positions or sym not in self.symbols:
                continue
            side = Side.LONG if snap.qty > 0 else Side.SHORT
            view = self.last_views.get(sym)
            atr = view.tf("5m").v("atr") if view and view.tf("5m").ok else snap.entry_price * 0.01
            if not atr or math.isnan(atr):
                atr = snap.entry_price * 0.01
            stop = norm_price(self.symbols[sym], snap.entry_price - side.sign * 2.0 * atr)
            pos = Position(id=short_id("P"), engine=self.name, symbol=sym, side=side, qty=abs(snap.qty),
                           entry_price=snap.entry_price, entry_time=now, stop=stop, take_profit=None, tp1=None,
                           initial_stop=stop, alpha="adopted", alphas=["adopted"], reason="거래소에서 발견된 외부 포지션을 인수",
                           confidence=0.0, regime="RANGE", risk_amount=abs(snap.qty) * 2.0 * atr, r_unit=2.0 * atr,
                           notional=abs(snap.qty) * snap.entry_price, leverage=snap.leverage or self.s.exchange_leverage,
                           params_version=self.params.version, atr=atr, trail_atr_mult=2.5, max_hold_bars=0,
                           status=PositionStatus.OPEN, filled_qty=abs(snap.qty), original_qty=abs(snap.qty),
                           extra={"tp1_frac": 0.0})
            self.positions[sym] = pos
            if not any(a.get("orderType", a.get("type")) == "STOP_MARKET" for a in algo_by_symbol.get(sym, [])):
                await self._place_brackets(pos)
            self._save(pos)
            await self._emit("reconcile", {"message": f"{sym} {side.value} 외부 포지션 인수, 보호 손절 {stop}"})
        # zombie algo orders on flat symbols
        for sym, lst in algo_by_symbol.items():
            if sym and sym not in ex_positions and sym not in self.positions and lst:
                try:
                    await self.account.cancel_all(sym)
                    log.info("[%s] cancelled %d orphan algo orders on %s", self.name, len(lst), sym)
                except Exception:  # noqa: BLE001
                    pass

    # --- snapshot for UIs ----------------------------------------------------------------------
    def snapshot(self, marks: dict[str, float] | None = None) -> dict:
        marks = marks or {}
        pos_rows = []
        for p in self.open_positions():
            m = marks.get(p.symbol) or p.extra.get("last_mark") or p.entry_price
            pos_rows.append({"id": p.id, "symbol": p.symbol, "side": p.side.value, "status": p.status.value, "qty": p.qty,
                             "entry": p.entry_price, "mark": m, "stop": p.stop, "tp": p.take_profit, "tp1": p.tp1,
                             "tp1_done": p.tp1_done, "unrealized": p.unrealized(m) if p.status is PositionStatus.OPEN else 0.0,
                             "pnl_pct": p.pnl_pct(m), "r": p.r_now(m), "alpha": p.alpha, "alphas": p.alphas,
                             "reason": p.reason, "regime": p.regime, "confidence": p.confidence,
                             "entry_time": p.entry_time, "bars_held": p.bars_held, "notional": p.notional,
                             "leverage": p.leverage, "risk": p.risk_amount, "expected_profit": p.expected_profit})
        return {"name": self.name, "paper": self.account.is_paper, "params_version": self.params.version,
                "equity": self.stats.equity, "balance": self.stats.balance, "start_equity": self.stats.start_equity,
                "realized_today": self.stats.realized_today, "trades_today": self.stats.trades_today,
                "wins_today": self.stats.wins_today, "total_trades": self.stats.total_trades,
                "paused": self.risk.state.paused, "halted": self.risk.state.halted_until > self.clock(),
                "halt_reason": self.risk.state.halt_reason, "risk_scale": self.risk.state.risk_scale,
                "positions": pos_rows, "skipped": self.stats.skipped, "entries_enabled": self.entries_enabled}
