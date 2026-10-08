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
from heartless.exchange.base import Account, OrderResult
from heartless.exchange.symbols import norm_price, norm_qty
from heartless.learning.bandit import AlphaBandit
from heartless.strategy.base import Context
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from heartless.strategy.risk import RiskManager
from heartless.util.ids import client_order_id, short_id
from heartless.util.timeutil import MS_MINUTE, now_ms

log = logging.getLogger(__name__)

# a reduce-only close that has not been confirmed within this window may be re-sent (on_tick / reconcile / owner)
CLOSE_RETRY_MS = 10_000
# order states after which executedQty can no longer change
TERMINAL_ORDER_STATES = ("FILLED", "CANCELED", "EXPIRED", "REJECTED")


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
                 notify: bool = False, meta=None, meta_learn: bool = True):
        self.name = name
        self.meta = meta  # MetaLabeler or None (heartless/learning/metalabel.py)
        self.meta_learn = meta_learn
        self.account = account
        self.params = params
        self.s = settings
        self.symbols = symbols
        self.store = store
        self.bus = bus
        self.bandit = bandit or AlphaBandit([a for a in params.alphas])
        self.ensemble = Ensemble(params, self.bandit, only_alpha=only_alpha)
        # store-backed whenever this engine persists (paper/live/challengers): loss halts, the drawdown pause, the weekly
        # risk scale and symbol cool-downs survive a restart; backtests (persist=False) keep a memory-only RiskState
        self.risk = RiskManager(settings, store if (persist and store is not None) else None, name)
        self.clock = clock or now_ms
        self.persist = persist and store is not None
        self.notify = notify
        self.positions: dict[str, Position] = {}
        self.stats = EngineStats()
        self.closed: list[TradeRecord] = []  # in-memory (backtests + recent)
        self.account.on_fill(self.on_fill)
        if hasattr(self.account, "on_algo_event"):
            self.account.on_algo_event(self.on_algo_event)
        self.last_views: dict[str, MarketView] = {}
        self._equity_cache: tuple[int, float] = (0, 0.0)
        self.entries_enabled = True
        self.tag = "".join(ch for ch in name.upper() if ch.isalnum())[:4]
        self._last_exit_ts: dict[str, int] = {}  # symbol -> clock() of the last finalize/cancel (reconcile freshness)

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
        meta_x = None
        if self.meta is not None:
            meta_x = self._meta_vector(view, ctx, d)
            allow, mult, er = self.meta.decide(sig.alpha, meta_x)
            if not allow:
                self.stats.skipped["meta veto"] = self.stats.skipped.get("meta veto", 0) + 1
                return
            d.size_mult *= mult
            if er is not None:
                d.reason += f" | 메타모델 기대 {er:+.2f}R"
        open_notional = sum(p.notional for p in self.open_positions())
        sz = self.risk.size(d, equity, info, entry_ref, open_notional)
        if sz.qty <= 0:
            self.stats.skipped["size"] = self.stats.skipped.get("size", 0) + 1
            return
        stop = norm_price(info, sig.stop)
        tp = norm_price(info, sig.take_profit) if sig.take_profit else None
        tp1 = norm_price(info, sig.tp1) if sig.tp1 else None
        # Bracket sanity against the price we will actually trade at (the signal's levels were computed from the bar
        # close; the touch can have moved since). A stop at or through the reference is already hit: Binance rejects
        # the STOP_MARKET (-2021), the software backstop flattens at once and the trade is a fee-only round trip with
        # seconds of no exchange stop. A take-profit behind the reference would self-trigger likewise, so drop it.
        if (stop - entry_ref) * d.side.sign >= 0:
            self.stats.skipped["stop"] = self.stats.skipped.get("stop", 0) + 1
            return
        if tp is not None and (tp - entry_ref) * d.side.sign <= 0:
            tp = None
        if tp1 is not None and ((tp1 - entry_ref) * d.side.sign <= 0 or (tp is not None and abs(tp1 - entry_ref) >= abs(tp - entry_ref))):
            tp1 = None
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
        if meta_x is not None:
            pos.extra["meta_x"] = [None if not (v == v) else float(v) for v in meta_x]
        self.positions[d.symbol] = pos
        self._save(pos)
        sent = False  # becomes True once an order call was reached (it may have been accepted despite an exception)
        try:
            await self.account.prepare_symbol(d.symbol, sz.leverage)
            sent = True
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
            if sent:
                # a transport error / timeout can hit AFTER Binance accepted the order: ask before forgetting it
                try:
                    chk = await self.account.query_order(d.symbol, "", cid)
                except Exception as e2:  # noqa: BLE001
                    log.warning("[%s] cannot verify entry %s after failure (%s); keeping it pending", self.name, d.symbol, e2)
                    self._save(pos)
                    return  # stays PENDING: _manage_pending verifies / cancels it by client id on later bars
                if chk.status in ("NEW", "PARTIALLY_FILLED", "FILLED"):
                    res = chk
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
            self._mark_settled(pos, res.order_id)
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

    @staticmethod
    def _mark_settled(pos: Position, order_id: str) -> None:
        """Record that an order's final executedQty has been credited from a REST query/result: every later stream
        fill for that order id is a duplicate."""
        if order_id:
            settled = pos.extra.setdefault("settled_orders", [])
            if order_id not in settled:
                settled.append(order_id)

    def _gone(self, pos: Position) -> bool:
        """True once a position stopped being the live OPEN position for its symbol (exited / replaced during an await)."""
        return pos.status is not PositionStatus.OPEN or self.positions.get(pos.symbol) is not pos

    async def _on_entry_fill(self, pos: Position, fill: Fill) -> None:
        if (fill.client_id and pos.entry_client_id and fill.client_id != pos.entry_client_id and fill.client_id.startswith("HLE")
                and fill.client_id not in pos.extra.get("entry_cids", [])):
            return  # a different (stale) entry order
        if fill.order_id and fill.order_id in pos.extra.get("settled_orders", []):
            return  # this order's final executed quantity was already credited from its REST state
        if pos.status is PositionStatus.CLOSED:
            # the entry's remainder filled under the cancel that _finalize sent: that quantity is live on the
            # exchange and no longer tracked here. reconcile adopts it with a protective stop on its next pass; say
            # so, because until then it is protected by nothing but that pass.
            log.warning("[%s] late entry fill %s %s after close; reconcile will adopt it", self.name, fill.symbol, fill.qty)
            await self._emit("error", {"message": f"{fill.symbol} 청산 후 진입 잔량 {fill.qty} 체결 — 다음 reconcile 에서 인수/보호"})
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
        pos.qty += fill.qty  # not `total`: a late entry fill must not resurrect quantity already exited
        pos.fees += fill.fee
        pos.notional = pos.qty * pos.entry_price
        # recompute R geometry from the real fills (every fill: a market order routinely executes as several trades);
        # measured against the initial stop so a late fill after a breakeven/trailing move cannot shrink the R unit
        pos.r_unit = abs(pos.entry_price - (pos.initial_stop or pos.stop)) or pos.r_unit
        pos.risk_amount = pos.filled_qty * (pos.r_unit + pos.entry_price * (2 * self.s.taker_fee + 0.0003))
        if pos.status is PositionStatus.PENDING:
            pos.status = PositionStatus.OPEN
            pos.entry_time = fill.ts or self.clock()
            await self._place_brackets(pos)
            self._save(pos)
            await self._emit("position_opened", {"position": pos, "symbol": pos.symbol})
        else:
            # additional partial fill of a limit entry: brackets must cover the new quantity (not while closing)
            if pos.status is PositionStatus.OPEN:
                await self._refresh_brackets(pos)
            self._save(pos)

    async def _place_brackets(self, pos: Position) -> None:
        if self._gone(pos):
            return
        info = self.symbols[pos.symbol]
        close_side = pos.side.close_order_side
        try:
            new_id = await self.account.place_stop(pos.symbol, close_side, pos.stop, qty=pos.qty, close_position=True,
                                                   client_id=client_order_id(f"S{self.tag}"))
        except Exception as e:  # noqa: BLE001
            new_id = None
            log.error("[%s] failed to place stop for %s: %s", self.name, pos.symbol, e)
            await self._emit("error", {"message": f"{pos.symbol} 손절 주문 실패 ({e}). 봇 내부 백스톱으로 보호 중"})
        if self._gone(pos):  # exited while the order was in flight: never leave a zombie closePosition stop behind
            await self._cancel_algo_quiet(pos.symbol, new_id)
            return
        if new_id is not None:
            pos.sl_algo_id = new_id
            pos.last_stop_update = self.clock()
        frac = float(pos.extra.get("tp1_frac", 0.5) or 0.0)
        placed_tp1 = 0.0  # TP1 quantity placed in THIS call (never a stale value from a previous bracket set)
        pos.extra.pop("tp1_qty", None)
        if pos.tp1 and not pos.tp1_done and frac > 0:
            q1 = norm_qty(info, pos.qty * frac)
            rest = norm_qty(info, pos.qty - q1)
            if q1 > 0 and rest > 0:
                try:
                    tp1_id = await self.account.place_take_profit(pos.symbol, close_side, pos.tp1, q1,
                                                                  client_id=client_order_id(f"T{self.tag}"))
                except Exception as e:  # noqa: BLE001
                    tp1_id = None
                    log.warning("[%s] TP1 placement failed for %s: %s", self.name, pos.symbol, e)
                if self._gone(pos):
                    await self._cancel_algo_quiet(pos.symbol, tp1_id)
                    return
                if tp1_id is not None:
                    pos.extra["tp1_algo_id"] = tp1_id
                    pos.extra["tp1_qty"] = q1
                    placed_tp1 = q1
            else:
                pos.tp1 = None
        if pos.take_profit:
            q_tp = norm_qty(info, pos.qty - placed_tp1)
            if q_tp > 0:
                try:
                    tp_id = await self.account.place_take_profit(pos.symbol, close_side, pos.take_profit, q_tp,
                                                                 client_id=client_order_id(f"T{self.tag}"))
                except Exception as e:  # noqa: BLE001
                    tp_id = None
                    log.warning("[%s] TP placement failed for %s: %s", self.name, pos.symbol, e)
                if self._gone(pos):
                    await self._cancel_algo_quiet(pos.symbol, tp_id)
                    return
                if tp_id is not None:
                    pos.tp_algo_id = tp_id

    async def _cancel_algo_quiet(self, symbol: str, algo_id: str | None) -> None:
        if not algo_id:
            return
        try:
            await self.account.cancel_algo(symbol, algo_id)
        except Exception:  # noqa: BLE001
            pass

    async def _ensure_stop(self, pos: Position) -> None:
        """Make sure an OPEN position has an exchange stop (used when a close failed and the position is handed back)."""
        if pos.sl_algo_id:
            return
        try:
            pos.sl_algo_id = await self.account.place_stop(pos.symbol, pos.side.close_order_side, pos.stop, qty=pos.qty,
                                                           close_position=True, client_id=client_order_id(f"S{self.tag}"))
            pos.last_stop_update = self.clock()
        except Exception as e:  # noqa: BLE001
            log.error("[%s] failed to restore stop for %s: %s", self.name, pos.symbol, e)
            await self._emit("error", {"message": f"{pos.symbol} 손절 재설정 실패 ({e}). 봇 내부 백스톱으로 보호 중"})

    def _meta_vector(self, view: MarketView, ctx: Context, d: Decision) -> list[float]:
        """Market context at signal time for the meta-labeler (same feature code as the discovery engine)."""
        import numpy as _np

        from heartless.learning.discovery import features_at
        from heartless.learning.metalabel import META_FEATURES, meta_vector

        feats: dict = {}
        try:
            arr = features_at(view.frames, None, _np.array([int(view.cursor_time)], dtype=_np.int64))
            feats = {n: float(arr[n][0]) for n in META_FEATURES if n in arr}
        except Exception:  # noqa: BLE001
            pass
        ex = ctx.extras or {}
        if ex and not ex.get("stale"):
            for n in META_FEATURES:
                if n.startswith("x.") and isinstance(ex.get(n[2:]), (int, float)):
                    feats[n] = float(ex[n[2:]])
        return meta_vector(feats, d.score, d.side.sign)

    async def on_algo_event(self, event: dict) -> None:
        """The exchange reports a conditional order's status. If our protective stop was rejected or expired
        without filling (margin check, price protection, system cancel), re-arm it right away instead of waiting for
        the next reconcile; a TP that died is simply forgotten (the stop still protects the position)."""
        pos = self.positions.get(event.get("symbol", ""))
        if pos is None or pos.status is not PositionStatus.OPEN:
            return
        status = event.get("status", "")
        if status not in ("REJECTED", "EXPIRED", "CANCELED"):
            return
        aid = event.get("algo_id", "")
        if aid and aid == pos.sl_algo_id:
            if status == "CANCELED" and self.clock() - pos.last_stop_update < 5_000:
                return  # our own replace/cancel sequence
            log.warning("[%s] exchange %s our stop on %s; re-arming", self.name, status.lower(), pos.symbol)
            pos.sl_algo_id = ""
            await self._ensure_stop(pos)
            self._save(pos)
            await self._emit("reconcile", {"message": f"{pos.symbol} 손절 주문이 거래소에서 {status} 되어 즉시 재설정했습니다"})
        elif aid and aid == pos.tp_algo_id and status in ("REJECTED", "EXPIRED"):
            pos.tp_algo_id = ""
            self._save(pos)
        elif aid and aid == pos.extra.get("tp1_algo_id") and status in ("REJECTED", "EXPIRED"):
            pos.extra.pop("tp1_algo_id", None)
            self._save(pos)

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
        if pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING):
            # a PENDING row holds nothing yet (the first entry fill flips it to OPEN), so a reducing fill on its
            # symbol belongs to something else: a late event of the previous trade (already finalized) or an
            # external position (reconcile's business). Booking it here would record a phantom trade against the
            # entry reference and cancel a working entry.
            if pos.status is PositionStatus.PENDING:
                log.warning("[%s] ignoring reducing fill %s %s %s on a pending entry", self.name, fill.symbol,
                            fill.order_side, fill.qty)
            return
        qty = min(fill.qty, pos.qty) if pos.qty > 0 else fill.qty
        if qty <= 0:
            return
        pnl = (fill.price - pos.entry_price) * qty * pos.side.sign
        pos.realized += pnl
        # a REST-synthesized close carries the whole order's fee; when part of that order was already booked from
        # stream trades only the uncredited share of the fee belongs here (otherwise fees are counted twice)
        pos.fees += fill.fee * (qty / fill.qty) if fill.qty > qty > 0 else fill.fee
        pos.qty = max(pos.qty - qty, 0.0)
        kind = fill.kind
        # the engine's own close orders (close_position sets CLOSING + exit_reason before sending; live ids start
        # with HLC, the REST-synthesized fill has no id) already carry the true reason: never relabel them
        engine_close = pos.status is PositionStatus.CLOSING or (fill.client_id or "").startswith("HLC")
        if kind in ("UNKNOWN", "CLOSE") and fill.reduce_only and not engine_close:
            # infer from price relative to brackets
            if pos.tp1 and not pos.tp1_done and abs(fill.price - pos.tp1) <= abs(fill.price - pos.stop):
                kind = "TP"
            elif (pos.side is Side.LONG and fill.price <= pos.stop * 1.002) or (pos.side is Side.SHORT and fill.price >= pos.stop * 0.998):
                kind = "SL"
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        remaining = pos.qty > step * 0.5 + 1e-12
        if remaining and self._is_dust(pos, fill.price):
            # a leftover below the exchange minimum (rounding residue, partial bracket fill) is not a position
            # worth managing and no bracket may cover it: flatten it now instead of holding it indefinitely
            main = {"SL": "손절(SL)" if not pos.be_moved else "본절 손절(BE)", "TP": "익절(TP)"}.get(kind, pos.exit_reason or "청산")
            self._save(pos)
            await self.close_position(pos.symbol, f"{main} (+잔량 정리)")
            return
        if remaining and kind == "TP" and pos.tp1 and not pos.tp1_done and not engine_close:
            pos.tp1_done = True
            pos.extra.pop("tp1_algo_id", None)
            pos.extra.pop("tp1_qty", None)
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

    def _is_dust(self, pos: Position, price: float) -> bool:
        info = self.symbols.get(pos.symbol)
        if info is None or pos.qty <= 0:
            return False
        px = price or pos.entry_price
        return pos.qty < info.min_qty * 0.999 or (px > 0 and pos.qty * px < info.min_notional * 0.999) or \
            (pos.original_qty > 0 and pos.qty < pos.original_qty * 0.02 and pos.qty <= info.step_size * 1.001)

    async def _move_stop_to_breakeven(self, pos: Position) -> None:
        info = self.symbols[pos.symbol]
        buffer = pos.entry_price * (self.s.taker_fee * 2 + 0.0002)
        be = pos.entry_price + pos.side.sign * buffer
        be = norm_price(info, be)
        better = be > pos.stop if pos.side is Side.LONG else be < pos.stop
        if better:
            if await self._replace_stop(pos, be):
                pos.be_moved = True
            # on failure leave be_moved False so the next bar retries instead of pretending protection exists
        else:
            pos.be_moved = True  # stop already at/above breakeven

    async def _replace_stop(self, pos: Position, new_stop: float) -> bool:
        """Move the exchange stop. Returns True when the position is protected at new_stop (or already better)."""
        info = self.symbols[pos.symbol]
        new_stop = norm_price(info, new_stop)
        if abs(new_stop - pos.stop) < info.tick_size * 0.5:
            return True
        if self._gone(pos):
            return False
        old_id = pos.sl_algo_id
        try:
            new_id = await self.account.place_stop(pos.symbol, pos.side.close_order_side, new_stop, qty=pos.qty,
                                                   close_position=True, client_id=client_order_id(f"S{self.tag}"))
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] replace stop failed on %s: %s", self.name, pos.symbol, e)
            return False
        if self._gone(pos):
            # the position exited (old stop / TP / manual close) while the new stop was in flight: _finalize already
            # cancelled the old ids, so the new closePosition stop would otherwise live on as a zombie
            await self._cancel_algo_quiet(pos.symbol, new_id)
            return False
        pos.sl_algo_id = new_id
        pos.stop = new_stop
        pos.last_stop_update = self.clock()
        if old_id:
            try:
                await self.account.cancel_algo(pos.symbol, old_id)
            except Exception:  # noqa: BLE001
                pass
        self._save(pos)
        return True

    # --- management ----------------------------------------------------------------------------
    async def _manage(self, pos: Position, view: MarketView, ctx: Context) -> None:
        now = self.clock()
        price = ctx.ticker.ref or view.price
        if pos.status is PositionStatus.PENDING:
            await self._manage_pending(pos, view, ctx)
            return
        if pos.status is not PositionStatus.OPEN or not price:
            return
        # a post-only entry that opened on a partial fill keeps its remainder working for two bars at most
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        if (pos.entry_style == EntryStyle.LIMIT.value and pos.filled_qty < pos.original_qty - step * 0.5 - 1e-12
                and now - pos.pending_since >= 2 * MS_MINUTE):
            await self._cancel_entry_remainder(pos)
            if pos.status is not PositionStatus.OPEN:
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
            if r_now < 0.5 or pos.bars_held >= 3 * pos.max_hold_bars:
                # not working at the deadline, or a winner held 3x past it: a scalp must not turn into a swing
                reason = "시간 초과 청산(time stop)" if r_now < 0.5 else "최대 보유시간 초과 청산"
                await self.close_position(pos.symbol, reason)
                return
            if not pos.be_moved:
                await self._move_stop_to_breakeven(pos)
            if pos.trail_atr_mult <= 0:
                # a working trade past its deadline keeps running only under a protective trail
                pos.trail_atr_mult = 1.5
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
            # a market order whose result never came back (status UNKNOWN / transport error) is verified on the very
            # next bar: if it did execute the position sits on the exchange without a stop until its fill is credited
            if age_bars >= 1:
                res = await self._credit_entry_fills(pos)  # credits a fill -> OPEN + brackets
                if res is None or pos.status is not PositionStatus.PENDING:
                    return  # unreachable (retry next bar) or it filled
                if res.status in ("CANCELED", "EXPIRED", "REJECTED", "UNKNOWN"):
                    await self._cancel_pending(pos, f"entry {res.status.lower()}")
            return
        # (a partially filled limit entry is OPEN, not PENDING: its remainder is handled in _manage / close / finalize)
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
        res = await self._settle_entry_order(pos)  # cancel the resting order and credit anything that filled meanwhile
        if pos.status is not PositionStatus.PENDING:
            return
        if res is None:
            return  # exchange unreachable: the old order may still rest, so do not quote a second one; retry next bar
        price = norm_price(info, touch, pos.side.order_side)
        cid = client_order_id(f"E{self.tag}")
        # remember every entry id this position ever used so fills are recognised whichever order they belong to,
        # whether they arrive before the REST response (new id) or late (old id, de-duplicated via settled_orders)
        cids = pos.extra.setdefault("entry_cids", [])
        for c in (pos.entry_client_id, cid):
            if c and c not in cids:
                cids.append(c)
        pos.requotes += 1  # counts attempts, so a flaky exchange cannot keep a position re-quoting forever
        rem = pos.original_qty - pos.filled_qty
        try:
            res = await self.account.limit_order(pos.symbol, pos.side.order_side, rem, price, post_only=True, client_id=cid)
            if res.status in ("EXPIRED", "REJECTED"):
                res = await self.account.market_order(pos.symbol, pos.side.order_side, rem, client_id=cid)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] re-quote failed for %s: %s", self.name, pos.symbol, e)
            await self._emit("error", {"message": f"{pos.symbol} 재주문 실패: {e}"})
            # the old order is already cancelled; the newest id is the only one that might rest on the exchange
            pos.entry_client_id = cid
            pos.entry_order_id = ""
            try:
                res = await self.account.query_order(pos.symbol, "", cid)
            except Exception:  # noqa: BLE001
                res = None
            if res is None or res.status not in ("NEW", "PARTIALLY_FILLED", "FILLED"):
                self._save(pos)
                return  # nothing confirmed under the new id: the next bar settles it and tries again
        if res.status == "REJECTED":
            await self._cancel_pending(pos, "재주문 실패")
            return
        pos.entry_client_id = cid
        pos.entry_order_id = res.order_id
        pos.pending_since = now
        pos.limit_price = price
        new_stop = pos.stop  # keep the structural stop; R shrinks/grows with the new price
        pos.r_unit = abs(price - new_stop)
        pos.entry_price = price
        if res.status == "FILLED" and pos.status is PositionStatus.PENDING and pos.filled_qty <= 0:
            await self.on_fill(Fill(pos.symbol, pos.side.order_side, res.filled_qty, res.avg_price,
                                    res.filled_qty * res.avg_price * self.s.taker_fee, now, cid, res.order_id, kind="ENTRY"))
            self._mark_settled(pos, res.order_id)
        self._save(pos)

    async def _settle_entry_order(self, pos: Position) -> OrderResult | None:
        """Cancel the working entry order, then read its final state and credit any fill that raced the cancel.

        Returns the query result, or None when the exchange could not be asked (the order may then still rest).
        Once the order is terminal its id is marked settled so a late stream fill for it cannot be double counted."""
        try:
            await self.account.cancel_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] cancel entry order on %s failed: %s", self.name, pos.symbol, e)
        return await self._credit_entry_fills(pos)

    async def _credit_entry_fills(self, pos: Position) -> OrderResult | None:
        """Read the entry order's state from the exchange and credit any executed quantity the engine has not seen
        (a fill whose stream event was missed: restart, user-stream gap, REST timeout). Crediting a first fill turns
        the PENDING row into an OPEN position and places its brackets. Returns the order state, or None when the
        exchange could not be asked. A terminal order is marked settled so a late stream fill cannot double count."""
        if not (pos.entry_order_id or pos.entry_client_id):
            return None
        try:
            res = await self.account.query_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] query entry order on %s failed: %s", self.name, pos.symbol, e)
            return None
        oid = res.order_id or pos.entry_order_id
        if res.filled_qty > pos.filled_qty + 1e-12:
            delta = res.filled_qty - pos.filled_qty
            fee_rate = self.s.maker_fee if pos.entry_style == EntryStyle.LIMIT.value else self.s.taker_fee
            await self.on_fill(Fill(pos.symbol, pos.side.order_side, delta, res.avg_price, delta * res.avg_price * fee_rate,
                                    self.clock(), pos.entry_client_id, oid, kind="ENTRY"))
        if res.status in TERMINAL_ORDER_STATES:
            self._mark_settled(pos, oid)
        return res

    async def _cancel_entry_remainder(self, pos: Position) -> None:
        """Cancel the unfilled rest of a partially filled entry so it cannot fill into an unmanaged position later."""
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        if not (pos.entry_order_id or pos.entry_client_id) or pos.filled_qty >= pos.original_qty - step * 0.5 - 1e-12:
            return
        res = await self._settle_entry_order(pos)
        if res is not None and res.status not in ("NEW", "PARTIALLY_FILLED"):
            pos.original_qty = pos.filled_qty  # nothing rests any more: the position is what was filled
            self._save(pos)

    async def _cancel_pending(self, pos: Position, reason: str) -> None:
        res = await self._settle_entry_order(pos)
        if pos.status is not PositionStatus.PENDING:
            return  # it filled under the cancel (in either ordering): it is a live position now, brackets placed
        if res is None:
            # neither the cancel nor the query could be confirmed: keep it pending rather than forgetting an order
            # that may still fill; _manage_pending retries on the next bar
            await self._emit("error", {"message": f"{pos.symbol} 진입 주문 취소 확인 실패; 다음 봉에 재시도"})
            return
        if pos.filled_qty > 0:  # partially filled: manage what we have
            pos.original_qty = pos.filled_qty
            return
        pos.status = PositionStatus.CANCELLED
        pos.exit_reason = reason
        pos.exit_time = self.clock()
        self._last_exit_ts[pos.symbol] = self.clock()
        self._save(pos)
        self.positions.pop(pos.symbol, None)
        await self._emit("entry_cancelled", {"position": pos, "symbol": pos.symbol, "reason": reason})

    async def on_tick(self, symbol: str, mark: float, bid: float, ask: float) -> None:
        """High-frequency backstop: enforce stops in software if the exchange stop is missing or late."""
        pos = self.positions.get(symbol)
        if pos is None or pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING) or not mark:
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
            if pos.status is PositionStatus.CANCELLED:
                return True
            if pos.status is not PositionStatus.OPEN:
                return False  # still pending (exchange unreachable); retried on the next bar
            # the entry filled under the cancel: flatten it like any open position
        if pos.status is PositionStatus.CLOSING:
            if self.clock() - int(pos.extra.get("close_sent", 0) or 0) < CLOSE_RETRY_MS:
                return False  # a close order is already in flight; its fill (or reconcile) settles it
        elif pos.status is not PositionStatus.OPEN:
            return False
        pos.status = PositionStatus.CLOSING
        pos.exit_reason = reason
        pos.extra["close_sent"] = self.clock()
        self._save(pos)  # a restart must see the close attempt (reconcile settles a CLOSING row either way)
        info = self.symbols[symbol]
        cid = client_order_id(f"C{self.tag}")
        # Send the reduce-only close FIRST and cancel the brackets only once it is accepted: if the order fails
        # (transport error, timeout, rejection) the exchange stop is still protecting the position. A closePosition
        # stop that outlives the position is harmless (and _finalize / reconcile cancel it).
        try:
            await self._cancel_entry_remainder(pos)
            qty = norm_qty(info, pos.qty) or pos.qty
            res = await self.account.market_order(symbol, pos.side.close_order_side, qty, reduce_only=True, client_id=cid)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] close order failed for %s", self.name, symbol)
            return await self._close_failed(pos, str(e))
        if res.status not in ("FILLED", "NEW", "PARTIALLY_FILLED"):
            # REJECTED, or UNKNOWN (the account could not tell whether the send reached Binance): never assume flat
            return await self._close_failed(pos, f"{res.status} {res.raw}")
        if res.status == "FILLED" and pos.status is PositionStatus.CLOSING and res.filled_qty > 0 and not self.account.is_paper:
            await self.on_fill(Fill(symbol, pos.side.close_order_side, res.filled_qty, res.avg_price,
                                    res.filled_qty * res.avg_price * self.s.taker_fee, self.clock(), cid, res.order_id,
                                    reduce_only=True, kind="CLOSE"))
        if pos.status is PositionStatus.CLOSING:  # fill not processed yet: the brackets are no longer needed
            await self._cancel_brackets(pos)
            self._save(pos)
        return True

    async def _close_failed(self, pos: Position, why: str) -> bool:
        """A reduce-only close did not go through. The brackets were left untouched, so the position is still protected
        on the exchange; find out what is left there and either book the exit or hand the position back."""
        symbol = pos.symbol
        try:
            snap = (await self.account.get_positions()).get(symbol)
        except Exception as e:  # noqa: BLE001
            log.error("[%s] close failed for %s (%s) and positions could not be read: %s", self.name, symbol, why, e)
            await self._emit("error", {"message": f"{symbol} 청산 실패 ({why}); 포지션 조회도 실패 ({e}). 손절 유지, 재시도 예정"})
            return False  # stays CLOSING: on_tick / reconcile / the owner retry after CLOSE_RETRY_MS
        if pos.status is PositionStatus.CLOSED or self.positions.get(symbol) is not pos:
            return True  # an exit fill arrived meanwhile and finalized it
        if snap is None or abs(snap.qty) <= 0:
            # nothing left on the exchange -> treat as closed at mark (the remainder is booked in _finalize)
            price = pos.extra.get("last_mark") or pos.entry_price
            await self._finalize(pos, price, self.clock(), (pos.exit_reason or "청산") + " (거래소 포지션 없음)")
            return True
        pos.status = PositionStatus.OPEN
        await self._ensure_stop(pos)
        self._save(pos)
        await self._emit("error", {"message": f"{symbol} 청산 주문 실패 ({why}); 포지션 유지, 손절 유지"})
        return False

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
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        if pos.qty > step * 0.5 + 1e-12:
            # quantity that left the exchange without a fill we processed (reconcile, rejected close, restart):
            # book it at the exit estimate so the trade record, risk cool-down and learning see the real outcome
            pos.realized += (exit_price - pos.entry_price) * pos.qty * pos.side.sign
            pos.fees += pos.qty * exit_price * self.s.taker_fee
            pos.extra["exit_estimated"] = True
        pos.qty = 0.0
        net = pos.net_pnl()
        pos.r_multiple = net / pos.risk_amount if pos.risk_amount else 0.0
        self._last_exit_ts[pos.symbol] = self.clock()
        await self._cancel_entry_remainder(pos)
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
        if self.meta is not None and self.meta_learn and pos.extra.get("meta_x"):
            try:
                self.meta.update(pos.alpha, [float("nan") if v is None else v for v in pos.extra["meta_x"]], pos.r_multiple)
            except Exception:  # noqa: BLE001
                log.exception("[%s] meta-label update failed", self.name)
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
        snap_ts = self.clock()  # anything the engine finalized after this instant is newer than the snapshot
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
        flipped: set[str] = set()
        # positions we think are open (or that a close left hanging)
        for sym, pos in list(self.positions.items()):
            if pos.status is PositionStatus.PENDING:
                # an entry that executed while its fill event was missed (restart, stream gap): the exchange holds
                # the position, nothing protects it and the symbol is ours so it is never adopted below -> credit
                # the fill now (which places the brackets) instead of waiting for _manage_pending's next bar
                snap = ex_positions.get(sym)
                if snap is not None and abs(snap.qty) > 0 and (snap.qty > 0) == (pos.side is Side.LONG):
                    await self._credit_entry_fills(pos)
                continue
            if pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING):
                continue
            snap = ex_positions.get(sym)
            if snap is None or abs(snap.qty) <= 0:
                mark = pos.extra.get("last_mark") or pos.stop
                reason = (f"{pos.exit_reason} (reconcile)" if pos.status is PositionStatus.CLOSING and pos.exit_reason
                          else "거래소에서 청산 확인(reconcile)")
                await self._finalize(pos, mark, now, reason)
                await self._emit("reconcile", {"message": f"{sym} 포지션이 거래소에서 이미 종료되어 정리했습니다"})
                continue
            if (snap.qty > 0) != (pos.side is Side.LONG):
                stale_ids = {str(x) for x in (pos.sl_algo_id, pos.tp_algo_id, pos.extra.get("tp1_algo_id", "")) if x}
                await self._finalize(pos, snap.mark or pos.entry_price, now, "방향 불일치(reconcile)")
                # the ids _finalize just cancelled must not count as protection for the position adopted below
                algo_by_symbol[sym] = [a for a in algo_by_symbol.get(sym, []) if self._algo_id(a) not in stale_ids]
                flipped.add(sym)
                continue
            retry_close = False
            if pos.status is PositionStatus.CLOSING:
                if now - int(pos.extra.get("close_sent", 0) or 0) < CLOSE_RETRY_MS:
                    continue  # the close order is still in flight
                # the close never executed (or we could not tell): hand the position back and try again
                log.warning("[%s] %s is still on the exchange after a close attempt; retrying", self.name, sym)
                pos.status = PositionStatus.OPEN
                retry_close = True
            if abs(abs(snap.qty) - pos.qty) > self.symbols[sym].step_size * 0.5:
                pos.qty = abs(snap.qty)
                pos.notional = pos.qty * pos.entry_price
            rows = self._stop_rows(pos, algo_by_symbol.get(sym, []))
            stop_row = rows[0] if rows else None
            if stop_row is not None:
                if self._algo_id(stop_row) and pos.sl_algo_id != self._algo_id(stop_row):
                    pos.sl_algo_id = self._algo_id(stop_row)  # keep the id so _cancel_brackets can clean it up
                # exactly one engine stop protects a position: every other same-side STOP_MARKET is a leftover (a
                # bracket whose cancel failed, the stop of the previous trade on this symbol that survived because
                # the symbol was re-entered before the flat-symbol sweep) and would close the position at a level
                # nobody chose
                for a in rows[1:]:
                    await self._cancel_algo_quiet(sym, self._algo_id(a))
            else:
                log.warning("[%s] %s has no exchange stop, re-placing", self.name, sym)
                try:
                    pos.sl_algo_id = await self.account.place_stop(sym, pos.side.close_order_side, pos.stop, qty=pos.qty,
                                                                   close_position=True, client_id=client_order_id(f"S{self.tag}"))
                    pos.last_stop_update = now
                except Exception as e:  # noqa: BLE001
                    await self._emit("error", {"message": f"{sym} 손절 재설정 실패: {e}"})
            # a STOP_MARKET on the wrong side can never protect this position: it is a zombie from an earlier one
            if not getattr(self.account, "hedge_mode", False):
                for a in algo_by_symbol.get(sym, []):
                    if a.get("orderType", a.get("type")) == "STOP_MARKET" and a.get("side") != pos.side.close_order_side:
                        await self._cancel_algo_quiet(sym, self._algo_id(a))
            self._save(pos)
            if retry_close:
                await self.close_position(sym, pos.exit_reason or "청산 재시도(reconcile)")
        # positions on the exchange we do not know about -> adopt with a protective stop
        candidates = [sym for sym in ex_positions if sym not in self.positions and sym in self.symbols]
        fresh = ex_positions
        if candidates:
            # adoption is rare and must never act on a stale snapshot (a stop may have fired during the awaits above)
            try:
                fresh = await self.account.get_positions()
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] reconcile: could not refresh positions before adopting: %s", self.name, e)
                fresh = {}
        for sym in candidates:
            snap = fresh.get(sym)
            if snap is None or abs(snap.qty) <= 0 or sym in self.positions:
                continue
            if self._last_exit_ts.get(sym, 0) >= snap_ts and sym not in flipped:
                continue  # the engine closed this symbol during this pass: the snapshot predates that exit
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
            stop_row = self._protective_stop(pos, algo_by_symbol.get(sym, []))
            if stop_row is None:
                await self._place_brackets(pos)
            else:
                pos.sl_algo_id = self._algo_id(stop_row)
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
        # resting entry orders nobody owns (e.g. the rest of a partially filled entry after the position closed)
        try:
            orders = await self.account.open_orders()
        except Exception as e:  # noqa: BLE001
            log.debug("[%s] reconcile: open orders unavailable: %s", self.name, e)
            orders = []
        prefix = f"HLE{self.tag}"
        for o in orders or []:
            sym = o.get("symbol", "")
            cid = str(o.get("clientOrderId", "") or "")
            if not cid.startswith(prefix):
                continue
            pos = self.positions.get(sym)
            if pos is not None and pos.status in (PositionStatus.PENDING, PositionStatus.OPEN) and cid == pos.entry_client_id:
                continue  # the working entry of a live position (its remainder is managed by _manage / close)
            try:
                await self.account.cancel_order(sym, str(o.get("orderId", "") or ""), cid)
                log.info("[%s] cancelled orphan entry order %s on %s", self.name, cid, sym)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _algo_id(a: dict) -> str:
        return str(a.get("algoId", a.get("algo_id", "")) or "")

    def _stop_rows(self, pos: Position, algos: list[dict]) -> list[dict]:
        """The STOP_MARKETs on this symbol that could protect pos (closing side, and position side in hedge mode),
        best first: the engine's own sl_algo_id, then by trigger distance to pos.stop. A stop on the other side is a
        leftover and counts for nothing."""
        want = pos.side.close_order_side
        hedge = getattr(self.account, "hedge_mode", False)
        rows = [a for a in algos if a.get("orderType", a.get("type")) == "STOP_MARKET" and a.get("side") == want
                and (not hedge or a.get("positionSide") in (None, "BOTH", pos.side.value))]

        def rank(a: dict) -> tuple[int, float]:
            own = bool(pos.sl_algo_id) and self._algo_id(a) == pos.sl_algo_id
            try:
                dist = abs(float(a.get("triggerPrice", a.get("trigger")) or pos.stop) - pos.stop)
            except (TypeError, ValueError):
                dist = float("inf")
            return (0 if own else 1, dist)

        return sorted(rows, key=rank)

    def _protective_stop(self, pos: Position, algos: list[dict]) -> dict | None:
        rows = self._stop_rows(pos, algos)
        return rows[0] if rows else None

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
