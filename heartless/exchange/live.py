"""Live Binance USDⓈ-M account implementing the Account interface."""
from __future__ import annotations

import asyncio
import logging

from heartless.core.models import AccountState, Fill
from heartless.exchange.base import Account, OrderResult, PositionSnapshot
from heartless.exchange.binance_rest import BinanceError, BinanceRest
from heartless.exchange.binance_ws import UserStream
from heartless.util.timeutil import now_ms

log = logging.getLogger(__name__)


class LiveAccount(Account):
    is_paper = False

    def __init__(self, rest: BinanceRest, name: str = "live", taker_fee: float = 0.0005):
        super().__init__()
        self.rest = rest
        self.name = name
        self.taker_fee = taker_fee
        self.user_stream = UserStream(rest, self._on_user_event)
        self._stream_task: asyncio.Task | None = None
        self.cached_positions: dict[str, PositionSnapshot] = {}
        self.last_state: AccountState | None = None
        self._prepared: dict[str, int] = {}
        self.hedge_mode = False
        self.events: int = 0

    async def start(self) -> None:
        await self.rest.sync_time()
        try:
            self.hedge_mode = await self.rest.get_position_mode()
            if self.hedge_mode:
                positions = await self.get_positions()
                if not positions:
                    await self.rest.set_position_mode(False)
                    self.hedge_mode = False
                    log.info("switched account to one-way position mode")
                else:
                    log.warning("account is in hedge mode with open positions; Heartless will use positionSide")
        except BinanceError as e:
            log.warning("could not read position mode: %s", e)
        await self.get_state()
        self._stream_task = asyncio.create_task(self.user_stream.run(), name="user-stream")

    async def stop(self) -> None:
        await self.user_stream.stop()
        if self._stream_task:
            self._stream_task.cancel()
        try:
            await self.rest.close_listen_key()
        except Exception:  # noqa: BLE001
            pass

    def _pos_side(self, order_side: str, reduce_only: bool) -> str | None:
        if not self.hedge_mode:
            return None
        if reduce_only:
            return "SHORT" if order_side == "BUY" else "LONG"
        return "LONG" if order_side == "BUY" else "SHORT"

    # --- queries -------------------------------------------------------------------------------
    async def get_state(self) -> AccountState:
        acc = await self.rest.account()
        bal = float(acc.get("totalWalletBalance", 0.0))
        unreal = float(acc.get("totalUnrealizedProfit", 0.0))
        avail = float(acc.get("availableBalance", bal))
        self.last_state = AccountState(balance=bal, equity=bal + unreal, available=avail, unrealized=unreal, ts=now_ms())
        return self.last_state

    async def get_positions(self) -> dict[str, PositionSnapshot]:
        rows = await self.rest.position_risk()
        out: dict[str, PositionSnapshot] = {}
        for r in rows:
            qty = float(r.get("positionAmt", 0.0))
            if abs(qty) <= 0:
                continue
            sym = r["symbol"]
            snap = PositionSnapshot(symbol=sym, qty=qty, entry_price=float(r.get("entryPrice", 0.0)),
                                    unrealized=float(r.get("unRealizedProfit", 0.0)),
                                    leverage=int(float(r.get("leverage", 0) or 0)),
                                    mark=float(r.get("markPrice", 0.0) or 0.0))
            if sym in out:  # hedge mode: net the two legs
                out[sym].qty += qty
            else:
                out[sym] = snap
        self.cached_positions = out
        return out

    async def open_orders(self, symbol: str | None = None) -> list[dict]:
        return await self.rest.open_orders(symbol)

    async def open_algo_orders(self, symbol: str | None = None) -> list[dict]:
        return await self.rest.open_algo_orders(symbol)

    async def prepare_symbol(self, symbol: str, leverage: int) -> None:
        if self._prepared.get(symbol) == leverage:
            return
        try:
            await self.rest.set_margin_type(symbol, "ISOLATED")
        except BinanceError as e:
            if e.code not in (-4046, -4048):  # already isolated / open position prevents change
                log.warning("set margin type %s failed: %s", symbol, e)
        try:
            await self.rest.set_leverage(symbol, leverage)
        except BinanceError as e:
            log.warning("set leverage %s=%s failed: %s", symbol, leverage, e)
            if e.code == -4028 and leverage > 1:  # leverage not valid -> step down
                await self.rest.set_leverage(symbol, max(1, leverage // 2))
        self._prepared[symbol] = leverage

    # --- trading -------------------------------------------------------------------------------
    @staticmethod
    def _parse_order(d: dict) -> OrderResult:
        return OrderResult(order_id=str(d.get("orderId", "")), client_id=d.get("clientOrderId", ""),
                           status=d.get("status", "NEW"), filled_qty=float(d.get("executedQty", 0.0) or 0.0),
                           avg_price=float(d.get("avgPrice", 0.0) or 0.0), raw=d)

    async def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False,
                           client_id: str = "") -> OrderResult:
        try:
            d = await self.rest.new_order(symbol, side, "MARKET", quantity=qty, reduce_only=reduce_only,
                                          client_id=client_id or None,
                                          position_side=self._pos_side(side, reduce_only))
        except BinanceError as e:
            log.error("market order %s %s %s failed: %s", symbol, side, qty, e)
            return OrderResult(client_id=client_id, status="REJECTED", raw={"code": e.code, "msg": e.msg})
        return self._parse_order(d)

    async def limit_order(self, symbol: str, side: str, qty: float, price: float, post_only: bool = True,
                          reduce_only: bool = False, client_id: str = "") -> OrderResult:
        try:
            d = await self.rest.new_order(symbol, side, "LIMIT", quantity=qty, price=price,
                                          time_in_force="GTX" if post_only else "GTC", reduce_only=reduce_only,
                                          client_id=client_id or None,
                                          position_side=self._pos_side(side, reduce_only))
        except BinanceError as e:
            if e.code == -5022:  # GTX would immediately match
                return OrderResult(client_id=client_id, status="EXPIRED", raw={"code": e.code, "msg": e.msg})
            log.error("limit order %s %s %s@%s failed: %s", symbol, side, qty, price, e)
            return OrderResult(client_id=client_id, status="REJECTED", raw={"code": e.code, "msg": e.msg})
        return self._parse_order(d)

    async def cancel_order(self, symbol: str, order_id: str = "", client_id: str = "") -> bool:
        try:
            await self.rest.cancel_order(symbol, order_id or None, client_id or None)
            return True
        except BinanceError as e:
            if e.code == -2011:  # unknown order (already filled/cancelled)
                return False
            log.warning("cancel order failed: %s", e)
            return False

    async def query_order(self, symbol: str, order_id: str = "", client_id: str = "") -> OrderResult:
        try:
            d = await self.rest.query_order(symbol, order_id or None, client_id or None)
        except BinanceError as e:
            return OrderResult(order_id=order_id, client_id=client_id, status="UNKNOWN", raw={"code": e.code})
        return self._parse_order(d)

    async def place_stop(self, symbol: str, side: str, trigger_price: float, qty: float | None = None,
                         close_position: bool = False, client_id: str = "") -> str:
        """STOP_MARKET via the Algo Order service. Prefers closePosition=true (covers the whole position even
        after partial fills); falls back to quantity+reduceOnly if the service rejects closePosition."""
        if close_position:
            try:
                d = await self.rest.new_algo_order(symbol, side, "STOP_MARKET", trigger_price, close_position=True,
                                                   client_algo_id=client_id or None,
                                                   position_side=self._pos_side(side, True))
                return str(d.get("algoId", ""))
            except BinanceError as e:
                if qty is None or e.code in (-2019, -4045, -1021, -1022):  # margin / quota / auth -> don't retry blindly
                    raise
                log.warning("closePosition stop rejected (%s); retrying with quantity+reduceOnly", e)
        if qty is None:
            raise ValueError("qty required when close_position is False")
        d = await self.rest.new_algo_order(symbol, side, "STOP_MARKET", trigger_price, quantity=qty, reduce_only=True,
                                           client_algo_id=client_id or None, position_side=self._pos_side(side, True))
        return str(d.get("algoId", ""))

    async def place_take_profit(self, symbol: str, side: str, trigger_price: float, qty: float,
                                client_id: str = "") -> str:
        d = await self.rest.new_algo_order(symbol, side, "TAKE_PROFIT_MARKET", trigger_price, quantity=qty,
                                           reduce_only=True, client_algo_id=client_id or None,
                                           position_side=self._pos_side(side, True))
        return str(d.get("algoId", ""))

    async def cancel_algo(self, symbol: str, algo_id: str) -> bool:
        try:
            await self.rest.cancel_algo_order(symbol, algo_id)
            return True
        except BinanceError as e:
            if e.code in (-2011, -4120, -1102):
                return False
            log.warning("cancel algo failed: %s", e)
            return False

    async def cancel_all(self, symbol: str) -> None:
        for fn in (self.rest.cancel_all_orders, self.rest.cancel_all_algo_orders):
            try:
                await fn(symbol)
            except BinanceError as e:
                if e.code not in (-2011,):
                    log.warning("cancel all (%s) failed: %s", fn.__name__, e)

    # --- user data stream ----------------------------------------------------------------------
    async def _on_user_event(self, msg: dict) -> None:
        self.events += 1
        et = msg.get("e")
        if et == "ORDER_TRADE_UPDATE":
            o = msg.get("o", {})
            if o.get("x") != "TRADE":
                return
            qty = float(o.get("l", 0.0) or 0.0)
            if qty <= 0:
                return
            price = float(o.get("L", 0.0) or 0.0)
            fee = float(o.get("n", 0.0) or 0.0)
            if o.get("N") not in (None, "USDT", "USDC", "FDUSD", "BUSD"):
                fee = qty * price * self.taker_fee * 0.9  # commission paid in BNB etc.
            ot = o.get("ot") or o.get("o")
            cid = o.get("c", "") or ""
            reduce_only = bool(o.get("R"))
            if ot == "STOP_MARKET" or cid.startswith("HLS"):
                kind = "SL"
            elif ot == "TAKE_PROFIT_MARKET" or cid.startswith("HLT"):
                kind = "TP"
            elif cid.startswith("HLC") or reduce_only:
                kind = "CLOSE"
            elif cid.startswith("HLE"):
                kind = "ENTRY"
            else:
                kind = "UNKNOWN"
            fill = Fill(symbol=o["s"], order_side=o["S"], qty=qty, price=price, fee=fee, ts=int(o.get("T", now_ms())),
                        client_id=cid, order_id=str(o.get("i", "")), reduce_only=reduce_only, maker=bool(o.get("m")),
                        kind=kind)
            await self._emit_fill(fill)
        elif et == "ACCOUNT_UPDATE":
            a = msg.get("a", {})
            for p in a.get("P", []):
                sym = p.get("s")
                qty = float(p.get("pa", 0.0) or 0.0)
                if abs(qty) > 0:
                    self.cached_positions[sym] = PositionSnapshot(symbol=sym, qty=qty, entry_price=float(p.get("ep", 0.0) or 0.0),
                                                                  unrealized=float(p.get("up", 0.0) or 0.0))
                else:
                    self.cached_positions.pop(sym, None)
            for b in a.get("B", []):
                if b.get("a") == "USDT" and self.last_state:
                    self.last_state.balance = float(b.get("wb", self.last_state.balance))
        elif et == "MARGIN_CALL":
            log.error("MARGIN CALL received: %s", msg)
        elif et == "ALGO_UPDATE":
            log.debug("algo update: %s", msg)
