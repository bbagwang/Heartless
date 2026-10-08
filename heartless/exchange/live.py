"""Live Binance USDⓈ-M account implementing the Account interface."""
from __future__ import annotations

import asyncio
import logging

import httpx

from heartless.core.models import AccountState, Fill
from heartless.exchange.base import Account, OrderResult, PositionSnapshot
from heartless.exchange.binance_rest import BinanceError, BinanceRest
from heartless.exchange.binance_ws import UserStream
from heartless.util.timeutil import now_ms

log = logging.getLogger(__name__)

# Binance codes whose meaning is "the request may or may not have been executed" (docs: -1000/-1001 internal
# error, -1007 "Send status unknown; execution status unknown"). Together with HTTP 5xx and transport errors
# on a POST they must never be read as a definitive rejection.
UNCERTAIN_CODES = (-1000, -1001, -1007)
ORDER_NOT_FOUND_CODES = (-2013, -2011)
STABLE_ASSETS = ("USDT", "USDC", "FDUSD", "BUSD")
UNKNOWN_RECHECK_DELAY = 1.0  # seconds between the two "does the order exist?" lookups after an uncertain send


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
        self._funding_seen: set[str] = set()

    async def start(self) -> None:
        await self.rest.sync_time()
        try:
            self.hedge_mode = await self.rest.get_position_mode()
            if self.hedge_mode:
                positions = await self.get_positions()
                if not positions:
                    try:
                        await self.rest.set_position_mode(False)
                        self.hedge_mode = False
                        log.info("switched account to one-way position mode")
                    except BinanceError as e:  # e.g. -4068 open orders: re-read so hedge_mode mirrors the exchange
                        log.warning("could not switch to one-way position mode: %s", e)
                        self.hedge_mode = await self.rest.get_position_mode()
                if self.hedge_mode:
                    log.warning("account is in hedge mode (open positions/orders); Heartless will send positionSide "
                                "and omit reduceOnly on every order")
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

    @staticmethod
    def _outcome_uncertain(e: Exception) -> bool:
        """True when Binance may have executed the order even though the call failed."""
        if isinstance(e, BinanceError):
            return e.status >= 500 or e.code in UNCERTAIN_CODES
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
            return False  # no connection was established: the request never reached Binance
        return True  # read/write timeout, reset, protocol error: the request was sent but not answered

    async def _order_error_result(self, symbol: str, client_id: str, e: Exception) -> OrderResult:
        """Translate a failed new-order call into an OrderResult without ever losing track of an order.

        4xx application errors are definitive rejections. For everything else (HTTP 5xx, -1000/-1001/-1007,
        transport errors) the order may have been executed, so it is looked up by its client id: a found order
        is returned as-is, an order that still does not exist after a second look was never accepted
        (REJECTED), and when the lookup itself fails the honest answer is status "UNKNOWN", never "REJECTED".
        """
        raw = {"code": getattr(e, "code", -1), "msg": getattr(e, "msg", None) or str(e)}
        if not self._outcome_uncertain(e):
            return OrderResult(client_id=client_id, status="REJECTED", raw=raw)
        raw["uncertain"] = True
        if client_id:
            for attempt in range(2):
                try:
                    d = await self.rest.query_order(symbol, None, client_id)
                except BinanceError as qe:
                    if qe.code not in ORDER_NOT_FOUND_CODES:
                        break
                    if attempt == 0:
                        await asyncio.sleep(UNKNOWN_RECHECK_DELAY)
                        continue
                    log.warning("order %s %s never reached the exchange (%s); treating as rejected", symbol, client_id, e)
                    return OrderResult(client_id=client_id, status="REJECTED", raw=raw)
                except httpx.HTTPError:
                    break
                log.warning("order %s %s resolved by lookup after uncertain send (%s): %s", symbol, client_id, e,
                            d.get("status"))
                return self._parse_order(d)
        log.error("order %s %s outcome UNKNOWN after %s", symbol, client_id or "<no client id>", e)
        return OrderResult(client_id=client_id, status="UNKNOWN", raw=raw)

    async def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False,
                           client_id: str = "") -> OrderResult:
        try:
            # Hedge mode: reduceOnly must not be sent together with positionSide (-1106); the opposite-side
            # order on the LONG/SHORT leg is implicitly reduce-only.
            d = await self.rest.new_order(symbol, side, "MARKET", quantity=qty,
                                          reduce_only=reduce_only and not self.hedge_mode,
                                          client_id=client_id or None,
                                          position_side=self._pos_side(side, reduce_only))
        except (BinanceError, httpx.HTTPError) as e:
            log.error("market order %s %s %s failed: %s", symbol, side, qty, e)
            return await self._order_error_result(symbol, client_id, e)
        return self._parse_order(d)

    async def limit_order(self, symbol: str, side: str, qty: float, price: float, post_only: bool = True,
                          reduce_only: bool = False, client_id: str = "") -> OrderResult:
        try:
            d = await self.rest.new_order(symbol, side, "LIMIT", quantity=qty, price=price,
                                          time_in_force="GTX" if post_only else "GTC",
                                          reduce_only=reduce_only and not self.hedge_mode,
                                          client_id=client_id or None,
                                          position_side=self._pos_side(side, reduce_only))
        except (BinanceError, httpx.HTTPError) as e:
            if isinstance(e, BinanceError) and e.code == -5022:  # GTX would immediately match
                return OrderResult(client_id=client_id, status="EXPIRED", raw={"code": e.code, "msg": e.msg})
            log.error("limit order %s %s %s@%s failed: %s", symbol, side, qty, price, e)
            return await self._order_error_result(symbol, client_id, e)
        return self._parse_order(d)

    async def cancel_order(self, symbol: str, order_id: str = "", client_id: str = "") -> bool:
        try:
            await self.rest.cancel_order(symbol, order_id or None, client_id or None)
            return True
        except (BinanceError, httpx.HTTPError) as e:
            if getattr(e, "code", None) == -2011:  # unknown order (already filled/cancelled)
                return False
            log.warning("cancel order failed: %s", e)
            return False

    async def query_order(self, symbol: str, order_id: str = "", client_id: str = "") -> OrderResult:
        """Status "UNKNOWN" means "the exchange has no such order" (a definitive answer the engine may act on).
        Transient failures (rate limit, 5xx, transport) are raised so the caller retries instead of cancelling."""
        try:
            d = await self.rest.query_order(symbol, order_id or None, client_id or None)
        except BinanceError as e:
            if e.code in ORDER_NOT_FOUND_CODES:
                return OrderResult(order_id=order_id, client_id=client_id, status="UNKNOWN",
                                   raw={"code": e.code, "msg": e.msg})
            raise
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
        d = await self.rest.new_algo_order(symbol, side, "STOP_MARKET", trigger_price, quantity=qty,
                                           reduce_only=not self.hedge_mode,
                                           client_algo_id=client_id or None, position_side=self._pos_side(side, True))
        return str(d.get("algoId", ""))

    async def place_take_profit(self, symbol: str, side: str, trigger_price: float, qty: float,
                                client_id: str = "") -> str:
        d = await self.rest.new_algo_order(symbol, side, "TAKE_PROFIT_MARKET", trigger_price, quantity=qty,
                                           reduce_only=not self.hedge_mode, client_algo_id=client_id or None,
                                           position_side=self._pos_side(side, True))
        return str(d.get("algoId", ""))

    async def cancel_algo(self, symbol: str, algo_id: str) -> bool:
        try:
            await self.rest.cancel_algo_order(symbol, algo_id)
            return True
        except (BinanceError, httpx.HTTPError) as e:
            if getattr(e, "code", None) in (-2011, -4120, -1102):
                return False
            log.warning("cancel algo failed: %s", e)
            return False

    async def cancel_all(self, symbol: str) -> None:
        for fn in (self.rest.cancel_all_orders, self.rest.cancel_all_algo_orders):
            try:
                await fn(symbol)
            except (BinanceError, httpx.HTTPError) as e:
                if getattr(e, "code", None) not in (-2011,):
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
            reduce_only = bool(o.get("R")) or bool(o.get("cp"))  # closePosition (cp) orders are reduce-only by nature
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
            if a.get("m") == "FUNDING_FEE":
                await self._on_funding_event(a, int(msg.get("T") or msg.get("E") or now_ms()))
        elif et == "MARGIN_CALL":
            log.error("MARGIN CALL received: %s", msg)
        elif et == "ALGO_UPDATE":
            # Binance futures user stream, conditional-order service: {"e":"ALGO_UPDATE","o":{"aid":..,"caid":..,"s":..,
            # "X": algo status, "o": order type, ...}}. Field names have varied between docs revisions, so read both
            # the short and the long spelling.
            o = msg.get("o") or msg.get("ao") or {}
            event = {"symbol": o.get("s") or o.get("symbol") or "",
                     "algo_id": str(o.get("aid") or o.get("algoId") or ""),
                     "client_id": o.get("caid") or o.get("clientAlgoId") or "",
                     "status": (o.get("X") or o.get("algoStatus") or "").upper(),
                     "order_type": o.get("o") or o.get("orderType") or o.get("type") or ""}
            log.debug("algo update: %s", event)
            if event["symbol"] and event["algo_id"]:
                await self._emit_algo_event(event)

    def _funding_price(self, symbol: str, fallback: float = 0.0) -> float:
        snap = self.cached_positions.get(symbol)
        return (snap.mark or snap.entry_price) if snap else fallback

    async def _on_funding_event(self, a: dict, ts: int) -> None:
        """Book a funding settlement as a FUNDING fill (fee > 0 = paid), the convention PaperAccount.apply_funding
        uses, so Position.funding / net_pnl() are comparable between live and paper.

        Binance pushes a FUNDING_FEE ACCOUNT_UPDATE with the balance change in B[].bc (negative when paid) and,
        for an isolated position, exactly one P entry naming the symbol. For cross margin P is empty and the
        stream carries only the aggregate, so the per-symbol amounts are read from the income ledger.
        """
        amount = sum(float(b.get("bc", 0.0) or 0.0) for b in a.get("B", []) if b.get("a") in STABLE_ASSETS)
        syms = {p.get("s") for p in a.get("P", []) if p.get("s")}
        hour = ts // 3_600_000  # funding settles at most once per hour per symbol
        if len(syms) == 1:
            sym = syms.pop()
            if amount:
                self._funding_seen.add(f"{sym}:{hour}")
                price = self._funding_price(sym, float(a["P"][0].get("ep", 0.0) or 0.0))
                await self._emit_fill(Fill(symbol=sym, order_side="FUNDING", qty=0.0, price=price, fee=-amount, ts=ts,
                                           kind="FUNDING"))
            return
        try:
            rows = await self.rest.income_history("FUNDING_FEE", start=ts - 10 * 60_000, end=ts + 60_000)
        except (BinanceError, httpx.HTTPError) as e:
            log.warning("funding income lookup failed: %s", e)
            return
        for r in rows or []:
            sym = r.get("symbol")
            income = float(r.get("income", 0.0) or 0.0)
            row_ts = int(r.get("time") or ts)
            keys = (str(r.get("tranId") or f"{sym}:{row_ts}"), f"{sym}:{row_ts // 3_600_000}")
            if not sym or not income or any(k in self._funding_seen for k in keys):
                continue
            self._funding_seen.update(keys)
            await self._emit_fill(Fill(symbol=sym, order_side="FUNDING", qty=0.0, price=self._funding_price(sym),
                                       fee=-income, ts=row_ts, kind="FUNDING"))
        if len(self._funding_seen) > 5000:
            self._funding_seen = set(list(self._funding_seen)[-2000:])
