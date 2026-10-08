"""Minimal, dependency-light Binance USDⓈ-M Futures REST client.

Covers market data, account, regular orders (/fapi/v1/order) and the Algo Order service that
Binance moved conditional orders (STOP_MARKET / TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET) to on
2025-12-09 (POST /fapi/v1/algoOrder, GET /fapi/v1/openAlgoOrders, DELETE /fapi/v1/algoOrder,
DELETE /fapi/v1/algoOpenOrders).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from heartless.core.models import Candle

log = logging.getLogger(__name__)

PROD_REST = "https://fapi.binance.com"
TEST_REST = "https://testnet.binancefuture.com"
PROD_WS = "wss://fstream.binance.com"
TEST_WS = "wss://stream.binancefuture.com"

CONDITIONAL_TYPES = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT", "TRAILING_STOP_MARKET"}


class BinanceError(Exception):
    def __init__(self, code: int, msg: str, status: int = 0):
        super().__init__(f"Binance error {code} (HTTP {status}): {msg}")
        self.code = code
        self.msg = msg
        self.status = status


class BinanceRest:
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False, recv_window: int = 5000,
                 timeout: float = 15.0):
        self.api_key = api_key
        self.api_secret = api_secret.encode() if api_secret else b""
        self.base = TEST_REST if testnet else PROD_REST
        self.ws_base = TEST_WS if testnet else PROD_WS
        self.recv_window = recv_window
        self._client = httpx.AsyncClient(base_url=self.base, timeout=timeout,
                                         headers={"X-MBX-APIKEY": api_key} if api_key else {})
        self.time_offset = 0
        self.used_weight = 0
        self._weight_reset_at = 0.0
        self._lock = asyncio.Lock()
        self._last_sync = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    # --- plumbing ------------------------------------------------------------------------------
    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        params["recvWindow"] = self.recv_window
        query = urlencode(params, doseq=True)
        params["signature"] = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        return params

    async def _throttle(self) -> None:
        # Binance futures: 2400 request weight / minute per IP. Back off when we get close.
        if self.used_weight > 2000:
            wait = max(0.0, 60 - (time.time() % 60)) + 0.5
            log.warning("rate limit guard: used weight %s, sleeping %.1fs", self.used_weight, wait)
            await asyncio.sleep(wait)
            self.used_weight = 0

    async def request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False,
                      retries: int = 3) -> Any:
        params = dict(params or {})
        attempt = 0
        while True:
            attempt += 1
            await self._throttle()
            p = self._sign(params) if signed else {k: v for k, v in params.items() if v is not None}
            try:
                # Binance accepts signed parameters in the query string for every method.
                resp = await self._client.request(method, path, params=p)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt > retries or method in ("POST", "DELETE") and signed:
                    raise
                await asyncio.sleep(min(2 ** attempt, 10))
                log.warning("transport error on %s %s (%s), retry %d", method, path, e, attempt)
                continue
            w = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if w:
                try:
                    self.used_weight = int(w)
                except ValueError:
                    pass
            if resp.status_code == 429 or resp.status_code == 418:
                ra = float(resp.headers.get("Retry-After", "5") or 5)
                log.warning("HTTP %s from Binance, sleeping %.0fs", resp.status_code, ra)
                await asyncio.sleep(ra + 1)
                if attempt <= retries:
                    continue
            if resp.status_code >= 500 and attempt <= retries and not (method in ("POST", "DELETE") and signed):
                await asyncio.sleep(min(2 ** attempt, 10))
                continue
            try:
                data = resp.json()
            except ValueError:
                data = {"code": -1, "msg": resp.text[:200]}
            body_err = isinstance(data, dict) and isinstance(data.get("code"), int) and data["code"] < 0
            if resp.status_code >= 400 or body_err:
                code = int(data.get("code", -1)) if isinstance(data, dict) else -1
                msg = data.get("msg", str(data)) if isinstance(data, dict) else str(data)
                if code == -1021 and attempt <= retries:  # timestamp out of recvWindow -> resync
                    await self.sync_time()
                    continue
                raise BinanceError(code, msg, resp.status_code)
            return data

    async def sync_time(self) -> None:
        data = await self.request("GET", "/fapi/v1/time")
        self.time_offset = int(data["serverTime"]) - int(time.time() * 1000)
        self._last_sync = time.time()
        log.debug("time offset %dms", self.time_offset)

    # --- market data ---------------------------------------------------------------------------
    async def exchange_info(self) -> dict:
        return await self.request("GET", "/fapi/v1/exchangeInfo")

    async def ticker_24h(self) -> list[dict]:
        return await self.request("GET", "/fapi/v1/ticker/24hr")

    async def premium_index(self, symbol: str | None = None) -> Any:
        return await self.request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol} if symbol else None)

    async def book_ticker(self, symbol: str | None = None) -> Any:
        return await self.request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol} if symbol else None)

    async def klines(self, symbol: str, interval: str = "1m", start: int | None = None, end: int | None = None,
                     limit: int = 1500) -> list[Candle]:
        data = await self.request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                                               "startTime": start, "endTime": end, "limit": limit})
        return [Candle.from_rest(r) for r in data]

    async def klines_range(self, symbol: str, start: int, end: int, interval: str = "1m") -> list[Candle]:
        """Page through klines between start and end (ms, inclusive)."""
        out: list[Candle] = []
        cur = start
        from heartless.util.timeutil import TF_MS

        step = TF_MS[interval]
        while cur <= end:
            batch = await self.klines(symbol, interval, start=cur, end=end, limit=1500)
            if not batch:
                break
            out.extend(batch)
            nxt = batch[-1].open_time + step
            if nxt <= cur:
                break
            cur = nxt
            if len(batch) < 1500:
                break
        return out

    async def funding_rate_history(self, symbol: str, start: int | None = None, end: int | None = None,
                                   limit: int = 1000) -> list[dict]:
        return await self.request("GET", "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start,
                                                                    "endTime": end, "limit": limit})

    async def open_interest(self, symbol: str) -> dict:
        return await self.request("GET", "/fapi/v1/openInterest", {"symbol": symbol})

    async def open_interest_hist(self, symbol: str, period: str = "15m", limit: int = 30) -> list[dict]:
        return await self.request("GET", "/futures/data/openInterestHist", {"symbol": symbol, "period": period,
                                                                             "limit": limit})

    async def top_long_short_ratio(self, symbol: str, period: str = "15m", limit: int = 10) -> list[dict]:
        return await self.request("GET", "/futures/data/topLongShortPositionRatio",
                                  {"symbol": symbol, "period": period, "limit": limit})

    async def top_long_short_account_ratio(self, symbol: str, period: str = "5m", limit: int = 10) -> list[dict]:
        return await self.request("GET", "/futures/data/topLongShortAccountRatio",
                                  {"symbol": symbol, "period": period, "limit": limit})

    async def global_long_short_account_ratio(self, symbol: str, period: str = "5m", limit: int = 10) -> list[dict]:
        return await self.request("GET", "/futures/data/globalLongShortAccountRatio",
                                  {"symbol": symbol, "period": period, "limit": limit})

    async def taker_long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 10) -> list[dict]:
        return await self.request("GET", "/futures/data/takerlongshortRatio",
                                  {"symbol": symbol, "period": period, "limit": limit})

    # --- account -------------------------------------------------------------------------------
    async def account(self) -> dict:
        try:
            return await self.request("GET", "/fapi/v3/account", signed=True)
        except BinanceError as e:
            if e.status in (404, 400) or e.code in (-1121, -1102):
                return await self.request("GET", "/fapi/v2/account", signed=True)
            raise

    async def balance(self) -> list[dict]:
        try:
            return await self.request("GET", "/fapi/v3/balance", signed=True)
        except BinanceError as e:
            if e.status in (404, 400):
                return await self.request("GET", "/fapi/v2/balance", signed=True)
            raise

    async def position_risk(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else None
        try:
            return await self.request("GET", "/fapi/v3/positionRisk", params, signed=True)
        except BinanceError as e:
            if e.status in (404, 400):
                return await self.request("GET", "/fapi/v2/positionRisk", params, signed=True)
            raise

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict | None:
        try:
            return await self.request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type},
                                      signed=True)
        except BinanceError as e:
            if e.code == -4046:  # No need to change margin type
                return None
            raise

    async def get_position_mode(self) -> bool:
        """True when the account is in hedge (dual-side) mode."""
        data = await self.request("GET", "/fapi/v1/positionSide/dual", signed=True)
        return bool(data.get("dualSidePosition"))

    async def set_position_mode(self, dual: bool) -> None:
        try:
            await self.request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if dual else "false"},
                               signed=True)
        except BinanceError as e:
            if e.code != -4059:  # No need to change position side
                raise

    async def income_history(self, income_type: str | None = None, start: int | None = None, end: int | None = None,
                             limit: int = 1000) -> list[dict]:
        return await self.request("GET", "/fapi/v1/income", {"incomeType": income_type, "startTime": start,
                                                              "endTime": end, "limit": limit}, signed=True)

    async def user_trades(self, symbol: str, start: int | None = None, limit: int = 500) -> list[dict]:
        return await self.request("GET", "/fapi/v1/userTrades", {"symbol": symbol, "startTime": start, "limit": limit},
                                  signed=True)

    # --- regular orders ------------------------------------------------------------------------
    async def new_order(self, symbol: str, side: str, type_: str, quantity: float | None = None,
                        price: float | None = None, time_in_force: str | None = None, reduce_only: bool = False,
                        client_id: str | None = None, position_side: str | None = None) -> dict:
        if type_ in CONDITIONAL_TYPES:
            raise ValueError("conditional orders must use new_algo_order()")
        params: dict[str, Any] = {"symbol": symbol, "side": side, "type": type_, "newOrderRespType": "RESULT"}
        if quantity is not None:
            params["quantity"] = _fmt(quantity)
        if price is not None:
            params["price"] = _fmt(price)
        if time_in_force:
            params["timeInForce"] = time_in_force
        if reduce_only:
            params["reduceOnly"] = "true"
        if client_id:
            params["newClientOrderId"] = client_id
        if position_side:
            params["positionSide"] = position_side
        return await self.request("POST", "/fapi/v1/order", params, signed=True)

    async def cancel_order(self, symbol: str, order_id: str | int | None = None, client_id: str | None = None) -> dict:
        return await self.request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id,
                                                               "origClientOrderId": client_id}, signed=True)

    async def cancel_all_orders(self, symbol: str) -> dict:
        return await self.request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    async def query_order(self, symbol: str, order_id: str | int | None = None, client_id: str | None = None) -> dict:
        return await self.request("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id,
                                                            "origClientOrderId": client_id}, signed=True)

    async def open_orders(self, symbol: str | None = None) -> list[dict]:
        return await self.request("GET", "/fapi/v1/openOrders", {"symbol": symbol} if symbol else None, signed=True)

    # --- algo (conditional) orders -------------------------------------------------------------
    async def new_algo_order(self, symbol: str, side: str, type_: str, trigger_price: float,
                             quantity: float | None = None, close_position: bool = False, reduce_only: bool = False,
                             working_type: str = "MARK_PRICE", price_protect: bool = True,
                             client_algo_id: str | None = None, position_side: str | None = None,
                             price: float | None = None, time_in_force: str | None = None) -> dict:
        if type_ not in CONDITIONAL_TYPES:
            raise ValueError(f"{type_} is not an algo order type")
        params: dict[str, Any] = {"symbol": symbol, "side": side, "type": type_, "algoType": "CONDITIONAL",
                                  "triggerPrice": _fmt(trigger_price), "workingType": working_type,
                                  "priceProtect": "true" if price_protect else "false"}
        if close_position:
            params["closePosition"] = "true"
        else:
            if quantity is None:
                raise ValueError("quantity required unless close_position")
            params["quantity"] = _fmt(quantity)
            if reduce_only:
                params["reduceOnly"] = "true"
        if price is not None:
            params["price"] = _fmt(price)
        if time_in_force:
            params["timeInForce"] = time_in_force
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id[:36]
        if position_side:
            params["positionSide"] = position_side
        return await self.request("POST", "/fapi/v1/algoOrder", params, signed=True)

    async def cancel_algo_order(self, symbol: str, algo_id: str | int | None = None,
                                client_algo_id: str | None = None) -> dict:
        return await self.request("DELETE", "/fapi/v1/algoOrder", {"symbol": symbol, "algoId": algo_id,
                                                                   "clientAlgoId": client_algo_id}, signed=True)

    async def cancel_all_algo_orders(self, symbol: str) -> dict:
        return await self.request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol}, signed=True)

    async def open_algo_orders(self, symbol: str | None = None) -> list[dict]:
        data = await self.request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol} if symbol else None, signed=True)
        if isinstance(data, dict):  # some deployments wrap the list
            return data.get("orders") or data.get("data") or []
        return data

    async def query_algo_order(self, symbol: str, algo_id: str | int | None = None,
                               client_algo_id: str | None = None) -> dict:
        return await self.request("GET", "/fapi/v1/algoOrder", {"symbol": symbol, "algoId": algo_id,
                                                                "clientAlgoId": client_algo_id}, signed=True)

    # --- user data stream ----------------------------------------------------------------------
    async def create_listen_key(self) -> str:
        data = await self.request("POST", "/fapi/v1/listenKey")
        return data["listenKey"]

    async def keepalive_listen_key(self) -> None:
        await self.request("PUT", "/fapi/v1/listenKey")

    async def close_listen_key(self) -> None:
        await self.request("DELETE", "/fapi/v1/listenKey")


def _fmt(x: float) -> str:
    """Format numbers without scientific notation or trailing zeros."""
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"
