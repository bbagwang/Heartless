"""Binance USDⓈ-M websocket streams with automatic reconnection."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

Callback = Callable[[dict], Awaitable[None] | None]


class MarketStream:
    """Combined market stream: 1m klines + bookTicker per symbol, mark price for all symbols."""

    def __init__(self, ws_base: str, on_kline: Callback, on_book: Callback, on_mark: Callback):
        self.ws_base = ws_base
        self.on_kline = on_kline
        self.on_book = on_book
        self.on_mark = on_mark
        self.symbols: list[str] = []
        self._restart = asyncio.Event()
        self._stop = False
        self.last_msg_ts = 0.0
        self.connected = False

    def set_symbols(self, symbols: list[str]) -> None:
        new = sorted(set(s.upper() for s in symbols))
        if new != self.symbols:
            self.symbols = new
            self._restart.set()

    def _url(self) -> str:
        streams = ["!markPrice@arr@1s"]
        for s in self.symbols:
            ls = s.lower()
            streams.append(f"{ls}@kline_1m")
            streams.append(f"{ls}@bookTicker")
        return f"{self.ws_base}/stream?streams={'/'.join(streams)}"

    async def stop(self) -> None:
        self._stop = True
        self._restart.set()

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop:
            if not self.symbols:
                await asyncio.sleep(1)
                continue
            self._restart.clear()
            url = self._url()
            try:
                async with connect(url, ping_interval=20, ping_timeout=20, max_size=2**23, open_timeout=20) as ws:
                    self.connected = True
                    backoff = 1.0
                    log.info("market stream connected (%d symbols)", len(self.symbols))
                    restart_task = asyncio.create_task(self._restart.wait())
                    try:
                        while not self._stop:
                            recv_task = asyncio.create_task(ws.recv())
                            done, _ = await asyncio.wait({recv_task, restart_task}, timeout=60,
                                                         return_when=asyncio.FIRST_COMPLETED)
                            if restart_task in done:
                                recv_task.cancel()
                                log.info("market stream restarting with new symbol list")
                                break
                            if recv_task not in done:
                                recv_task.cancel()
                                if time.time() - self.last_msg_ts > 90:
                                    log.warning("market stream silent for 90s, reconnecting")
                                    break
                                continue
                            raw = recv_task.result()
                            self.last_msg_ts = time.time()
                            await self._dispatch(raw)
                    finally:
                        restart_task.cancel()
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("market stream disconnected: %s", e)
            except Exception:  # noqa: BLE001
                log.exception("market stream error")
            finally:
                self.connected = False
            if self._stop:
                break
            if not self._restart.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        data = msg.get("data", msg)
        if isinstance(data, list):  # !markPrice@arr
            res = self.on_mark(data)
            if asyncio.iscoroutine(res):
                await res
            return
        et = data.get("e")
        if et == "kline":
            res = self.on_kline(data)
        elif et == "bookTicker":
            res = self.on_book(data)
        elif et == "markPriceUpdate":
            res = self.on_mark([data])
        else:
            return
        if asyncio.iscoroutine(res):
            await res


class UserStream:
    """User data stream (listenKey based) delivering order/position updates."""

    def __init__(self, rest, on_event: Callback):
        self.rest = rest
        self.on_event = on_event
        self._stop = False
        self.connected = False
        self.listen_key: str | None = None

    async def stop(self) -> None:
        self._stop = True

    async def _keepalive(self) -> None:
        while not self._stop:
            await asyncio.sleep(25 * 60)
            try:
                await self.rest.keepalive_listen_key()
            except Exception as e:  # noqa: BLE001
                log.warning("listenKey keepalive failed: %s", e)

    async def run(self) -> None:
        ka = asyncio.create_task(self._keepalive())
        backoff = 1.0
        try:
            while not self._stop:
                try:
                    self.listen_key = await self.rest.create_listen_key()
                    url = f"{self.rest.ws_base}/ws/{self.listen_key}"
                    async with connect(url, ping_interval=20, ping_timeout=20, max_size=2**22, open_timeout=20) as ws:
                        self.connected = True
                        backoff = 1.0
                        log.info("user stream connected")
                        async for raw in ws:
                            if self._stop:
                                break
                            try:
                                msg = json.loads(raw)
                            except ValueError:
                                continue
                            if msg.get("e") == "listenKeyExpired":
                                log.warning("listenKey expired, reconnecting")
                                break
                            res = self.on_event(msg)
                            if asyncio.iscoroutine(res):
                                await res
                except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                    log.warning("user stream disconnected: %s", e)
                except Exception:  # noqa: BLE001
                    log.exception("user stream error")
                finally:
                    self.connected = False
                if not self._stop:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        finally:
            ka.cancel()
