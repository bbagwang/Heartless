"""Binance USDⓈ-M websocket streams with automatic reconnection."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from heartless.util import perf
from heartless.util.timeutil import now_ms

log = logging.getLogger(__name__)

Callback = Callable[[dict], Awaitable[None] | None]

# F7: orjson parses stream frames 1.6-3.5x faster than json when it is installed (optional, never required). Both raise a
# ValueError subclass on bad input and return the same dicts / lists / str / int / float values.
try:  # pragma: no cover - depends on the environment
    import orjson as _orjson

    loads: Callable[[str | bytes], Any] = _orjson.loads
    JSON_ENGINE = "orjson"
except ImportError:  # pragma: no cover - depends on the environment
    loads = json.loads
    JSON_ENGINE = "json"

# F7: the smallest (local receive time - exchange event time) over a window of messages is the clock offset plus the
# shortest one-way delay; it is reported to perf.DRIFT once per window
LAG_WINDOW_S = 60.0


class StreamLag:
    """Minimum and last (local receive - event time E) per window, in ms."""
    __slots__ = ("source", "window_min", "window_end", "last", "min_last_window")

    def __init__(self, source: str):
        self.source = source
        self.window_min = float("inf")
        self.window_end = 0.0
        self.last = float("nan")
        self.min_last_window = float("nan")

    def observe(self, event_ms: int | float | None, recv_ms: float | None = None) -> None:
        if not event_ms:
            return
        recv = float(now_ms()) if recv_ms is None else float(recv_ms)  # the replay's virtual clock when one runs
        now = recv / 1000.0
        lag = recv - float(event_ms)
        self.last = lag
        if lag < self.window_min:
            self.window_min = lag
        if now >= self.window_end:
            if self.window_end and self.window_min != float("inf"):
                self.min_last_window = self.window_min
                perf.DRIFT.observe(self.window_min, self.source)
            self.window_min = float("inf")
            self.window_end = now + LAG_WINDOW_S


class MarketStream:
    """Combined market stream: 1m klines + bookTicker per traded symbol, mark price for all symbols, and 1m klines only
    for the `kline_only` symbols (the market-context universe members that are not traded: data/context.py needs
    their bars, nothing else)."""

    def __init__(self, ws_base: str, on_kline: Callback, on_book: Callback, on_mark: Callback):
        self.ws_base = ws_base
        self.on_kline = on_kline
        self.on_book = on_book
        self.on_mark = on_mark
        self.symbols: list[str] = []
        self.kline_only: list[str] = []
        self._restart = asyncio.Event()
        self._stop = False
        self.last_msg_ts = 0.0
        self.connected = False
        self.lag = StreamLag("stream")  # F7: kline event time vs local receive time

    def set_symbols(self, symbols: list[str], kline_only: list[str] | None = None) -> None:
        """Traded symbols, plus symbols subscribed to 1m klines only (None keeps the current kline-only list)."""
        new = sorted(set(s.upper() for s in symbols))
        extra = self.kline_only if kline_only is None else sorted(set(s.upper() for s in kline_only))
        extra = [s for s in extra if s not in new]
        if new != self.symbols or extra != self.kline_only:
            self.symbols = new
            self.kline_only = extra
            self._restart.set()

    def _url(self) -> str:
        streams = ["!markPrice@arr@1s"]
        for s in self.symbols:
            ls = s.lower()
            streams.append(f"{ls}@kline_1m")
            streams.append(f"{ls}@bookTicker")
        streams += [f"{s.lower()}@kline_1m" for s in self.kline_only]
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
                            try:
                                await self._dispatch(raw)
                            except Exception:  # noqa: BLE001
                                # Handler errors are not socket errors: log and keep the stream alive.
                                log.exception("market stream callback failed")
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
            msg = loads(raw)
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
            self.lag.observe(data.get("E"))
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
                                msg = loads(raw)
                            except ValueError:
                                continue
                            if msg.get("e") == "listenKeyExpired":
                                log.warning("listenKey expired, reconnecting")
                                break
                            try:
                                res = self.on_event(msg)
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:  # noqa: BLE001
                                # A failing handler must not be mistaken for a socket failure: keep the
                                # connection (and every subsequent ORDER_TRADE_UPDATE) instead of reconnecting.
                                log.exception("user stream callback failed (e=%s)", msg.get("e"))
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


class BookStream:
    """Depth streams for the local order books (F7, heartless/exchange/orderbook.py), on connections of their own: the
    kline / bookTicker connection keeps its URL and message rate, and a depth reconnect never interrupts the bars.

    mode "partial" subscribes `<symbol>@depth20@<speed>`, "diff" `<symbol>@depth@<speed>` (speed 100, 250 or 500 ms;
    250 ms is the streams' default and has no suffix). Symbols are split over connections of at most `per_connection`
    streams. Every depthUpdate goes to `on_depth(data)`; when a connection drops, `on_reset(symbols)` is called so the
    books of those symbols resync."""

    def __init__(self, ws_base: str, on_depth: Callback, mode: str = "partial", speed_ms: int = 100,
                 per_connection: int = 50, on_reset: Callable[[list[str]], None] | None = None):
        self.ws_base = ws_base
        self.on_depth = on_depth
        self.on_reset = on_reset
        self.mode = mode
        self.speed_ms = min((100, 250, 500), key=lambda v: abs(v - int(speed_ms)))
        self.per_connection = max(1, int(per_connection))
        self.symbols: list[str] = []
        self._restart = asyncio.Event()
        self._stop = False
        self.connected: dict[int, bool] = {}
        self.messages = 0
        self.lag = StreamLag("depth")

    def stream_name(self, symbol: str) -> str:
        suffix = "" if self.speed_ms == 250 else f"@{self.speed_ms}ms"
        kind = "depth20" if self.mode == "partial" else "depth"
        return f"{symbol.lower()}@{kind}{suffix}"

    def urls(self) -> list[tuple[list[str], str]]:
        out = []
        for i in range(0, len(self.symbols), self.per_connection):
            chunk = self.symbols[i:i + self.per_connection]
            out.append((chunk, f"{self.ws_base}/stream?streams={'/'.join(self.stream_name(s) for s in chunk)}"))
        return out

    def set_symbols(self, symbols: list[str]) -> None:
        new = sorted(set(s.upper() for s in symbols))
        if new != self.symbols:
            self.symbols = new
            self._restart.set()

    async def stop(self) -> None:
        self._stop = True
        self._restart.set()

    async def run(self) -> None:
        while not self._stop:
            if not self.symbols:
                await asyncio.sleep(1)
                continue
            self._restart.clear()
            tasks = [asyncio.create_task(self._conn(i, chunk, url), name=f"depth-stream-{i}")
                     for i, (chunk, url) in enumerate(self.urls())]
            try:
                await self._restart.wait()
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.connected.clear()

    async def _conn(self, idx: int, chunk: list[str], url: str) -> None:
        backoff = 1.0
        while not self._stop:
            try:
                async with connect(url, ping_interval=20, ping_timeout=20, max_size=2**23, open_timeout=20) as ws:
                    self.connected[idx] = True
                    backoff = 1.0
                    log.info("depth stream %d connected (%d symbols, %s)", idx, len(chunk), self.mode)
                    async for raw in ws:
                        await self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("depth stream %d disconnected: %s", idx, e)
            except Exception:  # noqa: BLE001
                log.exception("depth stream %d error", idx)
            finally:
                self.connected[idx] = False
            if self.on_reset is not None:
                self.on_reset(chunk)
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            msg = loads(raw)
        except ValueError:
            return
        data = msg.get("data", msg)
        if not isinstance(data, dict) or data.get("e") != "depthUpdate":
            return
        self.messages += 1
        self.lag.observe(data.get("E"))
        try:
            res = self.on_depth(data)
            if asyncio.iscoroutine(res):
                await res
        except Exception:  # noqa: BLE001
            log.exception("depth handler failed")
