"""Regression: an exception raised by a stream callback must not be treated as a socket failure.

Before the fix, a raising ``on_event`` / ``on_kline`` handler fell through to the outer ``except Exception``
in ``UserStream.run`` / ``MarketStream.run``, which tore the connection down and reconnected (new listenKey
for the user stream). Fills emitted during that gap were lost. Now a handler failure is logged and the
``async for`` / ``recv`` loop continues on the SAME connection.
"""
import asyncio
import contextlib
import json

from heartless.exchange import binance_ws
from heartless.exchange.binance_ws import MarketStream, UserStream


class FakeWS:
    """Minimal stand-in for a websockets connection: async-iterable and ``recv()``-able.

    Messages are popped from a list shared with the ``FakeConnect`` factory, so a reconnect (second ``connect``)
    continues with the remaining messages instead of hanging. Once drained, ``on_drained`` is awaited (the test
    uses it to stop the stream) and the connection then blocks / ends.
    """

    def __init__(self, messages: list, on_drained):
        self._messages = messages
        self._on_drained = on_drained

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._messages:
            return self._messages.pop(0)
        await self._on_drained()
        raise StopAsyncIteration

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        await self._on_drained()
        # Stream is stopping: block until MarketStream.run cancels this recv task via the restart event.
        await asyncio.Event().wait()


class FakeConnect:
    def __init__(self, messages: list, on_drained):
        self.messages = list(messages)
        self.on_drained = on_drained
        self.calls = 0
        self.urls: list[str] = []

    @contextlib.asynccontextmanager
    async def __call__(self, url, **kw):
        self.calls += 1
        self.urls.append(url)
        yield FakeWS(self.messages, self.on_drained)


class FakeRest:
    ws_base = "wss://fake"

    def __init__(self):
        self.listen_keys = 0

    async def create_listen_key(self):
        self.listen_keys += 1
        return f"lk{self.listen_keys}"

    async def keepalive_listen_key(self):
        return None


def _otu(symbol: str, client_id: str) -> dict:
    return {"e": "ORDER_TRADE_UPDATE", "T": 1, "o": {"s": symbol, "c": client_id, "x": "TRADE", "X": "FILLED"}}


def test_user_stream_callback_error_does_not_reconnect(monkeypatch):
    rest = FakeRest()
    delivered: list[dict] = []
    stream: UserStream | None = None

    async def on_event(msg):
        delivered.append(msg)
        if msg["o"]["c"] == "HLS-boom":
            raise RuntimeError("store.save_trade failed (disk full)")

    async def drained():
        await stream.stop()

    msgs = [json.dumps(_otu("ETHUSDT", "HLS-boom")), json.dumps(_otu("BTCUSDT", "HLT-ok"))]
    fake_connect = FakeConnect(msgs, drained)
    monkeypatch.setattr(binance_ws, "connect", fake_connect)

    stream = UserStream(rest, on_event)
    asyncio.run(asyncio.wait_for(stream.run(), timeout=10))

    # both events reached the handler, over a single connection with a single listenKey
    assert [m["o"]["c"] for m in delivered] == ["HLS-boom", "HLT-ok"]
    assert rest.listen_keys == 1, "a raising callback must not trigger a new listenKey / reconnect"
    assert fake_connect.calls == 1
    assert fake_connect.urls == ["wss://fake/ws/lk1"]
    assert stream.connected is False


def test_user_stream_sync_callback_error_does_not_reconnect(monkeypatch):
    """Same guarantee for a plain (non-coroutine) handler."""
    rest = FakeRest()
    delivered: list[str] = []
    stream: UserStream | None = None

    def on_event(msg):
        delivered.append(msg["o"]["c"])
        if len(delivered) == 1:
            raise ValueError("bad fill")

    async def drained():
        await stream.stop()

    fake_connect = FakeConnect([json.dumps(_otu("A", "1")), json.dumps(_otu("B", "2"))], drained)
    monkeypatch.setattr(binance_ws, "connect", fake_connect)
    stream = UserStream(rest, on_event)
    asyncio.run(asyncio.wait_for(stream.run(), timeout=10))
    assert delivered == ["1", "2"] and rest.listen_keys == 1 and fake_connect.calls == 1


def test_user_stream_listen_key_expired_still_reconnects(monkeypatch):
    """The fix must not swallow the protocol-level reconnect path."""
    rest = FakeRest()
    stream: UserStream | None = None
    delivered: list[str] = []

    async def on_event(msg):
        delivered.append(msg["o"]["c"])

    async def drained():
        await stream.stop()

    msgs = [json.dumps({"e": "listenKeyExpired"}), json.dumps(_otu("BTCUSDT", "after"))]
    fake_connect = FakeConnect(msgs, drained)
    monkeypatch.setattr(binance_ws, "connect", fake_connect)
    stream = UserStream(rest, on_event)
    asyncio.run(asyncio.wait_for(stream.run(), timeout=10))
    assert rest.listen_keys == 2 and fake_connect.calls == 2
    assert fake_connect.urls == ["wss://fake/ws/lk1", "wss://fake/ws/lk2"]
    assert delivered == ["after"]


def _frame(stream_name: str, data: dict) -> str:
    return json.dumps({"stream": stream_name, "data": data})


def test_market_stream_callback_error_does_not_reconnect(monkeypatch):
    klines: list[dict] = []
    books: list[dict] = []
    marks: list[list] = []
    stream: MarketStream | None = None

    def on_kline(d):
        klines.append(d)
        raise RuntimeError("engine on_tick blew up")

    async def on_book(d):
        books.append(d)

    def on_mark(rows):
        marks.append(rows)

    async def drained():
        await stream.stop()

    msgs = [
        _frame("btcusdt@kline_1m", {"e": "kline", "s": "BTCUSDT", "k": {"t": 1}}),
        _frame("btcusdt@bookTicker", {"e": "bookTicker", "s": "BTCUSDT", "b": "1", "a": "2"}),
        _frame("!markPrice@arr@1s", [{"e": "markPriceUpdate", "s": "BTCUSDT", "p": "1.5"}]),
    ]
    fake_connect = FakeConnect(msgs, drained)
    monkeypatch.setattr(binance_ws, "connect", fake_connect)

    stream = MarketStream("wss://fake", on_kline, on_book, on_mark)
    stream.set_symbols(["BTCUSDT"])
    asyncio.run(asyncio.wait_for(stream.run(), timeout=10))

    assert len(klines) == 1
    assert len(books) == 1 and books[0]["e"] == "bookTicker"
    assert len(marks) == 1 and marks[0][0]["s"] == "BTCUSDT"
    assert fake_connect.calls == 1, "a raising callback must not force a market stream reconnect"
    assert stream.connected is False
